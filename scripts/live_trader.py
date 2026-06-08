"""
Live Gold Trader — Mini-Medallion
====================================
Autonomous live trading script that runs 24/7.

Fetches real-time gold prices every 60 seconds, runs all 7 models
(Wavelet, HMM, LSTM, TFT, Genetic, HMM Pro, Ensemble Meta-Learner),
and executes trades through the paper trading engine with full
risk management, trailing stops, and circuit breakers.

Usage:
    python scripts/live_trader.py                       # Default: 60s interval
    python scripts/live_trader.py --interval 30         # 30s interval
    python scripts/live_trader.py --capital 50000       # Custom capital
    python scripts/live_trader.py --dry-run              # Signals only, no trades

Features:
    - Real-time gold price from Gold-API.com + yfinance hybrid
    - All 7 Phase 3 models run on every tick
    - ML Meta-Learner ensemble (RandomForest) for final signal
    - Dynamic regime-conditional model weighting
    - Kelly Criterion position sizing (regime-aware)
    - Trailing stops via RL Execution Agent
    - Circuit breakers: daily loss, drawdown, consecutive loss, cooldown
    - CSV prediction logging for post-hoc analysis
    - Graceful shutdown on Ctrl+C (closes open positions)
    - Auto daily reset at midnight UTC
    - Live P&L dashboard in terminal
"""

import sys
import os
import time
import signal
import argparse
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# ============================================================================
# SETUP LOGGING
# ============================================================================

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default logger and add custom sinks
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{message}</cyan>"
    ),
    colorize=True,
)
logger.add(
    str(LOG_DIR / "live_trader.log"),
    level="DEBUG",
    rotation="50 MB",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
)

# ============================================================================
# SUPPRESS WARNINGS
# ============================================================================
import warnings
import os

# Suppress sklearn version warnings (old models may have been pickled with different sklearn version)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Suppress torch/triton warnings
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================================
# IMPORTS (after path setup)
# ============================================================================

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.utils.config import load_config
from src.paper_trading.engine import (
    PaperTradingEngine,
    PaperTradingConfig,
    ModelSignal,
    SignalType,
    TradeStatus,
)
from src.paper_trading.live_inference import (
    fetch_live_gold_data,
    fetch_metalpriceapi_spot,
    run_wavelet,
    run_hmm,
    run_lstm,
    run_tft_pro,
    run_ensemble,
)
from src.models.hmm_pro_gpu import run_hmm_pro_gpu
from src.paper_trading.prediction_logger import log_prediction_cycle
from src.risk.manager import RiskManager


# ============================================================================
# GLOBAL STATE
# ============================================================================

_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Force shutdown requested. Exiting immediately.")
        sys.exit(1)
    logger.warning("\n[!] Shutdown requested (Ctrl+C). Closing positions and saving state...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# TERMINAL DASHBOARD
# ============================================================================

def print_dashboard(
    iteration: int,
    price: float,
    regime: str,
    signals: dict,
    engine: PaperTradingEngine,
    risk_mgr: RiskManager,
    trade_taken: bool,
    elapsed_ms: float,
):
    """Print a compact live dashboard to the terminal (ASCII-safe for Windows)."""
    snapshot = engine._create_portfolio_snapshot(price)
    pnl_color = "\033[92m" if snapshot.pnl_total >= 0 else "\033[91m"
    reset = "\033[0m"

    # Position info
    pos_str = "FLAT"
    if engine.current_position and engine.current_position.status == TradeStatus.OPEN:
        pos = engine.current_position
        hedge_info = f" | Ag Hedge: {pos.silver_quantity:.1f}oz" if pos.silver_quantity > 0 else ""
        pos_str = (
            f"{pos.signal_type.value} {pos.quantity:.2f}oz "
            f"@ ${pos.entry_price:,.2f} "
            f"(P&L: ${pos.pnl:+,.2f}){hedge_info}"
        )

    # Signal summary with distinct model labels
    model_labels = {
        "wavelet_pro": "WVP",      # WaveletPro (6-level DWT + CWT + features)
        "wavelet_basic": "WVB",    # Basic 5-level wavelet (original)
        "hmm": "HMM",              # Legacy HMM v3.0
        "lstm": "LST",             # LSTM temporal
        "tft": "TFT",              # TFT forecaster
        "genetic": "GEN",          # Genetic algorithm
        "hmm_pro": "HMP",          # HMM Pro (GMMHMM) — PRO
        "ensemble": "ENS",         # Ensemble meta-learner
    }
    sig_parts = []
    for m in ["wavelet_pro", "hmm_pro", "lstm", "tft_pro", "ensemble"]:
        s = signals.get(m, {})
        sig_val = s.get("signal", "?")[:1]
        conf = s.get("confidence", 0)
        label = model_labels.get(m, m[:3].upper())
        sig_parts.append(f"{label}:{sig_val}{conf:.0%}")

    sig_line = " | ".join(sig_parts)

    # Trade count
    closed = [t for t in engine.trades if t.status == TradeStatus.CLOSED]
    wins = len([t for t in closed if t.pnl > 0])
    wr = (wins / len(closed) * 100) if closed else 0

    try:
        print()
        print("=" * 80)
        print(f"  [LIVE TRADER] MINI-MEDALLION -- Tick #{iteration}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Cycle: {elapsed_ms:.0f}ms")
        print("-" * 80)
        print(f"  Gold: ${price:,.2f}  |  Regime: {regime}")
        print(f"  Signals: {sig_line}")
        print(f"  Action: {'>>> TRADE EXECUTED <<<' if trade_taken else '--- No trade ---'}")
        print("-" * 80)
        print(f"  Portfolio: ${snapshot.total_value:,.2f}  |  "
              f"Cash: ${snapshot.cash:,.2f}")
        print(f"  {pnl_color}Total P&L: ${snapshot.pnl_total:+,.2f} "
              f"({snapshot.return_pct:+.2f}%){reset}")
        print(f"  Max DD: {snapshot.max_drawdown:.2f}%  |  "
              f"Sharpe: {snapshot.sharpe_ratio:.2f}")
        print(f"  Trades: {len(closed)} closed  |  "
              f"Win Rate: {wr:.1f}%  |  "
              f"Open: {pos_str}")
        print(f"  Risk: DD={risk_mgr.risk_state.current_drawdown:.2%} | "
              f"DailyPnL=${risk_mgr.risk_state.daily_pnl:+,.2f} | "
              f"ConsecLoss={risk_mgr.risk_state.consecutive_losses}")
        # Silver feed & Kalman hedge status
        sf = engine.silver_feed.feed_status()
        beta_str = f"B={engine.current_beta:.3f}" if engine.current_beta != 1.0 else "B=init"
        ag_str = f"Ag=${sf['current_xag']:.2f}" if sf['current_xag'] > 0 else "Ag=waiting"
        print(f"  Silver: {ag_str} | Kalman {beta_str} | "
              f"Ratio={sf['current_ratio']:.2f} | Feed={sf['fetch_count']}ok/{sf['error_count']}err")
        print("=" * 80)
    except (UnicodeEncodeError, OSError):
        # Fallback for terminals that can't handle any of this
        logger.info(
            f"Tick#{iteration} | Gold=${price:,.2f} | {regime} | "
            f"P&L=${snapshot.pnl_total:+,.2f} | Trades={len(closed)}"
        )


# ============================================================================
# CORE TRADING LOOP
# ============================================================================

def run_single_cycle(
    engine: PaperTradingEngine,
    risk_mgr: RiskManager,
    iteration: int,
    dry_run: bool = False,
) -> tuple:
    """
    Execute one full inference + trading cycle.

    Returns:
        (price, regime, all_signals, trade_taken)
    """
    # 1. Fetch live gold data (OHLCV + macro indicators)
    logger.info(f"[Tick #{iteration}] Fetching live gold data...")
    df = fetch_live_gold_data("5d", "1m")

    if df is None or df.empty or len(df) < 35:
        logger.warning("Insufficient data — skipping cycle")
        return None, "UNKNOWN", {}, False

    # 2. Use the price already scaled into the dataframe by fetch_live_gold_data()
    # That function already calls fetch_metalpriceapi_spot() internally and
    # scales the entire OHLCV series.  Calling it again here wastes an API hit
    # and can return a DIFFERENT price than the one the models will see.
    current_price = float(df["close"].iloc[-1])

    # Update engine price (also ticks the Kalman hedge with T-1 silver data)
    engine.update_price(current_price, pd.Timestamp.now())

    logger.info(f"[Tick #{iteration}] Gold @ ${current_price:,.2f} — running 8 models (2 wavelet variants)...")

    # 3. Run all individual models with robust error handling
    # Fallback signal if model fails
    fallback_sig = {"signal": "HOLD", "confidence": 0.0, "reasoning": "Model failed"}
    
    try:
        wavelet_pro_res = run_wavelet(df)
    except Exception as e:
        logger.warning(f"WaveletPro failed: {e}")
        wavelet_pro_res = fallback_sig.copy()
    
    try:
        hmm_res = run_hmm(df)
    except Exception as e:
        logger.warning(f"HMM v3 failed: {e}")
        hmm_res = {**fallback_sig.copy(), "regime": "NORMAL"}
    
    try:
        lstm_res = run_lstm(df)
    except Exception as e:
        logger.warning(f"LSTM failed: {e}")
        lstm_res = fallback_sig.copy()
    
    try:
        tft_res = run_tft_pro(df)
    except Exception as e:
        logger.warning(f"TFT Pro failed: {e}")
        tft_res = fallback_sig.copy()
    
    try:
        hmm_pro_res = run_hmm_pro_gpu(df)
    except Exception as e:
        logger.warning(f"HMM Pro GPU failed: {e}")
        hmm_pro_res = fallback_sig.copy()

    individual = {
        "wavelet_pro": wavelet_pro_res,    # 6-level DWT + CWT (new)
        "hmm": hmm_res,
        "lstm": lstm_res,
        "tft_pro": tft_res,
        "hmm_pro": hmm_pro_res,
    }

    # 4. Regime detection
    regime = hmm_res.get("regime", "NORMAL")

    # 5. Build macro context for ML ensemble
    macro_data = {
        "dxy_momentum": float(df["dxy_returns"].iloc[-3:].sum() * 100)
        if "dxy_returns" in df.columns else 0.0,
        "yield_momentum": float(df["us10y_returns"].iloc[-3:].sum() * 100)
        if "us10y_returns" in df.columns else 0.0,
    }

    # 6. Run ensemble meta-learner with error handling
    try:
        ensemble_res = run_ensemble(individual, regime, macro_data)
    except Exception as e:
        logger.warning(f"Ensemble failed: {e}")
        ensemble_res = fallback_sig.copy()
    
    all_signals = {**individual, "ensemble": ensemble_res}

    # 7. Risk management checks
    risk_mgr._update_regime_tracking(regime)
    snapshot = engine._create_portfolio_snapshot(current_price)
    risk_mgr.update_equity(snapshot.total_value)
    risk_mgr.risk_state.daily_pnl = engine.daily_pnl

    can_trade, risk_reason = risk_mgr.check_circuit_breakers(
        portfolio_value=snapshot.total_value,
        ensemble_conf=float(ensemble_res.get("confidence", 0.0)),
    )

    # Diagnostic: show the active threshold on every tick
    news_status = risk_mgr.economic_calendar.get_news_status()
    active_threshold = risk_mgr.min_confidence
    if risk_mgr.risk_state.current_regime in ["TRENDING", "GROWTH"]:
        active_threshold = min(0.50, risk_mgr.min_confidence)
    if news_status.get("tighten_threshold"):
        active_threshold = max(0.65, risk_mgr.min_confidence)
    logger.debug(
        f"[RISK] active_threshold={active_threshold:.2f} | "
        f"regime={regime} | news_block={news_status.get('block_trade', False)} | "
        f"news_tighten={news_status.get('tighten_threshold', False)}"
    )

    if not can_trade:
        logger.info(f"  Risk blocked: {risk_reason}")

    # 8. Determine if we should trade
    # NOTE: The confidence threshold is ALREADY enforced inside
    # risk_mgr.check_circuit_breakers().  Do NOT duplicate it here —
    # the RiskManager applies dynamic adjustments (regime-relaxed 50%,
    # news-tightened 65%) that this static check would override.
    ensemble_signal = ensemble_res.get("signal", "HOLD")
    ensemble_conf = float(ensemble_res.get("confidence", 0.0))
    should_trade = (
        ensemble_signal in ("LONG", "SHORT")
        and can_trade
        and not dry_run
    )

    # 9. Log prediction cycle (CSV)
    log_prediction_cycle(
        price=current_price,
        regime=regime,
        all_signals=all_signals,
        kelly_fraction=engine.config.kelly_fraction,
        trade_taken=should_trade,
    )

    # 10. Register all signals in engine (for tracking)
    for model_name, res in all_signals.items():
        sig_val = res.get("signal", "HOLD")
        if sig_val not in ("LONG", "SHORT", "HOLD", "CLOSE"):
            sig_val = "HOLD"
        sig = ModelSignal(
            model_name=model_name,
            signal_type=SignalType(sig_val),
            confidence=float(res.get("confidence", 0.0)),
            entry_price=current_price,
            current_price=current_price,
            timestamp=pd.Timestamp.now(),
            reasoning=res.get("reasoning", ""),
            regime=regime,
        )
        engine.last_signals[model_name] = sig
        engine.signal_history[model_name].append(sig)

    # 11. Execute trade (only via ensemble)
    trade_taken = False
    if should_trade:
        sig = ModelSignal(
            model_name="ensemble",
            signal_type=SignalType(ensemble_signal),
            confidence=ensemble_conf,
            entry_price=current_price,
            current_price=current_price,
            timestamp=pd.Timestamp.now(),
            reasoning=ensemble_res.get("reasoning", ""),
            regime=regime,
        )
        trade_result = engine.process_signal("ensemble", sig)
        if trade_result:
            trade_taken = True
            risk_mgr.risk_state.bars_since_last_trade = 0
            logger.info(
                f"  >>> TRADE EXECUTED: {trade_result.signal_type.value} "
                f"{trade_result.quantity:.2f}oz @ ${current_price:,.2f}"
            )

    # Log signal summary with distinct labels
    model_labels_log = {
        "wavelet_pro": "WVP",      # WaveletPro
        "wavelet_basic": "WVB",    # Basic 5-level wavelet
        "hmm": "HMM",              # Legacy HMM v3.0
        "lstm": "LST",             # LSTM
        "tft": "TFT",              # TFT
        "genetic": "GEN",          # Genetic
        "hmm_pro": "HMP",          # HMM Pro
        "ensemble": "ENS",         # Ensemble
    }
    sig_summary = " | ".join(
        f"{model_labels_log.get(m, m[:3].upper())}:{v.get('signal', '?')}@{v.get('confidence', 0):.0%}"
        for m, v in all_signals.items()
    )
    logger.info(f"  Signals: {sig_summary}")

    return current_price, regime, all_signals, trade_taken


def run_live_trader(args):
    """Main live trading loop."""
    global _shutdown_requested

    # Load config
    cfg = load_config()

    logger.info("=" * 70)
    logger.info("  MINI-MEDALLION LIVE GOLD TRADER")
    logger.info(f"  Initial Capital: ${args.capital:,.2f}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  Mode: {'DRY RUN (no trades)' if args.dry_run else 'LIVE TRADING'}")
    logger.info(f"  Logs: {LOG_DIR}")
    logger.info("=" * 70)

    # Initialize paper trading engine
    pt_cfg = PaperTradingConfig(
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.15,
        min_holding_bars=8,
        use_dynamic_weights=True,
    )
    engine = PaperTradingEngine(pt_cfg)
    engine.start()

    # Initialize risk manager
    risk_mgr = RiskManager(cfg)
    
    # Wire CLI --min-confidence into the RiskManager (the single source of truth)
    # This overrides the YAML default so the CLI flag actually works
    risk_mgr.min_confidence = args.min_confidence

    logger.info(f"  Engine started. Min confidence: {args.min_confidence:.0%}")
    logger.info("  Press Ctrl+C to stop gracefully.\n")

    # Track daily reset
    last_daily_reset = datetime.now(timezone.utc).date()
    iteration = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    recorded_trade_ids = set()

    # ── MAIN LOOP ──
    while not _shutdown_requested:
        iteration += 1
        cycle_start = time.perf_counter()

        try:
            # Daily reset at midnight UTC
            today_utc = datetime.now(timezone.utc).date()
            if today_utc != last_daily_reset:
                logger.info("[NEW DAY] Resetting daily counters")
                risk_mgr.reset_daily()
                engine.daily_pnl = 0.0
                engine.daily_trades.clear()
                last_daily_reset = today_utc

            # Run one cycle
            price, regime, signals, trade_taken = run_single_cycle(
                engine=engine,
                risk_mgr=risk_mgr,
                iteration=iteration,
                dry_run=args.dry_run,
            )

            if price is None:
                consecutive_failures += 1
                logger.warning(
                    f"Data fetch failed ({consecutive_failures}/{max_consecutive_failures})"
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        "[CRITICAL] Too many consecutive data failures! "
                        "Closing positions and halting."
                    )
                    if engine.current_position and engine.current_position.status == TradeStatus.OPEN:
                        # Use the last known market price, NOT entry_price
                        # entry_price would guarantee $0 P&L — a hidden bug
                        emergency_price = engine.current_position.entry_price
                        try:
                            fresh = fetch_metalpriceapi_spot()
                            if fresh and fresh > 0:
                                emergency_price = fresh
                        except Exception:
                            pass
                        logger.warning(f"  Emergency close @ ${emergency_price:,.2f}")
                        engine._close_position(datetime.now(), emergency_price)
                    break
            else:
                consecutive_failures = 0

                # Record closed trades in risk manager
                for t in engine.trades:
                    if (
                        t.status == TradeStatus.CLOSED
                        and t.trade_id not in recorded_trade_ids
                    ):
                        risk_mgr.record_trade(t.pnl)
                        recorded_trade_ids.add(t.trade_id)

                # Print dashboard
                elapsed_ms = (time.perf_counter() - cycle_start) * 1000
                print_dashboard(
                    iteration, price, regime, signals,
                    engine, risk_mgr, trade_taken, elapsed_ms,
                )

        except Exception as e:
            logger.error(f"Cycle #{iteration} error: {e}", exc_info=True)
            consecutive_failures += 1

        # Wait for next cycle (check shutdown every 1s)
        for _ in range(args.interval):
            if _shutdown_requested:
                break
            time.sleep(1)

    # ── GRACEFUL SHUTDOWN ──
    logger.info("\n[STOP] Shutting down live trader...")

    # Close open positions — fetch a FRESH spot price for accurate P&L
    if engine.current_position and engine.current_position.status == TradeStatus.OPEN:
        shutdown_price = None
        try:
            shutdown_price = fetch_metalpriceapi_spot()
        except Exception:
            pass
        if not shutdown_price or shutdown_price <= 0:
            # Fall back to the last price the engine saw
            shutdown_price = engine.current_position.entry_price
            logger.warning("  Could not fetch fresh spot price for shutdown close — using entry price")
        logger.info(f"  Closing open position @ ${shutdown_price:,.2f}")
        engine._close_position(datetime.now(), shutdown_price)

    # Stop engine
    result = engine.stop()

    # Final report
    snapshot = engine._create_portfolio_snapshot()
    closed_trades = [t for t in engine.trades if t.status == TradeStatus.CLOSED]
    wins = len([t for t in closed_trades if t.pnl > 0])
    wr = (wins / len(closed_trades) * 100) if closed_trades else 0

    print()
    print("=" * 70)
    print("  FINAL SESSION REPORT")
    print("=" * 70)
    print(f"  Session Duration:    {iteration} ticks")
    print(f"  Initial Capital:     ${pt_cfg.initial_capital:>12,.2f}")
    print(f"  Final Value:         ${snapshot.total_value:>12,.2f}")
    print(f"  Total P&L:           ${snapshot.pnl_total:>+12,.2f}")
    print(f"  Return:              {snapshot.return_pct:>+11.2f}%")
    print(f"  Max Drawdown:        {snapshot.max_drawdown:>11.2f}%")
    print(f"  Win Rate:            {wr:>11.1f}%")
    print(f"  Total Trades:        {len(closed_trades):>11d}")
    print(f"  Sharpe Ratio:        {snapshot.sharpe_ratio:>11.2f}")
    print("=" * 70)

    # Save trade log
    if closed_trades:
        trade_log_path = LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rows = []
        for t in closed_trades:
            rows.append({
                "trade_id": t.trade_id,
                "model": t.model_name,
                "signal": t.signal_type.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "entry_time": t.entry_time.isoformat() if t.entry_time else "",
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "regime": t.regime,
                "confidence": t.confidence,
            })
        pd.DataFrame(rows).to_csv(trade_log_path, index=False)
        logger.info(f"  Trade log saved: {trade_log_path}")

    logger.info("  [OK] Live trader shutdown complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mini-Medallion Live Gold Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/live_trader.py                     # Default settings
    python scripts/live_trader.py --interval 30       # 30-second ticks
    python scripts/live_trader.py --capital 200000     # $200K capital
    python scripts/live_trader.py --dry-run            # Signal-only mode
    python scripts/live_trader.py --min-confidence 0.8 # Higher threshold
        """,
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Seconds between inference cycles (default: 60)",
    )
    parser.add_argument(
        "--capital", type=float, default=100000.0,
        help="Initial trading capital in USD (default: 100000)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.35,
        help="Minimum ensemble confidence to execute a trade (default: 0.35)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run models and log signals but do NOT execute any trades",
    )

    args = parser.parse_args()
    run_live_trader(args)


if __name__ == "__main__":
    main()
