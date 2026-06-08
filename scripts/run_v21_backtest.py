"""
v2.1 Execution & Risk Engine Backtest
=======================================
Evaluates the v2.1 mechanism (MCDropout + SignalPolicy) on historical data
to generate a quantitative tearsheet.
"""

import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

# Adjust path to import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.local_data_loader import load_merged_data
from src.models.lstm_features import LSTMFeatureEngineer
from src.models.lstm_preprocessor import LSTMPreprocessor
from src.models.lstm_temporal import GoldLSTMModel
from src.paper_trading.v21_execution import SignalPolicy
from scripts.cost_model import CostModel, TradeEvaluator

def run_backtest():
    logger.info("=" * 60)
    logger.info("  v2.1 EXECUTION ENGINE BACKTEST")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info("Loading recent data (last 3 months)...")
    df = load_merged_data(resample="5min")
    # Take last ~1000 bars to keep MCD inference fast
    df = df.iloc[-1000:].copy()
    logger.info(f"Loaded {len(df):,} 5-minute bars.")

    # 2. Setup v2.1 Engine
    logger.info("\nLoading v2.0 Model & Preprocessor...")
    try:
        model = GoldLSTMModel(device="cuda" if torch.cuda.is_available() else "cpu")
        model.load(str(PROJECT_ROOT / "models" / "lstm_v2.0_triple_barrier.pt"))
        preprocessor = LSTMPreprocessor.load(str(PROJECT_ROOT / "models" / "lstm_preprocessor_v2.0.joblib"))
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model/preprocessor: {e}")
        return

    policy = SignalPolicy(long_threshold=0.45, short_threshold=0.65, max_uncertainty=0.08)
    evaluator = TradeEvaluator(CostModel(spread_bps=2.0, slippage_bps=1.0, commission_bps=0.5))

    # 3. Feature Engineering
    logger.info("\nEngineering Features...")
    engineer = LSTMFeatureEngineer()
    features = engineer.transform(df)

    feature_list_path = PROJECT_ROOT / "models" / "v2_selected_features.json"
    if feature_list_path.exists():
        with open(feature_list_path, "r") as f:
            sel_feats = json.load(f)["selected_features"]
        sel_feats = [f for f in sel_feats if f in features.columns]
        features = features[sel_feats]
    else:
        logger.warning("v2_selected_features.json not found!")

    features = features.replace([np.inf, -np.inf], np.nan)
    features.dropna(inplace=True)
    df_aligned = df.loc[features.index].copy()
    prices = df_aligned["close"].values
    
    # 4. Sliding Window Inference
    logger.info(f"\nRunning MCDropout Inference over {len(features)} sequences...")
    seq_len = preprocessor.seq_len
    fwd_bars = 5

    raw_signals = []
    filtered_signals = []
    trade_returns = []
    
    t0 = time.time()
    for i in range(seq_len, len(features) - fwd_bars):
        # We process in batches of 1 for simplicity of time-series simulation
        window = features.iloc[i - seq_len : i]
        
        # Scale
        X = preprocessor.transform_live(window)
        
        # Run MCDropout
        mcd_res = model.predict_mcdropout(X, n_passes=20)  # Reduced to 20 for backtest speed
        
        # Run Policy
        final_res = policy.evaluate(mcd_res)
        
        raw_signals.append(mcd_res["signal"])
        filtered_signals.append(final_res["signal"])
        
        # Execute Trade
        if final_res["signal"] in ["LONG", "SHORT"]:
            entry = prices[i]
            exit_p = prices[i + fwd_bars]
            size = final_res["sizing_scalar"]
            
            # Simple simulation: hold for fwd_bars
            ret = evaluator.evaluate_trade(entry, exit_p, final_res["signal"])
            # Scale return by size
            trade_returns.append(ret * size)

        if i % 2000 == 0 and i > seq_len:
            elapsed = time.time() - t0
            logger.info(f"Processed {i}/{len(features)}... ({elapsed:.1f}s)")

    # 5. Tearsheet Generation
    logger.info("\n" + "=" * 60)
    logger.info("  v2.1 INSTITUTIONAL TEARSHEET (Out-Of-Sample)")
    logger.info("=" * 60)

    raw_trades = [s for s in raw_signals if s != "HOLD"]
    filt_trades = [s for s in filtered_signals if s != "HOLD"]
    
    logger.info(f"Raw LSTM Trades Attempted:   {len(raw_trades)}")
    logger.info(f"v2.1 Policy Trades Executed: {len(filt_trades)} ({(len(filt_trades)/len(raw_signals)):.1%} Turnover)")
    logger.info(f"Trades Rejected by Policy:   {len(raw_trades) - len(filt_trades)}")

    if len(trade_returns) > 0:
        returns = np.array(trade_returns)
        win_rate = (returns > 0).mean()
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else float("inf")
        net_ev = returns.mean()
        sharpe = (returns.mean() / returns.std()) * np.sqrt((252 * 24 * 12) / fwd_bars)  # Approx annualization
        
        cum_ret = np.cumsum(returns)
        drawdowns = cum_ret - np.maximum.accumulate(cum_ret)
        max_dd = drawdowns.min()

        logger.info(f"\nNet Expectancy (per trade): {net_ev * 10000:.2f} bps")
        logger.info(f"Win Rate:                   {win_rate:.1%}")
        logger.info(f"Profit Factor:              {profit_factor:.2f}")
        logger.info(f"Sharpe Ratio:               {sharpe:.2f}")
        logger.info(f"Max Drawdown:               {max_dd:.2%}")
        logger.info(f"Total Net P&L:              {cum_ret[-1]:.2%}")
    else:
        logger.warning("No trades executed by policy!")

if __name__ == "__main__":
    run_backtest()
