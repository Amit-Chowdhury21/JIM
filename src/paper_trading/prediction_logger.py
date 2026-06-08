"""
Prediction Cycle CSV Logger
=============================
Logs every model prediction cycle into a CSV for analysis,
backtesting validation, and P&L tracking.

One row per cycle. P&L is filled retroactively when a trade closes.

CSV Location: logs/prediction_log.csv
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional
from loguru import logger

# ============================================================================
# CONFIG
# ============================================================================

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
CSV_PATH = os.path.join(CSV_DIR, "prediction_log.csv")

HEADER = [
    "timestamp",
    "price",
    "regime",
    "wavelet_pro_signal", "wavelet_pro_conf",
    "hmm_pro_signal", "hmm_pro_conf",
    "lstm_signal", "lstm_conf",
    "tft_pro_signal", "tft_pro_conf",
    "ensemble_signal", "ensemble_conf",
    "kelly_fraction",
    "trade_taken",
    "pnl",
]


# ============================================================================
# INITIALIZATION — write header if file doesn't exist
# ============================================================================

def _ensure_csv_exists():
    """Create the CSV with a header row if it doesn't already exist."""
    os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        logger.info(f"Created prediction log: {CSV_PATH}")


# Run once on import
_ensure_csv_exists()


# ============================================================================
# MAIN LOG FUNCTION — called once per prediction cycle
# ============================================================================

def log_prediction_cycle(
    price: float,
    regime: str,
    all_signals: Dict[str, Dict],
    kelly_fraction: Optional[float] = None,
    trade_taken: bool = False,
):
    """
    Append one row to the prediction CSV.

    Args:
        price:          Current gold price at this cycle.
        regime:         Market regime (GROWTH / NORMAL / CRISIS).
        all_signals:    Dict of {model_name: {signal, confidence, ...}}
                        Must contain keys: wavelet_pro, wavelet_basic, hmm, lstm, tft, genetic, hmm_pro, ensemble
        kelly_fraction: Kelly fraction used for position sizing (if trade taken).
        trade_taken:    Whether the ensemble signal triggered a trade this cycle.
    """
    try:
        def _sig(model: str) -> str:
            return all_signals.get(model, {}).get("signal", "HOLD")

        def _conf(model: str) -> float:
            return float(all_signals.get(model, {}).get("confidence", 0.0))

        row = [
            datetime.now().isoformat(timespec="seconds"),
            round(price, 2),
            regime,
            _sig("wavelet_pro"),    round(_conf("wavelet_pro"), 4),
            _sig("hmm_pro"),        round(_conf("hmm_pro"), 4),
            _sig("lstm"),           round(_conf("lstm"), 4),
            _sig("tft_pro"),        round(_conf("tft_pro"), 4),
            _sig("ensemble"),       round(_conf("ensemble"), 4),
            round(kelly_fraction, 4) if kelly_fraction is not None else "",
            "YES" if trade_taken else "NO",
            "",  # pnl — filled retroactively
        ]

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    except Exception as e:
        logger.warning(f"Failed to log prediction cycle: {e}")


# ============================================================================
# RETROACTIVE P&L UPDATE — called when a trade closes
# ============================================================================

def update_pnl_for_trade(trade_timestamp_prefix: str, pnl: float):
    """
    Find the row whose timestamp starts with `trade_timestamp_prefix`
    and fill in the pnl column.

    This is called by the paper trading engine when a position is closed.

    Args:
        trade_timestamp_prefix: ISO timestamp prefix to match (e.g. "2026-05-22T19:18")
        pnl:                   Realized P&L of the closed trade.
    """
    try:
        if not os.path.exists(CSV_PATH):
            return

        # Read all rows
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) <= 1:
            return  # Only header, nothing to update

        header = rows[0]
        pnl_idx = header.index("pnl")
        trade_idx = header.index("trade_taken")

        # Walk backwards (most recent first) to find the matching row
        updated = False
        for i in range(len(rows) - 1, 0, -1):
            row = rows[i]
            if (len(row) > trade_idx
                    and row[trade_idx] == "YES"
                    and (len(row) <= pnl_idx or row[pnl_idx] == "")):
                row[pnl_idx] = str(round(pnl, 4))
                updated = True
                break

        if updated:
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            logger.debug(f"Updated P&L={pnl:+.2f} for cycle near {trade_timestamp_prefix}")

    except Exception as e:
        logger.warning(f"Failed to update P&L in prediction log: {e}")


def get_csv_path() -> str:
    """Return the absolute path to the prediction log CSV."""
    return CSV_PATH
