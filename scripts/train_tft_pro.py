"""
TFT_Pro Training Script
========================
Run this script overnight on your 5070 Ti to train the TFT_Pro model.

Usage:
    python scripts/train_tft_pro.py

What it does:
    1. Fetches 60 days of 1-minute gold data via yfinance
    2. Builds the full feature matrix (observed past + known future + static)
    3. Trains TFT_Pro with walk-forward validation
    4. Saves the best checkpoint to models/tft_pro_checkpoint.pt

Hardware:
    - GPU: RTX 5070 Ti (training + hyperparameter tuning)
    - CPU: Ryzen 9800X3D (data preprocessing, feature engineering)
    - RAM: 32GB (sufficient for single-instrument research)
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


def fetch_training_data(period: str = "60d", interval: str = "1m") -> pd.DataFrame:
    """Fetch historical gold data for training."""
    import yfinance as yf

    logger.info(f"Fetching gold data: period={period}, interval={interval}")

    tickers = "GC=F DX-Y.NYB ^TNX SI=F"
    df_all = yf.download(tickers, period=period, interval=interval,
                         group_by="ticker", progress=True, auto_adjust=False)

    if df_all.empty or "GC=F" not in df_all:
        raise ValueError("Failed to fetch gold data from yfinance")

    # Extract gold OHLCV
    df = df_all["GC=F"].copy()
    df.columns = [c.lower() for c in df.columns]

    # Drop rows with missing close
    df = df.dropna(subset=["close"])

    # Add returns
    df["returns"] = df["close"].pct_change().fillna(0)

    # Add DXY
    if "DX-Y.NYB" in df_all:
        dxy = df_all["DX-Y.NYB"]["Close"].reindex(df.index).ffill().bfill()
        df["dxy"] = dxy
        df["dxy_returns"] = dxy.pct_change().fillna(0)
    else:
        df["dxy"] = 0.0
        df["dxy_returns"] = 0.0

    # Add US10Y
    if "^TNX" in df_all:
        us10y = df_all["^TNX"]["Close"].reindex(df.index).ffill().bfill()
        df["us10y"] = us10y
        df["us10y_returns"] = us10y.pct_change().fillna(0)
    else:
        df["us10y"] = 0.0
        df["us10y_returns"] = 0.0

    # Add gold/silver ratio
    if "SI=F" in df_all:
        silver = df_all["SI=F"]["Close"].reindex(df.index).ffill().bfill()
        df["gold_silver_ratio"] = df["close"] / (silver + 1e-10)
    else:
        df["gold_silver_ratio"] = 80.0

    logger.info(f"Fetched {len(df)} bars of gold data")
    return df


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("TFT_Pro Training Pipeline")
    logger.info("=" * 60)

    # 1. Fetch data
    try:
        df = fetch_training_data(period="7d", interval="1m")
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        logger.info("Trying 5m interval as fallback...")
        try:
            df = fetch_training_data(period="60d", interval="5m")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return

    if len(df) < 500:
        logger.error(f"Not enough data: {len(df)} bars (need >= 500)")
        return

    logger.info(f"Training data: {len(df)} bars, "
                f"{df.index[0]} to {df.index[-1]}")

    # 2. Train
    from src.models.tft_pro import train_tft_pro

    result = train_tft_pro(
        df=df,
        regime="NORMAL",
        epochs=50,
        batch_size=64,
        lr=1e-3,
        val_split=0.15,
        device="auto",
    )

    # 3. Report results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Status: {result.get('status')}")
    logger.info(f"  Best val loss: {result.get('best_val_loss', 'N/A'):.6f}")
    logger.info(f"  Epochs trained: {result.get('epochs_trained', 'N/A')}")
    logger.info(f"  Checkpoint: {result.get('checkpoint_path', 'N/A')}")
    logger.info(f"  Device: {result.get('device', 'N/A')}")
    logger.info("=" * 60)
    logger.info("Restart your API server to load the new model.")


if __name__ == "__main__":
    main()
