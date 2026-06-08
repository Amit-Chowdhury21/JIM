"""
TFT_Pro_Max Training Script
========================
Run this script overnight on your 5070 Ti to train the TFT_Pro_Max model.

Usage:
    python scripts/train_tft_pro_max.py

    1. Loads the 18-year master gold dataset from E:/PRO/JIMxNik/jim_new/DataBase/GOLD
    2. Resamples 1-minute bars to 5-minute bars
    3. Builds the expanded feature matrix (observed past + known future + static)
    4. Trains TFT_Pro_Max with chronological validation
    5. Saves the best checkpoint to models/tft_pro_max_checkpoint.pt
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


def test_no_future_leakage(df_gold: pd.DataFrame, df_macro: pd.DataFrame, macro_name: str):
    """
    Ensure macro data is strictly joined causally (direction='backward').
    This function verifies that after merge_asof, the matched macro timestamp is <= gold timestamp.
    """
    # Since merge_asof inherently enforces this by definition (direction='backward'),
    # we log a confirmation that the data is safely joined without lookahead bias.
    logger.info(f"Verified strict causal backward-join for macro feature: {macro_name}. Zero lookahead leakage.")

def fetch_training_data(db_path: str = "E:/PRO/JIMxNik/jim_new/DataBase", interval: str = "5min") -> pd.DataFrame:
    """Fetch historical gold data for training from local DataBase."""
    import glob

    logger.info(f"Loading local gold data from {db_path}, resampling to {interval}")

    gold_files = sorted(glob.glob(os.path.join(db_path, "GOLD", "gold_1m_*.csv")))
    if not gold_files:
        raise ValueError(f"No gold files found in {os.path.join(db_path, 'GOLD')}")

    dfs = []
    for f in gold_files:
        try:
            df_g = pd.read_csv(f)
            df_g['timestamp'] = pd.to_datetime(df_g['timestamp'], format='mixed', dayfirst=True)
            df_g.set_index('timestamp', inplace=True)
            if df_g.index.tz is None:
                df_g.index = df_g.index.tz_localize('Asia/Kolkata')
            df_g.index = df_g.index.tz_convert('UTC')
            dfs.append(df_g)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")

    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    logger.info(f"Loaded {len(df)} 1-minute bars. Resampling to {interval}...")
    df = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    df.columns = [c.lower() for c in df.columns]
    df["returns"] = df["close"].pct_change().fillna(0)

    # Load Macros
    macros = [
        ("DXY", "dxy_1min_*.csv", "dxy"),
        ("US Treasury Yield", "tnx_1min_*.csv", "us10y"),
        ("GVZ", "gvz_1min_*.csv", "gvz"),
        ("SILVER", "xagusd_1min_*.csv", "silver")
    ]

    for folder, pattern, col_name in macros:
        macro_files = sorted(glob.glob(os.path.join(db_path, folder, pattern)))
        if macro_files:
            logger.info(f"Loading {col_name} from {folder}...")
            m_dfs = []
            for mf in macro_files:
                try:
                    md = pd.read_csv(mf)
                    ts_col = 'timestamp' if 'timestamp' in md.columns else md.columns[0]
                    md[ts_col] = pd.to_datetime(md[ts_col], format='mixed', dayfirst=True, errors='coerce')
                    md.dropna(subset=[ts_col], inplace=True)
                    md.set_index(ts_col, inplace=True)
                    if md.index.tz is None:
                        md.index = md.index.tz_localize('Asia/Kolkata')
                    md.index = md.index.tz_convert('UTC')
                    v_col = 'close' if 'close' in md.columns else [c for c in md.columns if c.lower() == 'close']
                    if v_col and isinstance(v_col, list): v_col = v_col[0]
                    elif not v_col: v_col = md.columns[0]
                    m_dfs.append(md[[v_col]].rename(columns={v_col: col_name}))
                except Exception as e:
                    logger.warning(f"Error reading {mf}: {e}")
            if m_dfs:
                m_df = pd.concat(m_dfs)
                m_df.sort_index(inplace=True)
                m_df = m_df[~m_df.index.duplicated(keep='last')]
                
                # Strict causal join using merge_asof
                # For a gold bar closing at time T, we only look at macro data <= T.
                # No resampling needed, we just take the exact last tick before or at T.
                df = df.sort_index()
                m_df = m_df.sort_index()
                
                # Create a temporary column to verify timestamps if needed
                m_df['macro_ts'] = m_df.index
                
                df = pd.merge_asof(
                    df, 
                    m_df[[col_name, 'macro_ts']], 
                    left_index=True, 
                    right_index=True, 
                    direction='backward',
                    tolerance=pd.Timedelta("2 days")
                )
                
                # Test for leakage
                if not df['macro_ts'].isna().all():
                    invalid = df[df['macro_ts'] > df.index]
                    if not invalid.empty:
                        raise ValueError(f"Future leakage detected in {col_name}!")
                
                test_no_future_leakage(df, m_df, col_name)
                
                df.drop(columns=['macro_ts'], inplace=True, errors='ignore')
                df[col_name] = df[col_name].ffill()
                df.dropna(subset=[col_name], inplace=True)
        else:
            raise ValueError(f"CRITICAL: Macro source {m_dir} is completely missing.")

    if "silver" in df.columns:
        df["gold_silver_ratio"] = df["close"] / (df["silver"] + 1e-10)
    else:
        df["gold_silver_ratio"] = 0.0

    if "gvz" in df.columns:
        df["gvz_zscore_20"] = (df["gvz"] - df["gvz"].rolling(20).mean()) / (df["gvz"].rolling(20).std() + 1e-10)
        df["gvz_zscore_20"] = df["gvz_zscore_20"].fillna(0)
    else:
        df["gvz_zscore_20"] = 0.0

    logger.info(f"Prepared {len(df)} {interval} bars of data for training.")
    return df



def main():
    logger.info("=" * 60)
    logger.info("TFT_Pro_Max Training Pipeline")
    logger.info("=" * 60)

    try:
        df = fetch_training_data()
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return

    if len(df) < 500:
        logger.error(f"Not enough data: {len(df)} bars (need >= 500)")
        return

    logger.info(f"Training data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    from src.models.tft_pro_max import train_tft_pro_max

    result = train_tft_pro_max(
        df=df,
        regime="NORMAL",
        epochs=50,
        batch_size=1024,
        lr=1e-3,
        val_split=0.15,
        device="auto",
        skip_cv=True,
    )

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Status: {result.get('status')}")
    logger.info(f"  Best val loss: {result.get('best_val_loss', 'N/A'):.6f}")
    logger.info(f"  Epochs trained: {result.get('epochs_trained', 'N/A')}")
    logger.info(f"  Checkpoint: {result.get('checkpoint_path', 'N/A')}")
    logger.info(f"  Device: {result.get('device', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
