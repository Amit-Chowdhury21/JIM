"""
Local Multi-Asset Data Loader
==============================
Loads, merges, and resamples 18+ years of 1-minute data from the local
DataBase directory into a single aligned DataFrame suitable for LSTM training.

Supports Parquet caching for fast reloads (<5s vs 3+ minutes from raw CSVs).

Assets:
    - Gold (XAUUSD): OHLCV from DataBase/GOLD/gold_1m_*.csv
    - DXY:           Close from DataBase/DXY/dxy_1min_*.csv
    - GVZ:           Close from DataBase/GVZ/gvz_1min_*.csv
    - Silver:        OHLCV from DataBase/SILVER/xagusd_1min_*.csv
    - US 10Y Yield:  Close from DataBase/US Treasury Yield/tnx_1min_*.csv

Usage:
    from scripts.local_data_loader import load_merged_data
    df = load_merged_data(resample="5min")
"""

import os
import sys
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "DataBase"
CACHE_DIR = DATABASE_DIR / ".cache"


# ══════════════════════════════════════════════════════════════
# ASSET DEFINITIONS
# ══════════════════════════════════════════════════════════════

ASSETS = {
    "gold": {
        "folder": "GOLD",
        "pattern": "gold_1m_*.csv",
        "columns": {
            "open": "gold_open",
            "high": "gold_high",
            "low": "gold_low",
            "close": "close",       # Primary target — kept as 'close'
            "volume": "gold_volume",
        },
        "is_primary": True,
    },
    "dxy": {
        "folder": "DXY",
        "pattern": "dxy_1min_*.csv",
        "columns": {"close": "dxy"},
        "is_primary": False,
    },
    "gvz": {
        "folder": "GVZ",
        "pattern": "gvz_1min_*.csv",
        "columns": {"close": "gvz"},
        "is_primary": False,
    },
    "silver": {
        "folder": "SILVER",
        "pattern": "xagusd_1min_*.csv",
        "columns": {
            "open": "silver_open",
            "high": "silver_high",
            "low": "silver_low",
            "close": "silver",
            "volume": "silver_volume",
        },
        "is_primary": False,
    },
    "us10y": {
        "folder": "US Treasury Yield",
        "pattern": "tnx_1min_*.csv",
        "columns": {"close": "us10y"},
        "is_primary": False,
    },
}


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def _load_single_asset(asset_name: str, config: dict) -> pd.DataFrame:
    """Load all yearly CSVs for a single asset into one DataFrame."""
    folder_path = DATABASE_DIR / config["folder"]
    pattern = str(folder_path / config["pattern"])
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        logger.warning(f"[{asset_name}] No CSV files found matching {pattern}")
        return pd.DataFrame()

    logger.info(f"[{asset_name}] Loading {len(csv_files)} files from {config['folder']}/")

    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Parse timestamp
            if "timestamp" in df.columns:
                # Try DD-MM-YYYY HH:MM format first, fall back to other formats
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
                except (ValueError, TypeError):
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)

                # Strip timezone info to ensure all timestamps are tz-naive
                # (2026 CSVs from live updater may have tz-aware timestamps)
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

                df.set_index("timestamp", inplace=True)

            # Normalize column names to lowercase
            df.columns = [c.lower().strip() for c in df.columns]

            frames.append(df)
        except Exception as e:
            logger.warning(f"[{asset_name}] Failed to load {Path(csv_file).name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    # Rename columns according to mapping
    rename_map = {}
    for src_col, dst_col in config["columns"].items():
        if src_col in combined.columns:
            rename_map[src_col] = dst_col

    combined = combined.rename(columns=rename_map)

    # Keep only the mapped columns
    keep_cols = [c for c in config["columns"].values() if c in combined.columns]
    combined = combined[keep_cols]

    logger.info(f"[{asset_name}] Loaded {len(combined):,} rows "
                f"({combined.index.min()} → {combined.index.max()})")

    return combined


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a DataFrame with mixed OHLCV and scalar columns.

    OHLCV columns get proper aggregation; scalar columns get last().
    """
    agg_rules = {}

    for col in df.columns:
        col_lower = col.lower()
        if "open" in col_lower:
            agg_rules[col] = "first"
        elif "high" in col_lower:
            agg_rules[col] = "max"
        elif "low" in col_lower:
            agg_rules[col] = "min"
        elif "volume" in col_lower:
            agg_rules[col] = "sum"
        else:
            # close, dxy, gvz, us10y, silver — all take last value
            agg_rules[col] = "last"

    resampled = df.resample(freq).agg(agg_rules)
    resampled.dropna(how="all", inplace=True)

    return resampled


# ══════════════════════════════════════════════════════════════
# MAIN API
# ══════════════════════════════════════════════════════════════

def load_merged_data(
    resample: str = "5min",
    use_cache: bool = True,
    years_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Load and merge all assets, resample, and return a single DataFrame.

    Args:
        resample: Resampling frequency ('1min', '5min', '15min', '1h', '1d').
                  Use '1min' for no resampling (WARNING: 8.7M+ rows).
        use_cache: Whether to use/create Parquet cache.
        years_range: Optional (start_year, end_year) tuple to limit data.

    Returns:
        pd.DataFrame with gold OHLCV + macro columns, indexed by timestamp.
    """
    cache_name = f"merged_{resample}_all_assets.parquet"
    cache_path = CACHE_DIR / cache_name

    # ── Try cache ──
    if use_cache and cache_path.exists():
        logger.info(f"Loading cached data from {cache_path.name}...")
        df = pd.read_parquet(cache_path)
        if years_range:
            mask = (df.index.year >= years_range[0]) & (df.index.year <= years_range[1])
            df = df[mask]
        logger.info(f"Cached data: {len(df):,} rows, {len(df.columns)} columns "
                     f"({df.index.min()} → {df.index.max()})")
        return df

    # ── Load all assets ──
    logger.info("=" * 60)
    logger.info("  LOADING LOCAL 1-MINUTE DATABASE")
    logger.info("=" * 60)

    asset_dfs = {}
    for name, config in ASSETS.items():
        asset_df = _load_single_asset(name, config)
        if not asset_df.empty:
            asset_dfs[name] = asset_df

    if "gold" not in asset_dfs or asset_dfs["gold"].empty:
        raise ValueError("Gold data is required but was not found!")

    # ── Merge on timestamp ──
    logger.info("Merging assets on timestamp index...")

    # Start with gold (primary)
    merged = asset_dfs["gold"].copy()

    # Join other assets
    for name, df in asset_dfs.items():
        if name == "gold":
            continue
        # Use outer join to keep all gold timestamps, fill missing with NaN
        merged = merged.join(df, how="left", rsuffix=f"_dup_{name}")

    # Drop any duplicate columns from join artifacts
    dup_cols = [c for c in merged.columns if "_dup_" in c]
    if dup_cols:
        merged.drop(columns=dup_cols, inplace=True)

    logger.info(f"Merged raw: {len(merged):,} rows, {len(merged.columns)} columns")

    # ── Resample ──
    if resample != "1min":
        logger.info(f"Resampling to {resample}...")
        merged = _resample_ohlcv(merged, resample)
        logger.info(f"After resampling: {len(merged):,} rows")

    # ── Forward-fill macro columns (they may have gaps) ──
    macro_cols = [c for c in merged.columns if c not in ["close", "gold_open", "gold_high", "gold_low", "gold_volume"]]
    for col in macro_cols:
        merged[col] = merged[col].ffill()

    # ── Drop rows with no gold close ──
    merged.dropna(subset=["close"], inplace=True)

    # ── Filter year range ──
    if years_range:
        mask = (merged.index.year >= years_range[0]) & (merged.index.year <= years_range[1])
        merged = merged[mask]

    # ── Rename gold OHLCV for compatibility with LSTMFeatureEngineer ──
    # The feature engineer expects: open, high, low, close, volume
    rename_back = {
        "gold_open": "open",
        "gold_high": "high",
        "gold_low": "low",
        "gold_volume": "volume",
    }
    merged.rename(columns=rename_back, inplace=True)

    # ── Add derived cross-asset columns expected by LSTMFeatureEngineer ──
    if "close" in merged.columns:
        merged["returns"] = merged["close"].pct_change()
    if "dxy" in merged.columns:
        merged["dxy_returns"] = merged["dxy"].pct_change()
    if "us10y" in merged.columns:
        merged["us10y_returns"] = merged["us10y"].pct_change()
    if "silver" in merged.columns:
        merged["silver_returns"] = merged["silver"].pct_change()
        merged["gold_silver_ratio"] = merged["close"] / (merged["silver"] + 1e-10)

    # ── Cache ──
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path)
        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Cached to {cache_path.name} ({cache_mb:.0f} MB)")

    logger.info(f"Final dataset: {len(merged):,} rows, {len(merged.columns)} columns "
                f"({merged.index.min()} → {merged.index.max()})")

    return merged


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and merge local 1-minute data")
    parser.add_argument("--resample", type=str, default="5min",
                        help="Resampling frequency (1min, 5min, 15min, 1h, 1d)")
    parser.add_argument("--no-cache", action="store_true", help="Force reload from CSVs")
    parser.add_argument("--info", action="store_true", help="Print dataset info and exit")
    args = parser.parse_args()

    df = load_merged_data(resample=args.resample, use_cache=not args.no_cache)

    if args.info:
        print(f"\nDataset Shape: {df.shape}")
        print(f"Date Range: {df.index.min()} → {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        print(f"\nSample (last 5 rows):")
        print(df.tail())
