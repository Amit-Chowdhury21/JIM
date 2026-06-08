"""
Asset 1-Minute Data Manager
===========================
Fetches missing and live 1-minute data for DXY, GVZ, Silver, and TNX.
Saves data into respective CSV files seamlessly.

Usage:
    python scripts/asset_1m_data_manager.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from loguru import logger
from tvDatafeed import TvDatafeed, Interval

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS = {
    "DXY": {
        "symbol": "DXY",
        "exchange": "CAPITALCOM",
        "csv_path": PROJECT_ROOT / "DataBase" / "DXY" / "dxy_1min_2026.csv",
        "columns": ["close"]
    },
    "GVZ": {
        "symbol": "GVZ",
        "exchange": "CBOE",
        "csv_path": PROJECT_ROOT / "DataBase" / "GVZ" / "gvz_1min_2026.csv",
        "columns": ["close"]
    },
    "SILVER": {
        "symbol": "SILVER",
        "exchange": "TVC",
        "csv_path": PROJECT_ROOT / "DataBase" / "SILVER" / "xagusd_1min_2026.csv",
        "columns": ["open", "high", "low", "close", "volume"]
    },
    "TNX": {
        "symbol": "US10Y",
        "exchange": "TVC",
        "csv_path": PROJECT_ROOT / "DataBase" / "US Treasury Yield" / "tnx_1min_2026.csv",
        "columns": ["close"]
    }
}

# Initialize TradingView Datafeed (nologin)
# Suppress the spammy "you are using nologin method" stdout message
import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    tv = TvDatafeed()

def update_asset(name: str, config: dict):
    """Fetch missing data and append to CSV for a specific asset using TradingView."""
    symbol = config["symbol"]
    exchange = config["exchange"]
    csv_path = config["csv_path"]
    
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    last_timestamp = None
    existing_df = pd.DataFrame()
    
    if csv_path.exists():
        try:
            # Read existing to find the last timestamp
            existing_df = pd.read_csv(csv_path, index_col="timestamp")
            if not existing_df.empty:
                try:
                    existing_df.index = pd.to_datetime(existing_df.index, format='%d-%m-%Y %H:%M')
                except Exception:
                    existing_df.index = pd.to_datetime(existing_df.index, utc=True)
                
                # TradingView returns data in local time if unauthenticated or tz-aware depending on settings
                # We normalize everything to UTC first
                if existing_df.index.tzinfo is None:
                    existing_df.index = existing_df.index.tz_localize('Asia/Kolkata').tz_convert('UTC')
                else:
                    existing_df.index = existing_df.index.tz_convert('UTC')
                
                target_cols = config.get("columns", ["close"])
                available_cols = [c for c in target_cols if c in existing_df.columns]
                
                if available_cols:
                    existing_df = existing_df[available_cols]
                else:
                    existing_df = pd.DataFrame()
                    
                if not existing_df.empty:
                    last_timestamp = existing_df.index.max()
        except Exception as e:
            logger.warning(f"[{name}] Could not read existing CSV: {e}")
            existing_df = pd.DataFrame()
    
    try:
        now_utc = datetime.now(timezone.utc)
        
        # Decide how many bars to fetch based on last_timestamp
        if pd.isna(last_timestamp) or last_timestamp is None:
            logger.info(f"[{name}] Fetching maximum available 1m data (5000 bars)...")
            n_bars = 5000
        else:
            # Calculate minutes elapsed since last_timestamp
            minutes_elapsed = int((now_utc - last_timestamp).total_seconds() / 60)
            
            if minutes_elapsed > 5000:
                logger.info(f"[{name}] Last timestamp too old. Fetching max 5000 bars...")
                n_bars = 5000
            else:
                logger.debug(f"[{name}] Fetching {max(10, minutes_elapsed + 5)} bars to catch up...")
                n_bars = max(10, minutes_elapsed + 5)

        # Fetch from TradingView
        new_data = tv.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=n_bars)
        
        if new_data is None or new_data.empty:
            logger.debug(f"[{name}] No new data found.")
            return

        # Ensure timezone (TradingView gives us local DatetimeIndex, but let's be safe and assume it's Asia/Kolkata since our server is there)
        # Actually, get_hist() returns index without tzinfo but it's local time of the machine.
        if new_data.index.tzinfo is None:
            new_data.index = new_data.index.tz_localize(datetime.now().astimezone().tzinfo).tz_convert('UTC')
        else:
            new_data.index = new_data.index.tz_convert('UTC')

        # Drop the 'symbol' column from TV output
        if 'symbol' in new_data.columns:
            new_data.drop(columns=['symbol'], inplace=True)
            
        # Standardize columns
        new_data.columns = [col.lower().replace(" ", "_") for col in new_data.columns]
        new_data.index.name = "timestamp"
        
        # Keep only requested columns
        target_cols = config.get("columns", ["close"])
        available_cols = [c for c in target_cols if c in new_data.columns]
        
        if available_cols:
            new_data = new_data[available_cols]
        else:
            logger.warning(f"[{name}] None of the requested columns found in fetched data!")
            return
        
        if not existing_df.empty:
            # Append and remove duplicates based on index (timestamp)
            combined = pd.concat([existing_df, new_data])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            added = len(combined) - len(existing_df)
        else:
            combined = new_data.sort_index()
            added = len(combined)
            
        if added > 0:
            # Final output must be Asia/Kolkata
            if combined.index.tzinfo is None:
                combined.index = combined.index.tz_localize('UTC')
            combined.index = combined.index.tz_convert('Asia/Kolkata')
            
            # Filter out weekends (Saturday=5, Sunday=6)
            combined = combined[combined.index.dayofweek < 5]
            
            combined.index = combined.index.strftime('%d-%m-%Y %H:%M')
            combined.to_csv(csv_path)
            logger.info(f"[{name}] Added {added} new rows. Total: {len(combined)}. Saved to {csv_path.name}")
        else:
            logger.debug(f"[{name}] Already up to date.")
            
    except Exception as e:
        logger.error(f"[{name}] Failed to update data: {e}")

def main():
    logger.info("=" * 60)
    logger.info("  REAL-TIME ASSET 1-MINUTE DATA MANAGER (TRADINGVIEW)")
    logger.info("=" * 60)
    
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"--- Tick #{iteration} ---")
        
        for asset_name, asset_config in ASSETS.items():
            update_asset(asset_name, asset_config)
            
        logger.info("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()
