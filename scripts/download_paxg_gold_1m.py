import ccxt
import pandas as pd
import time
from datetime import datetime
import os
import argparse

def update_paxg_data(output_dir, symbol='PAXG/USDT', timeframe='1m', days=365):
    """
    Downloads historical OHLCV data for PAXG/USDT from Binance or updates existing data.
    """
    filename = "gold_1m_2026.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Initialize the exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    now = exchange.milliseconds()
    
    existing_df = None
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_csv(filepath)
            
            # CORRUPTION GUARD: If file has fewer than 1000 rows, it was likely
            # corrupted by Excel overwriting. Treat as missing and re-download.
            if len(existing_df) < 1000:
                print(f"[WARNING] File has only {len(existing_df)} rows — likely corrupted by Excel. Re-downloading from scratch...")
                existing_df = None
                start_dt = pd.to_datetime(f"{datetime.now().year}-01-01 00:00:00").tz_localize('Asia/Kolkata')
                start_time = int(start_dt.timestamp() * 1000)
            elif not existing_df.empty:
                last_time_str = existing_df['timestamp'].iloc[-1]
                # If it's already in the new format DD-MM-YYYY HH:MM
                if isinstance(last_time_str, str) and '-' in last_time_str:
                    try:
                        last_dt = pd.to_datetime(last_time_str, format='%d-%m-%Y %H:%M').tz_localize('Asia/Kolkata')
                    except ValueError:
                        # Fallback for old format YYYY-MM-DD HH:MM:SS
                        last_dt = pd.to_datetime(last_time_str).tz_localize('Asia/Kolkata')
                    start_time = int(last_dt.timestamp() * 1000) + 60000
                else:
                    # Fallback if it's still an integer timestamp
                    start_time = int(last_time_str) + 60000
            else:
                start_dt = pd.to_datetime(f"{datetime.now().year}-01-01 00:00:00").tz_localize('Asia/Kolkata')
                start_time = int(start_dt.timestamp() * 1000)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            return False
    else:
        print(f"Starting fresh download for {symbol} from beginning of {datetime.now().year}...")
        start_dt = pd.to_datetime(f"{datetime.now().year}-01-01 00:00:00").tz_localize('Asia/Kolkata')
        start_time = int(start_dt.timestamp() * 1000)

    if start_time >= now:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data is already up to date!")
        return True

    all_ohlcv = []
    current_since = start_time
    limit = 1000
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking for new data...")
    
    while current_since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                break
                
            ohlcv = [candle for candle in ohlcv if candle[0] <= now]
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 60000
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            
    if not all_ohlcv:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No new complete candles fetched. Already up to date.")
        return True
        
    # Convert new data to DataFrame and format it immediately
    new_df = pd.DataFrame(all_ohlcv, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert to Kolkata timezone string right away
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp_ms'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%d-%m-%Y %H:%M')
    
    # Keep only the final columns
    new_df = new_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    if existing_df is not None and not existing_df.empty:
        # Both dataframes now have the same columns: timestamp, open, high, low, close, volume
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df
        
    # Deduplicate on the formatted timestamp string
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    
    # Filter out weekends (Saturday=5, Sunday=6)
    temp_dt = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    df = df[temp_dt.dt.dayofweek < 5].copy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_csv(filepath, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Update complete! Added {len(all_ohlcv)} new candles. Total records: {len(df)}")
        return True
    except PermissionError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: Permission denied. Please close '{filename}' if it's open in Excel and try again.")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error saving data: {e}")
        return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or continuously update Gold proxy (PAXG/USDT) 1m data.")
    parser.add_argument('--continuous', action='store_true', help="Run continuously in the background, updating every minute.")
    args = parser.parse_args()

    output_directory = r"E:\PRO\JIMxNik\jim_new\DataBase\GOLD"
    
    if args.continuous:
        print("Starting continuous auto-updater. Will check for new candles every 60 seconds...")
        print("Press Ctrl+C to stop.")
        while True:
            update_paxg_data(output_dir=output_directory, symbol='PAXG/USDT', timeframe='1m', days=365)
            # Sleep until the start of the next minute to avoid redundant API calls
            now = datetime.now()
            sleep_time = 60 - now.second
            time.sleep(sleep_time)
    else:
        update_paxg_data(output_dir=output_directory, symbol='PAXG/USDT', timeframe='1m', days=365)
