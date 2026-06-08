import ccxt
import pandas as pd
import time
from datetime import datetime
import pytz
import os

def download_custom_history():
    symbol = 'PAXG/USDT'
    timeframe = '1m'
    output_dir = r"E:\PRO\JIMxNik\jim_new\DataBase"
    filename = "gold_1m_2025.csv"
    filepath = os.path.join(output_dir, filename)
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Define start and end in Asia/Kolkata
    tz = pytz.timezone('Asia/Kolkata')
    
    # 2025-01-01 00:00:00
    start_dt = tz.localize(datetime(2025, 1, 1, 0, 0, 0))
    # 2025-12-31 23:59:00
    end_dt = tz.localize(datetime(2025, 12, 31, 23, 59, 0))
    
    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(end_dt.timestamp() * 1000)
    
    print(f"Downloading from {start_dt} to {end_dt}...")
    
    all_ohlcv = []
    current_since = start_time
    limit = 1000
    
    while current_since <= end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                break
                
            # Filter candles past our end_time
            ohlcv = [c for c in ohlcv if c[0] <= end_time]
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 60000
            
            last_dt = pd.to_datetime(ohlcv[-1][0], unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')
            print(f"Fetched {len(all_ohlcv)} candles. Last candle: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            
    print(f"Complete. Total candles: {len(all_ohlcv)}")
    
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
    else:
        print("No data was fetched!")

if __name__ == "__main__":
    download_custom_history()
