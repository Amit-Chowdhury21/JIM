import pandas as pd
import json
from datetime import datetime
from loguru import logger

def convert_forexfactory_csv_to_json(csv_path="DataBase/CalenderAPI/ff_calendar_2026.csv", json_path="DataBase/CalenderAPI/ff_calendar_2026.json"):
    """
    Reads a downloaded ForexFactory CSV export and converts it to the exact JSON schema
    used by the v2.3 Alpha Execution Engine.
    """
    try:
        # ForexFactory CSV headers typically: Title, Country, Date, Time, Impact, Forecast, Previous
        df = pd.read_csv(csv_path)
        
        # Filter for only High impact USD events
        df = df[(df['Country'] == 'USD') & (df['Impact'] == 'High')].copy()
        
        # Drop rows missing date or time
        df = df.dropna(subset=['Date', 'Time'])
        
        events = []
        for _, row in df.iterrows():
            try:
                # Combine Date and Time into a single datetime object
                # Note: FF CSV times are usually in Eastern Time (ET) or your local time depending on your account settings.
                # Assuming the CSV is exported in local time (e.g., 8:30am)
                date_str = f"{row['Date']} {row['Time']}"
                dt = pd.to_datetime(date_str)
                
                events.append({
                    "title": row['Title'],
                    "country": row['Country'],
                    "date": dt.isoformat(),
                    "impact": row['Impact'],
                    "forecast": str(row.get('Forecast', '')),
                    "previous": str(row.get('Previous', ''))
                })
            except Exception as parse_e:
                logger.warning(f"Skipped row due to parse error: {row['Title']} - {parse_e}")
                
        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4)
            
        logger.info(f"✅ Successfully converted {len(events)} high-impact USD events to {json_path}")
        return True
        
    except FileNotFoundError:
        logger.error(f"❌ Could not find {csv_path}. Please download it from ForexFactory.")
    except Exception as e:
        logger.error(f"❌ Error processing CSV: {e}")

if __name__ == "__main__":
    convert_forexfactory_csv_to_json()
