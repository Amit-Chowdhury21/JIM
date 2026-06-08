import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger
import os

def convert_forexfactory_html_to_json(html_path="DataBase/CalenderAPI/ff_calendar.html", json_path="DataBase/CalenderAPI/ff_calendar_2026.json", year="2026"):
    """
    Parses a saved ForexFactory HTML page and converts it to the exact JSON schema
    used by the v2.3 Alpha Execution Engine.
    """
    if not os.path.exists(html_path):
        logger.error(f"❌ Could not find {html_path}. Please press Ctrl+S on the webpage and save it as ff_calendar.html in the DataBase folder.")
        return

    try:
        with open(html_path, 'rb') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        events = []
        current_date_str = None
        
        # Find all calendar rows
        rows = soup.find_all('tr', class_='calendar__row')
        
        for row in rows:
            try:
                # 1. Parse Date (ForexFactory often only lists the date on the first event of the day)
                date_td = row.find('td', class_='calendar__date')
                if date_td and date_td.text.strip():
                    # Extract just the "Jan 1" part, ignore the "Mon"
                    raw_date = date_td.text.strip()
                    # Example: "Fri Jun 5" -> "Jun 5"
                    parts = raw_date.split(' ')
                    if len(parts) >= 2:
                        current_date_str = f"{parts[1]} {parts[2]} {year}"
                
                if not current_date_str:
                    continue
                    
                # 2. Parse Time
                time_td = row.find('td', class_='calendar__time')
                time_str = time_td.text.strip() if time_td else ""
                # Sometimes time is blank (meaning it uses the previous event's time)
                # For simplicity, we just use the raw text. If it's a High Impact event, we need the exact time.
                if not time_str:
                    # Inherit time from previous valid event if possible, but FF usually repeats or we skip
                    pass 
                
                # 3. Parse Currency
                currency_td = row.find('td', class_='calendar__currency')
                currency = currency_td.text.strip() if currency_td else ""
                
                # 4. Parse Impact
                impact_td = row.find('td', class_='calendar__impact')
                impact = ""
                if impact_td:
                    if impact_td.find('span', class_='icon--ff-impact-red'):
                        impact = "High"
                    elif impact_td.find('span', class_='icon--ff-impact-ora'):
                        impact = "Medium"
                    elif impact_td.find('span', class_='icon--ff-impact-yel'):
                        impact = "Low"
                        
                # 5. Parse Title
                event_td = row.find('td', class_='calendar__event')
                title = event_td.text.strip() if event_td else ""
                
                # ONLY KEEP HIGH IMPACT USD EVENTS
                if currency == "USD" and impact == "High" and time_str and "All Day" not in time_str:
                    # Combine Date and Time
                    dt_str = f"{current_date_str} {time_str}"
                    # e.g. "Jun 5 2026 8:30am"
                    try:
                        dt = datetime.strptime(dt_str, "%b %d %Y %I:%M%p")
                        
                        events.append({
                            "title": title,
                            "country": currency,
                            "date": dt.isoformat(),
                            "impact": impact,
                            "forecast": "",
                            "previous": ""
                        })
                    except ValueError as ve:
                        logger.warning(f"Could not parse datetime: {dt_str} - {ve}")
                        
            except Exception as e:
                logger.warning(f"Skipped a row due to parse error: {e}")
                
        # Save to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4)
            
        logger.info(f"✅ Successfully scraped {len(events)} High-Impact USD events!")
        logger.info(f"✅ Saved to {json_path}")
        
    except Exception as e:
        logger.error(f"❌ Error parsing HTML: {e}")

if __name__ == "__main__":
    convert_forexfactory_html_to_json()
