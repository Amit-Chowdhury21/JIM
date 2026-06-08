import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import zoneinfo
from loguru import logger

class EconomicCalendar:
    """
    Fetches and parses high-impact economic events to protect the trading system from 
    extreme volatility spikes during news releases (e.g., CPI, NFP, FOMC).
    """
    
    def __init__(self):
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        self.high_impact_events = []
        self.last_fetch_date = None

    def fetch_events(self):
        """Fetches the weekly XML feed and parses high impact USD events."""
        now_utc = datetime.now(timezone.utc)
        
        # Only fetch once per day
        if self.last_fetch_date == now_utc.date() and self.high_impact_events:
            return
            
        try:
            logger.info("Fetching weekly economic calendar from Forex Factory...")
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            events = []
            tz_et = zoneinfo.ZoneInfo("America/New_York")
            
            for event in root.findall('event'):
                impact = event.find('impact').text if event.find('impact') is not None else ''
                country = event.find('country').text if event.find('country') is not None else ''
                
                # Filter for High impact USD events (can add EUR/GBP if trading other pairs)
                if impact == 'High' and country == 'USD':
                    date_str = event.find('date').text
                    time_str = event.find('time').text
                    
                    if not date_str or not time_str or time_str == 'All Day':
                        continue
                        
                    # ForexFactory XML times are US Eastern Time (America/New_York)
                    dt_str = f"{date_str} {time_str}"
                    try:
                        dt_et = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=tz_et)
                        dt_utc = dt_et.astimezone(timezone.utc)
                        events.append(dt_utc)
                    except ValueError as e:
                        logger.debug(f"Could not parse event date/time: {dt_str} - {e}")
            
            self.high_impact_events = sorted(events)
            self.last_fetch_date = now_utc.date()
            logger.info(f"Loaded {len(self.high_impact_events)} high-impact USD events for the week.")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Economic calendar rate limited (429). Backing off for today.")
                self.last_fetch_date = now_utc.date()
            else:
                logger.error(f"Failed to fetch economic calendar: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")

    def get_news_status(self, current_time_utc: datetime = None) -> dict:
        """
        Returns status regarding proximity to high impact news.
        - block_trade: True if within +/- 5 minutes
        - tighten_threshold: True if within +/- 30 minutes
        """
        self.fetch_events()
        
        if current_time_utc is None:
            current_time_utc = datetime.now(timezone.utc)
            
        # Ensure current_time_utc has a timezone for comparison
        if current_time_utc.tzinfo is None:
            current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
            
        for event_time in self.high_impact_events:
            diff_seconds = abs((current_time_utc - event_time).total_seconds())
            
            if diff_seconds <= 5 * 60:
                return {"block_trade": True, "tighten_threshold": True, "event_time": event_time}
            elif diff_seconds <= 30 * 60:
                return {"block_trade": False, "tighten_threshold": True, "event_time": event_time}
                
        return {"block_trade": False, "tighten_threshold": False, "event_time": None}
