"""
v2.3 "Alpha" Execution & Policy Engine
=======================================
Extends v2.1 by integrating:
1. Regime-Aware Thresholds (HMM)
2. Volatility-Adjusted Sizing (Target Volatility)
3. Macro Event-State Filtering (ForexFactory Calendar)
"""

import os
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from loguru import logger

class ForexFactoryCalendar:
    """
    Fetches and caches ForexFactory JSON calendar to identify high-impact USD events.
    Caches the result for 1 hour to avoid API bans.
    """
    def __init__(self, historical_file="DataBase/CalenderAPI/ff_calendar_2026.json"):
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.cache_file = "ff_calendar_cache.json"
        self.historical_file = historical_file
        self.last_fetch_time = 0.0
        self.live_events: List[Dict] = []
        self.historical_events: List[Dict] = []
        
        # Pre-load historical events for backtesting
        if os.path.exists(self.historical_file):
            import json
            try:
                with open(self.historical_file, "r", encoding="utf-8") as f:
                    self.historical_events = json.load(f)
                logger.info(f"[v2.3 Calendar] Loaded {len(self.historical_events)} historical events from {self.historical_file}")
            except Exception as e:
                logger.error(f"[v2.3 Calendar] Failed to load historical events: {e}")
        
    def _fetch_events(self):
        now = time.time()
        # Fetch at most once per hour (3600 seconds)
        if now - self.last_fetch_time > 3600 or not self.live_events:
            cache_path = os.path.join("DataBase", "CalenderAPI", self.cache_file)
            try:
                resp = requests.get(self.url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    # Filter for High impact ("High") USD events
                    self.live_events = [
                        e for e in data
                        if e.get("country") == "USD" and e.get("impact") == "High"
                    ]
                    self.last_fetch_time = now
                    logger.debug(f"[v2.3 Calendar] Fetched {len(self.live_events)} high-impact USD events this week.")
                    
                    # Save to DataBase for resilience
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    import json
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(self.live_events, f)
                        
            except Exception as e:
                logger.warning(f"[v2.3 Calendar] Failed to fetch ForexFactory calendar: {e}")
                # Fallback to local cache if API fails
                if not self.live_events and os.path.exists(cache_path):
                    import json
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            self.live_events = json.load(f)
                        self.last_fetch_time = now  # Reset timer so we don't spam errors
                        logger.info(f"[v2.3 Calendar] Loaded {len(self.live_events)} events from local cache fallback.")
                    except Exception as cache_err:
                        logger.error(f"[v2.3 Calendar] Cache fallback also failed: {cache_err}")
                
    def is_event_window(self, current_dt: datetime, window_minutes: int = 15) -> bool:
        """
        Check if current_dt is within window_minutes of any high-impact USD event.
        current_dt must be timezone-aware (UTC).
        """
        self._fetch_events()
        all_events = self.live_events + self.historical_events
        for e in all_events:
            try:
                # ff_calendar returns ISO strings e.g. "2026-06-05T08:30:00-04:00"
                event_dt = datetime.fromisoformat(e["date"])
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    event_dt = event_dt.astimezone(timezone.utc)
                
                # Check absolute difference in minutes
                diff_minutes = abs((current_dt - event_dt).total_seconds()) / 60.0
                if diff_minutes <= window_minutes:
                    logger.info(f"[v2.3 Event Filter] Active: {e['title']} at {e['date']} (Impact: High)")
                    return True
            except Exception:
                continue
        return False

# Global Calendar Instance
_calendar = ForexFactoryCalendar()

class AlphaSignalPolicy:
    """
    v2.3 Execution Policy.
    Gates trades based on Epistemic Uncertainty, Regime-Aware Thresholds,
    Volatility Sizing, and Macro Events.
    """
    def __init__(
        self, 
        base_long: float = 0.45, 
        base_short: float = 0.65, 
        max_uncertainty: float = 0.08
    ):
        self.base_long = base_long
        self.base_short = base_short
        self.max_uncertainty = max_uncertainty
        
        # Regime multipliers for thresholds
        self.regime_weights = {
            "Growth": {"long": 0.88, "short": 0.84},  # L: 0.45*0.88=0.40, S: 0.65*0.84=0.55
            "Normal": {"long": 1.00, "short": 1.00},  # L: 0.45, S: 0.65
            "Crisis": {"long": 1.11, "short": 1.15},  # L: 0.45*1.11=0.50, S: 0.65*1.15=0.75
        }
        
    def evaluate(
        self, 
        mcd_result: Dict, 
        regime: str, 
        vol_10: float, 
        vol_10_mean: float
    ) -> Dict:
        """
        Evaluates MCDropout signal against v2.3 execution rules.
        """
        signal = mcd_result.get("signal", "HOLD")
        uncertainty = mcd_result.get("uncertainty", 0.0)
        probs = mcd_result.get("probabilities", {})
        
        long_prob = probs.get("LONG", 0.0)
        short_prob = probs.get("SHORT", 0.0)
        
        # 1. Macro Event Filter (15 minutes window)
        now_utc = datetime.now(timezone.utc)
        if _calendar.is_event_window(now_utc, window_minutes=15):
            return self._reject("HOLD", "Macro Event Shock Window (High-Impact USD)")
            
        # 2. Epistemic Uncertainty Gate
        if uncertainty > self.max_uncertainty:
            return self._reject("HOLD", f"High uncertainty ({uncertainty:.4f} > {self.max_uncertainty})")
            
        # 3. Regime-Aware Thresholds
        regime_factor = self.regime_weights.get(regime, self.regime_weights["Normal"])
        target_long_thresh = self.base_long * regime_factor["long"]
        target_short_thresh = self.base_short * regime_factor["short"]
        
        if signal == "LONG" and long_prob < target_long_thresh:
            return self._reject("HOLD", f"LONG prob ({long_prob:.4f}) below {regime} threshold ({target_long_thresh:.2f})")
            
        if signal == "SHORT" and short_prob < target_short_thresh:
            return self._reject("HOLD", f"SHORT prob ({short_prob:.4f}) below {regime} threshold ({target_short_thresh:.2f})")
            
        # 4. Volatility-Adjusted Sizing
        # Base sizing from uncertainty [1.0 down to 0.5]
        base_size = 1.0
        if uncertainty > 0 and self.max_uncertainty > 0:
            base_size = max(0.5, 1.0 - (uncertainty / self.max_uncertainty) * 0.5)
            
        # Adjust for volatility: If vol_10 is double the mean, size halves. If vol is half, size doubles (max 1.5x)
        vol_ratio = 1.0
        if vol_10_mean > 0:
            vol_ratio = vol_10 / vol_10_mean
            
        vol_scalar = 1.0
        if vol_ratio > 0.1:  # Avoid div by zero
            vol_scalar = min(1.5, max(0.2, 1.0 / vol_ratio))
            
        final_size = base_size * vol_scalar
        
        # If originally HOLD, pass through
        if signal == "HOLD":
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "uncertainty": uncertainty,
                "sizing_scalar": 0.0,
                "reasoning": "Model predicts HOLD"
            }
            
        # Passed all gates
        return {
            "signal": signal,
            "confidence": mcd_result.get("confidence", 0.0),
            "uncertainty": uncertainty,
            "sizing_scalar": round(final_size, 3),
            "reasoning": f"v2.3 Passed (Regime: {regime}, Vol Ratio: {vol_ratio:.2f}, Size: {final_size:.2f}x)"
        }
        
    def _reject(self, new_signal: str, reason: str) -> Dict:
        return {
            "signal": new_signal,
            "confidence": 0.0,
            "uncertainty": 0.0,
            "sizing_scalar": 0.0,
            "reasoning": f"v2.3 Rejected: {reason}"
        }
