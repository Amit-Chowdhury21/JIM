"""
Real-Time Macro Data Feed Collector
====================================

Collects and manages:
  - DXY (US Dollar Index) prices
  - VIX (Volatility Index) levels
  - US 10Y Treasury yields
  - Gold-Silver ratio
  - Credit spreads
  - Real yield proxy

Provides caching and fallback mechanisms for high-frequency access.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import time


class MacroDataFeed:
    """Fetches and caches real-time macro data."""
    
    # Ticker mappings
    TICKERS = {
        "dxy": "DX-Y.NYB",           # US Dollar Index
        "vix": "^VIX",               # Volatility Index
        "us10y": "^TNX",             # 10-Year Treasury yield (%)
        "us2y": "^IRX",              # 2-Year Treasury yield (%)
        "gold": "GC=F",              # Gold futures
        "silver": "SI=F",            # Silver futures
        "usd_jpy": "JPY=X",          # USD/JPY
        "eur_usd": "EURUSD=X",       # EUR/USD
        "realyield": "TIP",          # TIPS (inflation-protected bonds)
        "credit_spread": "LQD",      # Investment Grade Credit ETF
    }
    
    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Initialize macro data feed.
        
        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default 60s)
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict = {}
        self._cache_timestamp: Dict = {}
        self._last_fetch_error: Optional[str] = None
        
    def get_macro_data(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current macro data with caching.
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
        
        Returns:
            Dictionary with keys: dxy, vix, us10y, us2y, gold_silver_ratio, 
            real_yield_proxy, credit_spread, eur_usd, usd_jpy
        """
        now = datetime.now()
        
        # Check cache validity
        cache_age = (now - self._cache_timestamp.get("full", datetime.min)).total_seconds()
        
        if not force_refresh and cache_age < self.cache_ttl_seconds and "full" in self._cache:
            return self._cache["full"]
        
        # Fetch fresh data
        macro_data = self._fetch_macro_data()
        
        if macro_data:
            self._cache["full"] = macro_data
            self._cache_timestamp["full"] = now
            self._last_fetch_error = None
        else:
            if "full" in self._cache:
                logger.warning("Macro data fetch failed, using cached data")
            else:
                logger.error("Macro data fetch failed and no cache available")
                return self._get_fallback_data()
        
        return macro_data
    
    def _fetch_macro_data(self) -> Optional[Dict[str, float]]:
        """Fetch fresh macro data from yfinance."""
        try:
            # Fetch all tickers in one batch for efficiency
            tickers_str = " ".join(self.TICKERS.values())
            data = yf.download(tickers_str, period="1d", interval="1m", progress=False)
            
            if data.empty:
                self._last_fetch_error = "yfinance returned empty data"
                return None
            
            # Extract current (latest) values
            dxy = float(data["DX-Y.NYB"]["Close"].iloc[-1]) if "DX-Y.NYB" in data else 104.0
            vix = float(data["^VIX"]["Close"].iloc[-1]) if "^VIX" in data else 15.0
            us10y = float(data["^TNX"]["Close"].iloc[-1]) if "^TNX" in data else 4.5
            us2y = float(data["^IRX"]["Close"].iloc[-1]) if "^IRX" in data else 5.0
            gold = float(data["GC=F"]["Close"].iloc[-1]) if "GC=F" in data else 2000.0
            silver = float(data["SI=F"]["Close"].iloc[-1]) if "SI=F" in data else 25.0
            eur_usd = float(data["EURUSD=X"]["Close"].iloc[-1]) if "EURUSD=X" in data else 1.08
            usd_jpy = float(data["JPY=X"]["Close"].iloc[-1]) if "JPY=X" in data else 150.0
            
            # Calculate derived metrics
            gold_silver_ratio = gold / silver if silver > 0 else 80.0
            
            # Real yield proxy: 10Y yield - 2Y yield (term premium)
            real_yield_proxy = us10y - us2y
            
            # Credit spread proxy (would need additional data in production)
            credit_spread = 1.5  # Placeholder (use actual credit ETF data in production)
            
            # DXY shock: percentage change from baseline (104 = neutral)
            dxy_shock = (dxy - 104.0) / 104.0
            
            return {
                "dxy": dxy,
                "dxy_shock": dxy_shock,
                "vix": vix,
                "us10y": us10y,
                "us2y": us2y,
                "term_premium": us10y - us2y,
                "gold": gold,
                "silver": silver,
                "gold_silver_ratio": gold_silver_ratio,
                "real_yield_proxy": real_yield_proxy,
                "credit_spread": credit_spread,
                "eur_usd": eur_usd,
                "usd_jpy": usd_jpy,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            self._last_fetch_error = str(e)
            logger.error(f"Failed to fetch macro data: {e}")
            return None
    
    def _get_fallback_data(self) -> Dict[str, float]:
        """Return neutral/default macro data when fetching fails."""
        return {
            "dxy": 104.0,
            "dxy_shock": 0.0,
            "vix": 15.0,
            "us10y": 4.5,
            "us2y": 5.0,
            "term_premium": -0.5,
            "gold": 2000.0,
            "silver": 25.0,
            "gold_silver_ratio": 80.0,
            "real_yield_proxy": -0.5,
            "credit_spread": 1.5,
            "eur_usd": 1.08,
            "usd_jpy": 150.0,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_regime_indicators(self) -> Dict[str, float]:
        """
        Get macro indicators specifically for regime classification.
        
        Returns:
            Dictionary with: vix_level, dxy_shock, rates_shock, credit_stress
        """
        macro = self.get_macro_data()
        
        # Volatility shock (VIX deviation from neutral ~15)
        vix_shock = (macro["vix"] - 15.0) / 15.0
        
        # Rate shock (10Y deviation from 4% neutral)
        rates_shock = (macro["us10y"] - 4.0) / 4.0
        
        # Term premium inversion as proxy for stress
        term_inversion = 1.0 if macro["term_premium"] < -1.0 else 0.0
        
        # Credit stress (spread proxy)
        credit_stress = min(macro["credit_spread"] / 2.0, 1.0)  # Normalize [0-1]
        
        return {
            "vix_level": macro["vix"],
            "vix_shock": vix_shock,
            "dxy_shock": macro["dxy_shock"],
            "rates_shock": rates_shock,
            "term_inversion": term_inversion,
            "credit_stress": credit_stress,
        }
    
    def get_last_error(self) -> Optional[str]:
        """Get last fetch error message."""
        return self._last_fetch_error
    
    def clear_cache(self):
        """Clear cached data."""
        self._cache = {}
        self._cache_timestamp = {}


# Global instance
_macro_feed: Optional[MacroDataFeed] = None


def get_macro_feed() -> MacroDataFeed:
    """Get or create global macro data feed."""
    global _macro_feed
    if _macro_feed is None:
        _macro_feed = MacroDataFeed(cache_ttl_seconds=60)
    return _macro_feed


def fetch_macro_data(force_refresh: bool = False) -> Dict[str, float]:
    """Convenience function to fetch macro data."""
    return get_macro_feed().get_macro_data(force_refresh=force_refresh)


def get_regime_indicators() -> Dict[str, float]:
    """Convenience function to get regime-specific macro indicators."""
    return get_macro_feed().get_regime_indicators()
