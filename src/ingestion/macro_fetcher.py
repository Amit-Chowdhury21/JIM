"""
Macro-Correlate Data Fetcher
=============================
Fetches all macro variables that influence gold prices.
Sources: Yahoo Finance (yfinance) for market data, FRED API for economic data.
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from loguru import logger

from src.utils.config import get_config, PROJECT_ROOT


from src.utils.resilience import retry


class MacroFetcher:
    """Fetches macro-correlate data from Yahoo Finance and FRED."""

    YAHOO_SYMBOLS = {
        "dxy": "DX-Y.NYB", "vix": "^VIX", "tlt": "TLT",
        "tip": "TIP", "silver": "SI=F", "oil": "CL=F", "cny": "CNY=X",
    }

    FRED_SERIES = {
        "DFF": "Fed Funds Rate", "DFII10": "10Y Real Yield",
        "T10Y2Y": "Yield Curve", "DTWEXBGS": "Trade-Weighted USD",
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.data_config = self.config.get("data", {})
        self.macro_config = self.data_config.get("macro", {})
        self.fred_config = self.data_config.get("fred", {})
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.macro_config:
            self.YAHOO_SYMBOLS = {
                "dxy": self.macro_config.get("dxy_symbol", "DX-Y.NYB"),
                "vix": self.macro_config.get("vix_symbol", "^VIX"),
                "tlt": self.macro_config.get("tlt_symbol", "TLT"),
                "tip": self.macro_config.get("tips_symbol", "TIP"),
                "silver": self.macro_config.get("silver_symbol", "SI=F"),
                "oil": self.macro_config.get("oil_symbol", "CL=F"),
                "cny": self.macro_config.get("cny_symbol", "CNY=X"),
            }

    def fetch_all_yahoo_macro(self, period="10y", interval="1d") -> Dict[str, pd.DataFrame]:
        """Fetch all macro-correlate data from Yahoo Finance."""
        import yfinance as yf
        logger.info(f"Fetching {len(self.YAHOO_SYMBOLS)} macro feeds from Yahoo Finance...")
        macro_data = {}
        for name, symbol in self.YAHOO_SYMBOLS.items():
            try:
                df = self._fetch_single_yahoo(symbol, period, interval)
                if df is not None and not df.empty:
                    macro_data[name] = df
                    logger.info(f"  ✅ {name} ({symbol}): {len(df)} bars")
                else:
                    logger.warning(f"  ⚠️  {name} ({symbol}): No data returned")
            except Exception as e:
                logger.error(f"  ❌ {name} ({symbol}): {e}")
        logger.info(f"Yahoo macro: {len(macro_data)}/{len(self.YAHOO_SYMBOLS)} feeds fetched")
        return macro_data

    @retry(max_attempts=3, backoff_multiplier=2.0)
    def _fetch_single_yahoo(self, symbol, period, interval) -> Optional[pd.DataFrame]:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "timestamp"
        for drop_col in ["dividends", "stock_splits", "capital_gains"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
                
        # Phase 6: Automatic Outlier Correction & Gap Imputation
        df = self._clean_and_impute(df)
        
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def _clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 6: Data Quality & Imputation Engine.
        Automatically repairs missing data gaps and clips extreme outliers.
        """
        if df.empty:
            return df
            
        initial_len = len(df)
        
        # 1. Impute small gaps (up to 3 consecutive periods) with forward fill
        df = df.ffill(limit=3)
        
        # 2. Interpolate larger gaps linearly (up to 7 periods)
        df = df.interpolate(method='linear', limit=7)
        
        # 3. Outlier Correction (Clip extreme price spikes > 4 rolling std deviations)
        if len(df) > 30 and "close" in df.columns:
            rolling_median = df["close"].rolling(window=20, min_periods=5).median()
            rolling_std = df["close"].rolling(window=20, min_periods=5).std().bfill()
            
            # Identify extreme price spikes
            price_outliers = (df["close"] > rolling_median + 4*rolling_std) | (df["close"] < rolling_median - 4*rolling_std)
            if price_outliers.any():
                logger.info(f"Corrected {price_outliers.sum()} extreme macro outliers via median substitution")
                # Fix OHLC prices
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.loc[price_outliers, col] = rolling_median[price_outliers]
        
        # 4. Drop remaining NaNs (unrecoverable gaps at the start or > 10 periods)
        df = df.dropna()
        
        if len(df) < initial_len:
            logger.debug(f"Dropped {initial_len - len(df)} unrecoverable bad data rows from macro feed")
            
        return df

    def fetch_single_macro(self, name, period="10y", interval="1d") -> Optional[pd.DataFrame]:
        """Fetch a single macro feed by name (e.g., 'dxy', 'vix')."""
        symbol = self.YAHOO_SYMBOLS.get(name)
        if not symbol:
            logger.error(f"Unknown macro name: {name}")
            return None
        return self._fetch_single_yahoo(symbol, period, interval)

    def fetch_fred_data(self) -> Dict[str, pd.Series]:
        """Fetch Federal Reserve Economic Data (FRED)."""
        api_key = self.fred_config.get("api_key", "")
        series_list = self.fred_config.get("series", list(self.FRED_SERIES.keys()))
        
        has_api_key = api_key and api_key not in ("${FRED_API_KEY}", "your_fred_api_key_here")
        
        fred_data = {}
        logger.info(f"Fetching {len(series_list)} series from FRED...")
        
        if has_api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=api_key)
                for series_id in series_list:
                    try:
                        data = fred.get_series(series_id)
                        if data is not None and len(data) > 0:
                            fred_data[series_id] = data
                            logger.info(f"  ✅ {series_id} (API): {len(data)} observations")
                    except Exception as e:
                        logger.warning(f"  ⚠️  {series_id} (API) failed: {e}. Falling back to CSV...")
                        try:
                            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                            df = pd.read_csv(url, index_col=0, parse_dates=True)
                            if df is not None and not df.empty:
                                df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
                                df = df.dropna()
                                data = df[series_id]
                                fred_data[series_id] = data
                                logger.info(f"  ✅ {series_id} (CSV Fallback): {len(data)} observations")
                        except Exception as e_csv:
                            logger.error(f"  ❌ {series_id} (API & CSV) failed: {e_csv}")
                return fred_data
            except ImportError:
                logger.warning("fredapi not installed, falling back to public CSV method")
        else:
            logger.info("FRED API key not configured, using public CSV method")
            
        # Fallback to public CSV method for everything
        for series_id in series_list:
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                df = pd.read_csv(url, index_col=0, parse_dates=True)
                
                if df is not None and not df.empty:
                    df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
                    df = df.dropna()
                    
                    data = df[series_id]
                    fred_data[series_id] = data
                    logger.info(f"  ✅ {series_id} (CSV): {len(data)} observations")
            except Exception as e:
                logger.error(f"  ❌ {series_id} (CSV): {e}")
                
        return fred_data

    def fred_to_dataframe(self, fred_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Convert FRED series dict to aligned DataFrame, forward-filled."""
        if not fred_data:
            return pd.DataFrame()
        df = pd.DataFrame(fred_data)
        df.index.name = "timestamp"
        df = df.sort_index().ffill()
        return df

    def align_timeseries(self, base_df: pd.DataFrame, additional_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align all macro data to a base DataFrame's timezone-naive index."""
        aligned = base_df.copy()
        
        # Ensure base_df is timezone-naive
        if isinstance(aligned.index, pd.DatetimeIndex) and aligned.index.tz is not None:
            aligned.index = aligned.index.tz_localize(None)
            
        for name, data_obj in additional_data.items():
            if data_obj is None or data_obj.empty:
                continue
                
            df_to_merge = data_obj.copy()
            if isinstance(df_to_merge.index, pd.DatetimeIndex) and df_to_merge.index.tz is not None:
                df_to_merge.index = df_to_merge.index.tz_localize(None)
                
            if isinstance(df_to_merge, pd.Series):
                aligned[name] = df_to_merge.reindex(aligned.index, method="ffill")
            else:
                if "close" in df_to_merge.columns:
                    aligned[name] = df_to_merge["close"].reindex(aligned.index, method="ffill")
                else:
                    for col in df_to_merge.columns:
                        aligned[f"{name}_{col}"] = df_to_merge[col].reindex(aligned.index, method="ffill")
                        
        return aligned.ffill()

    def compute_ratios(self, gold_df, macro_data) -> pd.DataFrame:
        """Compute gold-relative ratios (Gold/Silver, Gold/Oil, Gold/DXY)."""
        ratios = pd.DataFrame(index=gold_df.index)
        for macro_name, col_name in [("silver", "gold_silver_ratio"), ("oil", "gold_oil_ratio"), ("dxy", "gold_dxy_ratio")]:
            if macro_name in macro_data and "close" in macro_data[macro_name].columns:
                aligned = macro_data[macro_name]["close"].reindex(gold_df.index, method="ffill")
                ratios[col_name] = gold_df["close"] / aligned.replace(0, np.nan)
        return ratios

    def save_macro_to_parquet(self, macro_data) -> Dict[str, Path]:
        paths = {}
        for name, df in macro_data.items():
            fp = self.raw_dir / f"macro_{name}.parquet"
            df.to_parquet(fp, engine="pyarrow")
            paths[name] = fp
        return paths

    def load_macro_from_parquet(self) -> Dict[str, pd.DataFrame]:
        macro_data = {}
        for name in self.YAHOO_SYMBOLS:
            fp = self.raw_dir / f"macro_{name}.parquet"
            if fp.exists():
                macro_data[name] = pd.read_parquet(fp, engine="pyarrow")
        return macro_data

    def save_fred_to_parquet(self, fred_df) -> Path:
        fp = self.raw_dir / "fred_data.parquet"
        fred_df.to_parquet(fp, engine="pyarrow")
        return fp

    def load_fred_from_parquet(self) -> Optional[pd.DataFrame]:
        fp = self.raw_dir / "fred_data.parquet"
        return pd.read_parquet(fp, engine="pyarrow") if fp.exists() else None

    # ──────────────────────────────────────────────────────
    # Unified methods for scheduler integration
    # ──────────────────────────────────────────────────────

    def fetch_and_persist_all(self, period: str = "10y") -> Dict[str, int]:
        """Run the full macro data pipeline: Yahoo + FRED + ratios → parquet.

        This is the single entry point the scheduler calls.

        Returns:
            Dict of source → row count for each data set persisted.
        """
        result = {}

        # 1. Yahoo Finance macro feeds
        logger.info("📡 Fetching Yahoo Finance macro feeds...")
        yahoo_data = self.fetch_all_yahoo_macro(period=period)
        if yahoo_data:
            self.save_macro_to_parquet(yahoo_data)
            for name, df in yahoo_data.items():
                result[f"yahoo_{name}"] = len(df)

        # 2. FRED economic data
        logger.info("📡 Fetching FRED economic data...")
        fred_raw = self.fetch_fred_data()
        if fred_raw:
            fred_df = self.fred_to_dataframe(fred_raw)
            self.save_fred_to_parquet(fred_df)
            result["fred"] = len(fred_df)
        else:
            result["fred"] = 0

        # 3. Google Trends (lightweight, no API key)
        logger.info("📡 Fetching Google Trends data...")
        trends_df = self.fetch_google_trends()
        if trends_df is not None and not trends_df.empty:
            fp = self.raw_dir / "google_trends.parquet"
            trends_df.to_parquet(fp, engine="pyarrow")
            result["google_trends"] = len(trends_df)
        else:
            result["google_trends"] = 0

        total = sum(result.values())
        logger.info(f"Macro pipeline complete: {total} total rows across {len(result)} sources")
        return result

    def get_all_macro_data(self) -> Dict[str, pd.DataFrame]:
        """Load all persisted macro data (Yahoo + FRED + Trends) from parquet.

        Returns:
            Unified dict of name → DataFrame.
        """
        data = {}

        # Yahoo macro feeds
        yahoo = self.load_macro_from_parquet()
        data.update(yahoo)

        # FRED data
        fred = self.load_fred_from_parquet()
        if fred is not None and not fred.empty:
            data["fred"] = fred

        # Google Trends
        trends_fp = self.raw_dir / "google_trends.parquet"
        if trends_fp.exists():
            try:
                data["google_trends"] = pd.read_parquet(trends_fp, engine="pyarrow")
            except Exception:
                pass

        logger.info(f"Loaded {len(data)} macro data sources from parquet")
        return data

    def fetch_google_trends(self, keywords: Optional[list] = None) -> Optional[pd.DataFrame]:
        """Fetch Google Trends data for gold-related keywords.

        Uses pytrends (free, no API key needed). Falls back gracefully
        if pytrends is not installed or rate-limited.

        Args:
            keywords: List of search terms. Defaults to config values.

        Returns:
            DataFrame with interest-over-time columns, or None.
        """
        if keywords is None:
            alt_cfg = self.config.get("data", {}).get("alternative", {})
            keywords = alt_cfg.get("google_trends_keywords", ["buy gold", "gold price", "safe haven"])

        try:
            # pyrefly: ignore [missing-import]
            from pytrends.request import TrendReq
        except ImportError:
            logger.debug("pytrends not installed — skipping Google Trends (pip install pytrends)")
            return None

        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pytrends.build_payload(keywords[:5], timeframe="today 5-y", geo="", gprop="")
            df = pytrends.interest_over_time()

            if df is not None and not df.empty:
                # Drop the 'isPartial' column if present
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                df.index.name = "timestamp"
                logger.info(f"  ✅ Google Trends: {len(df)} rows, keywords={keywords[:5]}")
                return df
            else:
                logger.warning("  ⚠️  Google Trends: no data returned")
                return None
        except Exception as e:
            logger.warning(f"  ⚠️  Google Trends failed: {e}")
            return None
