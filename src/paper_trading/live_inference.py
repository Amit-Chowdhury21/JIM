"""
Live Inference Loop
====================
Background asyncio task that continuously:
  1. Fetches latest gold price data via yfinance (every 60s)
  2. Runs all 6 Phase 3 models to generate fresh signals
  3. Injects signals into the paper trading engine
  4. Broadcasts model updates via WebSocket

All 6 models run on every tick:
  - Wavelet Denoiser      → trend direction + confidence
  - HMM Regime Detector   → market regime + regime-adjusted confidence
  - LSTM Temporal         → price sequence prediction
  - TFT Forecaster        → multi-horizon forecast
  - Genetic Algorithm     → evolved rule-based signal
  - Ensemble Stacking     → meta-learner aggregation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Callable, Any
import warnings
import numpy as np
import pandas as pd
from loguru import logger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")
warnings.filterwarnings("ignore", module="torch")

from src.models.hmm_pro import run_hmm_pro
from src.models.wavelet_pro import run_wavelet_pro
from src.paper_trading.prediction_logger import log_prediction_cycle, update_pnl_for_trade

# ============================================================================
# MODEL SIGNAL REGISTRY (shared state, updated by inference loop)
# ============================================================================

# Latest signal from each model — exposed via /paper-trading/live-signals
LIVE_MODEL_SIGNALS: Dict[str, Dict] = {
    model: {
        "signal": None,
        "confidence": 0.0,
        "regime": "UNKNOWN",
        "price": 0.0,
        "reasoning": "",
        "last_updated": None,
        "error": None,
    }
    for model in ["wavelet_pro", "wavelet_basic", "hmm", "lstm", "tft", "genetic", "hmm_pro", "ensemble"]
}

# Current gold price (updated on each fetch)
CURRENT_GOLD_PRICE: float = 0.0
LAST_PRICE_UPDATE: Optional[datetime] = None

# Current macro data
MACRO_DATA: Dict[str, float] = {
    "dxy": 0.0,
    "us10y": 0.0,
    "gold_silver_ratio": 0.0,
    "rl_kelly": 1.0,
    "rl_trailing": 0.015,
}


# ============================================================================
# GOLD DATA FETCHER
# ============================================================================

def fetch_live_gold_data(period: str = "5d", interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Fetch latest gold futures OHLCV data AND macro indicators (DXY, US10Y) via yfinance.
    Returns DataFrame with columns: open, high, low, close, volume, returns, dxy, us10y, dxy_returns, us10y_returns
    """
    try:
        import yfinance as yf
        import time
        
        # Group download is faster and aligns timestamps perfectly
        tickers = "GC=F PAXG-USD DX-Y.NYB ^TNX SI=F ^GVZ TIP"
        
        df_all = pd.DataFrame()
        for attempt in range(3):
            df_all = yf.download(tickers, period=period, interval=interval, group_by="ticker", progress=False, auto_adjust=False)
            if not df_all.empty and "GC=F" in df_all:
                break
            logger.warning(f"yfinance returned empty dataframe (attempt {attempt+1}/3), retrying...")
            time.sleep(2)

        if df_all.empty:
            logger.warning("yfinance returned empty dataframe")
            return None

        # Extract Gold (Merge GC=F futures with PAXG-USD 24/7 crypto gold to prevent stale prices on weekends/holidays)
        df = df_all["GC=F"].copy()
        if "PAXG-USD" in df_all:
            df_pax = df_all["PAXG-USD"].copy()
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns and col in df_pax.columns:
                    df[col] = df[col].fillna(df_pax[col])
            # For volume, crypto volume can be 0 or small, replace 0 with 1 to avoid zero-volume filter dropping valid rows
            if "Volume" in df.columns and "Volume" in df_pax.columns:
                df["Volume"] = df["Volume"].fillna(df_pax["Volume"].replace(0, 1))

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        
        # --- FROZEN BAR FIX (MODIFIED FOR INSTANT LIVE FEED) ---
        # We must NOT drop the most recent proxy bars even if they are flat/0 volume.
        # Crypto proxies often have 0 volume in 1m. Dropping them causes a 3-5 minute lag.
        if not df.empty:
            recent_cutoff = df.index[-1] - pd.Timedelta(minutes=10)
            is_recent = df.index >= recent_cutoff
            is_valid_volume = df["volume"] > 0
            is_not_flat = ~((df["high"] == df["low"]) & (df["open"] == df["close"]))
            
            # Keep if it's recent, OR (has volume AND is not flat)
            df = df[is_recent | (is_valid_volume & is_not_flat)].copy()
        
        # Extract DXY, US10Y and Silver closes (Forward fill to handle slightly misaligned ticks)
        df["dxy"] = df_all["DX-Y.NYB"]["Close"].ffill()
        df["us10y"] = df_all["^TNX"]["Close"].ffill()
        df["silver"] = df_all["SI=F"]["Close"].ffill()
        df["gvz"] = df_all["^GVZ"]["Close"].ffill()
        df["tip"] = df_all["TIP"]["Close"].ffill()
        
        # Forward fill up to the current wall-clock minute to eliminate ANY yfinance cache lag
        if not df.empty:
            current_minute = pd.Timestamp.now(tz=df.index.tz).floor("min")
            # If our last bar is behind the actual clock, fill the gap with synthetic bars
            if df.index[-1] < current_minute:
                missing_idx = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=1), end=current_minute, freq="min")
                if not missing_idx.empty:
                    missing_df = pd.DataFrame(index=missing_idx, columns=df.columns)
                    df = pd.concat([df, missing_df])
                    df = df.ffill()

        df.dropna(inplace=True)
        df["gvz_zscore_20"] = (df["gvz"] - df["gvz"].rolling(20).mean()) / df["gvz"].rolling(20).std()

        # Add returns
        df["returns"] = df["close"].pct_change()
        df["dxy_returns"] = df["dxy"].pct_change()
        df["us10y_returns"] = df["us10y"].pct_change()
        df["silver_returns"] = df["silver"].pct_change()
        df["tip_return"] = df["tip"].pct_change()
        df["gold_silver_ratio"] = df["close"] / df["silver"]
        
        # Negative real yield = Gold bullish
        df["real_yield_proxy"] = df["us10y"] - df["tip_return"].rolling(20).mean() * 100
        
        df.dropna(inplace=True)
        
        # Override the final bar with the true live spot price from MetalPriceAPI
        # and scale the entire historical series so the returns stay structurally intact
        # without introducing a massive gap on the final bar.
        try:
            spot_price = fetch_metalpriceapi_spot()
            if spot_price and spot_price > 0:
                current_last = df["close"].iloc[-1]
                scaling_factor = spot_price / current_last
                
                # Scale OHL prices
                df["open"] *= scaling_factor
                df["high"] *= scaling_factor
                df["low"] *= scaling_factor
                df["close"] *= scaling_factor
                
                # Recalculate GSR
                df["gold_silver_ratio"] = df["close"] / df["silver"]
                
                logger.debug(f"Scaled historical data by {scaling_factor:.4f} to match live MetalPriceAPI spot: ${spot_price:.2f}")
        except Exception as e:
            logger.debug(f"Failed to fetch live MetalPriceAPI spot for history override: {e}")

        logger.debug(f"Data fetched: {len(df)} bars. Gold: ${df['close'].iloc[-1]:.2f}, DXY: {df['dxy'].iloc[-1]:.2f}, GSR: {df['gold_silver_ratio'].iloc[-1]:.2f}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch gold data: {e}")
        return None

def fetch_metalpriceapi_spot() -> Optional[float]:
    """
    Fetch real-time spot price for Gold from the free Gold-API.com,
    with fallback to MetalPriceAPI if needed.
    """
    import os
    import requests
    from dotenv import load_dotenv

    # 1. Try free Gold-API.com first (unlimited, no key required)
    try:
        resp = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("price")
            if price and float(price) > 0:
                logger.debug(f"Fetched real-time gold spot price from Gold-API.com: ${price:.2f}")
                return round(float(price), 2)
    except Exception as e:
        logger.warning(f"Gold-API.com fetch failed: {e}")

    # 2. Fallback to MetalPriceAPI
    load_dotenv()
    api_key = os.environ.get("METALPRICE_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GOLDAPI_KEY", "")
    
    if api_key and api_key != "your_metalprice_key_here":
        try:
            resp = requests.get(
                f"https://api.metalpriceapi.com/v1/latest?api_key={api_key}&base=USD&currencies=XAU",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    rates = data.get("rates", {})
                    xau_rate = rates.get("XAU")
                    if xau_rate and float(xau_rate) > 0:
                        return round(1.0 / float(xau_rate), 2)
            else:
                logger.warning(f"MetalPriceAPI error {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"MetalPriceAPI fetch failed: {e}")
            
    return None

def fetch_metalpriceapi_gs_spot() -> tuple[Optional[float], Optional[float]]:
    """
    Fetch real-time spot prices for Gold and Silver from Gold-API.com,
    with fallback to MetalPriceAPI if needed.
    Returns: (gold_price, silver_price)
    """
    import os
    import requests
    from dotenv import load_dotenv

    # 1. Try free Gold-API.com first (unlimited, no key required)
    try:
        r_gold = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
        r_silver = requests.get("https://api.gold-api.com/price/XAG", timeout=5)
        
        gold_price = None
        silver_price = None
        
        if r_gold.status_code == 200:
            data = r_gold.json()
            price = data.get("price")
            if price and float(price) > 0:
                gold_price = round(float(price), 2)
                
        if r_silver.status_code == 200:
            data = r_silver.json()
            price = data.get("price")
            if price and float(price) > 0:
                silver_price = round(float(price), 2)
                
        if gold_price is not None or silver_price is not None:
            logger.debug(f"Fetched real-time spot prices from Gold-API.com: Gold=${gold_price}, Silver=${silver_price}")
            return gold_price, silver_price
    except Exception as e:
        logger.warning(f"Gold-API.com (G/S) fetch failed: {e}")

    # 2. Fallback to MetalPriceAPI
    load_dotenv()
    api_key = os.environ.get("METALPRICE_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GOLDAPI_KEY", "")
        
    if api_key and api_key != "your_metalprice_key_here":
        try:
            resp = requests.get(
                f"https://api.metalpriceapi.com/v1/latest?api_key={api_key}&base=USD&currencies=XAU,XAG",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    rates = data.get("rates", {})
                    xau = rates.get("XAU")
                    xag = rates.get("XAG")
                    gold_price = round(1.0 / float(xau), 2) if xau and float(xau) > 0 else None
                    silver_price = round(1.0 / float(xag), 2) if xag and float(xag) > 0 else None
                    return gold_price, silver_price
            else:
                logger.warning(f"MetalPriceAPI (G/S) error {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"MetalPriceAPI (G/S) fetch failed: {e}")
            
    return None, None


# ============================================================================
# MODEL RUNNERS — each returns (signal: str, confidence: float, reasoning: str)
# ============================================================================

def run_wavelet(df: pd.DataFrame) -> Dict:
    """
    Wavelet signal using WaveletPro (professional 6-level DWT model).
    
    This replaces the basic 5-level model with the production-ready WaveletPro
    that includes:
    - 6-level DWT decomposition
    - Wavelet Oscillator (D3 + D4)
    - Continuous Wavelet Transform (CWT)
    - 30+ engineered features
    """
    try:
        return run_wavelet_pro(df)
    except Exception as e:
        logger.warning(f"Wavelet Pro fallback to basic model: {e}")
        # Fallback to basic wavelet if WaveletPro fails
        try:
            prices = df["close"].values

            # Need at least 32 samples (2^5 for 5-level decomposition)
            if len(prices) < 32:
                return {"signal": "HOLD", "confidence": 0.0, "reasoning": "Insufficient data"}

            try:
                import pywt
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coeffs = pywt.wavedec(prices, "db4", level=5)
                    # Zero out detail coefficients to denoise
                    coeffs[1:] = [np.zeros_like(v) for v in coeffs[1:]]
                    trend = pywt.waverec(coeffs, "db4")
                pywt_available = True
            except ImportError:
                # Proxy if pywt is unavailable
                trend = pd.Series(prices).rolling(window=10).mean().bfill().values
                pywt_available = False

            # Signal: compare last 3 denoised values to determine trend
            if len(trend) >= 3:
                slope = (trend[-1] - trend[-3]) / (abs(trend[-3]) + 1e-8)
                
                # Dynamic confidence based on slope magnitude
                abs_slope = abs(slope)
                # Base 50% + up to 45% based on slope (0.001 slope adds 40%)
                base_conf = 0.50 + min(abs_slope * 400.0, 0.45)
                
                # Penalty if using fallback model
                confidence = base_conf if pywt_available else base_conf * 0.70

                if slope > 0.0001:
                    signal = "LONG"
                    reasoning = f"Wavelet trend slope +{slope:.4f} → uptrend detected"
                elif slope < -0.0001:
                    signal = "SHORT"
                    reasoning = f"Wavelet trend slope {slope:.4f} → downtrend detected"
                else:
                    signal = "HOLD"
                    confidence = 0.15
                    reasoning = f"Wavelet trend flat (slope={slope:.6f})"
            else:
                signal = "HOLD"
                confidence = 0.0
                reasoning = "Insufficient trend data"

            return {"signal": signal, "confidence": round(confidence, 3), "reasoning": reasoning}

        except Exception as e_fallback:
            logger.error(f"Both Wavelet Pro and basic fallback failed: {e_fallback}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Wavelet error: {str(e_fallback)[:80]}"
            }
        logger.warning(f"Wavelet model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}


def run_wavelet_basic(df: pd.DataFrame) -> Dict:
    """
    Basic 5-level wavelet signal (original model for comparison).
    
    Features:
    - 5-level DWT decomposition using Daubechies-4 (db4)
    - Simple trend detection via coefficient zeroing
    - Lightweight, fast execution
    - For comparison with WaveletPro (6-level + CWT + features)
    """
    try:
        prices = df["close"].values

        # Need at least 32 samples (2^5 for 5-level decomposition)
        if len(prices) < 32:
            return {"signal": "HOLD", "confidence": 0.0, "reasoning": "Insufficient data for 5-level DWT"}

        try:
            import pywt
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = pywt.wavedec(prices, "db4", level=5)
                # Zero out detail coefficients to denoise
                coeffs[1:] = [np.zeros_like(v) for v in coeffs[1:]]
                trend = pywt.waverec(coeffs, "db4")
            pywt_available = True
        except ImportError:
            # Proxy if pywt is unavailable
            trend = pd.Series(prices).rolling(window=10).mean().bfill().values
            pywt_available = False

        # Signal: compare last 3 denoised values to determine trend
        if len(trend) >= 3:
            slope = (trend[-1] - trend[-3]) / (abs(trend[-3]) + 1e-8)
            
            # Dynamic confidence based on slope magnitude
            abs_slope = abs(slope)
            # Base 50% + up to 45% based on slope
            base_conf = 0.50 + min(abs_slope * 400.0, 0.45)
            
            # Penalty if using fallback model
            confidence = base_conf if pywt_available else base_conf * 0.70

            if slope > 0.0001:
                signal = "LONG"
                reasoning = f"Basic 5-level: trend slope +{slope:.4f} → uptrend"
            elif slope < -0.0001:
                signal = "SHORT"
                reasoning = f"Basic 5-level: trend slope {slope:.4f} → downtrend"
            else:
                signal = "HOLD"
                confidence = 0.15
                reasoning = f"Basic 5-level: flat trend (slope={slope:.6f})"
        else:
            signal = "HOLD"
            confidence = 0.0
            reasoning = "Insufficient trend data"

        return {"signal": signal, "confidence": round(confidence, 3), "reasoning": reasoning}

    except Exception as e:
        logger.error(f"Basic wavelet failed: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}


_hmm_detector = None

def run_hmm(df: pd.DataFrame) -> Dict:
    """HMM regime detector: uses the real RegimeDetector.generate_signal()."""
    global _hmm_detector
    try:
        from src.models.hmm_regime import RegimeDetector

        if _hmm_detector is None:
            _hmm_detector = RegimeDetector(n_regimes=5, n_iter=200)

        if not getattr(_hmm_detector, 'is_trained', False):
            _hmm_detector.train(df)

        result = _hmm_detector.generate_signal(df)
        output = result.to_dict()
        # Ensure regime is always in the output for downstream consumers
        if "regime" not in output or not output["regime"]:
            output["regime"] = "NORMAL"
        return output

    except Exception as e:
        logger.warning(f"HMM model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "regime": "UNKNOWN", "reasoning": f"Error: {str(e)[:80]}"}


# ── CNN-LSTM-Attention Model (Phase 8-10: Real Deep Learning Inference) ──
# Loaded once at import time. Falls back to heuristic proxy if no trained model exists.
_lstm_model = None
_lstm_preprocessor = None
_lstm_feature_engineer = None
_lstm_model_available = False

try:
    import os as _os
    from src.models.lstm_temporal import GoldLSTMModel
    from src.models.lstm_features import LSTMFeatureEngineer
    from src.models.lstm_preprocessor import LSTMPreprocessor

    _lstm_model_path = _os.path.abspath(_os.path.join(
        _os.path.dirname(__file__), '..', '..', 'models', 'lstm_cnn_attention.pt'
    ))
    _lstm_preprocessor_path = _os.path.abspath(_os.path.join(
        _os.path.dirname(__file__), '..', '..', 'models', 'lstm_preprocessor.joblib'
    ))

    if _os.path.exists(_lstm_model_path) and _os.path.exists(_lstm_preprocessor_path):
        _lstm_model = GoldLSTMModel()
        _lstm_model.load(_lstm_model_path)
        _lstm_preprocessor = LSTMPreprocessor.load(_lstm_preprocessor_path)
        _lstm_feature_engineer = LSTMFeatureEngineer()
        _lstm_model_available = True
        logger.info(f"CNN-LSTM-Attention model loaded from {_lstm_model_path}")
    else:
        logger.info("No trained LSTM model found — using heuristic proxy. "
                     "Run 'python scripts/train_lstm_model.py' to train.")
except Exception as _e:
    logger.warning(f"LSTM model loading failed: {_e} — using heuristic proxy")


def _run_lstm_heuristic(df: pd.DataFrame) -> Dict:
    """Fallback heuristic LSTM proxy (EMA/MACD-based)."""
    closes = df["close"].values
    returns = df["returns"].values

    alpha = 0.15
    ew_return = returns[-1]
    for r in reversed(returns[-20:-1]):
        ew_return = alpha * ew_return + (1 - alpha) * r

    if len(returns) >= 5:
        accel = np.mean(np.diff(returns[-5:]))
    else:
        accel = 0.0

    ema_fast = pd.Series(closes).ewm(span=8).mean().iloc[-1]
    ema_slow = pd.Series(closes).ewm(span=21).mean().iloc[-1]
    macd = ema_fast - ema_slow

    macro_adjustment = 0.0
    if "dxy_returns" in df.columns and "us10y_returns" in df.columns:
        dxy_momentum = df["dxy_returns"].iloc[-3:].sum() * 100
        yield_momentum = df["us10y_returns"].iloc[-3:].sum() * 100
        macro_adjustment = (dxy_momentum + yield_momentum) * -0.3

    score = np.sign(ew_return) * 0.4 + np.sign(macd) * 0.3 + np.sign(accel) * 0.1 + macro_adjustment
    confidence = min(abs(score) * 0.8 + 0.3, 0.92)

    if score > 0.3:
        signal = "LONG"
    elif score < -0.3:
        signal = "SHORT"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "confidence": round(float(confidence), 3),
        "reasoning": f"LSTM-proxy: MACD={macd:.2f}, MacroDrag={macro_adjustment:.2f}, score={score:.2f}",
    }


def run_lstm(df: pd.DataFrame) -> Dict:
    """
    CNN-LSTM-Attention model: real deep learning inference.

    If a trained model exists (models/lstm_cnn_attention.pt), runs real
    GPU/CPU inference through the CNN-LSTM-Attention hybrid architecture.
    Otherwise falls back to the heuristic LSTM proxy.
    """
    try:
        if not _lstm_model_available:
            return _run_lstm_heuristic(df)

        import time
        t0 = time.perf_counter()

        # 1. Feature engineering
        features = _lstm_feature_engineer.transform(df)

        if len(features) < _lstm_preprocessor.seq_len:
            logger.debug(f"LSTM: insufficient features ({len(features)} < {_lstm_preprocessor.seq_len}), using proxy")
            return _run_lstm_heuristic(df)

        # 2. Preprocess (scale + sequence)
        X = _lstm_preprocessor.transform_live(features)

        # 3. Model inference
        result = _lstm_model.predict(X, temperature=1.2)

        latency = (time.perf_counter() - t0) * 1000

        return {
            "signal": result["signal"],
            "confidence": round(float(result["confidence"]), 3),
            "reasoning": (
                f"CNN-LSTM-Attn: {result['signal']} "
                f"P(S={result['probabilities']['SHORT']:.0%}/"
                f"H={result['probabilities']['HOLD']:.0%}/"
                f"L={result['probabilities']['LONG']:.0%}) "
                f"{latency:.0f}ms"
            ),
        }

    except Exception as e:
        logger.warning(f"LSTM model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}



def run_tft(df: pd.DataFrame) -> Dict:
    """TFT forecaster: multi-scale attention-based signal."""
    try:
        closes = df["close"].values
        returns = df["returns"].values
        highs = df["high"].values if "high" in df.columns else closes
        lows = df["low"].values if "low" in df.columns else closes

        # Multi-window RSI (approximates TFT's multi-horizon attention)
        def rsi(series, period=14):
            delta = np.diff(series)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-period:]) + 1e-10
            avg_loss = np.mean(loss[-period:]) + 1e-10
            rs = avg_gain / avg_loss
            return 100 - 100 / (1 + rs)

        rsi_14 = rsi(closes, 14)
        rsi_7 = rsi(closes, 7)

        # ATR-normalized momentum
        atr = np.mean(np.abs(highs[-14:] - lows[-14:]))
        momentum_14 = (closes[-1] - closes[-14]) / (atr + 1e-8)

        # Bollinger band position
        bb_mean = np.mean(closes[-20:])
        bb_std = np.std(closes[-20:])
        bb_pos = (closes[-1] - bb_mean) / (2 * bb_std + 1e-8)  # -1 to 1

        # TFT attention-inspired weighting across time scales
        short_signal = np.sign(rsi_7 - 50) * (abs(rsi_7 - 50) / 50)
        long_signal = np.sign(rsi_14 - 50) * (abs(rsi_14 - 50) / 50)
        trend_signal = np.sign(momentum_14) * min(abs(momentum_14) / 5, 1.0)

        combined = short_signal * 0.3 + long_signal * 0.4 + trend_signal * 0.3
        # Limit contrarian BB adjustment
        bb_adj = np.clip(bb_pos * 0.15, -0.25, 0.25)
        combined -= bb_adj
        
        # TFT Macro Attention: Yield & DXY Curve Inversion
        macro_reasoning = ""
        if "dxy_returns" in df.columns and "us10y_returns" in df.columns:
            dxy_roc = df["dxy_returns"].iloc[-5:].mean() * 1000
            yield_roc = df["us10y_returns"].iloc[-5:].mean() * 1000
            
            if dxy_roc > 1.5 or yield_roc > 1.5:
                combined -= 0.4 # Severe bearish override
                macro_reasoning = f" (MACRO FEAR: DXY/US10Y Spiking)"
            elif dxy_roc < -1.5 or yield_roc < -1.5:
                combined += 0.4 # Severe bullish override
                macro_reasoning = f" (MACRO GREED: DXY/US10Y Dropping)"

        # Gold-Silver Ratio adjustment (Investopedia)
        if "gold_silver_ratio" in df.columns:
            gs_ratio = df["gold_silver_ratio"].iloc[-1]
            if gs_ratio > 85:
                combined -= 0.15
                macro_reasoning += f" (GSR High: {gs_ratio:.1f})"
            elif gs_ratio < 75:
                combined += 0.15
                macro_reasoning += f" (GSR Low: {gs_ratio:.1f})"

        confidence = min(abs(combined) * 0.9 + 0.25, 0.94)

        if combined > 0.2:
            signal = "LONG"
        elif combined < -0.2:
            signal = "SHORT"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "reasoning": f"TFT-proxy: RSI14={rsi_14:.1f}, RSI7={rsi_7:.1f}, BB_pos={bb_pos:.2f}, score={combined:.3f}{macro_reasoning}",
        }

    except Exception as e:
        logger.warning(f"TFT model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}


def run_genetic(df: pd.DataFrame) -> Dict:
    """Genetic algorithm: evolved rule-based signal voting."""
    try:
        closes = df["close"].values
        returns = df["returns"].values

        votes = []
        reasons = []

        # Rule 1: SMA crossover (evolved: 10/30)
        if len(closes) >= 30:
            sma10 = np.mean(closes[-10:])
            sma30 = np.mean(closes[-30:])
            if sma10 > sma30 * 1.001:
                votes.append(1)
                reasons.append(f"SMA10>{sma30:.0f}")
            elif sma10 < sma30 * 0.999:
                votes.append(-1)
                reasons.append(f"SMA10<{sma30:.0f}")
            else:
                votes.append(0)

        # Rule 2: Momentum rule (evolved: 5-day return threshold)
        if len(returns) >= 5:
            mom5 = np.sum(returns[-5:])
            if mom5 > 0.008:
                votes.append(1)
                reasons.append(f"Mom5d=+{mom5:.3f}")
            elif mom5 < -0.008:
                votes.append(-1)
                reasons.append(f"Mom5d={mom5:.3f}")
            else:
                votes.append(0)

        # Rule 3: Volume-weighted return (evolved)
        if "volume" in df.columns and len(df) >= 10:
            recent = df.iloc[-10:]
            vwap_ret = np.average(recent["returns"], weights=recent["volume"] + 1)
            if vwap_ret > 0.001:
                votes.append(1)
                reasons.append(f"VWAP_ret=+{vwap_ret:.4f}")
            elif vwap_ret < -0.001:
                votes.append(-1)
                reasons.append(f"VWAP_ret={vwap_ret:.4f}")
            else:
                votes.append(0)

        # Rule 4: High-Low range expansion (volatility breakout)
        if len(closes) >= 20:
            range_now = df["high"].iloc[-1] - df["low"].iloc[-1] if "high" in df.columns else 0
            range_avg = (df["high"] - df["low"]).iloc[-20:].mean() if "high" in df.columns else 1
            if range_now > range_avg * 1.3 and returns[-1] > 0:
                votes.append(1)
                reasons.append("breakout_up")
            elif range_now > range_avg * 1.3 and returns[-1] < 0:
                votes.append(-1)
                reasons.append("breakout_dn")
            else:
                votes.append(0)

        # Rule 5: Recent reversal detection
        if len(returns) >= 3:
            if returns[-3] < -0.005 and returns[-2] < -0.003 and returns[-1] > 0.002:
                votes.append(1)
                reasons.append("reversal_long")
            elif returns[-3] > 0.005 and returns[-2] > 0.003 and returns[-1] < -0.002:
                votes.append(-1)
                reasons.append("reversal_short")
            else:
                votes.append(0)

        if not votes:
            return {"signal": "HOLD", "confidence": 0.0, "reasoning": "No rules fired"}

        score = np.mean(votes)
        # Confidence = agreement among rules
        agreement = abs(score)
        confidence = min(agreement * 0.85 + 0.20, 0.90)

        if score > 0.2:
            signal = "LONG"
        elif score < -0.2:
            signal = "SHORT"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "reasoning": f"Genetic ({len([v for v in votes if v != 0])}/{len(votes)} rules): " + ", ".join(reasons[:3]),
        }

    except Exception as e:
        logger.warning(f"Genetic model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}


from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import joblib
import os
from loguru import logger

# The Machine Learning Meta-Learner (preserved for future use)
# Currently DISABLED: the saved model was trained on proxy daily signals that
# don't match live 1-minute model outputs, causing contradictory predictions.
# To re-enable: retrain on actual live signal logs, then set _ml_validated = True.
_meta_learner_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'meta_learner.joblib'))
_meta_learner = None
_ml_validated = False  # Gate: only use ML if retrained on live data

try:
    _meta_learner = joblib.load(_meta_learner_path)
    logger.info(f"Loaded meta-learner from {_meta_learner_path} (DISABLED — using heuristic ensemble)")
except Exception as e:
    logger.info(f"No meta-learner found ({e}). Using heuristic ensemble.")


# ============================================================================
# REGIME-AWARE MODEL WEIGHTS
# ============================================================================

_REGIME_WEIGHTS = {
    "GROWTH": {
        "wavelet": 0.20, "hmm": 0.20, "lstm": 0.15,
        "tft": 0.13, "genetic": 0.13, "hmm_pro": 0.12,
    },
    "NORMAL": {
        "wavelet": 0.20, "hmm": 0.15, "lstm": 0.25,
        "tft": 0.18, "genetic": 0.08, "hmm_pro": 0.15,
    },
    "CRISIS": {
        "wavelet": 0.15, "hmm": 0.25, "lstm": 0.18,
        "tft": 0.18, "genetic": 0.11, "hmm_pro": 0.13,
    },
}


def run_ensemble(individual_signals: Dict[str, Dict], regime: str = "NORMAL", macro_data: Optional[Dict] = None) -> Dict:
    """
    Confidence-Weighted Voting Ensemble v2.0

    Replaces the stale ML meta-learner with a transparent, robust ensemble:
    1. Regime-aware model weights
    2. Confidence-weighted directional scoring (LONG vs SHORT tug-of-war)
    3. Agreement bonus / conflict penalty
    4. Minimum quorum (≥2 directional votes)
    5. Macro data adjustment (DXY / yields)
    
    NOTE: Handles both WaveletPro and basic wavelet for comparison.
          Uses the wavelet signal with higher confidence for ensemble voting.
    """
    try:
        # Prepare individual signals dict with unified wavelet vote
        signals_for_ensemble = individual_signals.copy()
        
        # If both wavelet models are present, use the one with higher confidence
        wavelet_pro_sig = individual_signals.get("wavelet_pro", {})
        wavelet_basic_sig = individual_signals.get("wavelet_basic", {})
        
        if wavelet_pro_sig and wavelet_basic_sig:
            pro_conf = float(wavelet_pro_sig.get("confidence", 0.0))
            basic_conf = float(wavelet_basic_sig.get("confidence", 0.0))
            # Use the wavelet model with higher confidence as the unified "wavelet" vote
            best_wavelet = wavelet_pro_sig if pro_conf >= basic_conf else wavelet_basic_sig
            signals_for_ensemble["wavelet"] = best_wavelet
        elif wavelet_pro_sig:
            signals_for_ensemble["wavelet"] = wavelet_pro_sig
        elif wavelet_basic_sig:
            signals_for_ensemble["wavelet"] = wavelet_basic_sig
        
        models = ["wavelet", "hmm", "lstm", "tft", "genetic", "hmm_pro"]

        # ── 1. Get regime-specific weights ──
        weights = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["NORMAL"]).copy()

        # ── 2. Collect votes and compute directional scores ──
        long_score = 0.0
        short_score = 0.0
        hold_score = 0.0
        long_voters = []
        short_voters = []
        hold_voters = []

        for model in models:
            sig = signals_for_ensemble.get(model, {})
            direction = sig.get("signal", "HOLD")
            conf = float(sig.get("confidence", 0.0))
            w = weights.get(model, 0.10)
            weighted_vote = w * conf

            if direction == "LONG":
                long_score += weighted_vote
                long_voters.append((model, conf))
            elif direction == "SHORT":
                short_score += weighted_vote
                short_voters.append((model, conf))
            else:
                hold_score += weighted_vote
                hold_voters.append((model, conf))

        n_long = len(long_voters)
        n_short = len(short_voters)
        n_hold = len(hold_voters)

        # ── 3. Net directional score ──
        net_score = long_score - short_score
        abs_net = abs(net_score)

        # Determine raw signal from net direction
        if abs_net < 0.01:
            # Scores are effectively equal — no clear direction
            raw_signal = "HOLD"
            raw_confidence = 0.0
        elif net_score > 0:
            raw_signal = "LONG"
            raw_confidence = abs_net
        else:
            raw_signal = "SHORT"
            raw_confidence = abs_net

        # ── 4. Quorum check: need ≥2 models voting in the winning direction ──
        winning_count = n_long if raw_signal == "LONG" else n_short
        if raw_signal != "HOLD" and winning_count < 2:
            raw_signal = "HOLD"
            raw_confidence = raw_confidence * 0.3
            quorum_note = "QuorumFail"
        else:
            quorum_note = ""

        # ── 5. Agreement / conflict modifiers ──
        total_directional = n_long + n_short
        agreement_mult = 1.0

        if total_directional > 0:
            # Check if models are fighting each other
            if n_long >= 2 and n_short >= 2:
                # Significant conflict — penalize confidence
                agreement_mult = 0.65
            elif winning_count >= 3:
                # Strong agreement — boost confidence
                agreement_mult = 1.25
            elif winning_count >= 4:
                # Very strong agreement
                agreement_mult = 1.40

        # ── 6. Consensus override ──
        # If 3+ models agree on a direction AND the ensemble disagrees, force it
        consensus_note = ""
        if n_long >= 3 and raw_signal != "LONG":
            avg_long_conf = np.mean([c for _, c in long_voters])
            if avg_long_conf > 0.40:
                raw_signal = "LONG"
                raw_confidence = long_score
                agreement_mult = 1.20
                consensus_note = f"ConsensusOverride({n_long}L)"
        elif n_short >= 3 and raw_signal != "SHORT":
            avg_short_conf = np.mean([c for _, c in short_voters])
            if avg_short_conf > 0.40:
                raw_signal = "SHORT"
                raw_confidence = short_score
                agreement_mult = 1.20
                consensus_note = f"ConsensusOverride({n_short}S)"

        # ── 7. Macro adjustment ──
        macro_note = ""
        if macro_data and raw_signal != "HOLD":
            dxy_mom = macro_data.get("dxy_momentum", 0.0)
            yield_mom = macro_data.get("yield_momentum", 0.0)

            # Strong dollar + rising yields = bearish gold
            macro_pressure = (dxy_mom + yield_mom) * 0.5

            if abs(macro_pressure) > 0.5:
                if macro_pressure > 0 and raw_signal == "LONG":
                    # Macro headwind against LONG
                    agreement_mult *= 0.85
                    macro_note = f"MacroHeadwind(DXY={dxy_mom:.2f})"
                elif macro_pressure < 0 and raw_signal == "SHORT":
                    # Macro headwind against SHORT
                    agreement_mult *= 0.85
                    macro_note = f"MacroHeadwind(DXY={dxy_mom:.2f})"
                elif macro_pressure > 0 and raw_signal == "SHORT":
                    # Macro tailwind for SHORT
                    agreement_mult *= 1.10
                    macro_note = f"MacroTailwind(DXY={dxy_mom:.2f})"
                elif macro_pressure < 0 and raw_signal == "LONG":
                    # Macro tailwind for LONG
                    agreement_mult *= 1.10
                    macro_note = f"MacroTailwind(DXY={dxy_mom:.2f})"

        # ── 8. Final confidence ──
        final_confidence = raw_confidence * agreement_mult

        # Scale to 0-1 range: the theoretical max weighted score is ~0.25 (one model at 100%)
        # In practice with multiple models, scores range 0.05 - 0.35
        final_confidence = min(final_confidence / 0.25, 1.0)
        final_confidence = max(0.0, min(final_confidence, 0.95))

        final_signal = raw_signal if final_confidence >= 0.10 else "HOLD"

        # ── 9. Build reasoning ──
        parts = [
            f"Ensemble({n_long}L/{n_short}S/{n_hold}H)",
            f"net={net_score:+.3f}",
            f"L={long_score:.3f}/S={short_score:.3f}/H={hold_score:.3f}",
        ]
        for note in [quorum_note, consensus_note, macro_note]:
            if note:
                parts.append(note)

        return {
            "signal": final_signal,
            "confidence": round(float(final_confidence), 3),
            "reasoning": " | ".join(parts),
        }

    except Exception as e:
        logger.warning(f"Ensemble error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}



# ============================================================================
# LIVE INFERENCE LOOP
# ============================================================================

class LiveInferenceLoop:
    """
    Background asyncio task that runs all 6 models on live gold data.

    Usage:
        loop = LiveInferenceLoop(engine, broadcast_fn)
        task = asyncio.create_task(loop.run())
        # To stop:
        loop.stop()
        await task
    """

    def __init__(
        self,
        engine,
        broadcast_fn: Optional[Callable] = None,
        interval_seconds: int = 60,
    ):
        self.engine = engine
        self.broadcast_fn = broadcast_fn  # async fn(event_type, data)
        self.interval_seconds = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.iteration = 0
        self.last_run: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0

    def stop(self):
        """Signal the loop to stop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def run(self):
        """Main inference loop — runs until stopped."""
        self._running = True
        logger.info(f"Live inference loop started (interval={self.interval_seconds}s)")

        while self._running:
            try:
                await self._run_cycle()
                self.iteration += 1
                self.last_run = datetime.now()
                self.last_error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Inference loop error (iteration {self.iteration}): {e}")

            # Wait for next cycle (in chunks so we respond to stop() quickly)
            for _ in range(self.interval_seconds * 2):  # 0.5s chunks
                if not self._running:
                    break
                await asyncio.sleep(0.5)

        logger.info(f"Live inference loop stopped after {self.iteration} iterations")

    async def _run_cycle(self):
        """Execute one full inference cycle across all 6 models."""
        global CURRENT_GOLD_PRICE, LAST_PRICE_UPDATE

        # 1. Fetch live gold data (run in thread pool to avoid blocking)
        # Bug 5 Fix: Changed interval from 15m to 1m so the 60s loop gets fresh data every tick
        df = await asyncio.get_event_loop().run_in_executor(
            None, fetch_live_gold_data, "5d", "1m"
        )

        if df is None or df.empty or len(df) < 35:
            logger.warning("Skipping inference cycle: insufficient gold data")
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3 and self.engine and self.engine.status == "RUNNING":
                logger.critical("Data feed down! Triggering GRACEFUL DEGRADATION -> HALTING TRADING!")
                # Liquidate open positions
                if self.engine.current_position:
                    self.engine._close_position(datetime.now(), CURRENT_GOLD_PRICE)
                self.engine.stop()
            return

        # Reset failures on success
        self.consecutive_failures = 0

        # Hybrid Approach: Override latest close price with true real-time spot from MetalPriceAPI
        spot_price = await asyncio.get_event_loop().run_in_executor(
            None, fetch_metalpriceapi_spot
        )
        
        if spot_price is not None and spot_price > 0:
            CURRENT_GOLD_PRICE = spot_price
            # Inject into the dataframe so the models see the exact current price
            df.loc[df.index[-1], "close"] = spot_price
            logger.debug(f"Hybrid mode: using MetalPriceAPI spot price ${spot_price:,.2f}")
        else:
            CURRENT_GOLD_PRICE = float(df["close"].iloc[-1])
            
        LAST_PRICE_UPDATE = datetime.now()
        current_price = CURRENT_GOLD_PRICE

        if self.engine and self.engine.status == "RUNNING":
            self.engine.update_price(current_price, datetime.now())

        logger.info(f"[Inference #{self.iteration}] Gold @ ${current_price:.2f} — running 8 models (2 wavelet variants)...")

        # 2. Run individual models (in executor to avoid blocking event loop)
        loop = asyncio.get_event_loop()

        wavelet_pro_res = await loop.run_in_executor(None, run_wavelet, df)
        wavelet_basic_res = await loop.run_in_executor(None, run_wavelet_basic, df)
        hmm_res = await loop.run_in_executor(None, run_hmm, df)
        lstm_res = await loop.run_in_executor(None, run_lstm, df)
        tft_res = await loop.run_in_executor(None, run_tft, df)
        genetic_res = await loop.run_in_executor(None, run_genetic, df)
        hmm_pro_res = await loop.run_in_executor(None, run_hmm_pro, df)

        individual = {
            "wavelet_pro": wavelet_pro_res,    # 6-level DWT + CWT + features (new)
            "wavelet_basic": wavelet_basic_res, # 5-level DWT only (original)
            "hmm": hmm_res,
            "lstm": lstm_res,
            "tft": tft_res,
            "genetic": genetic_res,
            "hmm_pro": hmm_pro_res,
        }

        # 3. Get Regime for Meta-Labeling
        regime = hmm_res.get("regime", "NORMAL")
        now_iso = datetime.now().isoformat()
        # Build macro data for the ML ensemble
        macro_data = {
            "dxy_momentum": float(df["dxy_returns"].iloc[-3:].sum() * 100) if "dxy_returns" in df.columns else 0.0,
            "yield_momentum": float(df["us10y_returns"].iloc[-3:].sum() * 100) if "us10y_returns" in df.columns else 0.0,
        }
        
        # Run ensemble synchronously with dynamic regime weights
        ensemble_res = run_ensemble(individual, regime, macro_data)

        all_results = {**individual, "ensemble": ensemble_res}

        # Check circuit breakers using the global risk manager if available
        _can_trade = True
        import src.api.paper_trading_routes as routes
        _risk_manager = getattr(routes, "_risk_manager", None)
        if _risk_manager is not None and self.engine is not None and self.engine.status == "RUNNING":
            # Set the daily PNL and update equity of risk manager before check
            _risk_manager.risk_state.daily_pnl = self.engine.daily_pnl
            _risk_manager.update_equity(self.engine._create_portfolio_snapshot().total_value)
            
            # Increment bars in risk manager to keep in sync
            if not hasattr(_risk_manager.risk_state, "bars_since_last_trade"):
                _risk_manager.risk_state.bars_since_last_trade = 100
            
            _can_trade, _reason = _risk_manager.check_circuit_breakers(
                portfolio_value=self.engine._create_portfolio_snapshot().total_value,
                ensemble_conf=float(ensemble_res.get("confidence", 0.0))
            )
            if not _can_trade:
                logger.debug(f"Prediction cycle trade check blocked by RiskManager: {_reason}")

        # ── CSV LOG: record this prediction cycle ──
        _trade_taken = (
            self.engine is not None
            and self.engine.status == "RUNNING"
            and ensemble_res.get("signal") in ("LONG", "SHORT")
            and float(ensemble_res.get("confidence", 0)) >= (self.engine.config.min_confidence if self.engine else 0.6)
            and _can_trade
        )
        _kelly = self.engine.config.kelly_fraction if self.engine else None
        log_prediction_cycle(
            price=current_price,
            regime=regime,
            all_signals=all_results,
            kelly_fraction=_kelly,
            trade_taken=_trade_taken,
        )

        # 4. Update LIVE_MODEL_SIGNALS registry

        for model, res in all_results.items():
            LIVE_MODEL_SIGNALS[model].update({
                "signal": res.get("signal", "HOLD"),
                "confidence": res.get("confidence", 0.0),
                "regime": res.get("regime", regime),
                "price": current_price,
                "reasoning": res.get("reasoning", ""),
                "last_updated": now_iso,
                "error": None,
            })

        # 4. Feed signals into paper trading engine (if running)
        if self.engine and self.engine.status == "RUNNING":
            from src.paper_trading.engine import ModelSignal, SignalType

            if _risk_manager is not None:
                MIN_BARS_BETWEEN_TRADES = getattr(_risk_manager, "min_bars_between_trades", 10)
            else:
                MIN_BARS_BETWEEN_TRADES = 10

            if not hasattr(self, "bars_since_last_trade"):
                self.bars_since_last_trade = MIN_BARS_BETWEEN_TRADES
            
            self.bars_since_last_trade += 1
            if _risk_manager is not None:
                _risk_manager.risk_state.bars_since_last_trade = self.bars_since_last_trade

            for model_name, res in all_results.items():
                try:
                    signal_val = res.get("signal", "HOLD")
                    confidence = float(res.get("confidence", 0.0))

                    # Only the ensemble model is allowed to execute actual trades!
                    # The other 5 individual models just register their status for the dashboard
                    # to prevent them from constantly whipsawing the single paper trading position.
                    
                    if model_name == "ensemble":
                        # Apply cooldown
                        if self.bars_since_last_trade < MIN_BARS_BETWEEN_TRADES and signal_val in ("LONG", "SHORT"):
                            # Demote to HOLD if in cooldown
                            signal_val = "HOLD"
                            
                        if signal_val in ("LONG", "SHORT") and confidence >= self.engine.config.min_confidence and _trade_taken:
                            sig = ModelSignal(
                                model_name=model_name,
                                signal_type=SignalType(signal_val),
                                confidence=confidence,
                                entry_price=current_price,
                                current_price=current_price,
                                timestamp=datetime.now(),
                                reasoning=res.get("reasoning", ""),
                                regime=str(res.get("regime", regime)),
                            )
                            trade_res = self.engine.process_signal(model_name, sig)
                            if trade_res:
                                self.bars_since_last_trade = 0  # Reset cooldown when trade executes
                                if _risk_manager is not None:
                                    _risk_manager.risk_state.bars_since_last_trade = 0
                            continue
                            
                    # Register the signal for dashboard display without executing a trade
                    from src.paper_trading.engine import SignalType as ST
                    sig = ModelSignal(
                        model_name=model_name,
                        signal_type=ST(signal_val) if signal_val in ("LONG", "SHORT", "HOLD") else ST.HOLD,
                        confidence=confidence,
                        entry_price=current_price,
                        current_price=current_price,
                        timestamp=datetime.now(),
                        reasoning=res.get("reasoning", ""),
                        regime=regime,
                    )
                    self.engine.last_signals[model_name] = sig
                    self.engine.signal_history[model_name].append(sig)

                except Exception as e:
                    logger.warning(f"Failed to process {model_name} signal: {e}")

        # 5. Broadcast model update via WebSocket
        if self.broadcast_fn:
            from src.models.rl_execution_agent import get_rl_agent
            rl_params = get_rl_agent().get_execution_parameters(regime, 0.02, ensemble_res["confidence"])
            
            dxy_val = float(df["dxy"].iloc[-1]) if df is not None and "dxy" in df.columns else 0.0
            us10y_val = float(df["us10y"].iloc[-1]) if df is not None and "us10y" in df.columns else 0.0
            gsr_val = float(df["gold_silver_ratio"].iloc[-1]) if df is not None and "gold_silver_ratio" in df.columns else 0.0
            
            MACRO_DATA["dxy"] = dxy_val
            MACRO_DATA["us10y"] = us10y_val
            MACRO_DATA["gold_silver_ratio"] = gsr_val
            MACRO_DATA["rl_kelly"] = rl_params["kelly_multiplier"]
            MACRO_DATA["rl_trailing"] = rl_params["trailing_stop_pct"]
            
            broadcast_payload = {
                "price": current_price,
                "regime": regime,
                "timestamp": now_iso,
                "macro": MACRO_DATA,
                "models": {
                    m: {
                        "signal": v["signal"],
                        "confidence": v["confidence"],
                        "reasoning": v["reasoning"],
                    }
                    for m, v in LIVE_MODEL_SIGNALS.items()
                },
            }
            try:
                await self.broadcast_fn("model_signals_update", broadcast_payload)
            except Exception as e:
                logger.debug(f"Broadcast failed: {e}")

        # Log summary
        sig_summary = " | ".join(
            f"{m[:3].upper()}:{v['signal']}@{v['confidence']:.0%}"
            for m, v in all_results.items()
        )
        logger.info(f"  Signals: {sig_summary}")
