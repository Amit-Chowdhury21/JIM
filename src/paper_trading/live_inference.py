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
    for model in ["wavelet_pro", "wavelet_basic", "hmm", "lstm", "lstm_v21", "tft", "tft_pro", "tft_pro_max", "genetic", "hmm_pro", "ensemble", "ensemble_v21", "ensemble_v23"]
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
                    df = df.reindex(df.index.union(missing_idx))
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

    # 0. Try tvDatafeed for instant OANDA XAUUSD spot price
    try:
        from tvDatafeed import TvDatafeed, Interval
        import logging
        logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)
        tv = TvDatafeed()
        df_tv = tv.get_hist(symbol='XAUUSD', exchange='OANDA', interval=Interval.in_1_minute, n_bars=1)
        if df_tv is not None and not df_tv.empty:
            price = df_tv["close"].iloc[-1]
            logger.debug(f"Fetched real-time gold spot price from tvDatafeed (OANDA): ${price:.2f}")
            return round(float(price), 2)
    except Exception as e:
        logger.debug(f"tvDatafeed fetch failed: {e}")

    # 1. Try free Gold-API.com first (unlimited, no key required, most reliable)
    try:
        resp = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("price")
            if price and float(price) > 0:
                logger.debug(f"Fetched real-time gold spot price from Gold-API.com: ${price:.2f}")
                return round(float(price), 2)
    except Exception as e:
        logger.debug(f"Gold-API.com fetch failed: {e}")

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

    # 0. Skip tvDatafeed for now - it's unreliable without login
    # Try free Gold-API.com first (unlimited, no key required)
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
        logger.debug(f"Gold-API.com (G/S) fetch failed: {e}")

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
    """
    try:
        return run_wavelet_pro(df)
    except Exception as e:
        logger.warning(f"Wavelet model error: {e}")
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


# ── CNN-LSTM-Attention Models ──
# v1.0 Model (Phase 1 Baseline)
_lstm_model_v1 = None
_lstm_preprocessor_v1 = None
_lstm_feature_engineer_v1 = None
_lstm_model_v1_available = False

# v2.x Models (Triple Barrier)
_lstm_model_v2 = None
_lstm_preprocessor_v2 = None
_lstm_feature_engineer_v2 = None
_lstm_model_v2_available = False

try:
    import os as _os
    from src.models.lstm_temporal import GoldLSTMModel
    from src.models.lstm_features import LSTMFeatureEngineer
    from src.models.lstm_preprocessor import LSTMPreprocessor

    # Load v1.0 Model
    _path_v1 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'models', 'lstm_cnn_attention_1m.pt'))
    _prep_v1 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'models', 'lstm_preprocessor_1m.joblib'))
    
    if _os.path.exists(_path_v1) and _os.path.exists(_prep_v1):
        try:
            _lstm_model_v1 = GoldLSTMModel()
            _lstm_model_v1.load(_path_v1)
        except Exception as _inner_e:
            if "kernel image is available" in str(_inner_e) or "CUDA error" in str(_inner_e):
                _lstm_model_v1 = GoldLSTMModel(device="cpu")
                _lstm_model_v1.load(_path_v1)
            else:
                raise _inner_e
        _lstm_preprocessor_v1 = LSTMPreprocessor.load(_prep_v1)
        _lstm_feature_engineer_v1 = LSTMFeatureEngineer()
        _lstm_model_v1_available = True
        logger.info(f"v1.0 Baseline Model loaded from {_path_v1}")

    # Load v2.0 Model
    _path_v2 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'models', 'lstm_v2.0_triple_barrier.pt'))
    _prep_v2 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'models', 'lstm_preprocessor_v2.0.joblib'))
    
    if _os.path.exists(_path_v2) and _os.path.exists(_prep_v2):
        try:
            _lstm_model_v2 = GoldLSTMModel()
            _lstm_model_v2.load(_path_v2)
        except Exception as _inner_e:
            if "kernel image is available" in str(_inner_e) or "CUDA error" in str(_inner_e):
                _lstm_model_v2 = GoldLSTMModel(device="cpu")
                _lstm_model_v2.load(_path_v2)
            else:
                raise _inner_e
        _lstm_preprocessor_v2 = LSTMPreprocessor.load(_prep_v2)
        _lstm_feature_engineer_v2 = LSTMFeatureEngineer()
        _lstm_model_v2_available = True
        logger.info(f"v2.x Model loaded from {_path_v2}")

except Exception as _e:
    logger.warning(f"LSTM model loading failed: {_e}")


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
        if not _lstm_model_v1_available:
            return _run_lstm_heuristic(df)

        import time
        t0 = time.perf_counter()

        # 1. Feature engineering
        features = _lstm_feature_engineer_v1.transform(df)

        if len(features) < _lstm_preprocessor_v1.seq_len:
            logger.debug(f"LSTM v1.0: insufficient features ({len(features)} < {_lstm_preprocessor_v1.seq_len}), using proxy")
            return _run_lstm_heuristic(df)

        # 2. Preprocess (scale + sequence)
        X = _lstm_preprocessor_v1.transform_live(features)

        # 3. Model inference
        result = _lstm_model_v1.predict(X, temperature=1.2)

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

def run_lstm_v2_base(df: pd.DataFrame) -> Dict:
    """v2.0 Base Execution Engine (Single pass on Triple Barrier Model)."""
    try:
        if not _lstm_model_v2_available:
            return _run_lstm_heuristic(df)

        import time
        t0 = time.perf_counter()

        features = _lstm_feature_engineer_v2.transform(df)
        if len(features) < _lstm_preprocessor_v2.seq_len:
            return _run_lstm_heuristic(df)

        X = _lstm_preprocessor_v2.transform_live(features)
        result = _lstm_model_v2.predict(X, temperature=1.2)
        
        latency = (time.perf_counter() - t0) * 1000
        return {
            "signal": result["signal"],
            "confidence": round(float(result["confidence"]), 3),
            "reasoning": (
                f"v2.0-Base: {result['signal']} "
                f"P(S={result['probabilities']['SHORT']:.0%}/"
                f"H={result['probabilities']['HOLD']:.0%}/"
                f"L={result['probabilities']['LONG']:.0%}) "
                f"{latency:.0f}ms"
            ),
        }
    except Exception as e:
        logger.warning(f"LSTM v2.0 Base error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}

def run_lstm_v21(df: pd.DataFrame) -> Dict:
    """
    v2.1 Execution Engine:
    Runs the CNN-LSTM-Attention model through the MCDropoutInferencer
    and filters it through the SignalPolicy.
    """
    try:
        from src.paper_trading.v21_execution import SignalPolicy
        policy = SignalPolicy(long_threshold=0.45, short_threshold=0.65, max_uncertainty=0.08)
        
        if not _lstm_model_v2_available:
            return policy._reject("HOLD", "Model not loaded (heuristic proxy active)")

        import time
        t0 = time.perf_counter()

        features = _lstm_feature_engineer_v2.transform(df)
        if len(features) < _lstm_preprocessor_v2.seq_len:
            return policy._reject("HOLD", f"Insufficient features ({len(features)})")

        X = _lstm_preprocessor_v2.transform_live(features)
        
        # MCDropout (30 stochastic passes)
        mcd_result = _lstm_model_v2.predict_mcdropout(X, n_passes=30, temperature=1.2)
        
        # Apply strict v2.1 execution policy
        final_result = policy.evaluate(mcd_result)
        
        latency = (time.perf_counter() - t0) * 1000
        final_result["reasoning"] = f"{final_result['reasoning']} ({latency:.0f}ms)"
        
        return final_result

    except Exception as e:
        logger.warning(f"LSTM v2.1 model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)[:80]}"}

def run_lstm_v23(df: pd.DataFrame, regime: str) -> Dict:
    """
    v2.3 Execution Engine:
    Runs the CNN-LSTM-Attention model through the MCDropoutInferencer
    and filters it through the AlphaSignalPolicy (Regime, Volatility, Event-State).
    """
    try:
        from src.paper_trading.v23_execution import AlphaSignalPolicy
        policy = AlphaSignalPolicy(base_long=0.45, base_short=0.65, max_uncertainty=0.08)
        
        if not _lstm_model_v2_available:
            return policy._reject("HOLD", "Model not loaded (heuristic proxy active)")

        import time
        t0 = time.perf_counter()

        features = _lstm_feature_engineer_v2.transform(df)
        if len(features) < _lstm_preprocessor_v2.seq_len:
            return policy._reject("HOLD", f"Insufficient features ({len(features)})")

        X = _lstm_preprocessor_v2.transform_live(features)
        
        # MCDropout (30 stochastic passes)
        mcd_result = _lstm_model_v2.predict_mcdropout(X, n_passes=30, temperature=1.2)
        
        # Extract volatility context for v2.3 policy
        if "close" in df.columns:
            rets = df["close"].pct_change()
            vol_10 = rets.rolling(10).std().iloc[-1] if len(rets) >= 10 else 0.001
            vol_10_mean = rets.rolling(10).std().mean() if len(rets) >= 10 else 0.001
        else:
            vol_10 = 0.001
            vol_10_mean = 0.001
            
        if pd.isna(vol_10) or pd.isna(vol_10_mean):
            vol_10, vol_10_mean = 0.001, 0.001
            
        # Apply strict v2.3 execution policy
        final_result = policy.evaluate(mcd_result, regime, vol_10, vol_10_mean)
        
        latency = (time.perf_counter() - t0) * 1000
        final_result["reasoning"] = f"{final_result['reasoning']} ({latency:.0f}ms)"
        
        return final_result

    except Exception as e:
        logger.warning(f"LSTM v2.3 model error: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "sizing_scalar": 0.0, "reasoning": f"Error: {str(e)[:80]}"}






# ── TFT_Pro: professional Temporal Fusion Transformer ──────────────────────

_tft_pro_forecaster = None

def run_tft_pro(df: pd.DataFrame, regime: str = "NORMAL",
                wavelet_info: Optional[Dict] = None,
                hmm_info: Optional[Dict] = None) -> Dict:
    """
    TFT_Pro: Professional multi-horizon quantile forecaster.

    Architecture: GRN + VSN + BiLSTM + Multi-Head Attention + Quantile Heads
    Outputs: 3 horizons × 3 quantiles (10th, 50th, 90th percentile)
    """
    global _tft_pro_forecaster
    try:
        # Lazy-load the forecaster on first call
        if _tft_pro_forecaster is None:
            from src.models.tft_pro import TFTProForecaster
            _tft_pro_forecaster = TFTProForecaster()

        result = _tft_pro_forecaster.predict(
            df, regime=regime,
            wavelet_info=wavelet_info,
            hmm_info=hmm_info,
        )

        if result is not None:
            return result

        # Fallback: no trained model or inference failed
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": "[TFT_Pro fallback] Inference returned None"}

    except Exception as e:
        logger.warning(f"TFT_Pro error: {e} — falling back to HOLD")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"[TFT_Pro error: {str(e)[:40]}]"}

# ── TFT_Pro_Max: professional Temporal Fusion Transformer (Max) ──

_tft_pro_max_forecaster = None

def run_tft_pro_max(df: pd.DataFrame, regime: str = "NORMAL") -> Dict:
    global _tft_pro_max_forecaster
    try:
        if _tft_pro_max_forecaster is None:
            from src.models.tft_pro_max import TFTProMaxForecaster
            _tft_pro_max_forecaster = TFTProMaxForecaster()

        result = _tft_pro_max_forecaster.predict(
            df, regime=regime
        )

        if result is not None:
            return result

        return {"signal": "HOLD", "confidence": 0.0, "reasoning": "[TFT_Pro_Max fallback] Inference returned None"}

    except Exception as e:
        logger.warning(f"TFT_Pro_Max error: {e} — falling back to HOLD")
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"[TFT_Pro_Max error: {str(e)[:40]}]"}





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
# REGIME-AWARE ENSEMBLE ORCHESTRATION
# ============================================================================

from src.paper_trading.dynamic_weights import get_weight_adjuster
from src.models.meta_decision_layer import MetaDecisionLayer

def run_ensemble(individual_signals: Dict[str, Dict], regime: str = "NORMAL", macro_data: Optional[Dict] = None) -> Dict:
    """
    Professional Gold Ensemble 
    
    Uses DynamicWeightAdjuster (Base Weights + Multipliers) and MetaDecisionLayer 
    (Position Sizing + Risk Gates) to compute the final ensemble score.
    """
    try:
        models = ["wavelet_pro", "hmm_pro", "lstm", "tft_pro", "tft_pro_max"]
        
        # 1. Parse individual signals
        class MockSignal:
            def __init__(self, d):
                from src.paper_trading.engine import SignalType
                raw_sig = d.get("signal", "HOLD")
                sig_val = raw_sig.value if hasattr(raw_sig, "value") else raw_sig
                self.signal_type = SignalType(sig_val)
                self.confidence = float(d.get("confidence", 0.0))
                
        current_signals = {}
        for model in models:
            if model in individual_signals:
                current_signals[model] = MockSignal(individual_signals[model])
                
        # 2. Calculate dynamic weights
        adjuster = get_weight_adjuster()
        weights = adjuster.get_weights(regime, current_signals)
        
        # 3. Prepare inputs for MetaDecisionLayer
        decision_layer = MetaDecisionLayer()
        weighted_signals = {}
        signal_confidences = {}
        
        for name, sig in current_signals.items():
            if sig.signal_type == "LONG":
                sig_val = 1.0
            elif sig.signal_type == "SHORT":
                sig_val = -1.0
            else:
                sig_val = 0.0
                
            weighted_signals[name] = sig_val
            signal_confidences[name] = sig.confidence
            
        direction_scores = [v for v in weighted_signals.values() if v != 0]
        import numpy as np
        signal_disagreement = np.std(direction_scores) if len(direction_scores) > 1 else 0.0
        
        # 4. Make Meta Decision
        decision = decision_layer.make_meta_decision(
            weighted_signals=weighted_signals,
            signal_confidences=signal_confidences,
            model_weights=weights,
            disagreement_penalty=1.0,  # dynamic_weights already handles agreement
            signal_disagreement=signal_disagreement,
            regime=regime.lower(),
            current_price=0.0,
            atr=0.0,
        )
        
        # 5. Map back to expected signal format
        if decision.position_side.value == "long":
            final_signal = "LONG"
        elif decision.position_side.value == "short":
            final_signal = "SHORT"
        else:
            final_signal = "HOLD"
            
        return {
            "signal": final_signal,
            "confidence": round(decision.ensemble_confidence, 3),
            "reasoning": f"Ensemble: {decision.reasoning} | W: { {k: round(v,2) for k,v in weights.items()} }"
        }
        
    except Exception as e:
        logger.warning(f"Ensemble execution error: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Ensemble Error: {e}"}


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

        logger.info(f"[Inference #{self.iteration}] Gold @ ${current_price:.2f} — running models...")

        # 2. Run individual models (in executor to avoid blocking event loop)
        loop = asyncio.get_event_loop()

        wavelet_res = await loop.run_in_executor(None, run_wavelet, df)
        hmm_res = await loop.run_in_executor(None, run_hmm, df)
        lstm_v1_res = await loop.run_in_executor(None, run_lstm, df)
        lstm_v2_base_res = await loop.run_in_executor(None, run_lstm_v2_base, df)
        lstm_v21_res = await loop.run_in_executor(None, run_lstm_v21, df)
        hmm_pro_res = await loop.run_in_executor(None, run_hmm_pro, df)

        # TFT_Pro needs regime + model outputs from HMM/Wavelet for cross-model features
        regime_for_tft = hmm_res.get("regime", "NORMAL")
        tft_pro_res = await loop.run_in_executor(
            None, run_tft_pro, df, regime_for_tft, wavelet_res, hmm_res
        )
        lstm_v23_res = await loop.run_in_executor(None, run_lstm_v23, df, regime_for_tft)
        tft_pro_max_res = await loop.run_in_executor(
            None, run_tft_pro_max, df, regime_for_tft
        )

        individual_v1 = {
            "wavelet_pro": wavelet_res,
            "hmm": hmm_res,
            "lstm": lstm_v1_res,
            "tft_pro": tft_pro_res,
            "tft_pro_max": tft_pro_max_res,
            "hmm_pro": hmm_pro_res,
        }
        
        individual_v2 = {
            "wavelet_pro": wavelet_res,
            "hmm": hmm_res,
            "lstm": lstm_v2_base_res,
            "tft_pro": tft_pro_res,
            "tft_pro_max": tft_pro_max_res,
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
        
        # Run v2.0 ensemble using lstm_v2_base
        ensemble_res = run_ensemble(individual_v2, regime, macro_data)
        
        # Create specialized v2.1 ensemble that uses the gated MCDropout signal
        individual_v21 = individual_v2.copy()
        individual_v21["lstm"] = lstm_v21_res
        ensemble_v21_res = run_ensemble(individual_v21, regime, macro_data)

        # Create specialized v2.3 ensemble that uses the Alpha Execution Policy
        individual_v23 = individual_v2.copy()
        individual_v23["lstm"] = lstm_v23_res
        ensemble_v23_res = run_ensemble(individual_v23, regime, macro_data)

        all_results = {
            **individual_v1, 
            "lstm": lstm_v1_res,
            "ensemble": ensemble_res, 
            "ensemble_v21": ensemble_v21_res,
            "ensemble_v23": ensemble_v23_res
        }

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
                    
                    if model_name in ("ensemble", "ensemble_v21", "ensemble_v23", "lstm"):
                        # Apply cooldown
                        if self.bars_since_last_trade < MIN_BARS_BETWEEN_TRADES and signal_val in ("LONG", "SHORT"):
                            # Demote to HOLD if in cooldown
                            signal_val = "HOLD"
                            
                        # ONLY EXECUTE IF AUTO TRADE IS ENABLED
                        can_execute = getattr(self.engine, "auto_trade_enabled", False)
                            
                        if can_execute and signal_val in ("LONG", "SHORT") and confidence >= self.engine.config.min_confidence and _trade_taken:
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
                    if model_name not in self.engine.signal_history:
                        self.engine.signal_history[model_name] = []
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

        # 4.5 CSV Logging (Hourly Rotation in E:\PRO\JIMxNik\LSTMLogs)
        try:
            import os
            import pytz
            
            log_dir = r"E:\PRO\JIMxNik\LSTMLogs"
            os.makedirs(log_dir, exist_ok=True)
            
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(pytz.utc).astimezone(ist)
            
            filename = f"lstm_logs_{now_ist.strftime('%Y%m%d_%H')}.csv"
            filepath = os.path.join(log_dir, filename)
            
            file_exists = os.path.exists(filepath)
            
            v1_sig = all_results.get("lstm", {}).get("signal", "HOLD")
            v2_sig = all_results.get("ensemble", {}).get("signal", "HOLD")
            v21_sig = all_results.get("ensemble_v21", {}).get("signal", "HOLD")
            v23_sig = all_results.get("ensemble_v23", {}).get("signal", "HOLD")
            
            exec_str = "TRADED" if getattr(self, "bars_since_last_trade", -1) == 0 else "HOLD"
            
            pnl_val = 0.0
            if self.engine and hasattr(self.engine, "engines"):
                # Total PnL across engines
                pnl_val = sum(e.daily_pnl for e in self.engine.engines.values())
            
            with open(filepath, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("timestamp(ist),price,v1.0,v2.0,v2.1,v2.3,Exec,PnL\n")
                f.write(f"{now_ist.strftime('%Y-%m-%d %H:%M:%S')},{current_price:.2f},{v1_sig},{v2_sig},{v21_sig},{v23_sig},{exec_str},{pnl_val:.2f}\n")
                
        except Exception as e:
            logger.error(f"Failed to write LSTM Logs: {e}")

        # Log summary
        sig_summary = " | ".join(
            f"{m[:3].upper()}:{v['signal']}@{v['confidence']:.0%}"
            for m, v in all_results.items()
        )
        logger.info(f"  Signals: {sig_summary}")
