"""
Hidden Markov Model - Regime Detector v3.0 (Super Upgrade)
============================================================
Jim Simons' Insight:
  "The market has different states. You don't trade the same way
   in a crisis as you do in a bull run. The first job is to know
   which state you're in."

Architecture (v3.0):
  1. Multi-Timeframe Regime Fusion (1m + 5m + 15m consensus)
  2. 5 Internal Regimes -> 3 External (GROWTH / NORMAL / CRISIS)
  3. HMM Ensemble (3 models with different configurations)
  4. Regime Transition Velocity tracking (early warning)
  5. Regime-Conditional Micro-Models (per-regime indicator sets)
  6. Soft Probabilities (calibrated confidence from full posterior)
  7. DXY Lead-Lag exploitation (anticipatory signals)

Internal Regimes (5, ordered by volatility):
  0 = QUIET_RANGE   -> GROWTH  (calm, mean-reverting)
  1 = TRENDING_UP   -> GROWTH  (strong directional trend)
  2 = NORMAL        -> NORMAL  (balanced, mixed signals)
  3 = VOLATILE_MR   -> CRISIS  (high-vol, mean-reverting)
  4 = CRISIS_TREND  -> CRISIS  (high-vol, strong down-trend)

External Interface (unchanged, backward compatible):
  regime in {GROWTH, NORMAL, CRISIS}
  signal in {LONG, SHORT, HOLD}
  confidence in [0.0, 1.0]
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, Counter
import warnings
from loguru import logger

# Suppress hmmlearn degenerate covariance warnings
warnings.filterwarnings("ignore", module="hmmlearn")

from src.models.base import BaseModel, ModelOutput

# GPU acceleration
try:
    from src.utils.gpu import get_dataframe_engine, get_array_engine
    pd_engine = get_dataframe_engine()
    cp = get_array_engine()
except ImportError:
    pd_engine = pd
    cp = np

# HMM library
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
    # Check for GMMHMM availability
    try:
        _test_gmmhmm = hmm.GMMHMM(n_components=2, n_mix=2)
        GMMHMM_AVAILABLE = True
    except Exception:
        GMMHMM_AVAILABLE = False
        logger.warning("GMMHMM not available in hmmlearn. RegimeDetector will fallback to GaussianHMM.")
except ImportError:
    HMM_AVAILABLE = False
    GMMHMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. RegimeDetector will use a simplified fallback.")


class RegimeDetector(BaseModel):
    """
    Production HMM-based market regime detector for XAU/USD - v3.0.

    Multi-timeframe, multi-model regime intelligence system optimized
    for 1-minute gold scalping. Runs 5 HMM models in parallel and
    fuses their regime beliefs for maximum accuracy.
    """

    # External regime names (backward compatible)
    REGIME_NAMES = {0: "GROWTH", 1: "NORMAL", 2: "CRISIS"}
    REGIME_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

    # Internal 5-regime classification
    REGIME_NAMES_INTERNAL = {
        0: "QUIET_RANGE",
        1: "TRENDING_UP",
        2: "NORMAL",
        3: "VOLATILE_MR",
        4: "CRISIS_TREND",
    }

    # Map internal -> external
    REGIME_EXTERNAL_MAP = {
        "QUIET_RANGE": "GROWTH",
        "TRENDING_UP": "GROWTH",
        "NORMAL": "NORMAL",
        "VOLATILE_MR": "CRISIS",
        "CRISIS_TREND": "CRISIS",
    }

    # Reverse map: external ID -> list of internal IDs
    _EXT_TO_INT = {0: [0, 1], 1: [2], 2: [3, 4]}  # GROWTH, NORMAL, CRISIS

    # Strategy mapping per regime (Jim Simons' framework)
    REGIME_STRATEGIES = {
        "GROWTH": {
            "approach": "Mean Reversion",
            "kelly_fraction": 0.65,
            "position_bias": "LONG",
            "description": "Low-vol environment, buy dips, revert to mean",
        },
        "NORMAL": {
            "approach": "Balanced",
            "kelly_fraction": 0.50,
            "position_bias": "NEUTRAL",
            "description": "Mixed signals, standard position sizing",
        },
        "CRISIS": {
            "approach": "Trend Following",
            "kelly_fraction": 0.25,
            "position_bias": "DEFENSIVE",
            "description": "High-vol, follow the trend, minimize exposure",
        },
    }

    def __init__(
        self,
        n_regimes: int = 5,
        covariance_type: str = "diag",
        n_iter: int = 1000,
        min_regime_duration: int = 30,
        regime_persistence_weight: float = 0.3,
        vol_windows: Optional[List[int]] = None,
        version: str = "3.0",
    ):
        super().__init__(name="RegimeDetector", version=version)
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_regime_duration = min_regime_duration
        self.regime_persistence_weight = regime_persistence_weight
        self.vol_windows = vol_windows or [10, 20, 50]

        # ── Primary HMM (5-regime on 1m data) ──
        self.model: Optional[Any] = None
        if GMMHMM_AVAILABLE:
            try:
                self.model = hmm.GMMHMM(
                    n_components=n_regimes,
                    n_mix=3,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=42,
                    tol=1e-4,
                )
            except Exception as e:
                logger.warning(f"GMMHMM initialization failed: {e}. Falling back to GaussianHMM.")
                self.model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=42,
                    tol=1e-4,
                )
        elif HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=42,
                tol=1e-4,
            )

        # ── HMM Ensemble (Upgrade 7): two additional 3-regime models ──
        self._ensemble_models: List[Any] = []
        self._ensemble_orders: List[Optional[Dict]] = []
        if GMMHMM_AVAILABLE:
            try:
                # Model A: 3-regime, 3-mix, diagonal covariance (coarse, stable)
                self._ensemble_models.append(hmm.GMMHMM(
                    n_components=3, n_mix=3, covariance_type="diag",
                    n_iter=n_iter, random_state=42, tol=1e-4,
                ))
                self._ensemble_orders.append(None)
                # Model B: 3-regime, 3-mix, full covariance (captures correlations)
                self._ensemble_models.append(hmm.GMMHMM(
                    n_components=3, n_mix=3, covariance_type="full",
                    n_iter=n_iter, random_state=42, tol=1e-4,
                ))
                self._ensemble_orders.append(None)
            except Exception as e:
                logger.warning(f"GMMHMM ensemble initialization failed: {e}. Falling back to GaussianHMM.")
                self._ensemble_models = []
                self._ensemble_orders = []
                # Model A: 3-regime, diagonal covariance (coarse, stable)
                self._ensemble_models.append(hmm.GaussianHMM(
                    n_components=3, covariance_type="diag",
                    n_iter=n_iter, random_state=42, tol=1e-4,
                ))
                self._ensemble_orders.append(None)
                # Model B: 3-regime, full covariance (captures correlations)
                self._ensemble_models.append(hmm.GaussianHMM(
                    n_components=3, covariance_type="full",
                    n_iter=n_iter, random_state=42, tol=1e-4,
                ))
                self._ensemble_orders.append(None)
        elif HMM_AVAILABLE:
            # Model A: 3-regime, diagonal covariance (coarse, stable)
            self._ensemble_models.append(hmm.GaussianHMM(
                n_components=3, covariance_type="diag",
                n_iter=n_iter, random_state=42, tol=1e-4,
            ))
            self._ensemble_orders.append(None)
            # Model B: 3-regime, full covariance (captures correlations)
            self._ensemble_models.append(hmm.GaussianHMM(
                n_components=3, covariance_type="full",
                n_iter=n_iter, random_state=42, tol=1e-4,
            ))
            self._ensemble_orders.append(None)

        # Fusion weights: [primary, ens_A, ens_B]
        self._ens_weights = [0.45, 0.30, 0.25]

        # ── Multi-TF HMMs (Upgrade 1) ──
        self._tf_models: Dict[str, Any] = {}
        self._tf_orders: Dict[str, Optional[Dict]] = {}
        self._tf_trained: Dict[str, bool] = {}
        self._tf_feature_stats: Dict[str, Tuple] = {}
        
        if GMMHMM_AVAILABLE:
            for tf in ["5m", "15m"]:
                try:
                    self._tf_models[tf] = hmm.GMMHMM(
                        n_components=3, n_mix=3, covariance_type="diag",
                        n_iter=min(n_iter, 500), random_state=42, tol=1e-4,
                    )
                except Exception as e:
                    logger.warning(f"GMMHMM Multi-TF initialization failed: {e}. Falling back to GaussianHMM.")
                    self._tf_models[tf] = hmm.GaussianHMM(
                        n_components=3, covariance_type="diag",
                        n_iter=min(n_iter, 500), random_state=42, tol=1e-4,
                    )
                self._tf_orders[tf] = None
                self._tf_trained[tf] = False
        elif HMM_AVAILABLE:
            for tf in ["5m", "15m"]:
                self._tf_models[tf] = hmm.GaussianHMM(
                    n_components=3, covariance_type="diag",
                    n_iter=min(n_iter, 500), random_state=42, tol=1e-4,
                )
                self._tf_orders[tf] = None
                self._tf_trained[tf] = False
        self._tf_weights = {"1m": 0.25, "5m": 0.35, "15m": 0.40}

        # Learned parameters (primary model)
        self._regime_order: Optional[Dict] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._regime_stats: Optional[Dict] = None
        self._transition_matrix: Optional[np.ndarray] = None

        # Internal regime classification
        self._internal_regime_map: Dict[int, str] = {}

        # Runtime state
        self._recent_regimes: List[int] = []
        self._regime_change_count: int = 0

        # ── Transition Velocity (Upgrade 3) ──
        self._prob_history: deque = deque(maxlen=20)

        # Adaptive retraining
        self._bars_since_retrain: int = 0
        self._retrain_interval: int = 500

    # ──────────────────────────────────────────────────────────
    # Feature Preparation
    # ──────────────────────────────────────────────────────────

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = False,
        ext_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Prepare multi-dimensional observation features for HMM.

        Args:
            df: OHLCV DataFrame.
            fit: If True, compute and store normalization stats.
            ext_stats: Optional external (mean, std) to use instead of stored.

        Returns:
            Tuple of (observation_matrix, datetime_index)
        """
        features = pd.DataFrame(index=df.index)

        # Core features
        smoothed_close = df["close"].rolling(5, min_periods=1).mean()
        raw_returns = df["close"].pct_change()

        features["returns"] = smoothed_close.pct_change()
        features["volatility"] = raw_returns.rolling(20, min_periods=5).std()
        features["abs_returns"] = raw_returns.abs().rolling(5, min_periods=1).mean()

        # Intraday range
        if "high" in df.columns and "low" in df.columns:
            features["range_pct"] = (df["high"] - df["low"]) / df["close"]

        # Volume regime
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20, min_periods=5).mean()
            vol_std = df["volume"].rolling(20, min_periods=5).std()
            features["volume_zscore"] = (df["volume"] - vol_ma) / (vol_std + 1e-10)

        # Momentum (5-bar cumulative return)
        features["momentum_5"] = features["returns"].rolling(5, min_periods=1).sum()

        # Return acceleration
        features["return_accel"] = features["returns"].diff()

        # Cross-asset features (gracefully skip if missing)
        if "dxy_returns" in df.columns:
            features["dxy_returns"] = df["dxy_returns"]
        if "us10y_returns" in df.columns:
            features["us10y_returns"] = df["us10y_returns"]
        if "gold_silver_ratio" in df.columns:
            gsr = df["gold_silver_ratio"]
            gsr_mean = gsr.rolling(20, min_periods=5).mean()
            gsr_std = gsr.rolling(20, min_periods=5).std()
            features["gsr_zscore"] = (gsr - gsr_mean) / (gsr_std + 1e-10)
        if "gvz_zscore_20" in df.columns:
            features["gvz_zscore"] = df["gvz_zscore_20"]
        if "real_yield_proxy" in df.columns:
            features["real_yield_chg"] = df["real_yield_proxy"].diff()

        features = features.dropna()
        X = features.values.astype(np.float64)

        if len(X) == 0:
            return X, features.index

        # Compute normalization stats
        if fit:
            self._feature_mean = X.mean(axis=0)
            self._feature_std = X.std(axis=0) + 1e-8

        # Determine which stats to use for normalization
        if ext_stats is not None:
            f_mean, f_std = ext_stats
        elif self._feature_mean is not None:
            f_mean, f_std = self._feature_mean, self._feature_std
        else:
            f_mean = X.mean(axis=0)
            f_std = X.std(axis=0) + 1e-8

        # Handle dimension mismatch (different feature counts across timeframes)
        if len(f_mean) == X.shape[1]:
            X = (X - f_mean) / f_std
        else:
            f_mean = X.mean(axis=0)
            f_std = X.std(axis=0) + 1e-8
            X = (X - f_mean) / f_std

        # ── CRITICAL FIX: Replace inf/nan with safe values before returning ──
        # This prevents "array must not contain infs or NaNs" errors downstream
        inf_mask = ~np.isfinite(X)
        if np.any(inf_mask):
            logger.debug(f"Replacing {np.sum(inf_mask)} inf/nan values in features")
            X[~np.isfinite(X)] = 0.0

        return X, features.index

    # ──────────────────────────────────────────────────────────
    # Multi-Timeframe Resampling (Upgrade 1)
    # ──────────────────────────────────────────────────────────

    def _resample_df(self, df: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        """Resample 1-minute OHLCV data to a higher timeframe."""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return None

            agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                agg_dict["volume"] = "sum"

            resampled = df.resample(rule).agg(agg_dict).dropna()
            if len(resampled) < 30:
                return None

            # Carry forward cross-asset columns
            for col in ["dxy_returns", "us10y_returns"]:
                if col in df.columns:
                    resampled[col] = df[col].resample(rule).sum()
            for col in ["gold_silver_ratio", "gvz_zscore_20", "real_yield_proxy"]:
                if col in df.columns:
                    resampled[col] = df[col].resample(rule).last()

            return resampled.dropna(subset=["close"])

        except Exception as e:
            logger.debug(f"Resample to {rule} failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y=None) -> Dict[str, Any]:
        """
        Train all HMMs: primary (5-regime), ensemble (3-regime x2), multi-TF.
        """
        # ── 1. Prepare features from 1m data ──
        obs, idx = self.prepare_features(X, fit=True)
        obs_cpu = np.asarray(obs) if not hasattr(obs, "get") else obs.get()

        if len(obs_cpu) < 30:
            logger.warning("HMM: Not enough data to train")
            return {}

        # ── DATA VALIDATION: Check for inf/nan before training ──
        if not np.all(np.isfinite(obs_cpu)):
            inf_count = np.sum(~np.isfinite(obs_cpu))
            logger.warning(f"Found {inf_count} inf/nan values in observation matrix, replacing with 0")
            obs_cpu[~np.isfinite(obs_cpu)] = 0.0

        # Save primary model stats (TF training will overwrite self._feature_mean/std)
        primary_mean = self._feature_mean.copy()
        primary_std = self._feature_std.copy()

        logger.info(
            f"Training HMM v3.0: {len(obs_cpu)} obs, "
            f"{self.n_regimes} primary regimes, {obs_cpu.shape[1]} features"
        )

        # ── 2. Train primary model ──
        if self.model is not None:
            try:
                self.model.fit(obs_cpu)
                states = self.model.predict(obs_cpu)
            except Exception as e:
                logger.warning(f"Primary HMM fit failed: {e}, using fallback")
                states, _ = self._fallback_regime_detection(obs_cpu)
        else:
            states, _ = self._fallback_regime_detection(obs_cpu)

        # Volatility-order regimes (0 = lowest vol, N-1 = highest vol)
        vol_by_state: Dict[int, float] = {}
        for s in range(self.n_regimes):
            mask = states == s
            vol_by_state[s] = float(np.std(obs_cpu[mask, 0])) if mask.any() else 0.0

        sorted_states = sorted(vol_by_state, key=lambda x: vol_by_state[x])
        self._regime_order = {old: new for new, old in enumerate(sorted_states)}

        ordered_states = np.array([self._regime_order[s] for s in states])

        # Classify internal regimes (Upgrade 2)
        self._classify_internal_regimes(obs_cpu, ordered_states)

        # Regime statistics
        self._regime_stats = {}
        for regime_id in range(self.n_regimes):
            mask = ordered_states == regime_id
            internal_name = self._internal_regime_map.get(regime_id, f"REGIME_{regime_id}")
            external_name = self.REGIME_EXTERNAL_MAP.get(internal_name, "NORMAL")
            durations = self._compute_regime_durations(ordered_states, regime_id)

            self._regime_stats[internal_name] = {
                "external": external_name,
                "pct_time": float(mask.mean()),
                "mean_return": float(obs_cpu[mask, 0].mean()) if mask.any() else 0.0,
                "volatility": float(obs_cpu[mask, 0].std()) if mask.any() else 0.0,
                "count": int(mask.sum()),
                "avg_duration_bars": float(np.mean(durations)) if durations else 0.0,
                "max_duration_bars": float(np.max(durations)) if durations else 0.0,
                "strategy": self.REGIME_STRATEGIES.get(external_name, {}),
            }
            logger.info(
                f"  {internal_name} -> {external_name}: {mask.mean():.1%} | "
                f"vol={obs_cpu[mask, 0].std():.4f} | "
                f"avg_dur={np.mean(durations):.0f} bars"
            )

        # Transition matrix
        if self.model is not None and hasattr(self.model, "transmat_"):
            raw_trans = self.model.transmat_
            reordered = np.zeros_like(raw_trans)
            for old, new in self._regime_order.items():
                for old2, new2 in self._regime_order.items():
                    reordered[new, new2] = raw_trans[old, old2]
            self._transition_matrix = reordered
        else:
            self._transition_matrix = (
                np.eye(self.n_regimes) * 0.8 + 0.2 / self.n_regimes
            )

        # ── 3. Train ensemble HMMs (Upgrade 7) ──
        for i, ens_model in enumerate(self._ensemble_models):
            label = chr(65 + i)
            try:
                # Extra validation before fit
                if not np.all(np.isfinite(obs_cpu)):
                    logger.debug(f"Ensemble HMM-{label}: Data has inf/nan, replacing...")
                    obs_cpu_clean = obs_cpu.copy()
                    obs_cpu_clean[~np.isfinite(obs_cpu_clean)] = 0.0
                else:
                    obs_cpu_clean = obs_cpu
                
                ens_model.fit(obs_cpu_clean)
                ens_states = ens_model.predict(obs_cpu_clean)
                ens_vol = {}
                for s in range(3):
                    mask = ens_states == s
                    ens_vol[s] = float(np.std(obs_cpu_clean[mask, 0])) if mask.any() else 0.0
                ens_sorted = sorted(ens_vol, key=lambda x: ens_vol[x])
                self._ensemble_orders[i] = {old: new for new, old in enumerate(ens_sorted)}
                logger.info(f"  Ensemble HMM-{label} trained (3-regime)")
            except Exception as e:
                logger.warning(f"  Ensemble HMM-{label} failed: {e}")
                self._ensemble_orders[i] = {0: 0, 1: 1, 2: 2}

        # ── 4. Train multi-TF HMMs (Upgrade 1) ──
        for tf, rule in [("5m", "5min"), ("15m", "15min")]:
            tf_df = self._resample_df(X, rule)
            if tf_df is not None and len(tf_df) >= 30:
                try:
                    tf_obs, _ = self.prepare_features(tf_df, fit=True)
                    tf_obs_cpu = np.asarray(tf_obs)
                    
                    # Validate before training
                    if not np.all(np.isfinite(tf_obs_cpu)):
                        logger.debug(f"Multi-TF HMM-{tf}: Data has inf/nan, replacing...")
                        tf_obs_cpu[~np.isfinite(tf_obs_cpu)] = 0.0

                    # Save TF-specific stats before they get overwritten
                    self._tf_feature_stats[tf] = (
                        self._feature_mean.copy(),
                        self._feature_std.copy(),
                    )

                    self._tf_models[tf].fit(tf_obs_cpu)
                    tf_states = self._tf_models[tf].predict(tf_obs_cpu)

                    tf_vol = {}
                    for s in range(3):
                        mask = tf_states == s
                        tf_vol[s] = float(np.std(tf_obs_cpu[mask, 0])) if mask.any() else 0.0
                    tf_sorted = sorted(tf_vol, key=lambda x: tf_vol[x])
                    self._tf_orders[tf] = {old: new for new, old in enumerate(tf_sorted)}
                    self._tf_trained[tf] = True
                    logger.info(f"  Multi-TF HMM-{tf} trained ({len(tf_obs_cpu)} bars)")
                except Exception as e:
                    logger.warning(f"  Multi-TF HMM-{tf} failed: {e}")
                    self._tf_trained[tf] = False
            else:
                self._tf_trained[tf] = False

        # Restore primary model stats
        self._feature_mean = primary_mean
        self._feature_std = primary_std

        self.is_trained = True
        self._train_timestamp = pd.Timestamp.now().isoformat()
        self._train_samples = len(obs_cpu)

        n_tf = sum(1 for v in self._tf_trained.values() if v)
        logger.info(
            f"  HMM v3.0 training complete: "
            f"1 primary + {len(self._ensemble_models)} ensemble + {n_tf} TF models"
        )

        return {
            "regimes": self._regime_stats,
            "transition_matrix": self._transition_matrix.tolist(),
            "n_features": obs_cpu.shape[1],
            "train_samples": len(obs_cpu),
            "internal_regime_map": dict(self._internal_regime_map),
        }

    def _classify_internal_regimes(
        self, obs: np.ndarray, ordered_states: np.ndarray
    ) -> None:
        """
        Classify volatility-ordered regimes into descriptive internal names.

        For n_regimes=5:
          - Regimes 0,1 (lowest vol): QUIET_RANGE vs TRENDING_UP
            (differentiated by mean return magnitude)
          - Regime 2 (medium vol): NORMAL
          - Regimes 3,4 (highest vol): VOLATILE_MR vs CRISIS_TREND
            (differentiated by trend consistency)

        For n_regimes=3: falls back to simple mapping.
        """
        self._internal_regime_map = {}

        if self.n_regimes >= 5:
            ret_std = float(np.std(obs[:, 0])) + 1e-10

            # Low-vol regimes -> GROWTH sub-types
            for r_id in [0, 1]:
                mask = ordered_states == r_id
                if mask.any():
                    mean_ret = float(np.mean(obs[mask, 0]))
                    if abs(mean_ret) > 0.1 * ret_std:
                        self._internal_regime_map[r_id] = "TRENDING_UP"
                    else:
                        self._internal_regime_map[r_id] = "QUIET_RANGE"
                else:
                    self._internal_regime_map[r_id] = "QUIET_RANGE"

            # Medium-vol regime -> NORMAL
            self._internal_regime_map[2] = "NORMAL"

            # High-vol regimes -> CRISIS sub-types
            for r_id in [3, 4]:
                mask = ordered_states == r_id
                if mask.any():
                    returns = obs[mask, 0]
                    pct_neg = float(np.mean(returns < 0))
                    pct_pos = float(np.mean(returns > 0))
                    consistency = max(pct_neg, pct_pos)
                    if consistency > 0.60:
                        self._internal_regime_map[r_id] = "CRISIS_TREND"
                    else:
                        self._internal_regime_map[r_id] = "VOLATILE_MR"
                else:
                    self._internal_regime_map[r_id] = "VOLATILE_MR"

            # Fill any remaining
            for r_id in range(5, self.n_regimes):
                self._internal_regime_map[r_id] = "NORMAL"

        elif self.n_regimes == 3:
            self._internal_regime_map = {
                0: "QUIET_RANGE", 1: "NORMAL", 2: "CRISIS_TREND",
            }
        else:
            for i in range(self.n_regimes):
                if i < self.n_regimes // 3:
                    self._internal_regime_map[i] = "QUIET_RANGE"
                elif i < 2 * self.n_regimes // 3:
                    self._internal_regime_map[i] = "NORMAL"
                else:
                    self._internal_regime_map[i] = "CRISIS_TREND"

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict regime for each timestep (primary model, mapped to 3 external)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        obs, idx = self.prepare_features(X)
        obs_cpu = np.asarray(obs) if not hasattr(obs, "get") else obs.get()

        if self.model is not None:
            raw_states = self.model.predict(obs_cpu)
            probs = self.model.predict_proba(obs_cpu)
        else:
            raw_states, probs = self._fallback_regime_detection(obs_cpu)

        # Map to volatility-ordered internal regimes, then to 3 external
        order = self._regime_order or {}
        ext_id_map = {}
        for internal_id, internal_name in self._internal_regime_map.items():
            ext_name = self.REGIME_EXTERNAL_MAP.get(internal_name, "NORMAL")
            ext_id_map[internal_id] = {"GROWTH": 0, "NORMAL": 1, "CRISIS": 2}[ext_name]

        ordered_states = np.array([
            ext_id_map.get(order.get(s, s), 1) for s in raw_states
        ])
        confidences = np.max(probs, axis=1)

        return ordered_states, confidences

    def _primary_probs_3(self, obs_cpu: np.ndarray) -> np.ndarray:
        """
        Get regime probabilities from primary model, mapped to 3 external
        regimes by summing internal probabilities (Upgrade 5).
        """
        if self.model is None or len(obs_cpu) == 0:
            return np.array([0.33, 0.34, 0.33])

        try:
            raw_probs = self.model.predict_proba(obs_cpu)[-1]
        except Exception:
            return np.array([0.33, 0.34, 0.33])

        order = self._regime_order or {}
        reordered = np.zeros(self.n_regimes)
        for old, new in order.items():
            if old < len(raw_probs):
                reordered[new] = raw_probs[old]

        # Sum into 3 external regimes
        if self.n_regimes >= 5:
            probs_3 = np.array([
                reordered[0] + reordered[1],  # GROWTH
                reordered[2],                  # NORMAL
                reordered[3] + reordered[4],   # CRISIS
            ])
            # Add any remaining to NORMAL
            if self.n_regimes > 5:
                probs_3[1] += reordered[5:].sum()
        elif self.n_regimes == 3:
            probs_3 = reordered.copy()
        else:
            third = max(self.n_regimes // 3, 1)
            probs_3 = np.array([
                reordered[:third].sum(),
                reordered[third:2 * third].sum(),
                reordered[2 * third:].sum(),
            ])

        total = probs_3.sum()
        if total > 0:
            probs_3 /= total
        return probs_3

    # ──────────────────────────────────────────────────────────
    # Multi-TF Fusion (Upgrade 1)
    # ──────────────────────────────────────────────────────────

    def _run_multi_tf(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run HMMs on 5m and 15m resampled data."""
        results = {}
        for tf, rule in [("5m", "5min"), ("15m", "15min")]:
            if not self._tf_trained.get(tf, False):
                continue
            try:
                tf_df = self._resample_df(df, rule)
                if tf_df is None or len(tf_df) < 20:
                    continue

                tf_obs, _ = self.prepare_features(
                    tf_df, fit=False,
                    ext_stats=self._tf_feature_stats.get(tf),
                )
                tf_obs_cpu = np.asarray(tf_obs)
                if len(tf_obs_cpu) == 0:
                    continue

                raw_probs = self._tf_models[tf].predict_proba(tf_obs_cpu)[-1]
                order = self._tf_orders.get(tf, {})
                reordered = np.zeros(3)
                for old, new in order.items():
                    if old < len(raw_probs):
                        reordered[new] = raw_probs[old]
                total = reordered.sum()
                if total > 0:
                    reordered /= total
                results[tf] = reordered

            except Exception as e:
                logger.debug(f"Multi-TF {tf} failed: {e}")
        return results

    # ──────────────────────────────────────────────────────────
    # Ensemble Fusion (Upgrade 7)
    # ──────────────────────────────────────────────────────────

    def _run_ensemble(self, obs_cpu: np.ndarray) -> Dict[str, np.ndarray]:
        """Run ensemble HMMs and return their 3-regime probability vectors."""
        results = {}
        for i, ens_model in enumerate(self._ensemble_models):
            try:
                raw_probs = ens_model.predict_proba(obs_cpu)[-1]
                order = self._ensemble_orders[i] or {}
                reordered = np.zeros(3)
                for old, new in order.items():
                    if old < len(raw_probs):
                        reordered[new] = raw_probs[old]
                total = reordered.sum()
                if total > 0:
                    reordered /= total
                results[f"ens_{chr(65 + i)}"] = reordered
            except Exception as e:
                logger.debug(f"Ensemble HMM-{chr(65 + i)} failed: {e}")
        return results

    # ──────────────────────────────────────────────────────────
    # Regime Transition Velocity (Upgrade 3)
    # ──────────────────────────────────────────────────────────

    def _compute_velocity(self, probs_3: np.ndarray) -> np.ndarray:
        """Compute the rate of change of regime probabilities."""
        self._prob_history.append(probs_3.copy())
        if len(self._prob_history) < 5:
            return np.zeros(3)
        return (probs_3 - self._prob_history[-5]) / 5.0

    # ──────────────────────────────────────────────────────────
    # Current Regime (full fusion pipeline)
    # ──────────────────────────────────────────────────────────

    def get_current_regime(
        self, X: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Backward-compatible: returns (external_regime_name, confidence).
        """
        ext, conf, _int, _pr, _vel = self._get_full_regime(X)
        return ext, conf

    def _get_full_regime(
        self, X: pd.DataFrame
    ) -> Tuple[str, float, str, np.ndarray, np.ndarray]:
        """
        Full regime detection: multi-TF + ensemble + persistence.

        Returns:
            (external_regime, confidence, internal_regime, fused_probs_3, velocity)
        """
        # 1. Primary model probabilities (5-regime -> 3)
        obs, idx = self.prepare_features(X)
        obs_cpu = np.asarray(obs) if not hasattr(obs, "get") else obs.get()

        if len(obs_cpu) == 0:
            return "NORMAL", 0.15, "NORMAL", np.array([0.33, 0.34, 0.33]), np.zeros(3)

        primary_probs = self._primary_probs_3(obs_cpu)

        # Get internal regime from primary model
        try:
            raw_state = self.model.predict(obs_cpu)[-1] if self.model else 0
            order = self._regime_order or {}
            ordered_state = order.get(int(raw_state), int(raw_state))
        except Exception:
            ordered_state = 0
        internal_name = self._internal_regime_map.get(ordered_state, "NORMAL")

        # 2. Ensemble probabilities (Upgrade 7)
        ensemble_probs = self._run_ensemble(obs_cpu)

        # 3. Multi-TF probabilities (Upgrade 1)
        tf_probs = self._run_multi_tf(X)

        # 4. Fuse all probability vectors with weighted average
        all_probs = {"primary": primary_probs}
        all_probs.update(ensemble_probs)
        all_probs.update(tf_probs)

        # Assign weights
        weights = {"primary": self._ens_weights[0]}
        for key in ensemble_probs:
            i = ord(key[-1]) - ord("A")
            w_idx = 1 + i
            weights[key] = self._ens_weights[w_idx] if w_idx < len(self._ens_weights) else 0.15
        for key in tf_probs:
            weights[key] = self._tf_weights.get(key, 0.2)

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        fused = np.zeros(3)
        for key, pvec in all_probs.items():
            fused += weights.get(key, 0) * pvec

        f_total = fused.sum()
        if f_total > 0:
            fused /= f_total

        # 5. Agreement scoring
        regime_ids = [int(np.argmax(p)) for p in all_probs.values()]
        vote_counts = Counter(regime_ids)
        dominant_id, dominant_count = vote_counts.most_common(1)[0]
        n_models = len(regime_ids)
        agreement = dominant_count / n_models if n_models > 0 else 0.5

        # 6. Determine consensus regime
        consensus_id = int(np.argmax(fused))
        raw_conf = float(fused[consensus_id])

        # Scale confidence by agreement
        if agreement >= 0.8:
            conf_mult = 1.0
        elif agreement >= 0.6:
            conf_mult = 0.7
        else:
            conf_mult = 0.4
        confidence = raw_conf * conf_mult

        # 7. Persistence scoring (anti-whipsaw)
        self._recent_regimes.append(consensus_id)
        if len(self._recent_regimes) > self.min_regime_duration * 3:
            self._recent_regimes = self._recent_regimes[-self.min_regime_duration * 3:]

        if len(self._recent_regimes) >= self.min_regime_duration:
            recent = self._recent_regimes[-self.min_regime_duration:]
            persistence = sum(1 for r in recent if r == consensus_id) / len(recent)
            w = self.regime_persistence_weight
            confidence = confidence * (1 - w) + persistence * w
            confidence = min(confidence, 1.0)

            # Hysteresis
            most_common, mc_count = Counter(recent).most_common(1)[0]
            if mc_count >= self.min_regime_duration * 0.6:
                consensus_id = most_common

        external_name = self.REGIME_NAMES.get(consensus_id, "NORMAL")

        # 8. Transition velocity (Upgrade 3)
        velocity = self._compute_velocity(fused)

        return external_name, confidence, internal_name, fused, velocity

    # ──────────────────────────────────────────────────────────
    # Micro-Models (Upgrade 4)
    # ──────────────────────────────────────────────────────────

    def _run_micro_model(
        self, df: pd.DataFrame, internal_regime: str,
        z_score: float, regime_conf: float,
    ) -> Tuple[str, float, str]:
        """
        Regime-specific indicator-based signal generation.

        Returns: (signal, confidence, reasoning_detail)
        """
        closes = df["close"].values
        if internal_regime == "QUIET_RANGE":
            return self._micro_quiet_range(closes, z_score, regime_conf)
        elif internal_regime == "TRENDING_UP":
            return self._micro_trending_up(closes, z_score, regime_conf)
        elif internal_regime == "VOLATILE_MR":
            return self._micro_volatile_mr(closes, z_score, regime_conf)
        elif internal_regime == "CRISIS_TREND":
            return self._micro_crisis_trend(df, closes, z_score, regime_conf)
        else:
            return self._micro_normal(closes, z_score, regime_conf)

    def _micro_quiet_range(self, closes, z_score, regime_conf):
        """QUIET_RANGE: Bollinger Band + RSI(7) mean-reversion."""
        n = len(closes)
        if n < 20:
            return "HOLD", regime_conf * 0.25, "QR:nodata"

        bb_mean = np.mean(closes[-20:])
        bb_std = np.std(closes[-20:]) + 1e-10
        bb_pos = (closes[-1] - bb_mean) / (2 * bb_std)  # [-1, +1]

        # RSI(7)
        delta = np.diff(closes[-8:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        rsi_7 = 100 - 100 / (1 + (np.mean(gains) + 1e-10) / (np.mean(losses) + 1e-10))

        # Strong signal: BB extreme + RSI confirmation
        if bb_pos < -0.7 and rsi_7 < 38:
            return "LONG", regime_conf * 0.85, f"QR:BuyDip BB={bb_pos:.2f} RSI={rsi_7:.0f}"
        elif bb_pos > 0.7 and rsi_7 > 62:
            return "SHORT", regime_conf * 0.85, f"QR:SellRip BB={bb_pos:.2f} RSI={rsi_7:.0f}"
        # Medium signal: BB or RSI alone
        elif bb_pos < -0.5 or rsi_7 < 30:
            return "LONG", regime_conf * 0.65, f"QR:LeanLong BB={bb_pos:.2f} RSI={rsi_7:.0f}"
        elif bb_pos > 0.5 or rsi_7 > 70:
            return "SHORT", regime_conf * 0.65, f"QR:LeanShort BB={bb_pos:.2f} RSI={rsi_7:.0f}"
        # Weak signal: z-score only
        elif z_score < -0.5:
            return "LONG", regime_conf * 0.50, f"QR:ZDip z={z_score:.2f}"
        elif z_score > 0.5:
            return "SHORT", regime_conf * 0.50, f"QR:ZRip z={z_score:.2f}"
        else:
            return "HOLD", regime_conf * 0.25, f"QR:Flat BB={bb_pos:.2f}"

    def _micro_trending_up(self, closes, z_score, regime_conf):
        """TRENDING_UP: MACD + EMA pullback."""
        n = len(closes)
        if n < 26:
            return "HOLD", regime_conf * 0.25, "TU:nodata"

        series = pd.Series(closes)
        ema_fast = series.ewm(span=8).mean().iloc[-1]
        ema_slow = series.ewm(span=21).mean().iloc[-1]
        macd = ema_fast - ema_slow

        series_prev = pd.Series(closes[:-1])
        macd_prev = series_prev.ewm(span=8).mean().iloc[-1] - series_prev.ewm(span=21).mean().iloc[-1]
        macd_accel = macd - macd_prev

        ema8_now = series.ewm(span=8).mean().iloc[-1]
        ema8_3ago = series.ewm(span=8).mean().iloc[-3] if n > 3 else ema8_now
        ema_slope = (ema8_now - ema8_3ago) / (abs(ema8_3ago) + 1e-10)

        # Strong: MACD positive + accelerating + EMA rising
        if macd > 0 and macd_accel > 0 and ema_slope > 0:
            return "LONG", regime_conf * 0.90, f"TU:StrongTrend MACD={macd:.2f}"
        # Medium: MACD positive + pullback
        elif macd > 0 and ema_slope > 0:
            return "LONG", regime_conf * 0.70, f"TU:Trend MACD={macd:.2f}"
        # Medium: MACD positive but decelerating (take profit zone)
        elif macd > 0 and macd_accel < 0:
            return "SHORT", regime_conf * 0.55, f"TU:Exhaustion MACD={macd:.2f} accel={macd_accel:.4f}"
        # Strong bearish: MACD negative and falling
        elif macd < 0 and macd_accel < 0:
            return "SHORT", regime_conf * 0.75, f"TU:TrendBreak MACD={macd:.2f}"
        # Weak bearish: MACD negative
        elif macd < 0:
            return "SHORT", regime_conf * 0.50, f"TU:Weak MACD={macd:.2f}"
        else:
            return "HOLD", regime_conf * 0.25, f"TU:Neutral MACD={macd:.2f}"

    def _micro_normal(self, closes, z_score, regime_conf):
        """NORMAL: Composite multi-indicator score."""
        n = len(closes)
        if n < 20:
            return "HOLD", regime_conf * 0.25, "NM:nodata"

        scores = []

        # Z-score momentum (activated at lower threshold)
        scores.append(
            np.sign(z_score) * min(abs(z_score) / 2.0, 1.0)
            if abs(z_score) > 0.5 else 0.0
        )

        # RSI(14)
        delta = np.diff(closes[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        rsi_14 = 100 - 100 / (1 + (np.mean(gains) + 1e-10) / (np.mean(losses) + 1e-10))
        scores.append(((rsi_14 - 50) / 50) * 0.5)

        # MACD
        series = pd.Series(closes)
        ema12 = series.ewm(span=12).mean().iloc[-1]
        ema26 = series.ewm(span=26).mean().iloc[-1]
        macd = ema12 - ema26
        scores.append(np.tanh(macd / (np.std(closes[-20:]) + 1e-10)) * 0.5)

        composite = float(np.mean(scores))

        # Tiered thresholds for better coverage
        if composite > 0.25:
            return "LONG", regime_conf * min(abs(composite) + 0.4, 1.0), f"NM:Bull score={composite:.2f}"
        elif composite < -0.25:
            return "SHORT", regime_conf * min(abs(composite) + 0.4, 1.0), f"NM:Bear score={composite:.2f}"
        elif composite > 0.10:
            return "LONG", regime_conf * 0.50, f"NM:LeanBull score={composite:.2f}"
        elif composite < -0.10:
            return "SHORT", regime_conf * 0.50, f"NM:LeanBear score={composite:.2f}"
        else:
            return "HOLD", regime_conf * 0.25, f"NM:Neutral score={composite:.2f}"

    def _micro_volatile_mr(self, closes, z_score, regime_conf):
        """VOLATILE_MR: Fade extreme moves in high-vol, mean-reverting regime."""
        n = len(closes)
        if n < 20:
            return "HOLD", regime_conf * 0.25, "VMR:nodata"

        atr = np.mean(np.abs(np.diff(closes[-15:]))) + 1e-10
        mean_20 = np.mean(closes[-20:])
        deviation = (closes[-1] - mean_20) / atr

        # Strong fade: price >1.5 ATR from mean
        if deviation < -1.5:
            conf = regime_conf * min(0.55 + abs(deviation) * 0.10, 0.90)
            return "LONG", conf, f"VMR:FadeDn dev={deviation:.2f}"
        elif deviation > 1.5:
            conf = regime_conf * min(0.55 + abs(deviation) * 0.10, 0.90)
            return "SHORT", conf, f"VMR:FadeUp dev={deviation:.2f}"
        # Medium fade: price 0.8-1.5 ATR from mean
        elif deviation < -0.8:
            return "LONG", regime_conf * 0.55, f"VMR:LeanDn dev={deviation:.2f}"
        elif deviation > 0.8:
            return "SHORT", regime_conf * 0.55, f"VMR:LeanUp dev={deviation:.2f}"
        # Z-score based
        elif z_score < -0.8:
            return "LONG", regime_conf * 0.45, f"VMR:ZFade z={z_score:.2f}"
        elif z_score > 0.8:
            return "SHORT", regime_conf * 0.45, f"VMR:ZFade z={z_score:.2f}"
        else:
            return "HOLD", regime_conf * 0.25, f"VMR:Flat dev={deviation:.2f}"

    def _micro_crisis_trend(self, df, closes, z_score, regime_conf):
        """CRISIS_TREND: Follow strong downtrends, fade extremes."""
        n = len(closes)
        if n < 20:
            return "HOLD", regime_conf * 0.25, "CT:nodata"

        recent_rets = np.diff(closes[-11:]) / (closes[-11:-1] + 1e-10)
        pct_down = float(np.mean(recent_rets < 0))
        cum_ret = float(np.sum(recent_rets))

        # Strong: consistent selling pressure
        if pct_down > 0.60 and cum_ret < -0.003:
            return "SHORT", regime_conf * 0.85, f"CT:StrongDn pct={pct_down:.0%} cum={cum_ret:.4f}"
        # Medium: majority of bars are down
        elif pct_down > 0.55:
            return "SHORT", regime_conf * 0.60, f"CT:MomDn pct={pct_down:.0%} cum={cum_ret:.4f}"
        # Z-score directional
        elif z_score < -0.8:
            return "SHORT", regime_conf * 0.55, f"CT:ZDn z={z_score:.2f}"
        # Counter-trend bounce on extreme oversold
        elif z_score > 1.5:
            return "LONG", regime_conf * 0.45, f"CT:Bounce z={z_score:.2f}"
        elif z_score > 0.5:
            return "LONG", regime_conf * 0.35, f"CT:LeanBounce z={z_score:.2f}"
        else:
            return "HOLD", regime_conf * 0.25, f"CT:Wait z={z_score:.2f}"

    # ──────────────────────────────────────────────────────────
    # DXY Lead-Lag (Upgrade 6)
    # ──────────────────────────────────────────────────────────

    def _check_dxy_lead_lag(
        self, df: pd.DataFrame, z_score: float,
        signal: str, confidence: float,
    ) -> Tuple[str, float, str]:
        """
        DXY lead-lag: DXY often leads gold by 1-3 minutes.
        Returns (adjusted_signal, adjusted_confidence, reason)
        """
        if "dxy_returns" not in df.columns or len(df) < 5:
            return signal, confidence, ""

        dxy_mom = float(df["dxy_returns"].iloc[-3:].sum())

        # DXY surging AND gold hasn't reacted yet
        if abs(dxy_mom) > 0.001 and abs(z_score) < 0.5:
            if dxy_mom > 0.001:
                # Strong dollar -> bearish gold
                if signal == "HOLD":
                    return "SHORT", min(confidence * 0.8 + 0.15, 1.0), f"DXY:LeadShort({dxy_mom:.4f})"
                elif signal == "LONG":
                    return signal, confidence * 0.6, f"DXY:WeakenLong({dxy_mom:.4f})"
            elif dxy_mom < -0.001:
                # Weak dollar -> bullish gold
                if signal == "HOLD":
                    return "LONG", min(confidence * 0.8 + 0.15, 1.0), f"DXY:LeadLong({dxy_mom:.4f})"
                elif signal == "SHORT":
                    return signal, confidence * 0.6, f"DXY:WeakenShort({dxy_mom:.4f})"

        return signal, confidence, ""

    # ──────────────────────────────────────────────────────────
    # Signal Generation (Full v3.0 Pipeline)
    # ──────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame) -> ModelOutput:
        """
        Generate a regime-aware trading signal using the full v3.0 pipeline:

        1. Multi-TF + Ensemble regime fusion
        2. Regime transition velocity analysis
        3. Regime-conditional micro-model
        4. DXY lead-lag adjustment
        5. Transition probability anticipatory signals
        6. Gradient confidence scaling
        """
        start_time = time.perf_counter()

        if len(df) < 30:
            return ModelOutput(
                signal="HOLD", confidence=0.0,
                reasoning="Insufficient data for HMM v3.0",
            )

        # Train if needed
        if not self.is_trained:
            self.train(df)

        # Adaptive retraining
        self._bars_since_retrain += 1
        if self._bars_since_retrain >= self._retrain_interval:
            logger.info(f"HMM v3.0 retrain ({self._bars_since_retrain} bars)")
            self.train(df)
            self._bars_since_retrain = 0

        # ── 1. Full regime detection ──
        external_regime, regime_conf, internal_regime, fused_probs, velocity = \
            self._get_full_regime(df)

        # ── 2. Z-score (smoothed over 3 bars to reduce single-bar noise) ──
        pct_changes = df["close"].pct_change()
        recent_return_3 = pct_changes.iloc[-3:].mean() if len(df) > 3 else pct_changes.iloc[-1]
        recent_vol = (
            pct_changes.rolling(10).std().iloc[-1]
            if len(df) > 10 else 0.01
        )
        z_score = recent_return_3 / (recent_vol + 1e-10)

        # ── 3. Micro-model (Upgrade 4) ──
        signal, confidence, micro_reason = self._run_micro_model(
            df, internal_regime, z_score, regime_conf,
        )

        # ── 4. Velocity override (Upgrade 3) ──
        vel_reason = ""
        if signal == "HOLD":
            crisis_vel = velocity[2] if len(velocity) > 2 else 0
            growth_vel = velocity[0] if len(velocity) > 0 else 0

            if crisis_vel > 0.05:
                signal = "SHORT"
                confidence = regime_conf * 0.65 * min(crisis_vel * 10, 1.0)
                vel_reason = f"VEL:CrisisAccel({crisis_vel:.3f})"
            elif growth_vel > 0.05:
                signal = "LONG"
                confidence = regime_conf * 0.65 * min(growth_vel * 10, 1.0)
                vel_reason = f"VEL:GrowthAccel({growth_vel:.3f})"

        # ── 5. DXY lead-lag (Upgrade 6) ──
        signal, confidence, dxy_reason = self._check_dxy_lead_lag(
            df, z_score, signal, confidence,
        )

        # ── 6. Transition probability override ──
        trans_reason = ""
        trans_probs = self.get_regime_transition_probabilities_3()
        crisis_trans = trans_probs.get(external_regime, {}).get("CRISIS", 0.0)
        growth_trans = trans_probs.get(external_regime, {}).get("GROWTH", 0.0)

        if signal == "HOLD":
            if crisis_trans > 0.30 and external_regime != "CRISIS":
                signal = "SHORT"
                confidence = regime_conf * 0.7 * min(crisis_trans / 0.5, 1.0)
                trans_reason = f"TRANS:CrisisWarn({crisis_trans:.0%})"
            elif growth_trans > 0.30 and external_regime != "GROWTH":
                signal = "LONG"
                confidence = regime_conf * 0.7 * min(growth_trans / 0.5, 1.0)
                trans_reason = f"TRANS:GrowthWarn({growth_trans:.0%})"

        # ── 7. Gradient confidence scaling ──
        # Floor at 70% so a near-zero z-score doesn't crush micro-model conviction
        if signal != "HOLD":
            z_strength = min(abs(z_score) / 2.0, 1.0)
            confidence = confidence * (0.70 + 0.30 * z_strength)

        confidence = max(0.0, min(confidence, 1.0))

        # Strategy info
        strategy = self.REGIME_STRATEGIES.get(
            external_regime, self.REGIME_STRATEGIES["NORMAL"]
        )

        # Build reasoning
        n_tf = sum(1 for v in self._tf_trained.values() if v)
        n_total = 1 + len(self._ensemble_models) + n_tf

        parts = [
            f"{external_regime}({internal_regime}) conf={regime_conf:.0%}",
            f"z={z_score:+.2f}",
            f"{strategy['approach']}",
            micro_reason,
        ]
        for extra in [vel_reason, dxy_reason, trans_reason]:
            if extra:
                parts.append(extra)
        parts.append(f"Fusion:{n_total}mdl")

        reasoning = " | ".join(parts)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return ModelOutput(
            signal=signal,
            confidence=round(confidence, 4),
            regime=external_regime,
            reasoning=reasoning,
            features_used=["close", "high", "low", "volume", "returns",
                           "dxy", "us10y", "gold_silver_ratio", "gvz"],
            metadata={
                "regime_confidence": round(regime_conf, 4),
                "internal_regime": internal_regime,
                "z_score": round(z_score, 4),
                "fused_probs": {
                    "GROWTH": round(float(fused_probs[0]), 4),
                    "NORMAL": round(float(fused_probs[1]), 4),
                    "CRISIS": round(float(fused_probs[2]), 4),
                },
                "velocity": {
                    "growth": round(float(velocity[0]), 4),
                    "normal": round(float(velocity[1]), 4),
                    "crisis": round(float(velocity[2]), 4),
                },
                "transition_probabilities": trans_probs.get(external_regime, {}),
                "expected_duration_bars": round(
                    self.get_expected_regime_duration(external_regime), 1
                ),
                "kelly_fraction": strategy.get("kelly_fraction", 0.5),
                "regime_stats": (self._regime_stats or {}).get(internal_regime, {}),
                "recent_return": round(float(recent_return_3), 6),
                "recent_volatility": round(float(recent_vol), 6),
                "crisis_probability": round(crisis_trans, 4),
                "growth_probability": round(growth_trans, 4),
                "n_models_fused": n_total,
                "micro_model": micro_reason,
            },
            latency_ms=latency_ms,
            timestamp=pd.Timestamp.now().isoformat(),
        )

    # ──────────────────────────────────────────────────────────
    # Transition Probabilities
    # ──────────────────────────────────────────────────────────

    def get_regime_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Get transition probabilities using internal regime names."""
        if self._transition_matrix is None:
            return {}

        result: Dict[str, Dict[str, float]] = {}
        for i in range(self.n_regimes):
            from_name = self._internal_regime_map.get(i, f"REGIME_{i}")
            result[from_name] = {}
            for j in range(self.n_regimes):
                to_name = self._internal_regime_map.get(j, f"REGIME_{j}")
                result[from_name][to_name] = float(self._transition_matrix[i, j])
        return result

    def get_regime_transition_probabilities_3(self) -> Dict[str, Dict[str, float]]:
        """Get transition probabilities mapped to 3 external regimes."""
        if self._transition_matrix is None:
            return {}

        # Map each internal regime to external ID {0: GROWTH, 1: NORMAL, 2: CRISIS}
        ext_id = {}
        for i in range(self.n_regimes):
            iname = self._internal_regime_map.get(i, "NORMAL")
            ename = self.REGIME_EXTERNAL_MAP.get(iname, "NORMAL")
            ext_id[i] = {"GROWTH": 0, "NORMAL": 1, "CRISIS": 2}[ename]

        # Aggregate transition matrix to 3x3
        ext_trans = np.zeros((3, 3))
        ext_count = np.zeros(3)

        for i in range(self.n_regimes):
            ei = ext_id[i]
            for j in range(self.n_regimes):
                ej = ext_id[j]
                ext_trans[ei, ej] += self._transition_matrix[i, j]
            ext_count[ei] += 1

        # Normalize rows
        for i in range(3):
            row_sum = ext_trans[i].sum()
            if row_sum > 0:
                ext_trans[i] /= row_sum

        names = {0: "GROWTH", 1: "NORMAL", 2: "CRISIS"}
        result: Dict[str, Dict[str, float]] = {}
        for i in range(3):
            result[names[i]] = {}
            for j in range(3):
                result[names[i]][names[j]] = float(ext_trans[i, j])
        return result

    def get_expected_regime_duration(self, regime_name: str) -> float:
        """Expected duration in bars for an external regime."""
        trans_3 = self.get_regime_transition_probabilities_3()
        p_self = trans_3.get(regime_name, {}).get(regime_name, 0.9)
        if p_self >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - p_self + 1e-10)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_regime_durations(states: np.ndarray, target_regime: int) -> List[int]:
        """Compute consecutive duration of being in a given regime."""
        durations = []
        current_duration = 0
        for s in states:
            if s == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return durations

    def _fallback_regime_detection(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback regime detection when hmmlearn is unavailable."""
        n_regimes = min(self.n_regimes, obs.shape[0])
        vol_col = obs[:, 1] if obs.shape[1] >= 2 else np.abs(obs[:, 0])

        boundaries = np.linspace(0, 100, n_regimes + 1)
        centers = []
        for i in range(n_regimes):
            low = np.percentile(vol_col, boundaries[i])
            high = np.percentile(vol_col, boundaries[i + 1])
            centers.append((low + high) / 2)
        centers = np.array(centers)

        probs = np.zeros((len(obs), n_regimes))
        for i, v in enumerate(vol_col):
            dists = np.abs(v - centers)
            weights = 1.0 / (dists + 1e-6)
            probs[i] = weights / np.sum(weights)

        states = np.argmax(probs, axis=1)
        return states, probs

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _get_state(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "regime_order": self._regime_order,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "regime_stats": self._regime_stats,
            "transition_matrix": (
                self._transition_matrix.tolist()
                if self._transition_matrix is not None else None
            ),
            "is_trained": self.is_trained,
            "internal_regime_map": self._internal_regime_map,
            "ensemble_models": self._ensemble_models,
            "ensemble_orders": self._ensemble_orders,
            "tf_models": self._tf_models,
            "tf_orders": self._tf_orders,
            "tf_trained": self._tf_trained,
            "tf_feature_stats": self._tf_feature_stats,
        }

    def _set_state(self, state: Dict) -> None:
        self.model = state.get("model", self.model)
        self._regime_order = state.get("regime_order")
        self._feature_mean = state.get("feature_mean")
        self._feature_std = state.get("feature_std")
        self._regime_stats = state.get("regime_stats")
        trans = state.get("transition_matrix")
        self._transition_matrix = np.array(trans) if trans is not None else None
        self._internal_regime_map = state.get("internal_regime_map", {})
        if "ensemble_models" in state:
            self._ensemble_models = state["ensemble_models"]
        if "ensemble_orders" in state:
            self._ensemble_orders = state["ensemble_orders"]
        if "tf_models" in state:
            self._tf_models = state["tf_models"]
        if "tf_orders" in state:
            self._tf_orders = state["tf_orders"]
        if "tf_trained" in state:
            self._tf_trained = state["tf_trained"]
        if "tf_feature_stats" in state:
            self._tf_feature_stats = state["tf_feature_stats"]

    def save(self, path: str) -> None:
        """Save model state to disk."""
        import joblib
        joblib.dump(self._get_state(), path)
        logger.info(f"RegimeDetector v3.0 saved to {path}")

    def load(self, path: str) -> None:
        """Load model state from disk."""
        import joblib
        state = joblib.load(path)
        self._set_state(state)
        self.is_trained = True
        logger.info(f"RegimeDetector v3.0 loaded from {path}")
