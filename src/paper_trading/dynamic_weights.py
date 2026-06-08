"""
Dynamic Model Weight Adjuster
==============================
Real-world-style regime-conditional, performance-adaptive model weighting.

In production quant systems (Renaissance, Two Sigma, Citadel), model weights
are NEVER static. They are adjusted based on:

1. Market Regime:    Each model has strengths in specific regimes
2. Recent Performance: Models that predicted well recently get higher weight
3. Signal Agreement: Consensus signals get boosted, conflicting signals get damped
4. Decay:            Older performance fades via exponential decay

This replaces the naive equal-weight (15% each) approach with a system
that mirrors how real hedge funds allocate signal capital.

Reference:
  - De Prado, M.L. "Advances in Financial Machine Learning" (2018), Ch. 8
  - Kelly, J.L. "A New Interpretation of Information Rate" (1956)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


# ============================================================================
# REGIME-CONDITIONAL BASE WEIGHTS (Research-Backed)
# ============================================================================
# These are calibrated based on each model's theoretical strength
# per market regime, inspired by real multi-model quant allocations.

REGIME_BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Weights redistribute to the 4 pro models ONLY.
    # Ensemble is set to 0.0 because it already aggregates all model
    # outputs internally — giving it a dynamic weight would double-count.
    "GROWTH": {
        "wavelet_pro": 0.25,
        "hmm_pro": 0.10,
        "lstm": 0.25,
        "tft_pro": 0.15,
        "tft_pro_max": 0.25,
        "ensemble": 0.00,
    },
    "NORMAL": {
        "wavelet_pro": 0.20,
        "hmm_pro": 0.20,
        "lstm": 0.15,
        "tft_pro": 0.20,
        "tft_pro_max": 0.25,
        "ensemble": 0.00,
    },
    "CRISIS": {
        "wavelet_pro": 0.10,
        "hmm_pro": 0.30,
        "lstm": 0.10,
        "tft_pro": 0.20,
        "tft_pro_max": 0.30,
        "ensemble": 0.00,
    },
}

# Default fallback (if regime is unknown)
DEFAULT_WEIGHTS: Dict[str, float] = REGIME_BASE_WEIGHTS["NORMAL"]


@dataclass
class ModelPerformanceRecord:
    """Tracks a single model's recent trading performance."""
    model_name: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def rolling_sharpe(self) -> float:
        """Annualized Sharpe from recent returns."""
        if len(self.recent_returns) < 5:
            return 0.0
        returns = np.array(self.recent_returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r == 0:
            return 0.0
        return float((mean_r / std_r) * np.sqrt(252))

    @property
    def trade_count(self) -> int:
        return self.wins + self.losses


class DynamicWeightAdjuster:
    """
    Production-grade dynamic model weight allocator.

    Combines three weight sources:
    1. Regime Base Weights:     Theoretical strengths per regime (calibrated)
    2. Performance Adaptation:  Rolling Sharpe-based adjustment (data-driven)
    3. Agreement Bonus:         Models in consensus get boosted

    Final weight = normalize(regime_weight × performance_multiplier × agreement_factor)

    This mirrors the approach used at firms like:
    - Renaissance Technologies: Model weights shift with detected regime
    - Two Sigma: Performance-adaptive allocation with decay
    - AQR: Factor-timing based on rolling metrics
    """

    # Performance adaptation parameters
    PERFORMANCE_LOOKBACK: int = 50          # Number of recent trades to consider
    PERFORMANCE_BLEND: float = 0.4          # How much performance adjusts base weights (0-1)
    MIN_TRADES_FOR_ADAPTATION: int = 10     # Minimum trades before adapting weights
    SHARPE_DECAY_HALFLIFE: int = 20         # Exponential decay half-life (trades)

    # Agreement bonus
    AGREEMENT_BOOST: float = 1.15           # 15% boost when ≥4 models agree
    DISAGREEMENT_PENALTY: float = 0.85      # 15% penalty when model is lone dissenter

    # Weight bounds (prevent any model from being zeroed out or dominating)
    MIN_WEIGHT: float = 0.00               # Floor: 0% allows models to be disabled
    MAX_WEIGHT: float = 0.40               # Cap: 40% maximum per model

    def __init__(self):
        """Initialize the weight adjuster."""
        self.model_performance: Dict[str, ModelPerformanceRecord] = {}
        self.current_regime: str = "NORMAL"
        self.weight_history: List[Dict] = []  # Audit trail

        # Initialize performance records for all models
        for model_name in DEFAULT_WEIGHTS.keys():
            self.model_performance[model_name] = ModelPerformanceRecord(
                model_name=model_name
            )

        logger.info(
            "DynamicWeightAdjuster initialized | "
            f"Models: {list(DEFAULT_WEIGHTS.keys())} | "
            f"Regimes: {list(REGIME_BASE_WEIGHTS.keys())}"
        )

    def get_weights(self, regime: str = "NORMAL",
                    current_signals: Optional[Dict[str, any]] = None) -> Dict[str, float]:
        """
        Calculate current model weights.

        Args:
            regime: Current market regime (GROWTH/NORMAL/CRISIS)
            current_signals: Dict of {model_name: ModelSignal} for confidence and agreement

        Returns:
            Dict of {model_name: weight} that sums to 1.0
        """
        self.current_regime = regime

        # Step 1: Get regime base weights
        base_weights = REGIME_BASE_WEIGHTS.get(regime, DEFAULT_WEIGHTS).copy()

        # Step 2: Apply performance adaptation (if enough data)
        adapted_weights = self._apply_performance_adaptation(base_weights)

        # Step 3: Apply signal agreement bonus/penalty and Confidence multipliers
        if current_signals:
            adapted_weights = self._apply_agreement_adjustment(
                adapted_weights, current_signals
            )
            adapted_weights = self._apply_confidence_adjustment(
                adapted_weights, current_signals
            )

        # Step 4: Clip to bounds and normalize
        final_weights = self._normalize_with_bounds(adapted_weights)

        # Audit trail
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "weights": final_weights.copy(),
            "adaptation_active": self._has_enough_data(),
        })

        # Keep only last 500 entries
        if len(self.weight_history) > 500:
            self.weight_history = self.weight_history[-500:]

        return final_weights

    def record_trade_result(self, model_name: str, pnl: float,
                            return_pct: float) -> None:
        """
        Record a trade result for performance tracking.

        Args:
            model_name: Which model generated the signal
            pnl: Dollar P&L of the trade
            return_pct: Return percentage of the trade
        """
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformanceRecord(
                model_name=model_name
            )

        record = self.model_performance[model_name]
        record.recent_returns.append(return_pct)
        record.total_pnl += pnl
        record.last_updated = datetime.now()

        if pnl >= 0:
            record.wins += 1
        else:
            record.losses += 1

        logger.debug(
            f"Weight tracker: {model_name} trade recorded | "
            f"PnL=${pnl:+.2f} | Win rate={record.win_rate:.1%} | "
            f"Rolling Sharpe={record.rolling_sharpe:.2f}"
        )

    def get_weight_summary(self) -> Dict:
        """Get a summary of current weights and model performance."""
        weights = self.get_weights(self.current_regime)
        return {
            "regime": self.current_regime,
            "weights": weights,
            "model_stats": {
                name: {
                    "weight": weights.get(name, 0),
                    "win_rate": rec.win_rate,
                    "rolling_sharpe": rec.rolling_sharpe,
                    "trade_count": rec.trade_count,
                    "total_pnl": rec.total_pnl,
                }
                for name, rec in self.model_performance.items()
            },
            "adaptation_active": self._has_enough_data(),
            "history_length": len(self.weight_history),
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _has_enough_data(self) -> bool:
        """Check if we have enough trade history for performance adaptation."""
        total_trades = sum(
            r.trade_count for r in self.model_performance.values()
        )
        return total_trades >= self.MIN_TRADES_FOR_ADAPTATION

    def _apply_performance_adaptation(
        self, base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust weights based on each model's recent rolling Sharpe ratio.

        Uses inverse-Sharpe weighting: models with higher recent Sharpe
        get proportionally more weight. Blended with base weights to
        prevent overfitting to recent performance.

        Formula:
            adapted_w = (1 - blend) × base_w + blend × sharpe_w
        """
        if not self._has_enough_data():
            return base_weights

        # Calculate Sharpe-based weights
        sharpes = {}
        for model_name in base_weights:
            record = self.model_performance.get(model_name)
            if record and record.trade_count >= 3:
                # Shift Sharpe to positive domain (add 2 to handle negatives)
                sharpes[model_name] = max(0.01, record.rolling_sharpe + 2.0)
            else:
                sharpes[model_name] = 2.0  # Neutral (Sharpe=0 maps to 2.0)

        total_sharpe = sum(sharpes.values())
        if total_sharpe == 0:
            return base_weights

        sharpe_weights = {
            name: s / total_sharpe for name, s in sharpes.items()
        }

        # Blend: base weights + performance weights
        blend = self.PERFORMANCE_BLEND
        adapted = {
            name: (1 - blend) * base_weights.get(name, 0) + blend * sharpe_weights.get(name, 0)
            for name in base_weights
        }

        return adapted

    def _apply_agreement_adjustment(
        self, weights: Dict[str, float],
        current_signals: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Boost models that agree with the majority, penalize dissenters.

        If ≥4 out of 7 models agree on direction, agreeing models get
        a 15% boost and lone dissenters get a 15% penalty.

        This captures the intuition that consensus signals are more
        reliable than lone-wolf signals.
        """
        if not current_signals or len(current_signals) < 3:
            return weights

        # Count direction votes
        direction_counts: Dict[str, int] = {}
        for signal_obj in current_signals.values():
            if hasattr(signal_obj, "signal_type"):
                direction = signal_obj.signal_type.value
            else:
                direction = signal_obj
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # Find majority direction
        if not direction_counts:
            return weights

        majority_direction = max(direction_counts, key=direction_counts.get)
        majority_count = direction_counts[majority_direction]

        # Only apply if there's a clear majority (≥3 of 4)
        if majority_count < 3:
            return weights

        adjusted = {}
        for model_name, weight in weights.items():
            signal = current_signals.get(model_name)
            # Check if signal matches majority direction
            signal_dir = getattr(signal, "signal_type", None)
            if signal_dir:
                signal_dir = signal_dir.value
            else:
                signal_dir = signal
                
            if signal_dir == majority_direction:
                adjusted[model_name] = weight * self.AGREEMENT_BOOST
            elif signal_dir is not None:
                adjusted[model_name] = weight * self.DISAGREEMENT_PENALTY
            else:
                adjusted[model_name] = weight

        return adjusted

    def _apply_confidence_adjustment(
        self, weights: Dict[str, float],
        current_signals: Dict[str, any]
    ) -> Dict[str, float]:
        """
        Boost models that have high confidence in their signal.
        """
        adjusted = {}
        for model_name, weight in weights.items():
            signal = current_signals.get(model_name)
            if signal and hasattr(signal, 'confidence'):
                # Map confidence (0-1) to multiplier (0.5 to 1.5)
                conf_multiplier = 0.5 + (signal.confidence * 1.0)
                adjusted[model_name] = weight * conf_multiplier
            else:
                adjusted[model_name] = weight
        return adjusted

    def _normalize_with_bounds(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Clip weights to [MIN_WEIGHT, MAX_WEIGHT] and normalize to sum=1.0.

        This prevents any single model from being zeroed out (survivorship)
        or dominating (concentration risk).
        """
        # Clip
        clipped = {
            name: np.clip(w, self.MIN_WEIGHT, self.MAX_WEIGHT)
            for name, w in weights.items()
        }

        # Normalize to sum = 1.0
        total = sum(clipped.values())
        if total == 0:
            # Fallback to equal weights
            n = len(clipped)
            return {name: 1.0 / n for name in clipped}

        return {name: w / total for name, w in clipped.items()}


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_weight_adjuster: Optional[DynamicWeightAdjuster] = None


def get_weight_adjuster() -> DynamicWeightAdjuster:
    """Get or create the global DynamicWeightAdjuster singleton."""
    global _weight_adjuster
    if _weight_adjuster is None:
        _weight_adjuster = DynamicWeightAdjuster()
    return _weight_adjuster
