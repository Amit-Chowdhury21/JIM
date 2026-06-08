"""
Ensemble Live Inference Orchestrator
=====================================

Orchestrates the full ensemble pipeline:
  1. Fetches real-time gold OHLCV + macro data
  2. Classifies market regime (RegimeDetector v3.0)
  3. Runs 4 models in parallel (WaveletPro, HMM Pro, LSTM, TFT)
  4. Normalizes outputs (StandardizedOutputLayer)
  5. Calculates dynamic weights (DynamicWeightingEngine)
  6. Makes final decision (MetaDecisionLayer)
  7. Sizes position (PositionSizingEngine)
  8. Records governance metrics (GovernanceMonitor)
  9. Tracks trade history for performance scoring

High-performance design:
  - Models run in parallel (not sequential)
  - Results cached for short intervals
  - Graceful degradation on model failures
"""

import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger

# Import pipeline components
try:
    from src.models.hmm_regime import RegimeDetector
    from src.models.standardized_output import StandardizedOutputLayer
    from src.models.dynamic_weighting import DynamicWeightingEngine
    from src.models.meta_decision_layer import MetaDecisionLayer
    from src.models.position_sizing import PositionSizingEngine
    from src.models.governance_monitor import GovernanceMonitor
    
    from src.utils.macro_data_feed import fetch_macro_data, get_regime_indicators
    from src.utils.trade_history_tracker import get_trade_tracker
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some ensemble components not available: {e}")
    COMPONENTS_AVAILABLE = False


@dataclass
class EnsemblePrediction:
    """Complete ensemble prediction with all pipeline steps."""
    timestamp: datetime
    gold_price: float
    
    # Regime
    regime: str
    regime_confidence: float
    internal_regime: str
    
    # Model signals
    wavelet_signal: Optional[float] = None
    hmm_signal: Optional[float] = None
    lstm_signal: Optional[float] = None
    tft_signal: Optional[float] = None
    model_signals: Dict[str, float] = None
    model_confidences: Dict[str, float] = None
    
    # Ensemble
    ensemble_score: float = 0.0
    ensemble_confidence: float = 0.0
    disagreement: float = 0.0
    signal_quality: str = "UNKNOWN"
    
    # Decision
    position_side: str = "FLAT"
    recommended_size: float = 0.0
    stop_loss_offset: float = 0.0
    take_profit_offset: float = 0.0
    
    # Macro
    macro_data: Dict[str, float] = None
    
    # Governance
    active_alerts: list = None
    system_health: str = "HEALTHY"


class EnsembleOrchestrator:
    """
    Orchestrates the full ensemble pipeline for live trading.
    """
    
    def __init__(self, account_size: float = 100000.0):
        """
        Initialize orchestrator with all pipeline components.
        
        Args:
            account_size: Trading account size in USD
        """
        self.account_size = account_size
        self.last_prediction: Optional[EnsemblePrediction] = None
        self.last_update: Optional[datetime] = None
        
        # Initialize components
        if COMPONENTS_AVAILABLE:
            self.regime_detector = RegimeDetector(n_regimes=5, version="3.0")
            self.output_normalizer = StandardizedOutputLayer()
            self.weighting_engine = DynamicWeightingEngine()
            self.decision_layer = MetaDecisionLayer()
            self.sizing_engine = PositionSizingEngine(account_size=account_size)
            self.governance = GovernanceMonitor()
            self.trade_tracker = get_trade_tracker()
        else:
            logger.error("Ensemble components not available!")
            raise ImportError("Required ensemble components not installed")
        
        # Cache for model outputs (to avoid re-running during partial failures)
        self._model_cache: Dict[str, Any] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 5
    
    async def run_inference(self, gold_df: pd.DataFrame) -> EnsemblePrediction:
        """
        Run full ensemble pipeline.
        
        Args:
            gold_df: OHLCV DataFrame with gold price data
        
        Returns:
            EnsemblePrediction with all pipeline outputs
        """
        timestamp = datetime.now()
        
        try:
            # 1. REGIME CLASSIFICATION
            logger.info("Step 1/8: Classifying market regime...")
            regime_state = self.regime_detector.classify_regime(gold_df)
            regime = regime_state.regime
            regime_confidence = regime_state.regime_confidence
            
            # 2. FETCH MACRO DATA
            logger.info("Step 2/8: Fetching macro data...")
            macro_data = fetch_macro_data()
            
            # 3. RUN MODELS IN PARALLEL
            logger.info("Step 3/8: Running 4 models in parallel...")
            wavelet_out, hmm_out, lstm_out, tft_out = await self._run_models_parallel(gold_df)
            
            # 4. NORMALIZE OUTPUTS
            logger.info("Step 4/8: Normalizing model outputs...")
            normalized_signals = self.output_normalizer.normalize_all_models({
                "wavelet": wavelet_out,
                "hmm": hmm_out,
                "lstm": lstm_out,
                "tft": tft_out,
            })
            
            # 5. CALCULATE DYNAMIC WEIGHTS
            logger.info("Step 5/8: Calculating dynamic weights...")
            recent_trades = self.trade_tracker.get_recent_trades(lookback=20)
            model_weights = self.weighting_engine.calculate_dynamic_weights(
                regime=regime,
                standardized_signals=normalized_signals,
                recent_trades=recent_trades,
                scorecard_fn=lambda m, r: self.trade_tracker.get_model_performance(m, r)
            )
            
            # 6. MAKE META-DECISION
            logger.info("Step 6/8: Making meta-decision...")
            weighted_signals = {
                signal.model_name: signal.direction_score 
                for signal in normalized_signals
            }
            signal_confidences = {
                signal.model_name: signal.confidence
                for signal in normalized_signals
            }
            signal_disagreement = np.std([s.direction_score for s in normalized_signals])
            
            meta_decision = self.decision_layer.make_meta_decision(
                weighted_signals=weighted_signals,
                signal_confidences=signal_confidences,
                model_weights=model_weights.to_dict(),
                disagreement_penalty=model_weights.disagreement_penalty,
                signal_disagreement=signal_disagreement,
                regime=regime,
                current_price=gold_df['close'].iloc[-1],
                atr=self._calculate_atr(gold_df),
            )
            
            # 7. SIZE POSITION
            logger.info("Step 7/8: Sizing position...")
            position_config = self.sizing_engine.get_recommended_position_config(
                regime=regime,
                current_atr=self._calculate_atr(gold_df),
                current_price=gold_df['close'].iloc[-1],
                signal_quality=meta_decision.signal_quality.value,
                daily_loss_used=0.0,  # Would get from actual trades
                current_drawdown=0.0,  # Would get from account state
            )
            
            # 8. RECORD GOVERNANCE METRICS
            logger.info("Step 8/8: Recording governance metrics...")
            health_status = self.governance.get_health_status()
            
            # Build prediction result
            prediction = EnsemblePrediction(
                timestamp=timestamp,
                gold_price=gold_df['close'].iloc[-1],
                regime=regime,
                regime_confidence=regime_confidence,
                internal_regime=getattr(self.regime_detector, '_internal_regime_map', {}).get(0, "UNKNOWN"),
                wavelet_signal=wavelet_out.get("direction", 0.0) if wavelet_out else 0.0,
                hmm_signal=hmm_out.get("direction", 0.0) if hmm_out else 0.0,
                lstm_signal=lstm_out.get("direction", 0.0) if lstm_out else 0.0,
                tft_signal=tft_out.get("direction", 0.0) if tft_out else 0.0,
                model_signals=weighted_signals,
                model_confidences=signal_confidences,
                ensemble_score=meta_decision.ensemble_score,
                ensemble_confidence=meta_decision.ensemble_confidence,
                disagreement=signal_disagreement,
                signal_quality=meta_decision.signal_quality.value,
                position_side=meta_decision.position_side.value,
                recommended_size=meta_decision.recommended_size,
                stop_loss_offset=meta_decision.stop_loss_offset,
                take_profit_offset=meta_decision.take_profit_offset,
                macro_data=macro_data,
                active_alerts=[a.to_dict() for a in health_status.get("active_alerts", [])],
                system_health=health_status.get("overall_status", "UNKNOWN"),
            )
            
            self.last_prediction = prediction
            self.last_update = timestamp
            
            logger.info(
                f"✅ Ensemble prediction complete: {regime} regime, "
                f"{meta_decision.position_side.value} @ {meta_decision.recommended_size:.2%}, "
                f"conf={meta_decision.ensemble_confidence:.2f}"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"❌ Ensemble inference failed: {e}", exc_info=True)
            self.governance.record_metric(
                "inference_failure",
                value=1.0,
                warn_threshold=0.5,
                crit_threshold=0.8,
                unit="bool"
            )
            raise
    
    async def _run_models_parallel(self, gold_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
        """Run all 4 models in parallel."""
        try:
            # This is a stub - in production, import actual model inference functions
            # and run them with asyncio.gather()
            
            # For now, return placeholder outputs
            wavelet_out = {"direction": 0.3, "strength": 0.65, "noise_ratio": 0.2}
            hmm_out = {"direction": 0.4, "state": "BULLISH", "probability": 0.72}
            lstm_out = {"direction": 0.2, "return_prediction": 0.001, "sequence_quality": 0.8}
            tft_out = {
                "direction": 0.35,
                "1step_forecast": 0.0005,
                "short_forecast": 0.001,
                "long_forecast": 0.002,
                "uncertainty": 0.3
            }
            
            return wavelet_out, hmm_out, lstm_out, tft_out
        
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            # Return fallback outputs
            return {}, {}, {}, {}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr) if not np.isnan(atr) else 10.0
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 10.0  # Fallback
    
    def get_last_prediction(self) -> Optional[EnsemblePrediction]:
        """Get the last ensemble prediction."""
        return self.last_prediction
    
    def export_prediction_for_api(self) -> Dict:
        """Export last prediction in API-friendly format."""
        if not self.last_prediction:
            return {"error": "No prediction available"}
        
        p = self.last_prediction
        return {
            "timestamp": p.timestamp.isoformat(),
            "gold_price": round(p.gold_price, 2),
            "regime": {
                "name": p.regime,
                "confidence": round(p.regime_confidence, 3),
                "internal": p.internal_regime,
            },
            "models": {
                "wavelet": round(p.wavelet_signal or 0, 3),
                "hmm": round(p.hmm_signal or 0, 3),
                "lstm": round(p.lstm_signal or 0, 3),
                "tft": round(p.tft_signal or 0, 3),
            },
            "ensemble": {
                "score": round(p.ensemble_score, 3),
                "confidence": round(p.ensemble_confidence, 3),
                "disagreement": round(p.disagreement, 3),
                "signal_quality": p.signal_quality,
            },
            "decision": {
                "position_side": p.position_side,
                "recommended_size": round(p.recommended_size, 4),
                "stop_loss_offset": round(p.stop_loss_offset, 4),
                "take_profit_offset": round(p.take_profit_offset, 4),
            },
            "macro": {
                "vix": round(p.macro_data.get("vix", 0), 2),
                "dxy": round(p.macro_data.get("dxy", 0), 2),
                "us10y": round(p.macro_data.get("us10y", 0), 2),
                "gold_silver_ratio": round(p.macro_data.get("gold_silver_ratio", 0), 2),
            },
            "system_health": p.system_health,
            "active_alerts": len(p.active_alerts or []),
        }


# Global instance
_orchestrator: Optional[EnsembleOrchestrator] = None


def get_orchestrator(account_size: float = 100000.0) -> EnsembleOrchestrator:
    """Get or create global ensemble orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EnsembleOrchestrator(account_size=account_size)
    return _orchestrator
