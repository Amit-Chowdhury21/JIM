"""
Multi-Engine Orchestrator
=========================
Manages multiple PaperTradingEngine instances simultaneously to allow side-by-side
comparison of different model versions (e.g., v1.0, v2.0, v2.1) on the same live feed.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import copy

from src.paper_trading.engine import PaperTradingEngine, PaperTradingConfig, ModelSignal

class MultiEngineOrchestrator:
    def __init__(self, base_config: Optional[PaperTradingConfig] = None):
        if base_config is None:
            base_config = PaperTradingConfig()
            
        # v1.0: Raw LSTM, no confidence thresholds, fixed sizing
        cfg_v1 = copy.deepcopy(base_config)
        cfg_v1.min_confidence = 0.0
        cfg_v1.use_dynamic_weights = False
        
        # v2.0: CNN-LSTM Ensemble, basic confidence threshold
        cfg_v2 = copy.deepcopy(base_config)
        cfg_v2.min_confidence = 0.50
        cfg_v2.use_dynamic_weights = False
        
        # v2.1: Full Risk Engine, dynamic weights, high confidence thresholds
        cfg_v21 = copy.deepcopy(base_config)
        cfg_v21.min_confidence = 0.60
        cfg_v21.use_dynamic_weights = True

        # v2.3: Alpha Execution Engine (Event, Volatility, Regime)
        cfg_v23 = copy.deepcopy(base_config)
        cfg_v23.min_confidence = 0.60
        cfg_v23.use_dynamic_weights = True

        self.engines: Dict[str, PaperTradingEngine] = {
            "v1.0": PaperTradingEngine(cfg_v1),
            "v2.0": PaperTradingEngine(cfg_v2),
            "v2.1": PaperTradingEngine(cfg_v21),
            "v2.3": PaperTradingEngine(cfg_v23)
        }
        # Status and state
        self.status = "INITIALIZED"
        self.started_at: Optional[datetime] = None
        self.auto_trade_enabled = False

    def start(self) -> Dict[str, Any]:
        if self.status == "RUNNING":
            return {"status": "RUNNING", "message": "Already started"}
            
        results = {}
        for name, engine in self.engines.items():
            res = engine.start()
            results[name] = res
            
        self.status = "RUNNING"
        self.started_at = datetime.now()
        logger.info(f"MultiEngineOrchestrator started with {len(self.engines)} engines.")
        return {"status": "RUNNING", "started_at": self.started_at.isoformat(), "details": results}

    def stop(self) -> Dict[str, Any]:
        if self.status != "RUNNING":
            return {"status": self.status, "message": "Not running"}
            
        results = {}
        for name, engine in self.engines.items():
            res = engine.stop()
            results[name] = res
            
        self.status = "STOPPED"
        logger.info("MultiEngineOrchestrator stopped.")
        return {"status": "STOPPED", "details": results}

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "engines": {name: engine.get_status() for name, engine in self.engines.items()}
        }

    def process_signal(self, model_name: str, signal: ModelSignal):
        """Routes the signal to the correct engine based on the model."""
        trades = {}
        
        # v1.0 only trades on the raw 'lstm' model
        if model_name == "lstm":
            trade = self.engines["v1.0"].process_signal(model_name, signal)
            if trade: trades["v1.0"] = trade
            
        # v2.0 trades on the standard 'ensemble' model
        elif model_name == "ensemble":
            trade2 = self.engines["v2.0"].process_signal(model_name, signal)
            if trade2: trades["v2.0"] = trade2
            
        # v2.1 trades exclusively on the 'ensemble_v21' model (MCDropout + SignalPolicy)
        elif model_name == "ensemble_v21":
            trade21 = self.engines["v2.1"].process_signal(model_name, signal)
            if trade21: trades["v2.1"] = trade21
            
        # v2.3 trades exclusively on the 'ensemble_v23' model (AlphaSignalPolicy)
        elif model_name == "ensemble_v23":
            trade23 = self.engines["v2.3"].process_signal(model_name, signal)
            if trade23: trades["v2.3"] = trade23
            
        # Update price for all engines regardless of trade to keep portfolios in sync
        for name, engine in self.engines.items():
            if signal.current_price > 0:
                engine.update_price(signal.current_price, signal.timestamp)
                
        return trades

    def _create_portfolio_snapshots(self) -> Dict[str, Any]:
        return {name: engine._create_portfolio_snapshot() for name, engine in self.engines.items()}
