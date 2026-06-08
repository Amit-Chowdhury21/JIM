"""
Trade History Tracker
=====================

Tracks executed trades and feeds performance data to:
  - GovernanceMonitor (for degradation detection)
  - DynamicWeightingEngine (for performance multipliers)
  - PositionSizingEngine (for Kelly calculation)

Features:
  - CSV logging of all trades
  - In-memory trade cache
  - Performance calculation by model + regime
  - Automatic P&L computation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json


@dataclass
class Trade:
    """Represents a single executed trade."""
    trade_id: str
    timestamp: datetime
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    side: str = "LONG"  # LONG or SHORT
    quantity: float = 1.0
    position_size_pct: float = 0.0
    
    # Model signals at entry
    ensemble_score: float = 0.0
    ensemble_confidence: float = 0.0
    signal_quality: str = "MODERATE"  # STRONG, MODERATE, WEAK
    contributing_models: Dict[str, float] = None  # model_name -> direction_score
    
    # Regime at entry
    regime: str = "NORMAL"
    regime_confidence: float = 0.0
    
    # Execution details
    stop_loss: float = 0.0
    take_profit: float = 0.0
    actual_stop_triggered: bool = False
    actual_tp_triggered: bool = False
    
    # P&L
    pnl_pips: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Closure reason
    exit_reason: str = ""  # "TP", "SL", "Manual", "Timeout"
    
    def __post_init__(self):
        if self.contributing_models is None:
            self.contributing_models = {}
    
    def close_trade(self, exit_price: float, exit_timestamp: datetime = None, 
                   exit_reason: str = "", stop_triggered: bool = False, 
                   tp_triggered: bool = False):
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp or datetime.now()
        self.exit_reason = exit_reason
        self.actual_stop_triggered = stop_triggered
        self.actual_tp_triggered = tp_triggered
        
        # Calculate P&L
        if self.side == "LONG":
            self.pnl_pips = (exit_price - self.entry_price)
        else:  # SHORT
            self.pnl_pips = (self.entry_price - exit_price)
        
        self.pnl_usd = self.pnl_pips * self.quantity
        self.pnl_pct = (self.pnl_pips / self.entry_price) * 100 if self.entry_price > 0 else 0
    
    def is_winner(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl_pips is not None and self.pnl_pips > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling datetime serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        d["exit_timestamp"] = self.exit_timestamp.isoformat() if self.exit_timestamp else None
        return d


class TradeHistoryTracker:
    """Tracks trade history and performance metrics."""
    
    def __init__(self, log_dir: Path = Path("logs")):
        """
        Initialize trade tracker.
        
        Args:
            log_dir: Directory for CSV logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}  # trade_id -> Trade
        
        # Performance tracking by model and regime
        self.model_performance: Dict[str, Dict[str, Dict]] = {}  # model -> regime -> metrics
        
        # CSV file for logging
        self.csv_path = self.log_dir / "trades.csv"
        self._ensure_csv_header()
    
    def _ensure_csv_header(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            header = "trade_id,timestamp,entry_price,exit_price,exit_timestamp,side,quantity,position_size_pct,ensemble_score,ensemble_confidence,signal_quality,regime,regime_confidence,stop_loss,take_profit,actual_stop_triggered,actual_tp_triggered,pnl_pips,pnl_usd,pnl_pct,exit_reason\n"
            self.csv_path.write_text(header)
            logger.info(f"Created trade log at {self.csv_path}")
    
    def log_trade_entry(self, trade: Trade):
        """Log a new trade entry."""
        self.trades.append(trade)
        self.open_trades[trade.trade_id] = trade
        
        # Log to CSV
        with open(self.csv_path, "a") as f:
            row_dict = trade.to_dict()
            values = [
                row_dict["trade_id"],
                row_dict["timestamp"],
                row_dict["entry_price"],
                row_dict["exit_price"] or "",
                row_dict["exit_timestamp"] or "",
                row_dict["side"],
                row_dict["quantity"],
                row_dict["position_size_pct"],
                row_dict["ensemble_score"],
                row_dict["ensemble_confidence"],
                row_dict["signal_quality"],
                row_dict["regime"],
                row_dict["regime_confidence"],
                row_dict["stop_loss"],
                row_dict["take_profit"],
                row_dict["actual_stop_triggered"],
                row_dict["actual_tp_triggered"],
                row_dict["pnl_pips"] or "",
                row_dict["pnl_usd"] or "",
                row_dict["pnl_pct"] or "",
                row_dict["exit_reason"],
            ]
            f.write(",".join(str(v) for v in values) + "\n")
        
        logger.info(f"Logged trade entry: {trade.trade_id} @ {trade.entry_price}")
    
    def close_trade(self, trade_id: str, exit_price: float, 
                   exit_reason: str = "", stop_triggered: bool = False, 
                   tp_triggered: bool = False):
        """Close a trade and record P&L."""
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return
        
        trade = self.open_trades.pop(trade_id)
        trade.close_trade(exit_price, exit_reason=exit_reason, 
                         stop_triggered=stop_triggered, tp_triggered=tp_triggered)
        
        self.closed_trades.append(trade)
        
        # Update performance tracking
        self._update_model_performance(trade)
        
        logger.info(
            f"Closed trade: {trade_id} | Exit: {exit_price} | "
            f"P&L: {trade.pnl_pips:.1f}pips ({trade.pnl_pct:.2f}%)"
        )
    
    def _update_model_performance(self, trade: Trade):
        """Update performance metrics for each model in the trade."""
        if not trade.is_winner() or trade.regime not in ["GROWTH", "NORMAL", "CRISIS"]:
            return
        
        # For each model that contributed to this trade
        for model_name, direction_score in trade.contributing_models.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {}
            
            if trade.regime not in self.model_performance[model_name]:
                self.model_performance[model_name][trade.regime] = {
                    "trades": 0,
                    "winners": 0,
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                }
            
            metrics = self.model_performance[model_name][trade.regime]
            metrics["trades"] += 1
            if trade.is_winner():
                metrics["winners"] += 1
            metrics["total_pnl"] += trade.pnl_usd or 0
            metrics["total_pnl_pct"] += trade.pnl_pct or 0
    
    def get_model_performance(self, model_name: str, regime: str, lookback: int = 100) -> Dict:
        """
        Get performance metrics for a model in a specific regime.
        
        Returns:
            Dictionary with: hit_rate, avg_return, sharpe, sample_count
        """
        if model_name not in self.model_performance:
            return {
                "hit_rate": 0.5,
                "avg_return": 0.0,
                "sharpe": 0.0,
                "sample_count": 0,
            }
        
        if regime not in self.model_performance[model_name]:
            return {
                "hit_rate": 0.5,
                "avg_return": 0.0,
                "sharpe": 0.0,
                "sample_count": 0,
            }
        
        metrics = self.model_performance[model_name][regime]
        trades = max(metrics["trades"], 1)
        
        hit_rate = metrics["winners"] / trades if trades > 0 else 0.5
        avg_return = metrics["total_pnl_pct"] / trades if trades > 0 else 0.0
        sharpe = avg_return / (1.0 if avg_return == 0 else abs(avg_return) * 0.5)  # Simplified
        
        return {
            "hit_rate": hit_rate,
            "avg_return": avg_return,
            "sharpe": sharpe,
            "sample_count": metrics["trades"],
        }
    
    def get_recent_trades(self, lookback: int = 20) -> List[Dict]:
        """Get recent closed trades for performance analysis."""
        recent = self.closed_trades[-lookback:]
        return [t.to_dict() for t in recent]
    
    def get_summary(self) -> Dict:
        """Get overall performance summary."""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "hit_rate": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
            }
        
        total_trades = len(self.closed_trades)
        winning = sum(1 for t in self.closed_trades if t.is_winner())
        losing = total_trades - winning
        total_pnl = sum(t.pnl_usd or 0 for t in self.closed_trades)
        avg_pnl_pct = np.mean([t.pnl_pct or 0 for t in self.closed_trades])
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "hit_rate": winning / total_trades if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "total_pnl_pct": avg_pnl_pct,
            "open_trades": len(self.open_trades),
        }


# Global instance
_tracker: Optional[TradeHistoryTracker] = None


def get_trade_tracker(log_dir: Path = Path("logs")) -> TradeHistoryTracker:
    """Get or create global trade tracker."""
    global _tracker
    if _tracker is None:
        _tracker = TradeHistoryTracker(log_dir)
    return _tracker
