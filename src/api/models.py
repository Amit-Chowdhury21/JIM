"""
FastAPI Models (Pydantic Schemas)
==================================
Request/response schemas for REST API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TradingSignal(str, Enum):
    """Trading signals."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (ok/degraded/error)")
    timestamp: datetime = Field(default_factory=datetime.now)
    gpu_available: bool = Field(..., description="Whether GPU is usable via PyTorch CUDA")
    gpu_count: int = Field(..., description="Number of CUDA GPUs available")
    rapids_available: bool = Field(..., description="Whether RAPIDS is available")
    database_connected: bool = Field(..., description="QuestDB connectivity")
    redis_connected: bool = Field(..., description="Redis connectivity")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    # Hardware GPU presence (independent of PyTorch CUDA build)
    hardware_gpu_detected: bool = Field(default=False, description="GPU hardware found via nvidia-smi")
    hardware_gpu_names: List[str] = Field(default_factory=list, description="GPU hardware names from nvidia-smi")
    # Phase 6: Extended metrics (optional)
    sla_compliant: Optional[bool] = Field(default=None, description="SLA compliance status")
    uptime_percent: Optional[float] = Field(default=None, description="Uptime percentage")


class CurrentSignalResponse(BaseModel):
    """Current trading signal response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    signal: TradingSignal = Field(..., description="Trading signal: LONG/SHORT/FLAT")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0-1]")
    regime: str = Field(..., description="Current market regime (GROWTH/NORMAL/CRISIS)")
    price: float = Field(..., description="Current gold price")
    volatility_24h: float = Field(..., description="24h realized volatility")
    reasons: List[str] = Field(default_factory=list, description="Signal generation reasons")


class BacktestRequest(BaseModel):
    """Backtest request parameters."""
    strategy: str = Field(..., description="Strategy name to backtest")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000, gt=0, description="Initial capital in USD")
    max_position_pct: float = Field(default=0.05, ge=0.01, le=0.5, description="Max position size %")
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0, description="Kelly criterion fraction")


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    total_return: float = Field(..., description="Total return %")
    annual_return: float = Field(..., description="Annualized return %")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown %")
    win_rate: float = Field(..., description="Win rate %")
    profit_factor: float = Field(..., description="Profit factor (gross profit / gross loss)")
    num_trades: int = Field(..., description="Total number of trades")
    num_winning_trades: int = Field(..., description="Number of winning trades")
    num_losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: float = Field(..., description="Average win $")
    avg_loss: float = Field(..., description="Average loss $")
    best_trade: float = Field(..., description="Best single trade return $")
    worst_trade: float = Field(..., description="Worst single trade return $")
    calmar_ratio: float = Field(..., description="Calmar ratio (return / max DD)")


class BacktestResponse(BaseModel):
    """Backtest response."""
    strategy: str = Field(..., description="Strategy name")
    period: str = Field(..., description="Backtest period")
    metrics: BacktestMetrics
    equity_curve: List[float] = Field(..., description="Daily equity values")
    drawdown_curve: List[float] = Field(..., description="Daily drawdown %")
    trades: List[Dict[str, Any]] = Field(default_factory=list, description="List of trades")


class FeatureRequest(BaseModel):
    """Request current engineered features."""
    symbol: str = Field(default="XAUUSD=X", description="Gold symbol")
    lookback_days: int = Field(default=100, ge=1, le=1000, description="Historical days to include")


class FeaturesResponse(BaseModel):
    """Engineered features response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    symbol: str
    num_features: int = Field(..., description="Number of features computed")
    feature_groups: Dict[str, int] = Field(..., description="Features per category")
    sample_features: Dict[str, float] = Field(..., description="Sample of features (first 10)")


class RegimeResponse(BaseModel):
    """Current regime detection response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    regime: str = Field(..., description="Current regime: GROWTH/NORMAL/CRISIS")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Regime confidence")
    volatility: float = Field(..., description="Current volatility")
    regime_duration_days: int = Field(..., description="How long in current regime")
    regime_probabilities: Dict[str, float] = Field(..., description="Probability of each regime")


class ModelStatusResponse(BaseModel):
    """Status of loaded models."""
    models: Dict[str, Dict[str, Any]] = Field(..., description="Model status details")
    last_training: Optional[datetime] = Field(None, description="Last model training time")
    next_training: Optional[datetime] = Field(None, description="Next scheduled training")


class DataQualityResponse(BaseModel):
    """Data quality report."""
    timestamp: datetime = Field(default_factory=datetime.now)
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality [0-100]")
    gold_data_quality: float = Field(..., description="Gold price data quality %")
    macro_data_quality: float = Field(..., description="Macro data quality %")
    missing_values_pct: float = Field(..., description="Missing values %")
    outliers_detected: int = Field(..., description="Number of outliers detected")
    latest_update: Optional[datetime] = Field(None, description="Latest data timestamp")
    issues: List[str] = Field(default_factory=list, description="Quality issues found")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error description")
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionRequest(BaseModel):
    """Request prediction from ensemble."""
    use_cache: bool = Field(default=True, description="Use cached features if available")
    include_reasons: bool = Field(default=True, description="Include reasoning behind signal")


class EnsembleResponse(BaseModel):
    """Ensemble model prediction response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    direction: TradingSignal = Field(..., description="Predicted direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence [0-1]")
    
    # Individual model outputs
    wavelet_signal: Optional[TradingSignal] = None
    wavelet_confidence: Optional[float] = None
    
    regime_signal: Optional[TradingSignal] = None
    regime_confidence: Optional[float] = None
    
    # Meta-learner details
    meta_learner_confidence: float = Field(..., description="Meta-learner (Critic) confidence")
    recommended_position_size_pct: float = Field(..., description="Kelly-sized position %")
    
    # Risk context
    current_regime: str = Field(..., description="Current market regime")
    current_volatility: float = Field(..., description="Current volatility")
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")


class RetrainingRequest(BaseModel):
    """Model retraining request."""
    models: Optional[List[str]] = Field(
        default=None,
        description="List of model names to retrain (e.g., ['wavelet', 'hmm', 'lstm', 'tft']). If None, defaults to all."
    )
    trigger_reason: Optional[str] = Field(
        default="manual",
        description="Reason for retraining (manual, degradation, scheduled)"
    )


class RetrainingResponse(BaseModel):
    """Model retraining response."""
    job_id: str = Field(..., description="Unique retraining job ID")
    status: str = Field(..., description="Job status (scheduled, in_progress, completed, failed)")
    timestamp: datetime = Field(default_factory=datetime.now)
    models: Optional[List[str]] = Field(default=None, description="Models to retrain")
    trigger_reason: Optional[str] = Field(default=None, description="Retraining trigger reason")
