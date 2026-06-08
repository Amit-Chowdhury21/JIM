"""
Paper Trading API Routes - Phase 6B Implementation (COMPLETE)
=============================================================

REST API endpoints for paper trading control, monitoring, and real-time updates.

Endpoints (10 total):
- POST /paper-trading/start          - Initialize and start paper trading engine
- GET  /paper-trading/status         - Get current engine status
- GET  /paper-trading/performance    - Get performance metrics
- POST /paper-trading/stop           - Stop engine and close positions
- GET  /paper-trading/trades         - Get trade history (paginated)
- GET  /paper-trading/portfolio      - Get portfolio snapshot
- GET  /paper-trading/risk-report    - Comprehensive risk analysis
- POST /paper-trading/signal         - Inject a trading signal
- POST /paper-trading/config         - Update engine configuration
- POST /paper-trading/reset-daily    - Reset daily P&L and counters
- POST /paper-trading/reset-circuit-breakers - Hard reset all circuit breakers
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Security, Request
from src.api.security import verify_api_key, limiter
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
import asyncio
import json

# Import paper trading modules
try:
    from src.paper_trading.multi_engine import MultiEngineOrchestrator
    from src.paper_trading.engine import (
        PaperTradingEngine,
        PaperTradingConfig,
        TradeExecution,
        ModelSignal,
        SignalType,
        TradeStatus,
    )
    # pyrefly: ignore [missing-import]
    from src.risk.manager import RiskManager
    # pyrefly: ignore [missing-import]
    from src.paper_trading.live_inference import (
        LiveInferenceLoop,
        LIVE_MODEL_SIGNALS,
        CURRENT_GOLD_PRICE,
        LAST_PRICE_UPDATE,
    )
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    logger.warning("Paper trading modules not available")
    PaperTradingEngine = None
    PaperTradingConfig = None
    TradeExecution = None
    ModelSignal = None
    SignalType = None
    TradeStatus = None
    RiskManager = None
    RiskLimits = None
    LiveInferenceLoop = None
    LIVE_MODEL_SIGNALS = None
    CURRENT_GOLD_PRICE = None
    LAST_PRICE_UPDATE = None
    PAPER_TRADING_AVAILABLE = False


# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================================

class PaperTradingStartRequest(BaseModel):
    """Request to start paper trading."""
    initial_capital: float = Field(default=100000.0, gt=0, description="Starting capital in USD")
    kelly_fraction: float = Field(default=0.25, ge=0.05, le=1.0, description="Kelly criterion fraction")
    max_position_pct: float = Field(default=0.10, ge=0.01, le=0.50, description="Max position as % of capital")
    max_daily_loss_pct: float = Field(default=0.02, ge=0.005, le=0.10, description="Max daily loss %")
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.50, description="Max drawdown %")
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0, description="Min signal confidence to trade")
    commission_per_trade: float = Field(default=5.0, ge=0.0, description="Fixed commission per trade in USD")
    slippage_pct: float = Field(default=0.01, ge=0.0, description="Slippage as % of price")


class PaperTradingStatusResponse(BaseModel):
    """Current paper trading status."""
    status: str  # INITIALIZED, RUNNING, STOPPED
    started_at: Optional[str] = None
    current_time: str
    uptime_seconds: Optional[float] = None
    portfolio: Dict[str, Any]
    trading: Dict[str, Any]
    models: Dict[str, Any]
    silver: Optional[Dict[str, Any]] = None


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics for paper trading."""
    total_value: float
    cash: float
    position_quantity: float
    position_value: float
    pnl_total: float
    pnl_realized: float
    pnl_unrealized: float
    pnl_daily: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    daily_trades: int


class TradeHistoryResponse(BaseModel):
    """Trade record from history."""
    trade_id: str
    model_name: str
    signal_type: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    silver_quantity: float = 0.0
    silver_entry_price: Optional[float] = None
    silver_exit_price: Optional[float] = None
    entry_time: str
    exit_time: Optional[str]
    pnl: float
    pnl_pct: float
    status: str
    regime: str = "NORMAL"
    confidence: float = 0.0


class PortfolioSnapshotResponse(BaseModel):
    """Portfolio state at a point in time."""
    timestamp: str
    total_value: float
    cash: float
    position_quantity: float
    position_value: float
    pnl_total: float
    daily_pnl: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int = 0


class SignalInjectionRequest(BaseModel):
    """Request to inject a trading signal."""
    model_name: str = Field(..., description="Model name: wavelet, hmm, lstm, tft, genetic, ensemble")
    signal_type: str = Field(..., description="Signal: LONG, SHORT, CLOSE, HOLD")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence [0-1]")
    price: float = Field(..., gt=0, description="Current gold price")
    regime: str = Field(default="NORMAL", description="Market regime: GROWTH, NORMAL, CRISIS")
    reasoning: str = Field(default="", description="Signal reasoning")


class ConfigUpdateRequest(BaseModel):
    """Request to update paper trading configuration."""
    kelly_fraction: Optional[float] = Field(None, ge=0.05, le=1.0)
    max_position_pct: Optional[float] = Field(None, ge=0.01, le=0.50)
    max_daily_loss_pct: Optional[float] = Field(None, ge=0.005, le=0.10)
    max_drawdown_pct: Optional[float] = Field(None, ge=0.05, le=0.50)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    trading_enabled: Optional[bool] = None


# ============================================================================
# GLOBAL STATE
# ============================================================================

_orchestrator: Optional['MultiEngineOrchestrator'] = None
_paper_trading_engine: Optional['PaperTradingEngine'] = None
_paper_trading_config: Optional['PaperTradingConfig'] = None
_risk_manager: Optional['RiskManager'] = None
_websocket_clients: List[WebSocket] = []
_inference_loop: Optional['LiveInferenceLoop'] = None
_inference_task: Optional[asyncio.Task] = None


class QueuedSignal:
    def __init__(self, model_name: str, signal: 'ModelSignal', future: asyncio.Future):
        self.model_name = model_name
        self.signal = signal
        self.future = future

class TradingQueueWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._task = None

    async def start(self):
        logger.info("Starting TradingQueueWorker task...")
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("TradingQueueWorker task stopped.")

    async def _loop(self):
        global _paper_trading_engine, _risk_manager
        while True:
            try:
                item: QueuedSignal = await self.queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error getting from queue: {e}")
                continue

            async with self._lock:
                try:
                    if _paper_trading_engine is None:
                        item.future.set_result((None, "Engine not initialized"))
                        self.queue.task_done()
                        continue

                    # 1. Risk checks (only check for new entries: LONG or SHORT)
                    signal_type = item.signal.signal_type
                    if _risk_manager is not None and signal_type in (SignalType.LONG, SignalType.SHORT):
                        can_trade, reason = _risk_manager.check_circuit_breakers(
                            portfolio_value=_orchestrator.engines['v2.1']._create_portfolio_snapshot().total_value,
                            ensemble_conf=item.signal.confidence
                        )
                        if not can_trade:
                            logger.warning(f"Trade blocked by RiskManager in queue worker: {reason}")
                            item.future.set_result((None, f"Blocked by RiskManager: {reason}"))
                            self.queue.task_done()
                            continue

                    # 2. Process signal through engine
                    trade = _orchestrator.process_signal(item.model_name, item.signal)
                    if trade is not None and _risk_manager is not None:
                        _risk_manager.risk_state.bars_since_last_trade = 0
                    item.future.set_result((trade, "OK"))
                except Exception as e:
                    logger.error(f"Error processing signal in queue worker: {e}")
                    if not item.future.done():
                        item.future.set_exception(e)
                finally:
                    self.queue.task_done()


_trading_queue_worker: Optional[TradingQueueWorker] = None


def get_engine():
    """Get the paper trading engine instance (for testing injection)."""
    return _paper_trading_engine


def set_engine(engine, config=None, risk_mgr=None):
    """Set the paper trading engine instance (for testing injection)."""
    global _paper_trading_engine, _paper_trading_config, _risk_manager
    _paper_trading_engine = engine
    _paper_trading_config = config
    _risk_manager = risk_mgr


# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(
    prefix="/paper-trading",
    tags=["paper-trading"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# WEBSOCKET BROADCAST HELPER
# ============================================================================

async def broadcast_update(event_type: str, data: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients."""
    message = json.dumps({"event": event_type, "data": data, "timestamp": datetime.now().isoformat()})
    disconnected = []
    for ws in _websocket_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _websocket_clients.remove(ws)


# ============================================================================
# ROUTE HANDLERS
# ============================================================================

@router.post("/start", response_model=Dict[str, Any], dependencies=[Security(verify_api_key)])
@limiter.limit("5/minute")
async def start_paper_trading(request: Request, start_request: PaperTradingStartRequest) -> Dict[str, Any]:
    """
    Start paper trading engine.
    
    Initializes the engine with specified capital, risk limits, and signal
    configuration. Returns initialization details on success.
    """
    global _paper_trading_engine, _paper_trading_config, _risk_manager, _trading_queue_worker, _orchestrator
    
    if not PAPER_TRADING_AVAILABLE or PaperTradingConfig is None or PaperTradingEngine is None or RiskManager is None:
        raise HTTPException(status_code=500, detail="Paper trading module not available")
    
    if _paper_trading_engine is not None and _paper_trading_engine.status == "RUNNING":
        raise HTTPException(status_code=409, detail="Paper trading is already running. Stop it first.")
    
    try:
        # Create configuration
        _paper_trading_config = PaperTradingConfig(
            initial_capital=start_request.initial_capital,
            kelly_fraction=start_request.kelly_fraction,
            max_position_pct=start_request.max_position_pct,
            max_daily_loss_pct=start_request.max_daily_loss_pct,
            max_drawdown_pct=start_request.max_drawdown_pct,
            min_confidence=start_request.min_confidence,
            commission_per_trade=start_request.commission_per_trade,
            slippage_pct=start_request.slippage_pct,
        )
        
        # Initialize engine
        _orchestrator = MultiEngineOrchestrator(_paper_trading_config)
        _paper_trading_engine = _orchestrator.engines['v2.1']
        
        # Initialize risk manager with config dict instead of deprecated RiskLimits
        risk_cfg = {
            "risk": {
                "kelly": {
                    "max_position_pct": start_request.max_position_pct,
                    "fraction": start_request.kelly_fraction,
                },
                "circuit_breakers": {
                    "daily_loss_limit": start_request.max_daily_loss_pct,
                    "drawdown_stop": start_request.max_drawdown_pct,
                    "min_confidence": start_request.min_confidence,
                }
            }
        }
        _risk_manager = RiskManager(risk_cfg)
        
        # Start engine
        result = _orchestrator.start()

        # Start queue worker
        _trading_queue_worker = TradingQueueWorker()
        await _trading_queue_worker.start()

        # ── Start live inference loop ──────────────────────────────────────
        global _inference_loop, _inference_task
        if LiveInferenceLoop is not None:
            _inference_loop = LiveInferenceLoop(
                engine=_paper_trading_engine,
                broadcast_fn=broadcast_update,
                interval_seconds=60,
            )
            _inference_task = asyncio.create_task(_inference_loop.run())
            logger.info("Live inference loop started (60s cadence, all 6 models)")
        # ─────────────────────────────────────────────────────────────────

        logger.info(f"Paper trading started with ${start_request.initial_capital:,.0f} capital")
        
        # Broadcast to WebSocket clients
        await broadcast_update("engine_started", result)
        
        return {
            "status": "success",
            "message": "Paper trading engine started",
            "config": {
                "initial_capital": start_request.initial_capital,
                "kelly_fraction": start_request.kelly_fraction,
                "max_position_pct": start_request.max_position_pct,
                "max_daily_loss_pct": start_request.max_daily_loss_pct,
                "min_confidence": start_request.min_confidence,
            },
            "details": result,
        }
    
    except Exception as e:
        logger.error(f"Failed to start paper trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start: {str(e)}")


@router.get("/status", response_model=Dict[str, Any])
async def get_paper_trading_status() -> Dict[str, Any]:
    """
    Get current paper trading status including portfolio, trading stats,
    and per-model signal information.
    """
    global _paper_trading_engine
    
    if _paper_trading_engine is None:
        raise HTTPException(
            status_code=404,
            detail="Paper trading engine not initialized. Call POST /paper-trading/start first."
        )
    
    try:
        status_dict = _orchestrator.get_status()
        
        # Calculate uptime
        uptime = None
        if _paper_trading_engine.started_at:
            uptime = (datetime.now() - _paper_trading_engine.started_at).total_seconds()
        
        return status_dict
    
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics including P&L breakdown, Sharpe ratio,
    drawdown, and win rate.
    """
    global _paper_trading_engine
    
    if _paper_trading_engine is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        snapshots = _orchestrator._create_portfolio_snapshots()
        
        return {
            name: {
                "total_value": snap.total_value,
                "cash": snap.cash,
                "position_quantity": snap.position_quantity,
                "position_value": snap.position_value,
                "pnl_total": snap.pnl_total,
                "pnl_realized": snap.pnl_realized,
                "pnl_unrealized": snap.pnl_unrealized,
                "pnl_daily": snap.daily_pnl,
                "return_pct": snap.return_pct,
                "sharpe_ratio": snap.sharpe_ratio,
                "max_drawdown": snap.max_drawdown,
                "win_rate": snap.win_rate,
                "num_trades": snap.num_trades,
                "daily_trades": len(_orchestrator.engines[name].daily_trades),
            } for name, snap in snapshots.items()
        }
    
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/stop", response_model=Dict[str, Any], dependencies=[Security(verify_api_key)])
@limiter.limit("5/minute")
async def stop_paper_trading(request: Request) -> Dict[str, Any]:
    """
    Stop paper trading and close all open positions.
    Returns final P&L summary and session statistics.
    """
    global _paper_trading_engine, _trading_queue_worker
    
    if _paper_trading_engine is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        result = _orchestrator.stop()

        # Stop queue worker
        if _trading_queue_worker is not None:
            await _trading_queue_worker.stop()
            _trading_queue_worker = None

        # ── Stop live inference loop ──────────────────────────────────────
        global _inference_loop, _inference_task
        if _inference_loop is not None:
            _inference_loop.stop()
            _inference_loop = None
            logger.info("Live inference loop stopped")
        # ─────────────────────────────────────────────────────────────────

        logger.info(f"Paper trading stopped. Final P&L: ${result.get('total_pnl', 0):.2f}")
        
        # Broadcast to WebSocket clients
        await broadcast_update("engine_stopped", result)
        
        return {
            "status": "success",
            "message": "Paper trading stopped",
            "details": result,
        }
    
    except Exception as e:
        logger.error(f"Failed to stop paper trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop: {str(e)}")


@router.get("/trades", response_model=List[TradeHistoryResponse])
async def get_trade_history(
    limit: int = Query(50, ge=1, le=500, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Number of trades to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by status: OPEN, CLOSED, ALL"),
) -> List[TradeHistoryResponse]:
    """
    Get trade history with pagination and optional status filtering.
    """
    global _orchestrator
    
    if _orchestrator is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        trades = []
        for engine in _orchestrator.engines.values():
            trades.extend(engine.trades)
        trades.sort(key=lambda t: t.entry_time, reverse=True)
        
        # Apply status filter
        if status_filter and status_filter != "ALL":
            trades = [t for t in trades if t.status.value == status_filter]
        
        # Apply pagination
        trades = trades[offset:offset + limit]
        
        return [
            TradeHistoryResponse(
                trade_id=t.trade_id,
                model_name=t.model_name,
                signal_type=t.signal_type.value,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                quantity=t.quantity,
                silver_quantity=t.silver_quantity,
                silver_entry_price=t.silver_entry_price,
                silver_exit_price=t.silver_exit_price,
                entry_time=t.entry_time.isoformat(),
                exit_time=t.exit_time.isoformat() if t.exit_time else None,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                status=t.status.value,
                regime=t.regime,
                confidence=t.confidence,
            )
            for t in trades
        ]
    
    except Exception as e:
        logger.error(f"Failed to get trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trades: {str(e)}")


@router.get("/portfolio", response_model=Dict[str, Any])
async def get_portfolio_snapshot() -> Dict[str, Any]:
    """
    Get current portfolio snapshot with mark-to-market valuation.
    """
    global _paper_trading_engine
    
    if _paper_trading_engine is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        snapshot = _paper_trading_engine._create_portfolio_snapshot()
        
        return PortfolioSnapshotResponse(
            timestamp=snapshot.timestamp.isoformat(),
            total_value=snapshot.total_value,
            cash=snapshot.cash,
            position_quantity=snapshot.position_quantity,
            position_value=snapshot.position_value,
            pnl_total=snapshot.pnl_total,
            daily_pnl=snapshot.daily_pnl,
            return_pct=snapshot.return_pct,
            sharpe_ratio=snapshot.sharpe_ratio,
            max_drawdown=snapshot.max_drawdown,
            win_rate=snapshot.win_rate,
            num_trades=snapshot.num_trades,
        )
    
    except Exception as e:
        logger.error(f"Failed to get portfolio snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio: {str(e)}")


@router.get("/risk-report", response_model=Dict[str, Any])
async def get_risk_report() -> Dict[str, Any]:
    """
    Get comprehensive risk report with current limits, violations,
    and exposure analysis.
    """
    global _paper_trading_engine, _risk_manager
    
    if _paper_trading_engine is None or _risk_manager is None:
        raise HTTPException(
            status_code=404,
            detail="Paper trading or risk manager not initialized"
        )
    
    try:
        snapshot = _paper_trading_engine._create_portfolio_snapshot()
        
        report = _risk_manager.get_risk_report(
            current_equity=snapshot.total_value,
            daily_pnl=snapshot.daily_pnl
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "risk_report": report,
            "portfolio_summary": {
                "total_value": snapshot.total_value,
                "return_pct": snapshot.return_pct,
                "max_drawdown": snapshot.max_drawdown,
            },
        }
    
    except Exception as e:
        logger.error(f"Failed to get risk report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk report: {str(e)}")


@router.post("/signal", response_model=Dict[str, Any], dependencies=[Security(verify_api_key)])
@limiter.limit("5/second")
async def inject_signal(request: Request, signal_request: SignalInjectionRequest) -> Dict[str, Any]:
    """
    Inject a trading signal from a model into the paper trading engine.
    The engine will evaluate the signal against risk limits and execute
    if all checks pass.
    """
    global _paper_trading_engine, _trading_queue_worker
    
    if _paper_trading_engine is None or SignalType is None or ModelSignal is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    if _paper_trading_engine.status != "RUNNING":
        raise HTTPException(status_code=409, detail="Paper trading engine is not running")
    
    # Validate model name
    valid_models = ["wavelet_pro", "hmm_pro", "lstm", "tft_pro", "tft_pro_max", "ensemble"]
    if signal_request.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name. Must be one of: {valid_models}"
        )
    
    # Validate signal type
    try:
        signal_type = SignalType(signal_request.signal_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signal_type. Must be one of: LONG, SHORT, CLOSE, HOLD"
        )
    
    try:
        # Create ModelSignal
        signal = ModelSignal(
            model_name=signal_request.model_name,
            signal_type=signal_type,
            confidence=signal_request.confidence,
            entry_price=signal_request.price,
            current_price=signal_request.price,
            timestamp=datetime.now(),
            reasoning=signal_request.reasoning,
            regime=signal_request.regime,
        )
        
        # In testing environments, standard synchronous TestClient blocks the event loop
        # so background queue worker task is never executed. Bypass queue worker in tests.
        from src.api.security import is_testing
        if is_testing:
            if _risk_manager is not None and signal_type in (SignalType.LONG, SignalType.SHORT):
                can_trade, reason = _risk_manager.check_circuit_breakers(
                    portfolio_value=_orchestrator.engines['v2.1']._create_portfolio_snapshot().total_value,
                    ensemble_conf=signal_request.confidence
                )
                if not can_trade:
                    logger.warning(f"Trade blocked by RiskManager: {reason}")
                    return {
                        "status": "success", 
                        "signal_processed": True, 
                        "trade_executed": False, 
                        "reason": f"Blocked by RiskManager: {reason}"
                    }
            trade = _orchestrator.process_signal(signal_request.model_name, signal)
            if trade is not None and _risk_manager is not None:
                _risk_manager.risk_state.bars_since_last_trade = 0
        else:
            # Ensure queue worker is running
            if _trading_queue_worker is None:
                _trading_queue_worker = TradingQueueWorker()
                await _trading_queue_worker.start()

            # Enqueue signal and wait for queue worker execution
            future = asyncio.get_running_loop().create_future()
            queued_item = QueuedSignal(signal_request.model_name, signal, future)
            await _trading_queue_worker.queue.put(queued_item)
            
            trade, status_msg = await future
            
            if status_msg != "OK":
                return {
                    "status": "success",
                    "signal_processed": True,
                    "trade_executed": False,
                    "reason": status_msg
                }
        
        result: Dict[str, Any] = {
            "status": "success",
            "signal_processed": True,
            "model": signal_request.model_name,
            "signal_type": signal_request.signal_type,
            "confidence": signal_request.confidence,
            "trade_executed": trade is not None,
        }
        
        if trade:
            result["trade"] = {
                "trade_id": trade.trade_id,
                "signal_type": trade.signal_type.value,
                "entry_price": trade.entry_price,
                "quantity": trade.quantity,
                "status": trade.status.value,
            }
            # Broadcast trade to WebSocket clients
            await broadcast_update("trade_executed", result["trade"])
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to process signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process signal: {str(e)}")


@router.post("/config", response_model=Dict[str, Any], dependencies=[Security(verify_api_key)])
async def update_config(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update paper trading configuration dynamically.
    Only provided fields are updated; others remain unchanged.
    """
    global _paper_trading_engine, _paper_trading_config
    
    if _paper_trading_engine is None or _paper_trading_config is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        updated_fields = {}
        
        if request.kelly_fraction is not None:
            _paper_trading_config.kelly_fraction = request.kelly_fraction
            _paper_trading_engine.config.kelly_fraction = request.kelly_fraction
            updated_fields["kelly_fraction"] = request.kelly_fraction
        
        if request.max_position_pct is not None:
            _paper_trading_config.max_position_pct = request.max_position_pct
            _paper_trading_engine.config.max_position_pct = request.max_position_pct
            updated_fields["max_position_pct"] = request.max_position_pct
        
        if request.max_daily_loss_pct is not None:
            _paper_trading_config.max_daily_loss_pct = request.max_daily_loss_pct
            _paper_trading_engine.config.max_daily_loss_pct = request.max_daily_loss_pct
            updated_fields["max_daily_loss_pct"] = request.max_daily_loss_pct
        
        if request.max_drawdown_pct is not None:
            _paper_trading_config.max_drawdown_pct = request.max_drawdown_pct
            _paper_trading_engine.config.max_drawdown_pct = request.max_drawdown_pct
            updated_fields["max_drawdown_pct"] = request.max_drawdown_pct
        
        if request.min_confidence is not None:
            _paper_trading_config.min_confidence = request.min_confidence
            _paper_trading_engine.config.min_confidence = request.min_confidence
            updated_fields["min_confidence"] = request.min_confidence
        
        if request.trading_enabled is not None:
            _paper_trading_config.trading_enabled = request.trading_enabled
            _paper_trading_engine.config.trading_enabled = request.trading_enabled
            updated_fields["trading_enabled"] = request.trading_enabled
        
        logger.info(f"Paper trading config updated: {updated_fields}")
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "updated_fields": updated_fields,
        }
    
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/reset-daily", response_model=Dict[str, Any], dependencies=[Security(verify_api_key)])
async def reset_daily_counters() -> Dict[str, Any]:
    """
    Reset daily P&L counters and trade limits.
    Called automatically at market open or manually for testing.
    """
    global _paper_trading_engine, _risk_manager
    
    if _paper_trading_engine is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
    
    try:
        # Reset engine daily state
        previous_daily_pnl = _paper_trading_engine.daily_pnl
        _paper_trading_engine.daily_pnl = 0.0
        _paper_trading_engine.daily_trades = []
        _paper_trading_engine.day_started_at = datetime.now()
        
        # Reset risk manager daily state
        if _risk_manager:
            current_equity = _orchestrator.engines['v2.1']._create_portfolio_snapshot().total_value
            _risk_manager.reset_daily()
        
        logger.info(f"Daily counters reset. Previous daily P&L: ${previous_daily_pnl:.2f}")
        
        return {
            "status": "success",
            "message": "Daily counters reset",
            "previous_daily_pnl": previous_daily_pnl,
            "reset_time": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Failed to reset daily counters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")


@router.post("/reset-circuit-breakers", dependencies=[Security(verify_api_key)])
async def reset_circuit_breakers() -> Dict[str, Any]:
    """
    Hard reset all circuit breakers (drawdown, consecutive losses, daily limits)
    to allow the engine to resume trading after a halt.
    """
    global _paper_trading_engine, _risk_manager
    
    if _paper_trading_engine is None or _risk_manager is None:
        raise HTTPException(status_code=404, detail="Paper trading engine not initialized")
        
    try:
        current_equity = _orchestrator.engines['v2.1']._create_portfolio_snapshot().total_value
        
        # 1. Reset Risk Manager
        _risk_manager.reset_circuit_breakers(current_equity)
        
        # 2. Reset Engine Daily P&L so Daily Loss CB clears
        _paper_trading_engine.daily_pnl = 0.0
        
        logger.warning(f"CIRCUIT BREAKERS RESET BY USER! Equity baseline: ${current_equity:.2f}")
        
        return {
            "status": "success",
            "message": "Circuit breakers successfully reset. Trading can now resume.",
            "equity_baseline": current_equity
        }
    except Exception as e:
        logger.error(f"Failed to reset circuit breakers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LIVE INFERENCE ENDPOINTS
# ============================================================================

@router.get("/live-signals", response_model=Dict[str, Any])
async def get_live_signals() -> Dict[str, Any]:
    """
    Get the latest signal from every model, updated every 60 seconds by the
    live inference loop running in the background.

    Returns per-model: signal, confidence, regime, price, reasoning, last_updated.
    This is the primary data source for the Models tab live view.
    """
    try:
        import src.paper_trading.live_inference as li
        signals = dict(li.LIVE_MODEL_SIGNALS)
        return {
            "status": "ok",
            "current_price": li.CURRENT_GOLD_PRICE,
            "last_price_update": li.LAST_PRICE_UPDATE.isoformat() if li.LAST_PRICE_UPDATE else None,
            "inference_running": _inference_loop is not None and _inference_loop._running,
            "iteration": _inference_loop.iteration if _inference_loop else 0,
            "macro": li.MACRO_DATA,
            "models": signals,
        }
    except Exception as e:
        logger.error(f"Failed to get live signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inference-status", response_model=Dict[str, Any])
async def get_inference_status() -> Dict[str, Any]:
    """
    Get live inference loop health status: running state, cadence, iteration count,
    last run time, and any errors.
    """
    if _inference_loop is None:
        return {
            "running": False,
            "message": "Inference loop not started. Start the paper trading engine first.",
            "iteration": 0,
            "last_run": None,
            "last_error": None,
            "interval_seconds": 60,
        }
    return {
        "running": _inference_loop._running,
        "iteration": _inference_loop.iteration,
        "last_run": _inference_loop.last_run.isoformat() if _inference_loop.last_run else None,
        "last_error": _inference_loop.last_error,
        "interval_seconds": _inference_loop.interval_seconds,
        "message": "Live inference loop running — all 6 models updating every 60s",
    }


@router.get("/model-weights", response_model=Dict[str, Any])
async def get_model_weights() -> Dict[str, Any]:
    """
    Get current dynamic model weights per regime.

    Returns the regime-conditional, performance-adaptive weights for each model,
    along with per-model stats (win rate, rolling Sharpe, trade count) and
    whether performance adaptation is active.

    This endpoint powers the Dynamic Weights visualisation in the dashboard.
    """
    try:
        from src.paper_trading.dynamic_weights import (
            get_weight_adjuster, REGIME_BASE_WEIGHTS,
        )

        adjuster = get_weight_adjuster()

        # If engine is running, use its regime; otherwise default to NORMAL
        current_regime = "NORMAL"
        if _paper_trading_engine and hasattr(_paper_trading_engine, 'weight_adjuster'):
            current_regime = _paper_trading_engine.weight_adjuster.current_regime or "NORMAL"

        summary = adjuster.get_weight_summary()

        return {
            "status": "ok",
            "regime": current_regime,
            "weights": summary["weights"],
            "model_stats": summary["model_stats"],
            "adaptation_active": summary["adaptation_active"],
            "regime_base_weights": REGIME_BASE_WEIGHTS,
        }
    except Exception as e:
        logger.error(f"Failed to get model weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PREDICTION LOG
# ============================================================================

@router.get("/prediction-log", summary="Get Model Prediction Logs")
def get_prediction_log(limit: int = Query(default=100, ge=1, le=1000)):
    """
    Get the prediction cycle CSV log.
    Returns the most recent `limit` rows.
    """
    try:
        from src.paper_trading.prediction_logger import get_csv_path
        import os
        import csv
        
        csv_path = get_csv_path()
        if not os.path.exists(csv_path):
            return {"status": "ok", "logs": []}
            
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        # Return the most recent 'limit' rows
        return {
            "status": "ok",
            "logs": rows[-limit:] if limit > 0 else rows,
        }
    except Exception as e:
        logger.error(f"Failed to read prediction log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation-matrix", summary="Get Model Correlation Matrix")
def get_correlation_matrix(limit: int = Query(default=500, ge=10, le=5000)):
    """
    Calculate the correlation matrix and consensus metrics dynamically from prediction_log.csv.
    """
    try:
        from src.paper_trading.prediction_logger import get_csv_path
        import os
        import csv
        
        csv_path = get_csv_path()
        if not os.path.exists(csv_path):
            return {"status": "ok", "correlationMatrix": {}, "avgPairwiseAgreement": {}, "consensusAnalysis": []}
            
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        recent_rows = rows[-limit:] if limit > 0 else rows
        
        # We define agreement as matching exactly (LONG == LONG, SHORT == SHORT, HOLD == HOLD)
        models_map = {
            "wavelet": "wavelet_pro_signal",
            "hmm": "hmm_pro_signal",
            "lstm": "lstm_signal",
            "tft": "tft_signal",
            "tft_pro": "tft_pro_signal",
            "tft_pro_max": "tft_pro_max_signal",
            "genetic": "genetic_signal",
            "ensemble": "ensemble_signal"
        }
        
        model_keys = list(models_map.keys())
        base_models = [m for m in model_keys if m != "ensemble"]
        
        # Initialize counts
        agree_counts = {m: {m2: 0 for m2 in model_keys} for m in model_keys}
        valid_rows_count = 0
        
        for row in recent_rows:
            # Check if all keys exist in row, if not skip
            missing = False
            for k, csv_col in models_map.items():
                if csv_col not in row or not row[csv_col]:
                    missing = True
                    break
            if missing:
                continue
            
            valid_rows_count += 1
            for m1 in model_keys:
                sig1 = row[models_map[m1]]
                for m2 in model_keys:
                    sig2 = row[models_map[m2]]
                    if sig1 == sig2:
                        agree_counts[m1][m2] += 1
                        
        if valid_rows_count == 0:
            return {"status": "ok", "correlationMatrix": {}, "avgPairwiseAgreement": {}, "consensusAnalysis": []}
            
        # 1. Correlation Matrix
        correlation_matrix = {}
        for m1 in model_keys:
            correlation_matrix[m1] = {}
            for m2 in model_keys:
                correlation_matrix[m1][m2] = int(round((agree_counts[m1][m2] / valid_rows_count) * 100))
                
        # 2. Avg Pairwise Agreement (among base models)
        avg_pairwise = {}
        for m1 in base_models:
            total_agree = 0
            for m2 in base_models:
                if m1 != m2:
                    total_agree += correlation_matrix[m1][m2]
            avg_pairwise[m1] = int(round(total_agree / (len(base_models) - 1))) if len(base_models) > 1 else 100
            
        # 3. Consensus Analysis
        consensus_analysis = []
        for m1 in base_models:
            consensus_analysis.append({
                "model": m1,
                "name": m1.capitalize(),
                "agreement": correlation_matrix[m1]["ensemble"]
            })
            
        return {
            "status": "ok",
            "correlationMatrix": correlation_matrix,
            "avgPairwiseAgreement": avg_pairwise,
            "consensusAnalysis": consensus_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# SAVE PREDICTION LOGS TO LOCAL FILESYSTEM
# ============================================================================

@router.post("/save-prediction-logs", summary="Save Prediction Logs to Local Storage")
def save_prediction_logs(logs: List[Dict[str, Any]] = None):
    """
    Save prediction logs to local filesystem with session-wise organization.
    
    Saves logs to: E:\PRO\JIMxNik\PredictionLogs\PredictionLogs_DDMMYYYY_HHMMSS.csv
    Creates a new session log on first call, then saves every 5 minutes.
    
    Args:
        logs: List of prediction log entries to save (if None, reads from API)
    
    Returns:
        {
            "status": "success",
            "file_path": "...",
            "records_saved": N,
            "session_id": "..."
        }
    """
    try:
        import os
        import csv
        from datetime import datetime
        
        # Define local storage path
        LOGS_DIR = r"E:\PRO\JIMxNik\PredictionLogs"
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # If logs not provided, read from API CSV
        if logs is None:
            from src.paper_trading.prediction_logger import get_csv_path
            csv_path = get_csv_path()
            if not os.path.exists(csv_path):
                return {"status": "ok", "records_saved": 0, "message": "No logs available"}
            
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                logs = list(reader)
        
        if not logs:
            return {"status": "ok", "records_saved": 0, "message": "No logs to save"}
        
        # Create hour-wise session file
        from datetime import timedelta
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        
        # Filter logs to only include data for the current hour
        filtered_logs = []
        for log in logs:
            try:
                # Handle possible 'Z' or timezone info if present, though typically it's 'YYYY-MM-DDTHH:MM:SS'
                ts_str = log.get("timestamp", "").replace("Z", "+00:00")
                log_time = datetime.fromisoformat(ts_str)
                # If log_time is timezone aware and hour_start is naive, we need to make hour_start aware
                if log_time.tzinfo is not None and hour_start.tzinfo is None:
                    # Strip timezone for comparison assuming local time is intended, or use aware
                    log_time = log_time.replace(tzinfo=None)
                    
                if hour_start <= log_time < hour_end:
                    filtered_logs.append(log)
            except (ValueError, TypeError):
                pass
                
        logs = filtered_logs
        
        if not logs:
            return {"status": "ok", "records_saved": 0, "message": "No logs available for current hour"}
        
        # Format: PredictionLogs_DDMMYYYY_HH00_to_HH00.csv
        session_id = f"{hour_start.strftime('%d%m%Y_%H00')}_to_{hour_end.strftime('%H00')}"
        filename = f"PredictionLogs_{session_id}.csv"
        filepath = os.path.join(LOGS_DIR, filename)
        
        # Get headers from first log entry
        headers = list(logs[0].keys())
        
        # Check if session file already exists
        records_saved = len(logs)
        if os.path.exists(filepath):
            # Read existing logs
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_logs = list(reader)
            
            # Deduplicate by timestamp
            existing_timestamps = {log.get("timestamp") for log in existing_logs}
            new_logs = [log for log in logs if log.get("timestamp") not in existing_timestamps]
            records_saved = len(new_logs)
            
            # Append new logs only
            if new_logs:
                with open(filepath, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerows(new_logs)
        else:
            # Create new session file with headers
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(logs)
        
        # Count total records in file
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            total_records = sum(1 for _ in f) - 1  # Subtract header row
        
        logger.info(f"Saved {records_saved} prediction logs to: {filepath}")
        
        return {
            "status": "success",
            "file_path": filepath,
            "records_saved": records_saved,
            "session_id": session_id,
            "total_records_in_session": total_records
        }
        
    except Exception as e:
        logger.error(f"Failed to save prediction logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time paper trading updates.
    
    Sends periodic portfolio snapshots and trade notifications.
    Connect at: ws://localhost:8000/paper-trading/ws
    """
    await websocket.accept()
    _websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(_websocket_clients)}")
    
    try:
        # Send initial status
        if _paper_trading_engine:
            status = _paper_trading_engine.get_status()
            await websocket.send_text(json.dumps({
                "event": "connected",
                "data": status,
                "timestamp": datetime.now().isoformat(),
            }))
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                
                # Handle client commands
                try:
                    cmd = json.loads(data)
                    if cmd.get("type") == "ping":
                        await websocket.send_text(json.dumps({"event": "pong"}))
                    elif cmd.get("type") == "get_status" and _paper_trading_engine:
                        status = _paper_trading_engine.get_status()
                        await websocket.send_text(json.dumps({
                            "event": "status_update",
                            "data": status,
                            "timestamp": datetime.now().isoformat(),
                        }))
                except (json.JSONDecodeError, KeyError):
                    pass
                    
            except asyncio.TimeoutError:
                # Send periodic snapshot if engine is running
                if _paper_trading_engine and _paper_trading_engine.status == "RUNNING":
                    snapshots = _orchestrator._create_portfolio_snapshots()
                    await websocket.send_text(json.dumps({
                        "event": "portfolio_update",
                        "data": {
                            "total_value": snapshot.total_value,
                            "cash": snapshot.cash,
                            "pnl_total": snapshot.pnl_total,
                            "daily_pnl": snapshot.daily_pnl,
                            "return_pct": snapshot.return_pct,
                            "num_trades": snapshot.num_trades,
                        },
                        "timestamp": datetime.now().isoformat(),
                    }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in _websocket_clients:
            _websocket_clients.remove(websocket)
        logger.info(f"WebSocket clients remaining: {len(_websocket_clients)}")

@router.post("/enable-auto-trade")
async def enable_auto_trade():
    """Enable automatic trading execution for LSTM models."""
    global _orchestrator
    if _orchestrator is None:
        raise HTTPException(status_code=400, detail="Orchestrator not started")
    _orchestrator.auto_trade_enabled = True
    return {"status": "success", "message": "Auto trading enabled"}

@router.get("/lstm-logs")
async def get_lstm_logs(lines: int = 20):
    """Fetch the most recent lines from the current hour's LSTM log CSV."""
    import os
    import pytz
    log_dir = r"E:\PRO\JIMxNik\LSTMLogs"
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(pytz.utc).astimezone(ist)
    filename = f"lstm_logs_{now_ist.strftime('%Y%m%d_%H')}.csv"
    filepath = os.path.join(log_dir, filename)
    
    if not os.path.exists(filepath):
        return {"logs": ["No logs generated for the current hour yet."]}
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            
        header = all_lines[0] if len(all_lines) > 0 else ""
        data_lines = all_lines[1:] if len(all_lines) > 1 else []
        recent_lines = data_lines[-lines:] if len(data_lines) > 0 else []
        
        return {"logs": [header] + recent_lines}
    except Exception as e:
        logger.error(f"Failed to read LSTM Logs: {e}")
        return {"logs": [f"Error reading logs: {e}"]}
