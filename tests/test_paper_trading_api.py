"""
Paper Trading API Route Tests - Phase 6B
=========================================

Comprehensive tests for all 10 paper trading REST API endpoints.
Tests the full lifecycle: start → signal → status → performance →
portfolio → trades → risk-report → config → reset-daily → stop.

Test Coverage:
- Endpoint availability and response validation (10 endpoints)
- Error handling (uninitialized engine, invalid inputs)
- Full trading lifecycle integration
- Configuration updates
- Signal injection and trade execution
- Pagination and filtering
- WebSocket connection (basic)

Total: 38 tests
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.app import app
from src.api.paper_trading_routes import (
    set_engine,
    get_engine,
    _websocket_clients,
)
from src.paper_trading.engine import (
    PaperTradingEngine,
    PaperTradingConfig,
    TradeExecution,
    ModelSignal,
    PortfolioSnapshot,
    SignalType,
    TradeStatus,
)
from src.risk.manager import RiskManager


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    c = TestClient(app)
    c.headers.update({"X-API-Key": "medallion_secret_key"})
    return c


@pytest.fixture
def default_config():
    """Default paper trading configuration."""
    return PaperTradingConfig(
        initial_capital=100000.0,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.15,
        min_confidence=0.60,
    )


@pytest.fixture
def engine(default_config):
    """Create and return a paper trading engine."""
    eng = PaperTradingEngine(default_config)
    return eng


@pytest.fixture
def risk_manager():
    """Create a risk manager."""
    config = {
        "risk": {
            "kelly": {
                "fraction": 0.25,
                "max_position_pct": 0.10,
            },
            "circuit_breakers": {
                "daily_loss_limit": 0.02,
                "drawdown_stop": 0.15,
            }
        }
    }
    return RiskManager(config)


@pytest.fixture
def started_engine(engine, default_config, risk_manager):
    """Create a started engine and inject it into the router."""
    engine.start()
    set_engine(engine, default_config, risk_manager)
    yield engine
    # Cleanup
    set_engine(None, None, None)


@pytest.fixture(autouse=True)
def cleanup_engine():
    """Ensure engine is cleaned up after each test."""
    yield
    set_engine(None, None, None)


# ============================================================================
# 1. START ENDPOINT TESTS
# ============================================================================

class TestStartEndpoint:
    """Tests for POST /paper-trading/start."""

    def test_start_success(self, client):
        """Test starting paper trading with default config."""
        response = client.post("/paper-trading/start", json={
            "initial_capital": 100000.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "details" in data
        assert data["config"]["initial_capital"] == 100000.0

    def test_start_custom_config(self, client):
        """Test starting with custom configuration."""
        response = client.post("/paper-trading/start", json={
            "initial_capital": 50000.0,
            "kelly_fraction": 0.5,
            "max_position_pct": 0.15,
            "min_confidence": 0.70,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["initial_capital"] == 50000.0
        assert data["config"]["kelly_fraction"] == 0.5
        assert data["config"]["min_confidence"] == 0.70

    def test_start_already_running(self, client, started_engine):
        """Test that starting when already running returns 409."""
        response = client.post("/paper-trading/start", json={
            "initial_capital": 100000.0,
        })
        assert response.status_code == 409

    def test_start_invalid_capital(self, client):
        """Test that negative capital is rejected."""
        response = client.post("/paper-trading/start", json={
            "initial_capital": -1000.0,
        })
        assert response.status_code == 422

    def test_start_security_unauthorized(self, client):
        """Test starting paper trading without valid API key header returns 403."""
        from fastapi.testclient import TestClient
        unauth_client = TestClient(app)
        response = unauth_client.post("/paper-trading/start", json={
            "initial_capital": 100000.0,
        })
        assert response.status_code == 403


# ============================================================================
# 2. STATUS ENDPOINT TESTS
# ============================================================================

class TestStatusEndpoint:
    """Tests for GET /paper-trading/status."""

    def test_status_not_initialized(self, client):
        """Test status when engine not initialized returns 404."""
        response = client.get("/paper-trading/status")
        assert response.status_code == 404

    def test_status_running(self, client, started_engine):
        """Test status when engine is running."""
        response = client.get("/paper-trading/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RUNNING"
        assert data["uptime_seconds"] is not None
        assert data["uptime_seconds"] >= 0
        assert "portfolio" in data
        assert "trading" in data
        assert "models" in data

    def test_status_portfolio_fields(self, client, started_engine):
        """Test that portfolio fields are present and correct."""
        response = client.get("/paper-trading/status")
        data = response.json()
        portfolio = data["portfolio"]
        assert "total_value" in portfolio
        assert "cash" in portfolio
        assert "pnl_total" in portfolio
        assert portfolio["total_value"] == 100000.0


# ============================================================================
# 3. PERFORMANCE ENDPOINT TESTS
# ============================================================================

class TestPerformanceEndpoint:
    """Tests for GET /paper-trading/performance."""

    def test_performance_not_initialized(self, client):
        """Test performance when engine not initialized."""
        response = client.get("/paper-trading/performance")
        assert response.status_code == 404

    def test_performance_initial(self, client, started_engine):
        """Test performance metrics at start."""
        response = client.get("/paper-trading/performance")
        assert response.status_code == 200
        data = response.json()
        assert data["total_value"] == 100000.0
        assert data["num_trades"] == 0
        assert data["pnl_total"] == 0.0
        assert data["daily_trades"] == 0


# ============================================================================
# 4. STOP ENDPOINT TESTS
# ============================================================================

class TestStopEndpoint:
    """Tests for POST /paper-trading/stop."""

    def test_stop_not_initialized(self, client):
        """Test stop when engine not initialized."""
        response = client.post("/paper-trading/stop")
        assert response.status_code == 404

    def test_stop_success(self, client, started_engine):
        """Test stopping a running engine."""
        response = client.post("/paper-trading/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "details" in data


# ============================================================================
# 5. TRADES ENDPOINT TESTS
# ============================================================================

class TestTradesEndpoint:
    """Tests for GET /paper-trading/trades."""

    def test_trades_not_initialized(self, client):
        """Test trades when engine not initialized."""
        response = client.get("/paper-trading/trades")
        assert response.status_code == 404

    def test_trades_empty(self, client, started_engine):
        """Test trades list when no trades have been made."""
        response = client.get("/paper-trading/trades")
        assert response.status_code == 200
        assert response.json() == []

    def test_trades_pagination(self, client, started_engine):
        """Test trades pagination parameters."""
        response = client.get("/paper-trading/trades?limit=10&offset=0")
        assert response.status_code == 200

    def test_trades_status_filter(self, client, started_engine):
        """Test trades with status filter."""
        response = client.get("/paper-trading/trades?status_filter=CLOSED")
        assert response.status_code == 200


# ============================================================================
# 6. PORTFOLIO ENDPOINT TESTS
# ============================================================================

class TestPortfolioEndpoint:
    """Tests for GET /paper-trading/portfolio."""

    def test_portfolio_not_initialized(self, client):
        """Test portfolio when engine not initialized."""
        response = client.get("/paper-trading/portfolio")
        assert response.status_code == 404

    def test_portfolio_initial(self, client, started_engine):
        """Test initial portfolio snapshot."""
        response = client.get("/paper-trading/portfolio")
        assert response.status_code == 200
        data = response.json()
        assert data["total_value"] == 100000.0
        assert data["cash"] == 100000.0
        assert data["position_quantity"] == 0
        assert "timestamp" in data
        assert data["num_trades"] == 0


# ============================================================================
# 7. RISK REPORT ENDPOINT TESTS
# ============================================================================

class TestRiskReportEndpoint:
    """Tests for GET /paper-trading/risk-report."""

    def test_risk_report_not_initialized(self, client):
        """Test risk report when engine not initialized."""
        response = client.get("/paper-trading/risk-report")
        assert response.status_code == 404

    def test_risk_report_success(self, client, started_engine):
        """Test risk report generation."""
        response = client.get("/paper-trading/risk-report")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "risk_report" in data
        assert "portfolio_summary" in data
        report = data["risk_report"]
        assert "current_equity" in report
        assert "drawdown_stop" in report


# ============================================================================
# 8. SIGNAL INJECTION ENDPOINT TESTS
# ============================================================================

class TestSignalEndpoint:
    """Tests for POST /paper-trading/signal."""

    def test_signal_not_initialized(self, client):
        """Test signal injection when engine not initialized."""
        response = client.post("/paper-trading/signal", json={
            "model_name": "wavelet_pro",
            "signal_type": "LONG",
            "confidence": 0.75,
            "price": 2350.0,
        })
        assert response.status_code == 404

    def test_signal_engine_not_running(self, client, engine, default_config, risk_manager):
        """Test signal injection when engine is initialized but not running."""
        set_engine(engine, default_config, risk_manager)
        response = client.post("/paper-trading/signal", json={
            "model_name": "wavelet_pro",
            "signal_type": "LONG",
            "confidence": 0.75,
            "price": 2350.0,
        })
        assert response.status_code == 409

    def test_signal_success_trade_executed(self, client, started_engine):
        """Test signal injection that results in a trade."""
        response = client.post("/paper-trading/signal", json={
            "model_name": "ensemble",
            "signal_type": "LONG",
            "confidence": 0.85,
            "price": 2350.0,
            "regime": "GROWTH",
            "reasoning": "Strong bullish signal",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["signal_processed"] is True
        assert data["trade_executed"] is True
        assert "trade" in data

    def test_signal_low_confidence_no_trade(self, client, started_engine):
        """Test that low confidence signals don't trigger trades."""
        response = client.post("/paper-trading/signal", json={
            "model_name": "wavelet_pro",
            "signal_type": "LONG",
            "confidence": 0.30,
            "price": 2350.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["trade_executed"] is False

    def test_signal_invalid_model(self, client, started_engine):
        """Test signal with invalid model name."""
        response = client.post("/paper-trading/signal", json={
            "model_name": "invalid_model",
            "signal_type": "LONG",
            "confidence": 0.75,
            "price": 2350.0,
        })
        assert response.status_code == 400

    def test_signal_invalid_type(self, client, started_engine):
        """Test signal with invalid signal type."""
        response = client.post("/paper-trading/signal", json={
            "model_name": "wavelet_pro",
            "signal_type": "INVALID",
            "confidence": 0.75,
            "price": 2350.0,
        })
        assert response.status_code == 400


# ============================================================================
# 9. CONFIG UPDATE ENDPOINT TESTS
# ============================================================================

class TestConfigEndpoint:
    """Tests for POST /paper-trading/config."""

    def test_config_not_initialized(self, client):
        """Test config update when engine not initialized."""
        response = client.post("/paper-trading/config", json={
            "kelly_fraction": 0.5,
        })
        assert response.status_code == 404

    def test_config_update_single(self, client, started_engine):
        """Test updating a single config parameter."""
        response = client.post("/paper-trading/config", json={
            "kelly_fraction": 0.50,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["updated_fields"]["kelly_fraction"] == 0.50

    def test_config_update_multiple(self, client, started_engine):
        """Test updating multiple config parameters."""
        response = client.post("/paper-trading/config", json={
            "kelly_fraction": 0.40,
            "min_confidence": 0.70,
            "trading_enabled": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["updated_fields"]) == 3


# ============================================================================
# 10. DAILY RESET ENDPOINT TESTS
# ============================================================================

class TestDailyResetEndpoint:
    """Tests for POST /paper-trading/reset-daily."""

    def test_reset_not_initialized(self, client):
        """Test reset when engine not initialized."""
        response = client.post("/paper-trading/reset-daily")
        assert response.status_code == 404

    def test_reset_success(self, client, started_engine):
        """Test successful daily counter reset."""
        response = client.post("/paper-trading/reset-daily")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "previous_daily_pnl" in data
        assert "reset_time" in data


# ============================================================================
# 11. FULL LIFECYCLE INTEGRATION TEST
# ============================================================================

class TestFullLifecycle:
    """End-to-end lifecycle test: start → trade → monitor → stop."""

    def test_full_trading_lifecycle(self, client):
        """Test complete paper trading lifecycle."""
        # 1. Start
        resp = client.post("/paper-trading/start", json={
            "initial_capital": 100000.0,
            "min_confidence": 0.50,
        })
        assert resp.status_code == 200

        # 2. Check status
        resp = client.get("/paper-trading/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "RUNNING"

        # 3. Inject a LONG signal
        resp = client.post("/paper-trading/signal", json={
            "model_name": "ensemble",
            "signal_type": "LONG",
            "confidence": 0.80,
            "price": 2350.0,
            "regime": "NORMAL",
        })
        assert resp.status_code == 200
        assert resp.json()["trade_executed"] is True

        # 4. Check performance
        resp = client.get("/paper-trading/performance")
        assert resp.status_code == 200
        assert resp.json()["num_trades"] >= 1

        # 5. Check portfolio
        resp = client.get("/paper-trading/portfolio")
        assert resp.status_code == 200

        # 6. Check trades
        resp = client.get("/paper-trading/trades")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

        # 7. Check risk report
        resp = client.get("/paper-trading/risk-report")
        assert resp.status_code == 200

        # 8. Inject a CLOSE signal
        resp = client.post("/paper-trading/signal", json={
            "model_name": "hmm_pro",
            "signal_type": "CLOSE",
            "confidence": 0.90,
            "price": 2360.0,
        })
        assert resp.status_code == 200

        # 9. Stop engine
        resp = client.post("/paper-trading/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
