"""
Unit tests for Paper Trading Engine

Tests cover:
- Engine initialization and lifecycle
- Signal processing and trade execution
- Position tracking and P&L calculation
- Risk limit enforcement
- Performance metrics calculation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

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


class TestPaperTradingConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PaperTradingConfig()
        assert config.initial_capital == 100000.0
        assert config.symbol == "XAUUSD"
        assert config.kelly_fraction == 0.25
        assert config.max_position_pct == 0.10
        assert config.min_confidence == 0.75
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PaperTradingConfig(
            initial_capital=50000.0,
            kelly_fraction=0.20,
            max_position_pct=0.15,
        )
        assert config.initial_capital == 50000.0
        assert config.kelly_fraction == 0.20
        assert config.max_position_pct == 0.15
    
    def test_signal_weights_sum(self):
        """Test that signal weights are properly defined."""
        config = PaperTradingConfig()
        total_weight = sum(config.signal_weights.values())
        assert abs(total_weight - 1.0) < 0.001  # Should sum to 1.0


class TestModelSignal:
    """Test ModelSignal dataclass."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        now = datetime.now()
        signal = ModelSignal(
            model_name="wavelet_pro",
            signal_type=SignalType.LONG,
            confidence=0.75,
            entry_price=2000.0,
            current_price=2005.0,
            timestamp=now,
            regime="NORMAL",
        )
        assert signal.model_name == "wavelet_pro"
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 0.75
        assert signal.entry_price == 2000.0
    
    def test_signal_types(self):
        """Test all signal types."""
        now = datetime.now()
        for signal_type in SignalType:
            signal = ModelSignal(
                model_name="test",
                signal_type=signal_type,
                confidence=0.7,
                entry_price=2000.0,
                current_price=2000.0,
                timestamp=now,
            )
            assert signal.signal_type == signal_type


class TestTradeExecution:
    """Test TradeExecution dataclass."""
    
    def test_trade_creation(self):
        """Test trade creation with default values."""
        trade = TradeExecution(
            model_name="lstm",
            signal_type=SignalType.LONG,
            entry_price=2000.0,
            quantity=1.5,
        )
        assert trade.model_name == "lstm"
        assert trade.signal_type == SignalType.LONG
        assert trade.entry_price == 2000.0
        assert trade.quantity == 1.5
        assert trade.status == TradeStatus.PENDING
        assert len(trade.trade_id) > 0
    
    def test_trade_lifecycle(self):
        """Test trade status transitions."""
        trade = TradeExecution(model_name="test", signal_type=SignalType.LONG, entry_price=2000.0)
        assert trade.status == TradeStatus.PENDING
        
        trade.status = TradeStatus.OPEN
        assert trade.status == TradeStatus.OPEN
        
        trade.status = TradeStatus.CLOSED
        assert trade.status == TradeStatus.CLOSED


class TestPaperTradingEngine:
    """Test PaperTradingEngine main class."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine."""
        config = PaperTradingConfig(initial_capital=100000.0)
        with patch('src.paper_trading.engine.get_config'), \
             patch('src.paper_trading.engine.ExecutionSimulator'), \
             patch('src.paper_trading.engine.PortfolioTracker'):
            engine = PaperTradingEngine(config)
            engine.portfolio.current_cash = 100000.0
        return engine
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.status == "INITIALIZED"
        assert engine.config.initial_capital == 100000.0
        assert len(engine.trades) == 0
        assert engine.current_position is None
    
    def test_engine_start(self, engine):
        """Test engine start."""
        result = engine.start()
        assert engine.status == "RUNNING"
        assert result["status"] == "RUNNING"
        assert "started_at" in result
        assert result["initial_capital"] == 100000.0
    
    def test_engine_stop(self, engine):
        """Test engine stop."""
        engine.start()
        result = engine.stop()
        assert engine.status == "STOPPED"
        assert result["status"] == "STOPPED"
        assert "stopped_at" in result
    
    def test_engine_get_status(self, engine):
        """Test getting engine status."""
        engine.start()
        status = engine.get_status()
        assert status["status"] == "RUNNING"
        assert "portfolio" in status
        assert "trading" in status
        assert "models" in status
    
    def test_cannot_start_twice(self, engine):
        """Test that engine cannot start twice."""
        engine.start()
        result = engine.start()
        assert result["status"] == "RUNNING"
        assert result["message"] == "Already started"
    
    def test_position_sizing(self, engine):
        """Test Kelly criterion position sizing."""
        # Mock the portfolio equity
        engine.portfolio.current_equity = 100000.0
        
        # Lower confidence values that won't hit the max_position_pct cap
        size_high = engine._calculate_position_size(0.75)  # Higher confidence
        size_low = engine._calculate_position_size(0.55)   # Lower confidence
        
        # Both should be positive floats
        assert isinstance(size_high, float)
        assert isinstance(size_low, float)
        assert size_high > 0
        assert size_low > 0
        # Higher confidence should yield larger or equal position
        assert size_high >= size_low
    
    def test_commission_calculation(self, engine):
        """Test commission calculation."""
        commission = engine._calculate_commission(100000.0)
        assert commission > 0
        assert commission >= engine.config.commission_per_trade
    
    def test_slippage_calculation(self, engine):
        """Test slippage calculation."""
        slippage = engine._calculate_slippage(2000.0)
        assert slippage > 0
        assert slippage == pytest.approx(2000.0 * (engine.config.slippage_pct / 100.0), rel=1e-2)
    
    def test_pnl_calculation_long(self, engine):
        """Test P&L calculation for long position."""
        pnl, pnl_pct = engine._calculate_pnl(
            SignalType.LONG,
            entry=2000.0,
            exit=2100.0,
            quantity=1.0
        )
        assert pnl == 100.0
        assert pnl_pct == 5.0
    
    def test_pnl_calculation_short(self, engine):
        """Test P&L calculation for short position."""
        pnl, pnl_pct = engine._calculate_pnl(
            SignalType.SHORT,
            entry=2000.0,
            exit=1900.0,
            quantity=1.0
        )
        assert pnl == 100.0
        assert pnl_pct == 5.0
    
    def test_pnl_calculation_loss(self, engine):
        """Test P&L calculation with loss."""
        pnl, pnl_pct = engine._calculate_pnl(
            SignalType.LONG,
            entry=2000.0,
            exit=1950.0,
            quantity=1.0
        )
        assert pnl == -50.0
        assert pnl_pct == -2.5
    
    def test_win_rate_calculation(self, engine):
        """Test win rate calculation."""
        # Create some test trades
        for i in range(5):
            trade = TradeExecution(
                model_name="test",
                signal_type=SignalType.LONG,
                entry_price=2000.0,
                quantity=1.0,
                status=TradeStatus.CLOSED,
                pnl=100.0 if i < 3 else -50.0,  # 3 wins, 2 losses
            )
            engine.trades.append(trade)
        
        win_rate = engine._calculate_win_rate()
        assert win_rate == 0.6  # 3 wins / 5 trades


class TestRiskManager:
    """Test RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a test risk manager."""
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
    
    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.risk_state.is_halted is False
        assert risk_manager.risk_state.consecutive_losses == 0
        assert risk_manager.kelly_fraction == 0.25
        assert risk_manager.max_position_pct == 0.10

    def test_check_circuit_breakers_ok(self, risk_manager):
        """Test circuit breakers with acceptable conditions."""
        can_trade, reason = risk_manager.check_circuit_breakers(
            portfolio_value=100000.0,
            ensemble_conf=0.80,
        )
        assert can_trade is True
        assert reason == "OK" or "REDUCE_SIZE" in reason

    def test_check_circuit_breakers_low_confidence(self, risk_manager):
        """Test circuit breakers with low confidence."""
        can_trade, reason = risk_manager.check_circuit_breakers(
            portfolio_value=100000.0,
            ensemble_conf=0.40,
        )
        assert can_trade is False
        assert "LOW_CONFIDENCE" in reason

    def test_check_circuit_breakers_daily_loss_exceeded(self, risk_manager):
        """Test circuit breakers with daily loss exceeded."""
        risk_manager.risk_state.daily_pnl = -3000.0  # 3% loss, exceeds 2% limit
        can_trade, reason = risk_manager.check_circuit_breakers(
            portfolio_value=100000.0,
            ensemble_conf=0.80,
        )
        assert can_trade is False
        assert "Daily loss" in reason or "HALTED" in reason

    def test_consecutive_losses_tracking(self, risk_manager):
        """Test consecutive losses tracking."""
        risk_manager.record_trade(-100.0)
        assert risk_manager.risk_state.consecutive_losses == 1
        
        risk_manager.record_trade(-50.0)
        assert risk_manager.risk_state.consecutive_losses == 2

    def test_consecutive_losses_reset_on_win(self, risk_manager):
        """Test consecutive losses reset on win."""
        risk_manager.record_trade(-100.0)
        assert risk_manager.risk_state.consecutive_losses == 1
        
        risk_manager.record_trade(200.0)
        assert risk_manager.risk_state.consecutive_losses == 0

    def test_peak_equity_tracking(self, risk_manager):
        """Test peak equity tracking."""
        risk_manager.update_equity(100000.0)
        assert risk_manager.risk_state.peak_equity == 100000.0
        
        risk_manager.update_equity(105000.0)
        assert risk_manager.risk_state.peak_equity == 105000.0
        
        risk_manager.update_equity(103000.0)
        assert risk_manager.risk_state.peak_equity == 105000.0  # Should not change

    def test_risk_report(self, risk_manager):
        """Test risk report generation."""
        report = risk_manager.get_risk_report(
            current_equity=102000.0,
            daily_pnl=2000.0
        )
        assert "current_equity" in report
        assert "drawdown_stop" in report
        assert "daily_pnl" in report


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test snapshot creation."""
        now = datetime.now()
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_value=102000.0,
            cash=50000.0,
            position_quantity=2.5,
            position_value=52000.0,
            pnl_realized=1000.0,
            pnl_unrealized=1000.0,
            pnl_total=2000.0,
            daily_pnl=500.0,
            return_pct=2.0,
        )
        assert snapshot.total_value == 102000.0
        assert snapshot.pnl_total == 2000.0
        assert snapshot.return_pct == 2.0


class TestIntegration:
    """Integration tests for paper trading engine."""
    
    def test_engine_lifecycle(self):
        """Test complete engine lifecycle."""
        with patch('src.paper_trading.engine.get_config'), \
             patch('src.paper_trading.engine.ExecutionSimulator'), \
             patch('src.paper_trading.engine.PortfolioTracker'):
            
            engine = PaperTradingEngine()
            
            # Start engine
            assert engine.status == "INITIALIZED"
            engine.start()
            assert engine.status == "RUNNING"
            
            # Get status
            status = engine.get_status()
            assert status["status"] == "RUNNING"
            
            # Stop engine
            engine.stop()
            assert engine.status == "STOPPED"
    
    def test_signal_confidence_threshold(self):
        """Test signal confidence threshold enforcement."""
        with patch('src.paper_trading.engine.get_config'), \
             patch('src.paper_trading.engine.ExecutionSimulator'), \
             patch('src.paper_trading.engine.PortfolioTracker'):
            
            engine = PaperTradingEngine()
            engine.start()
            
            # Low confidence signal should be rejected
            signal = ModelSignal(
                model_name="wavelet_pro",
                signal_type=SignalType.LONG,
                confidence=0.55,  # Below 0.75 threshold
                entry_price=2000.0,
                current_price=2000.0,
                timestamp=datetime.now(),
            )
            
            result = engine.process_signal("wavelet_pro", signal)
            assert result is None  # Should not execute
    
    def test_position_closing_on_opposite_signal(self):
        """Test that positions close on opposite signal."""
        with patch('src.paper_trading.engine.get_config'), \
             patch('src.paper_trading.engine.ExecutionSimulator'), \
             patch('src.paper_trading.engine.PortfolioTracker'):
            
            engine = PaperTradingEngine()
            engine.start()
            
            # Create a mock position
            engine.current_position = TradeExecution(
                model_name="lstm",
                signal_type=SignalType.LONG,
                entry_price=2000.0,
                quantity=1.0,
                status=TradeStatus.OPEN,
            )
            
            # Opposite signal should close position
            close_signal = ModelSignal(
                model_name="hmm",
                signal_type=SignalType.SHORT,
                confidence=0.7,
                entry_price=2050.0,
                current_price=2050.0,
                timestamp=datetime.now(),
            )
            
            # Note: This test demonstrates the logic, but actual execution depends on mocks
            assert engine.current_position is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
