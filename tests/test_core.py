"""
Unit tests for core components.
Run: pytest tests/test_core.py -v

Phase 1 enhanced test suite covering:
- Wavelet denoiser (4 tests)
- HMM regime detector (2 tests)
- Risk manager with new circuit breakers (10 tests)
- Feature engine with volume features (3 tests)
- Execution engine with order tracking (6 tests)
- GPU helpers (3 tests)
- Infrastructure health checks (2 tests)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# Helper: Generate sample gold-like data
# =============================================================================

def _make_sample_data(n=500):
    """Create sample gold-like OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    price = 1800 + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({
        "open": price + np.random.randn(n) * 2,
        "high": price + abs(np.random.randn(n) * 5),
        "low": price - abs(np.random.randn(n) * 5),
        "close": price,
        "volume": np.random.randint(10000, 100000, n),
    }, index=dates)




# =============================================================================
# HMM Regime Detector Tests
# =============================================================================

class TestRegimeDetector:
    """Tests for the HMM regime detector."""

    def test_train_and_predict(self):
        """Should train and produce regime predictions."""
        from src.models.hmm_regime import RegimeDetector
        model = RegimeDetector(n_regimes=3, n_iter=100)
        df = _make_sample_data()

        metrics = model.train(df)
        assert model.is_trained
        assert "regimes" in metrics

        states, confs = model.predict(df)
        assert len(states) > 0
        assert all(s in [0, 1, 2] for s in states)
        assert all(0 <= c <= 1 for c in confs)

    def test_get_current_regime(self):
        """Should return a named regime with confidence."""
        from src.models.hmm_regime import RegimeDetector
        model = RegimeDetector(n_regimes=3, n_iter=100)
        df = _make_sample_data()
        model.train(df)

        regime, conf = model.get_current_regime(df)
        assert regime in ["GROWTH", "NORMAL", "CRISIS"]
        assert 0 <= conf <= 1


# =============================================================================
# Risk Manager Tests (Enhanced)
# =============================================================================

class TestRiskManager:
    """Tests for the risk management system."""

    def setup_method(self):
        from src.risk.manager import RiskManager
        self.rm = RiskManager()

    def test_kelly_positive_edge(self):
        """Positive edge should give positive position size."""
        size = self.rm.calculate_kelly_size(
            win_prob=0.55, avg_win=100, avg_loss=90,
            portfolio_value=100000, regime="NORMAL"
        )
        assert size > 0

    def test_kelly_no_edge(self):
        """No edge should give zero position size."""
        size = self.rm.calculate_kelly_size(
            win_prob=0.40, avg_win=100, avg_loss=100,
            portfolio_value=100000, regime="NORMAL"
        )
        assert size == 0

    def test_kelly_crisis_reduces_size(self):
        """Crisis regime should produce smaller position than normal."""
        normal = self.rm.calculate_kelly_size(
            win_prob=0.55, avg_win=100, avg_loss=90,
            portfolio_value=100000, regime="NORMAL"
        )
        crisis = self.rm.calculate_kelly_size(
            win_prob=0.55, avg_win=100, avg_loss=90,
            portfolio_value=100000, regime="CRISIS"
        )
        assert crisis < normal

    def test_kelly_growth_increases_size(self):
        """Growth regime should produce larger position than normal."""
        # Use lower win_prob so Kelly doesn't hit the 5% max cap
        normal = self.rm.calculate_kelly_size(
            win_prob=0.52, avg_win=100, avg_loss=98,
            portfolio_value=100000, regime="NORMAL"
        )
        growth = self.rm.calculate_kelly_size(
            win_prob=0.52, avg_win=100, avg_loss=98,
            portfolio_value=100000, regime="GROWTH"
        )
        assert growth > normal

    def test_kelly_max_cap(self):
        """Position should never exceed max_position_pct."""
        size = self.rm.calculate_kelly_size(
            win_prob=0.95, avg_win=1000, avg_loss=10,
            portfolio_value=100000, regime="NORMAL"
        )
        assert size <= 100000 * self.rm.max_position_pct

    def test_circuit_breaker_daily_loss(self):
        """Should halt after hitting daily loss limit."""
        self.rm.risk_state.daily_pnl = -3500  # -3.5% on 100K
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert not can_trade
        assert "Daily loss" in reason

    def test_circuit_breaker_drawdown(self):
        """Should halt after hitting max drawdown."""
        self.rm.risk_state.current_drawdown = 0.11  # 11%
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert not can_trade
        assert "drawdown" in reason.lower()

    def test_circuit_breaker_latency(self):
        """Should halt when data latency exceeds threshold."""
        can_trade, reason = self.rm.check_circuit_breakers(
            100000, data_latency_ms=600
        )
        assert not can_trade
        assert "latency" in reason.lower()

    def test_circuit_breaker_min_confidence(self):
        """Should halt/skip trade when ensemble confidence is below the min threshold."""
        # When confidence is set and below the limit (default 0.65)
        can_trade, reason = self.rm.check_circuit_breakers(
            100000, ensemble_conf=0.60
        )
        assert not can_trade
        assert "LOW_CONFIDENCE" in reason

        # When confidence is above or equal to limit
        can_trade, reason = self.rm.check_circuit_breakers(
            100000, ensemble_conf=0.80
        )
        assert can_trade
        assert reason == "OK"

        # When confidence is 0 (signaling to ignore or no signal)
        can_trade, reason = self.rm.check_circuit_breakers(
            100000, ensemble_conf=0.0
        )
        assert can_trade
        assert reason == "OK"

    def test_circuit_breaker_model_disagreement(self):
        """Should return MIN_POSITION when models disagree."""
        can_trade, reason = self.rm.check_circuit_breakers(
            100000, model_disagreement=0.80
        )
        assert can_trade  # Not halted, just reduced
        assert reason == "MIN_POSITION"

    def test_daily_reset(self):
        """Should clear daily counters."""
        self.rm.risk_state.daily_pnl = -1000
        self.rm.risk_state.is_halted = True
        self.rm.reset_daily()
        assert self.rm.risk_state.daily_pnl == 0
        assert not self.rm.risk_state.is_halted

    def test_consecutive_loss_tracking(self):
        """Should track and reset consecutive losses."""
        self.rm.record_trade(-100)
        self.rm.record_trade(-50)
        self.rm.record_trade(-75)
        assert self.rm.risk_state.consecutive_losses == 3

        self.rm.record_trade(200)  # Win resets streak
        assert self.rm.risk_state.consecutive_losses == 0

    def test_rolling_stats(self):
        """Should compute rolling win rate and profit factor."""
        self.rm.record_trade(100)
        self.rm.record_trade(-50)
        self.rm.record_trade(80)
        self.rm.record_trade(-30)

        stats = self.rm.get_rolling_stats()
        assert stats["trade_count"] == 4
        assert stats["win_rate"] == 0.5
        assert stats["profit_factor"] > 1.0  # 180/80 = 2.25

    def test_regime_cooldown_blocks_trade(self):
        """Should temporarily halt trading immediately after a regime switch."""
        self.rm.allowed_regimes = ["NORMAL", "CRISIS"]
        # Initial state should not block (starts high)
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert can_trade
        assert reason == "OK"

        # Change regime from NORMAL to CRISIS (requires 3 consecutive bars to confirm)
        for _ in range(3):
            self.rm.calculate_kelly_size(0.55, 100, 90, 100000, regime="CRISIS")
        
        # Bars since switch should reset to 0
        assert self.rm.risk_state.bars_since_regime_switch == 0

        # Should now block trading
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert not can_trade
        assert "REGIME_COOLDOWN" in reason

        # Increment bars to 4 (under limit of 5)
        for _ in range(4):
            self.rm.calculate_kelly_size(0.55, 100, 90, 100000, regime="CRISIS")
        
        assert self.rm.risk_state.bars_since_regime_switch == 4
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert not can_trade

        # Increment to 5 (limit met)
        self.rm.calculate_kelly_size(0.55, 100, 90, 100000, regime="CRISIS")
        assert self.rm.risk_state.bars_since_regime_switch == 5
        
        # Should now allow trading again
        can_trade, reason = self.rm.check_circuit_breakers(100000)
        assert can_trade
        assert reason == "OK"


# =============================================================================
# Feature Engine Tests (Enhanced)
# =============================================================================

class TestFeatureEngine:
    """Tests for the feature engineering pipeline."""

    def test_generates_features(self):
        """Should produce a DataFrame with many more columns."""
        from src.features.engine import FeatureEngine
        engine = FeatureEngine()
        df = _make_sample_data(300)

        result = engine.generate_all(df)

        assert len(result.columns) > len(df.columns)
        assert len(result) > 0
        assert not result.isnull().any().any()

    def test_no_lookahead(self):
        """Feature values should not depend on future data."""
        from src.features.engine import FeatureEngine
        engine = FeatureEngine()
        df = _make_sample_data(500)

        full_features = engine.generate_all(df)
        partial_features = engine.generate_all(df.iloc[:400])

        # The 399th row (0-indexed) should be identical in both
        common_cols = set(full_features.columns) & set(partial_features.columns)
        last_partial_idx = partial_features.index[-1]

        if last_partial_idx in full_features.index:
            for col in list(common_cols)[:10]:  # Check first 10 features
                val_full = full_features.loc[last_partial_idx, col]
                val_partial = partial_features.loc[last_partial_idx, col]
                if not np.isnan(val_full) and not np.isnan(val_partial):
                    assert abs(val_full - val_partial) < 1e-10, f"Lookahead in {col}"

    def test_volume_features_generated(self):
        """Should generate OBV, VWAP, and volume ratio features."""
        from src.features.engine import FeatureEngine
        engine = FeatureEngine()
        df = _make_sample_data(300)

        result = engine.generate_all(df)
        feature_names = engine.get_feature_names(result)

        # Check volume-based features exist
        assert any("obv" in f for f in feature_names), "OBV feature missing"
        assert any("vwap" in f for f in feature_names), "VWAP feature missing"
        assert any("volume_ratio" in f for f in feature_names), "Volume ratio feature missing"

    def test_feature_count_above_80(self):
        """Enhanced engine should produce 80+ features."""
        from src.features.engine import FeatureEngine
        engine = FeatureEngine()
        df = _make_sample_data(300)

        result = engine.generate_all(df)
        feature_names = engine.get_feature_names(result)

        assert len(feature_names) >= 80, f"Only {len(feature_names)} features (expected 80+)"

    def test_feature_pruning_limits_feature_count(self):
        """Pruned engine should keep strictly less than 35 features including OHLCV and Golden Features."""
        from src.features.engine import FeatureEngine
        # Initialize with pruning turned on explicitly
        config = {
            "features": {
                "prune_features": True,
                "max_features_ceiling": 25
            }
        }
        engine = FeatureEngine(config)
        df = _make_sample_data(300)

        result = engine.generate_all(df)
        feature_names = engine.get_feature_names(result)

        # Should only contain OHLCV, returns, and existing Golden Features
        assert len(feature_names) <= 25
        assert "volatility_20" in feature_names or "macd" in feature_names


# =============================================================================
# Execution Engine Tests (NEW)
# =============================================================================

class TestExecutionEngine:
    """Tests for the execution engine."""

    def setup_method(self):
        from src.execution.engine import ExecutionEngine
        config = {
            "execution": {
                "broker": "paper",
                "max_slippage_bps": 1.0,
            }
        }
        self.engine = ExecutionEngine(config)
        self.engine.connect()

    def test_connect(self):
        """Should connect successfully in paper mode."""
        assert self.engine.is_connected
        assert self.engine.connection["provider"] == "paper"

    def test_submit_order_generates_id(self):
        """Submitted order should have a unique order ID."""
        result = self.engine.submit_order(
            symbol="XAUUSD", quantity=1.0, side="buy",
            order_type="limit", limit_price=2000.0
        )
        assert "order_id" in result
        assert len(result["order_id"]) > 0
        assert result["status"] == "filled"

    def test_order_history_tracking(self):
        """Should track all submitted orders in history."""
        self.engine.submit_order("XAUUSD", 1.0, "buy", limit_price=2000.0)
        self.engine.submit_order("XAUUSD", 0.5, "sell", limit_price=2010.0)

        history = self.engine.get_order_history()
        assert len(history) == 2
        assert history[0]["side"] == "buy"
        assert history[1]["side"] == "sell"

    def test_latency_stats(self):
        """Should compute latency percentiles after orders."""
        for _ in range(10):
            self.engine.submit_order("XAUUSD", 1.0, "buy", limit_price=2000.0)

        stats = self.engine.get_latency_stats()
        assert stats["count"] == 10
        assert stats["p50"] >= 0
        assert stats["p95"] >= stats["p50"]
        assert stats["p99"] >= stats["p95"]

    def test_cancel_order(self):
        """Should cancel an existing order."""
        result = self.engine.submit_order("XAUUSD", 1.0, "buy", limit_price=2000.0)
        cancel_result = self.engine.cancel_order(result["order_id"])
        assert cancel_result["status"] == "cancelled"

    def test_health_check(self):
        """Health check should report connected status and order count."""
        self.engine.submit_order("XAUUSD", 1.0, "buy", limit_price=2000.0)
        health = self.engine.health_check()

        assert health["connected"] is True
        assert health["broker"] == "paper"
        assert health["total_orders"] == 1


# =============================================================================
# GPU Helpers Tests (NEW)
# =============================================================================

class TestGPUHelpers:
    """Tests for GPU utility functions."""

    def test_detect_gpu_returns_dict(self):
        """detect_gpu should return a well-formed dict."""
        from src.utils.gpu import detect_gpu
        info = detect_gpu()

        assert isinstance(info, dict)
        assert "gpu_available" in info
        assert "device_count" in info
        assert isinstance(info["gpu_available"], bool)

    def test_get_dataframe_engine(self):
        """Should return either cuDF or Pandas module."""
        from src.utils.gpu import get_dataframe_engine
        engine = get_dataframe_engine()

        # Should have DataFrame class (both Pandas and cuDF have it)
        assert hasattr(engine, "DataFrame")

    def test_get_array_engine(self):
        """Should return either CuPy or NumPy module."""
        from src.utils.gpu import get_array_engine
        engine = get_array_engine()

        # Should have array function (both NumPy and CuPy have it)
        assert hasattr(engine, "array")


# =============================================================================
# Infrastructure Health Check Tests (NEW)
# =============================================================================

class TestInfrastructure:
    """Tests for infrastructure health check utilities."""

    def test_check_stack_returns_all_services(self):
        """Should return checks for all 7 infrastructure components."""
        from src.utils.infra import check_stack
        checks = check_stack()

        expected_services = {"docker-compose", "questdb", "redis", "minio", "mlflow", "prometheus", "grafana"}
        assert set(checks.keys()) == expected_services

        for service, result in checks.items():
            assert "required" in result
            assert "available" in result
            assert "details" in result

    def test_health_summary(self):
        """Should compute health summary from checks."""
        from src.utils.infra import get_health_summary
        summary = get_health_summary()

        assert "total_services" in summary
        assert "available" in summary
        assert "health_pct" in summary
        assert isinstance(summary["all_healthy"], bool)
        assert summary["total_services"] >= 7
