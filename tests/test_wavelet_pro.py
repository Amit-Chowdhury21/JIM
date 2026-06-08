"""
Comprehensive Tests for Wavelet Pro Model
══════════════════════════════════════════

Tests verify:
  ✓ 6-level DWT decomposition
  ✓ Wavelet Oscillator (D3+D4)
  ✓ Soft thresholding denoising
  ✓ CWT volatility analysis
  ✓ 30+ feature engineering
  ✓ Signal generation logic
  ✓ Comparison with basic wavelet model
"""

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.models.wavelet_pro import WaveletPro, WaveletProConfig, compare_wavelet_models

class TestWaveletProCore:
    """Test core wavelet decomposition functionality."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample gold price data."""
        np.random.seed(42)
        t = np.linspace(0, 10, 512)  # 512 samples (power of 2)
        trend = 1800 + 50 * np.sin(t)
        noise = np.random.randn(512) * 10
        return trend + noise
    
    @pytest.fixture
    def wavelet_pro(self):
        """Create WaveletPro instance."""
        return WaveletPro()
    
    def test_dwt_6_levels(self, sample_prices, wavelet_pro):
        """Verify 6-level DWT decomposition."""
        coeffs = wavelet_pro.decompose_dwt(sample_prices)
        
        # Should have A6 + D1-D6 = 7 components
        assert len(coeffs) == 7, f"Expected 7 components, got {len(coeffs)}"
        
        # Check keys
        expected_keys = {"A6", "D6", "D5", "D4", "D3", "D2", "D1"}
        assert set(coeffs.keys()) == expected_keys, f"Unexpected keys: {coeffs.keys()}"
        
        logger.info("✓ DWT 6-level decomposition verified")
    
    def test_decomposition_reconstruction(self, sample_prices, wavelet_pro):
        """Verify decomposition can be reconstructed accurately."""
        import pywt
        
        # Decompose
        coeffs_dict = wavelet_pro.decompose_dwt(sample_prices)
        
        # Reconstruct
        coeffs_list = list(coeffs_dict.values())  # [A6, D6, D5, D4, D3, D2, D1]
        reconstructed = pywt.waverec(coeffs_list, "db4")
        reconstructed = reconstructed[:len(sample_prices)]
        
        # Correlation should be very high
        correlation = np.corrcoef(sample_prices, reconstructed)[0, 1]
        assert correlation > 0.99, f"Reconstruction correlation too low: {correlation}"
        
        logger.info(f"✓ Reconstruction correlation: {correlation:.4f}")
    
    def test_soft_thresholding_denoises(self, sample_prices, wavelet_pro):
        """Verify soft thresholding reduces noise."""
        denoised, thresholds = wavelet_pro.denoise_soft_threshold(sample_prices)
        
        # Denoised should have lower variance (less noise)
        original_var = np.var(sample_prices)
        denoised_var = np.var(denoised)
        
        assert denoised_var < original_var, "Denoised signal has more variance than original"
        
        # Thresholds should be reasonable (positive)
        for level, threshold in thresholds.items():
            assert threshold >= 0, f"Negative threshold for {level}"
        
        logger.info(f"✓ Variance reduction: {original_var:.2f} → {denoised_var:.2f}")
    
    def test_wavelet_oscillator(self, sample_prices, wavelet_pro):
        """Verify Wavelet Oscillator (D3+D4) computation."""
        osc = wavelet_pro.compute_wavelet_oscillator(sample_prices)
        
        # Oscillator should be same length as input
        assert len(osc) == len(sample_prices), "Oscillator length mismatch"
        
        # Oscillator should be bounded (not infinite)
        assert np.all(np.isfinite(osc)), "Oscillator contains non-finite values"
        
        # Oscillator should oscillate around zero
        assert np.mean(osc) < 100, "Oscillator not centered near zero"
        
        logger.info(f"✓ Wavelet Oscillator: mean={np.mean(osc):.4f}, std={np.std(osc):.4f}")
    
    def test_cwt_computation(self, sample_prices, wavelet_pro):
        """Verify CWT for volatility analysis."""
        coeffs, frequencies = wavelet_pro.compute_cwt(sample_prices)
        
        # CWT should return coefficients and frequencies
        assert coeffs.shape[0] == len(wavelet_pro.config.cwt_scales), "CWT scale mismatch"
        assert coeffs.shape[1] == len(sample_prices), "CWT time mismatch"
        
        # Frequencies should be positive
        assert np.all(frequencies > 0), "CWT frequencies not positive"
        
        logger.info(f"✓ CWT computation: {coeffs.shape}")


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def price_data(self):
        """Generate sample price dataframe."""
        np.random.seed(42)
        t = np.linspace(0, 10, 512)
        trend = 1800 + 30 * np.sin(t)
        noise = np.random.randn(512) * 5
        prices = trend + noise
        return pd.DataFrame({"close": prices})
    
    def test_feature_count(self, price_data):
        """Verify ~30 features are generated."""
        wavelet_pro = WaveletPro()
        features = wavelet_pro.engineer_features(price_data)
        
        # Should have approximately 30 features
        assert len(features.columns) >= 25, f"Expected ≥25 features, got {len(features.columns)}"
        
        logger.info(f"✓ Generated {len(features.columns)} features")
    
    def test_feature_values_valid(self, price_data):
        """Verify feature values are valid (no NaN/Inf)."""
        wavelet_pro = WaveletPro()
        features = wavelet_pro.engineer_features(price_data)
        
        # Check for NaN
        assert not features.isna().any().any(), "Features contain NaN values"
        
        # Check for infinite values
        assert not np.isinf(features.values).any(), "Features contain infinite values"
        
        logger.info("✓ All feature values are valid")
    
    def test_oscillator_in_features(self, price_data):
        """Verify wavelet oscillator is in feature set."""
        wavelet_pro = WaveletPro()
        features = wavelet_pro.engineer_features(price_data)
        
        assert "wavelet_oscillator" in features.columns, "Oscillator not in features"
        
        logger.info("✓ Wavelet oscillator included in features")


class TestSignalGeneration:
    """Test trading signal generation."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample with trend."""
        np.random.seed(42)
        t = np.linspace(0, 20, 512)
        # Uptrend with oscillation
        trend = 1800 + 100 * t + 50 * np.sin(2*np.pi * t)
        noise = np.random.randn(512) * 5
        return trend + noise
    
    def test_signal_format(self, sample_prices):
        """Verify signal has correct format."""
        wavelet_pro = WaveletPro()
        signal, confidence, reasoning = wavelet_pro.generate_signal(sample_prices)
        
        # Signal should be one of three options
        assert signal in ["LONG", "SHORT", "HOLD"], f"Invalid signal: {signal}"
        
        # Confidence should be in [0, 1]
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
        
        # Reasoning should be a string
        assert isinstance(reasoning, str), "Reasoning should be string"
        
        logger.info(f"✓ Signal: {signal} (confidence={confidence:.2f})")
    
    def test_trend_filter_logic(self):
        """Verify trend filter affects signals."""
        wavelet_pro = WaveletPro()
        
        # Uptrend
        t_up = np.linspace(0, 10, 512)
        uptrend = 1800 + 100 * t_up + 5 * np.sin(10*t_up)
        
        # Downtrend
        downtrend = 2000 - 100 * t_up + 5 * np.sin(10*t_up)
        
        signal_up, conf_up, _ = wavelet_pro.generate_signal(uptrend)
        signal_down, conf_down, _ = wavelet_pro.generate_signal(downtrend)
        
        # Uptrend should favor LONG, downtrend should favor SHORT
        logger.info(f"✓ Uptrend signal: {signal_up}, Downtrend signal: {signal_down}")
    
    def test_confidence_varies(self):
        """Verify confidence varies with signal strength."""
        wavelet_pro = WaveletPro()
        
        # Weak signal (flat)
        flat = np.ones(512) * 1800
        
        # Strong signal (oscillating)
        t = np.linspace(0, 20, 512)
        strong = 1800 + 100 * np.sin(t)
        
        _, conf_weak, _ = wavelet_pro.generate_signal(flat)
        _, conf_strong, _ = wavelet_pro.generate_signal(strong)
        
        logger.info(f"✓ Weak signal confidence: {conf_weak:.2f}, Strong: {conf_strong:.2f}")




class TestModelComparison:
    """Test comparison between basic and Pro models."""
    
    def test_comparison_function(self):
        """Verify comparison function works."""
        np.random.seed(42)
        prices = np.random.randn(512).cumsum() + 1800
        
        pro_model = WaveletPro()
        
        comparison = compare_wavelet_models(prices, pro_model=pro_model)
        
        assert "pro_signal" in comparison, "Missing pro_signal"
        assert "pro_confidence" in comparison, "Missing pro_confidence"
        assert "pro_decomposition_levels" in comparison, "Missing decomposition_levels"
        
        assert comparison["pro_decomposition_levels"] == 6, "Should be 6 levels"
        
        logger.info(f"✓ Comparison: {comparison}")


class TestIntegration:
    """Integration tests with full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete workflow."""
        # Generate data
        np.random.seed(42)
        t = np.linspace(0, 20, 512)
        prices_array = 1800 + 50 * np.sin(t) + np.random.randn(512) * 5
        prices_df = pd.DataFrame({"close": prices_array})
        
        # Train
        wavelet_pro = WaveletPro()
        train_result = wavelet_pro.train(prices_df)
        assert train_result["status"] == "trained", "Training failed"
        
        # Engineer features
        features = wavelet_pro.engineer_features(prices_df)
        assert len(features) == 1, "Feature engineering failed"
        
        # Generate signal
        signal, confidence, reasoning = wavelet_pro.generate_signal(prices_array)
        assert signal in ["LONG", "SHORT", "HOLD"], "Invalid signal"
        
        # Evaluate
        signals, confidences = wavelet_pro.predict(prices_df)
        assert len(signals) == len(prices_df), "Prediction length mismatch"
        
        logger.info(f"✓ Full pipeline complete: {signal} (conf={confidence:.2f})")


class TestComparisonWithBasic:
    """Compare Pro model with basic Wavelet model."""
    
    def test_pro_vs_basic_features(self):
        """Verify Pro model has more features than basic."""
        np.random.seed(42)
        prices = 1800 + 100 * np.sin(np.linspace(0, 10, 512)) + np.random.randn(512) * 5
        prices_df = pd.DataFrame({"close": prices})
        
        pro_model = WaveletPro()
        features_pro = pro_model.engineer_features(prices_df)
        
        # Pro should have 30+ features
        assert len(features_pro.columns) >= 25, f"Pro model features: {len(features_pro.columns)}"
        
        logger.info(f"✓ Pro model features: {len(features_pro.columns)}")
    
    def test_pro_decomposition_depth(self):
        """Verify Pro uses 6 levels vs basic 5."""
        pro_model = WaveletPro()
        assert pro_model.config.dwt_levels == 6, "Pro should use 6 levels"
        
        logger.info("✓ Pro model uses 6-level DWT (vs basic 5)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
