

# Mini-Medallion HMM Philosophy

This document outlines the design philosophy, architectural differences, and key configuration parameters between the two Hidden Markov Model (HMM) systems operating within the Mini-Medallion trading architecture.

---

## The Core Difference: System Roles

The system uses two independent HMMs that serve completely different purposes. 

### 1. The Original HMM (`hmm_regime.py`) — **The "Weather Forecaster"**
- **Primary Job**: Regime Detection. It determines the current market environment.
- **System Impact**: It acts as the system coordinator. It does **not** tell the system *what* to trade. Instead, it tells the system *how* to trade by adjusting:
  - Which models to trust (Dynamic Weighting).
  - How much risk to take (Kelly Fraction scaling).
  - Which trading strategy to employ (Mean Reversion vs. Trend Following).
- **Output**: `GROWTH`, `NORMAL`, or `CRISIS`.

### 2. HMM Pro (`hmm_pro.py`) — **The "Trading Expert"**
- **Primary Job**: Signal Generation. It acts as one of the core voting models in the ensemble.
- **System Impact**: It replaced the old, weak NLP Sentiment model. It uses advanced mathematics to directly predict the next price movement based on historical patterns and macro data.
- **Output**: `LONG`, `SHORT`, or `HOLD` (with a confidence score from 0.0 to 1.0).

> **Analogy**: If the trading system is a ship, `hmm_regime.py` is the Captain looking at the horizon saying, *"A storm is coming (Crisis Regime). Reduce our sails and trust the radar."* Meanwhile, `hmm_pro.py` is the Master Navigator at the helm saying, *"Based on the wind and stars, turn the wheel 10 degrees right (LONG signal)."*

---

## Architectural Comparison

| Feature | Original HMM (`hmm_regime.py`) | HMM Pro (`hmm_pro.py`) |
| :--- | :--- | :--- |
| **Math Engine** | `GaussianHMM` | `GMMHMM` (Gaussian Mixture HMM) |
| **Complexity** | Assumes data follows a single bell curve per state. | Highly advanced; assumes data can have multiple overlapping patterns (mixtures) per state. |
| **Ensemble Logic** | Uses an **ensemble of 5 separate simple HMMs** running across different timeframes (1m, 5m, 15m) to reach a consensus. | Uses a **single, highly complex model** with 4 hidden states and 3 sub-mixtures per state to capture nuanced price action. |
| **Hidden States** | 5 internal states mapped down to 3 external regimes (Growth, Normal, Crisis). | 4 directional states mapped to signals (Bullish, Neutral, Bearish, Reversal). |
| **Failure Fallback**| Falls back to a simple volatility percentile ranking. | Gracefully degrades to `GaussianHMM`, then to a heuristic if needed. |

---

## Data Inputs (Features)

### Original HMM Features (Macro-Volatility Focus)
Looks primarily at how chaotic the market is across multiple timeframes:
- Price momentum and return acceleration
- Volatility (Standard deviation of returns)
- Volume z-scores

### HMM Pro Features (XAU/USD Trading Focus)
Heavily engineered specifically for Gold (XAU/USD), utilizing a 15-20 dimension array:
- **Momentum Indicators:** RSI(14), MACD, Bollinger Band width.
- **Volatility:** ATR(14), 20-bar realized volatility.
- **Macro Cross-Asset:** DXY (US Dollar Index), US10Y (Treasury Yields), Gold-Silver Ratio.
- **Session Data:** Hour-of-day cyclicality (Sine/Cosine of time).

---

## Key Parameters for Users (`configs/base.yaml`)

Both models can be tuned via `configs/base.yaml`. Here is what the parameters mean:

### Original HMM Parameters
```yaml
hmm:
  n_regimes: 3                 # External regimes to map to (Growth/Normal/Crisis)
  covariance_type: "diag"      # 'diag' is faster/stable, 'full' captures feature correlations
  n_iter: 1000                 # Maximum EM algorithm iterations for training
  min_regime_duration: 5       # Anti-whipsaw: minimum bars before confirming a regime change
  regime_persistence_weight: 0.3  # How heavily to weigh recent regimes to prevent flickering
  vol_windows: [10, 20, 50]    # Lookback windows for calculating market volatility
  retrain_frequency: "daily"   # How often the regime model re-fits to historical data
```

### HMM Pro Parameters
```yaml
hmm_pro:
  model_type: "GMMHMM"         # GMMHMM (complex) or GaussianHMM (simple)
  n_states: 4                  # Number of hidden directional states (Bull/Neutral/Bear/Reversal)
  n_components: 3              # Number of Gaussian mixtures per state (captures multimodal data)
  covariance_type: "diag"      # Keep as 'diag' for parameter efficiency with 15+ features
  max_iter: 200                # EM iterations (lower than original HMM because GMMs are heavier)
  tol: 1e-4                    # Convergence tolerance
  min_covariance: 1e-3         # Prevents model from collapsing on zero-variance features
  retrain_interval: 500        # Bars between live inference retraining (adaptive)
  random_state: 42             # Seed for reproducible results
```
