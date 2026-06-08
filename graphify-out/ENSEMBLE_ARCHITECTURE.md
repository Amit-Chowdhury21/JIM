# Professional Ensemble Trading Pipeline Architecture
## Gold (XAU/USD) Scalping System — v3.0

**Generated**: 2026-06-03 | **Graph Nodes**: 11,572 | **Edges**: 18,255 | **Communities**: 933

---

## 🎯 Executive Summary

This document describes the **17-phase professional ensemble pipeline** for XAU/USD trading. The system combines four specialized deep learning models (WaveletPro, HMM Pro, LSTM, TFT) with a sophisticated meta-decision layer, dynamic weighting engine, and governance monitoring system.

### Key Innovation: Regime-Aware Adaptive Weighting

Unlike naive ensemble averaging, this system:
- **Classifies market regime** (Growth/Normal/Crisis) using HMM v3.0 RegimeDetector
- **Adapts model weights dynamically** based on: performance history + confidence + regime suitability
- **Gates risky trades** with multi-layer risk evaluation (disagreement, uncertainty, staleness)
- **Sizes positions** using Kelly Criterion + volatility + regime multipliers + circuit breakers
- **Monitors system health** with real-time alerts and model degradation detection

---

## 📊 Pipeline Architecture (17-Phase Implementation)

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1-4: DATA INGESTION & MACRO REGIME CLASSIFICATION            │
│ ├─ Live 1-minute OHLCV data from gold spot prices                  │
│ ├─ Macro data: DXY, VIX, yields, spreads, gaps                     │
│ └─ HMM v3.0 Multi-Timeframe Regime Detector (1m + 5m + 15m)        │
│    Output: GROWTH | NORMAL | CRISIS (+ confidence + velocity)      │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5-8: PARALLEL MODEL INFERENCE (ALL 4 MODELS RUN IN PARALLEL) │
│                                                                      │
│ 1️⃣  WaveletPro                 3️⃣  LSTM                            │
│     └─ Daubechies-4 DWT       └─ 3-layer bidirectional             │
│     └─ CWT volatility analysis └─ 128 hidden units                  │
│     └─ Soft thresholding       └─ 100-timestep sequences            │
│     Output: trend +strength    Output: direction (3 categories)    │
│                                                                      │
│ 2️⃣  HMM Pro                    4️⃣  TFT                              │
│     └─ GMMHMM, 4 latent states └─ Temporal Fusion Transformer      │
│     └─ XAU-specific features   └─ 4-head attention                  │
│     └─ <20ms inference         └─ Multi-horizon (1,5,10,20 bars)   │
│     Output: BULLISH/NEUTRAL/   Output: quantile forecast 0.1-0.9   │
│             BEARISH + prob                                          │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 9: STANDARDIZED OUTPUT NORMALIZATION                          │
│ └─ All 4 models → [-1,1] direction_score + [0,1] confidence       │
│ └─ Regime suitability scoring per model                            │
│ └─ Risk warning flags (stale, uncertain, etc)                      │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 10: DYNAMIC WEIGHTING ENGINE                                  │
│ ├─ Base weights by regime:                                          │
│ │  GROWTH:  Wavelet 30%, HMM 15%, LSTM 30%, TFT 25%                │
│ │  NORMAL:  Wavelet 25%, HMM 25%, LSTM 20%, TFT 30%                │
│ │  CRISIS:  Wavelet 15%, HMM 35%, LSTM 10%, TFT 40%                │
│ │                                                                   │
│ ├─ Multipliers applied sequentially:                                │
│ │  1. Performance Multiplier (recent hit rate: [0.5-1.5])          │
│ │  2. Confidence Multiplier (signal confidence: [0.5-1.5])         │
│ │  3. Suitability Multiplier (regime performance: [0.5-1.5])       │
│ │  4. Disagreement Penalty (std dev of signals)                    │
│ │                                                                   │
│ └─ Normalized to 100% (hard caps: max 50%, min 5%)                 │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 11-12: META-DECISION LAYER                                    │
│ ├─ Ensemble Score: weighted average of direction_scores [-1,1]     │
│ ├─ Ensemble Confidence: weighted average × disagreement_penalty    │
│ ├─ Signal Quality Classification:                                  │
│ │  STRONG (conf ≥ 0.75 + disagreement < 0.5)                      │
│ │  MODERATE (conf ≥ 0.55)                                         │
│ │  WEAK (conf ≥ 0.35)                                             │
│ │  CONFLICTED (high disagreement: std > 0.6)                       │
│ │  BLOCKED (risk gates active)                                    │
│ │                                                                   │
│ └─ Risk Gates (ANY active = NO TRADE):                             │
│    • Low confidence gate (< 0.35)                                  │
│    • Extreme disagreement gate (std > 0.9)                         │
│    • Uncertainty gate (TFT quantile spread > 0.7)                  │
│    • Stale data gate (> 5 minutes old)                             │
│    • Crisis-specific gates (crisis + low conf < 0.6)               │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 13: POSITION SIZING ENGINE                                    │
│ ├─ Kelly Criterion: f* = (p*b - q) / b                             │
│ │  └─ Assumptions: 55% win rate, 2.0 profit factor                 │
│ │  └─ Half-Kelly for safety (reduce by 50%)                        │
│ │                                                                   │
│ ├─ Volatility Adjustment: (ref_atr / current_atr)^0.5              │
│ │  └─ High vol = smaller positions; low vol = larger               │
│ │                                                                   │
│ ├─ Regime Multipliers:                                             │
│ │  GROWTH: 1.2x (aggressive)                                       │
│ │  NORMAL: 1.0x (baseline)                                         │
│ │  CRISIS: 0.6x (conservative)                                     │
│ │                                                                   │
│ ├─ Max Position Caps by Regime:                                    │
│ │  GROWTH: 10%                                                     │
│ │  NORMAL: 5%                                                      │
│ │  CRISIS: 2%                                                      │
│ │                                                                   │
│ └─ Circuit Breakers:                                               │
│    • Daily loss limit: 2% of account                               │
│    • Drawdown limit: 10% from peak                                 │
│    • Blocks new trades if breached                                 │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 14-15: RISK MANAGEMENT & STOPS/TARGETS                       │
│ ├─ Stop Loss: 1.5x ATR (adjusted for volatility)                   │
│ ├─ Take Profit: Stop × Risk/Reward Ratio (default 2.0)             │
│ ├─ Position Side: LONG | SHORT | FLAT | UNDEFINED                  │
│ └─ Final Recommendation: position_side + size + stops + quality    │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 16: GOVERNANCE MONITORING & ALERTS                            │
│ ├─ System Health Metrics (1000-sample history):                    │
│ │  • Model degradation detection (hit_rate < 45%, Sharpe < -1.0)  │
│ │  • Data staleness detection (no update > 5 min)                  │
│ │  • Disagreement spike detection (std > 0.8)                      │
│ │  • Risk gate activation logging                                  │
│ │                                                                   │
│ ├─ Performance Tracking by Model × Regime:                         │
│ │  • Hit rate, Sharpe ratio, sample count                          │
│ │  • Regime-specific performance scoring                           │
│ │                                                                   │
│ └─ Alert Levels: INFO | WARNING | CRITICAL                         │
│    └─ Resolves alerts as conditions clear                          │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 17: DASHBOARD & API EXPOSURE                                  │
│                                                                      │
│ 📊 React Dashboard Pages:                                           │
│   1. EnsembleMetrics - Real-time weights, confidence, disagreement  │
│   2. RegimeMonitor - Regime probabilities, macro indicators, velocity
│   3. ModelPerformance - Hit rates, Sharpe, regime-specific scoring  │
│   4. PositionSizing - Kelly fraction, regime multipliers, limits   │
│   5. Governance - Health status, alerts, model scorecards          │
│                                                                      │
│ 🔌 REST API Endpoints:                                             │
│   GET /api/ensemble/metrics           → Current weights + confidence
│   GET /api/ensemble/regime            → Regime + velocity + probs   │
│   GET /api/ensemble/model-performance → Scorecards by regime        │
│   GET /api/ensemble/position-sizing   → Sizing recommendations     │
│   GET /api/ensemble/governance        → System health + alerts      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Core Components (6 Backend Modules)

### 1. **RegimeDetector (hmm_regime.py)** — Market State Classifier
- **Version**: 3.0 (Multi-Timeframe HMM Ensemble)
- **Purpose**: Classify market regime into Growth/Normal/Crisis
- **Architecture**:
  - Primary HMM: 5 internal regimes (Quiet/Trending/Normal/Volatile/Crisis)
  - Ensemble HMMs: 2 additional 3-regime models for fusion
  - Multi-TF HMMs: 5m and 15m consensus models
  - Transition Velocity: Rate of regime probability change (early warning)
  - Micro-Models: Per-regime specific indicator sets
- **Output**: External regime + confidence + transition velocity + internal regime
- **Latency**: < 50ms per classification

### 2. **StandardizedOutputLayer (standardized_output.py)** — Model Output Normalization
- **Purpose**: Normalize all 4 models to consistent schema
- **Normalizers**:
  - WaveletPro: trend_direction → [-1,1], noise_ratio → confidence reduction
  - HMM Pro: BULLISH→0.8, NEUTRAL→0.0, BEARISH→-0.8
  - LSTM: direction_category + return magnitude → tanh normalized
  - TFT: average forecast horizons, reduce confidence by uncertainty
- **Per-Model Output**:
  - `direction_score`: [-1, 1] (bullish to bearish)
  - `confidence`: [0, 1] (model certainty)
  - `horizon`: SHORT (1-5 bars) | MEDIUM (5-20) | LONG (20+)
  - `regime_suitability`: {growth, normal, crisis} → [0,1] performance history
  - `risk_warning`: String describing concerns (stale, high_uncertainty, etc)
- **Performance Tracking**: Hit rate + Sharpe ratio per model-regime pair

### 3. **DynamicWeightingEngine (dynamic_weighting.py)** — Adaptive Ensemble Weights
- **Base Weights by Regime** (normalized to 100%):
  - **Growth** (low vol): Wavelet 30%, HMM 15%, LSTM 30%, TFT 25%
  - **Normal** (balanced): Wavelet 25%, HMM 25%, LSTM 20%, TFT 30%
  - **Crisis** (high vol): Wavelet 15%, HMM 35%, LSTM 10%, TFT 40%
- **Multipliers** (each [0.5, 1.5], sequentially applied):
  1. Performance: Recent hit rate premium (1.0 = neutral baseline)
  2. Confidence: Model signal confidence (1.0 = neutral, <1.0 = uncertain)
  3. Suitability: Historical performance in current regime
  4. Disagreement Penalty: Std dev of model signals
- **Weight Caps**: Max 50% per model, min 5%, normalized after multipliers
- **History**: Last 100 weight updates stored for charting

### 4. **MetaDecisionLayer (meta_decision_layer.py)** — Trade Decision Logic
- **Ensemble Aggregation**:
  - Direction: Weighted average of normalized signals [-1,1]
  - Confidence: Weighted average of confidences × disagreement_penalty
  - Disagreement: Std deviation of model signals
- **Signal Quality Classification**:
  - `STRONG`: Confidence ≥ 0.75 + disagreement < 0.5
  - `MODERATE`: Confidence ≥ 0.55
  - `WEAK`: Confidence ≥ 0.35
  - `CONFLICTED`: High disagreement (std > 0.6)
  - `BLOCKED`: Risk gates active
- **Risk Gates** (any active blocks trade):
  1. Low confidence (< 0.35)
  2. Extreme disagreement (std > 0.9)
  3. High uncertainty (TFT quantile spread > 0.7)
  4. Stale data (> 5 minutes old)
  5. Crisis-specific (crisis regime + low confidence < 0.6)
- **Position Recommendation**:
  - Side: LONG | SHORT | FLAT | UNDEFINED
  - Size: Confidence × (1-disagreement) × regime_multiplier (clamped to regime max)
  - Stop Loss: 1.5×ATR (volatility adjusted)
  - Take Profit: Stop × 2.0 (risk/reward ratio)

### 5. **PositionSizingEngine (position_sizing.py)** — Risk-Based Sizing
- **Kelly Criterion Implementation**:
  - Formula: `f* = (p*b - q) / b` where p=win_rate, b=profit_factor, q=1-p
  - Assumed: 55% win rate, 2.0 profit factor
  - Applied: Half-Kelly (50% reduction for safety)
- **Volatility Adjustment**: `(reference_atr / current_atr)^0.5`
  - Inverse relationship: high vol = smaller positions
  - Clipped [0.5, 1.5]
- **Regime Multipliers**:
  - GROWTH: 1.2x (aggressive positioning)
  - NORMAL: 1.0x (baseline)
  - CRISIS: 0.6x (conservative)
- **Max Position Caps**:
  - GROWTH: 10% of account
  - NORMAL: 5%
  - CRISIS: 2%
- **Circuit Breakers**:
  - Daily loss limit: 2% of account (blocks new trades)
  - Drawdown limit: 10% from peak equity (blocks new trades)
- **Output**: Full position config with all adjustments applied

### 6. **GovernanceMonitor (governance_monitor.py)** — System Health & Alerts
- **Health Metrics** (1000-sample history per metric):
  - Data freshness (age of last update)
  - Model degradation (hit_rate < 45% or Sharpe < -1.0 = alert)
  - Disagreement spikes (std > 0.8)
  - Risk gate activations
  - Regime transitions
- **Performance Tracking**:
  - Per-model: total trades, hit rate, average return, Sharpe ratio
  - Per-regime: model performance breakdown
  - Regime-specific scoring
- **Alert Management**:
  - Levels: INFO | WARNING | CRITICAL
  - Automatic resolution when conditions clear
  - Active alert list sorted by timestamp
- **System Report** includes:
  - Overall health status (healthy/warning/critical)
  - Alert counts by level
  - Model scorecards
  - Current regime + confidence
  - System metrics snapshot

---

## 🎨 Dashboard Pages (React/Recharts)

### Page 1: EnsembleMetrics
- **Pie Chart**: Model weight distribution (WaveletPro, HMM, LSTM, TFT)
- **Bar Chart**: Confidence multipliers per model [0-1.5]
- **Metric Cards**: Ensemble score (green/red), confidence, disagreement, recommended size
- **Update Freq**: 5 seconds

### Page 2: RegimeMonitor
- **Regime Box**: Current regime (Growth/Normal/Crisis) with gradient background + confidence
- **Bar Chart**: Regime probabilities (3 bars: growth/normal/crisis)
- **Metrics Grid**: Realized vol %ile, VIX level, DXY shock, gap frequency, spread widening
- **Model Inputs**: HMM bullish probability, Wavelet noise ratio (progress bars)
- **Update Freq**: 5 seconds

### Page 3: ModelPerformance
- **Performance Cards**: 4 cards (one per model) with hit rate % and Sharpe ratio
- **Hit Rate Chart**: Bar comparison across 4 models
- **Sharpe Chart**: Bar comparison across 4 models
- **Grouped Chart**: Hit rates by regime (Growth/Normal/Crisis) for each model
- **Update Freq**: 10 seconds

### Page 4: PositionSizing
- **Summary Box**: Account size, max %, recommended %, risk per trade
- **Kelly Section**: Kelly fraction % (gradient bar), Half-Kelly explanation
- **Regime Multipliers Chart**: Growth/Normal/Crisis sizing multipliers (1.2x/1.0x/0.6x)
- **Risk Guidelines**: 3 cards with regime-specific limits
- **Update Freq**: 5 seconds

### Page 5: Governance
- **Health Status Badge**: Green/Yellow/Red indicator with text
- **Alert Counts**: Critical, warning, info, total active
- **Alerts List**: Scrollable, showing severity icon + title + message + timestamp
- **Model Scorecards**: 2-column grid (hit rate, Sharpe, samples per model)
- **Metrics Grid**: Top 6 system metrics with color-coded severity
- **Update Freq**: 5 seconds

---

## 🔌 API Endpoints

All endpoints return JSON with proper error handling and logging.

### GET `/api/ensemble/metrics`
Returns current model weights and confidence metrics.
```json
{
  "wavelet": 0.30,
  "hmm": 0.15,
  "lstm": 0.30,
  "tft": 0.25,
  "ensemble_score": 0.25,
  "ensemble_confidence": 0.68,
  "disagreement": 0.35,
  "recommended_size": 0.045,
  "timestamp": "2026-06-03T14:30:45.123Z"
}
```

### GET `/api/ensemble/regime`
Returns current regime classification from HMM v3.0 RegimeDetector.
```json
{
  "regime": "normal",
  "confidence": 0.72,
  "internal_regime": "NORMAL",
  "probabilities": {
    "growth": 0.25,
    "normal": 0.72,
    "crisis": 0.03
  },
  "transition_velocity": {
    "growth": 0.002,
    "normal": -0.015,
    "crisis": 0.013
  },
  "version": "v3.0",
  "detector_type": "HMM Multi-Timeframe Ensemble",
  "timestamp": "2026-06-03T14:30:45.123Z"
}
```

### GET `/api/ensemble/model-performance`
Returns hit rates and Sharpe ratios per model by regime.
```json
{
  "models": {
    "wavelet": {
      "overall_hit_rate": 0.52,
      "overall_sharpe": 0.95,
      "by_regime": {
        "growth": {"hit_rate": 0.58, "sharpe": 1.20, "samples": 150},
        "normal": {"hit_rate": 0.50, "sharpe": 0.85, "samples": 320},
        "crisis": {"hit_rate": 0.45, "sharpe": 0.60, "samples": 80}
      }
    },
    "hmm": {...},
    "lstm": {...},
    "tft": {...}
  }
}
```

### GET `/api/ensemble/position-sizing`
Returns position sizing recommendations.
```json
{
  "account_size": 100000,
  "max_position_pct": 0.05,
  "recommended_position_pct": 0.032,
  "recommended_position_usd": 3200,
  "kelly_fraction": 0.045,
  "kelly_adjusted": 0.0225,
  "regime_multiplier": 1.0,
  "vol_adjustment": 0.95,
  "daily_loss_used_pct": 0.005,
  "drawdown_limit_active": false,
  "timestamp": "2026-06-03T14:30:45.123Z"
}
```

### GET `/api/ensemble/governance`
Returns system health status and alerts.
```json
{
  "overall_status": "healthy",
  "critical_count": 0,
  "warning_count": 1,
  "info_count": 3,
  "active_alerts": [
    {
      "id": "alert_001",
      "level": "warning",
      "title": "High Disagreement Spike",
      "message": "Model disagreement (std=0.82) exceeded threshold (0.8)",
      "component": "ensemble",
      "created_at": "2026-06-03T14:28:10Z",
      "metadata": {"disagreement": 0.82}
    }
  ],
  "model_scorecards": {
    "wavelet": {...},
    "hmm": {...},
    "lstm": {...},
    "tft": {...}
  },
  "timestamp": "2026-06-03T14:30:45.123Z"
}
```

---

## 📈 Regime-Based Strategy Framework

| Regime | Volatility | Approach | Kelly Fraction | Position Bias | Model Emphasis |
|--------|-----------|----------|----------------|--------------|-----------------|
| **GROWTH** | Low (< 30%ile) | Mean Reversion | 65% | LONG | Wavelet 30%, LSTM 30% |
| **NORMAL** | Medium (30-75%ile) | Balanced | 50% | NEUTRAL | HMM 25%, TFT 30% |
| **CRISIS** | High (> 75%ile) | Trend Following | 25% | DEFENSIVE | HMM 35%, TFT 40% |

---

## ⚠️ Risk Management Layers

1. **Disagreement Penalty**: Reduces all weights when models conflict
2. **Confidence Gates**: Blocks trades if confidence < 0.35
3. **Uncertainty Gates**: Blocks trades if TFT quantile spread > 0.7
4. **Staleness Gates**: Blocks trades if data > 5 minutes old
5. **Regime Gates**: Extra caution in crisis (confidence threshold ≥ 0.6)
6. **Kelly Fraction**: Position size capped by mathematical edge calculation
7. **Volatility Adjustment**: High vol automatically reduces sizing
8. **Circuit Breakers**: Daily loss & drawdown limits prevent catastrophic losses

---

## 🔧 Integration Points

### Live Model Inference Connection (In Progress)
```python
from src.models.hmm_regime import RegimeDetector
from src.models.standardized_output import StandardizedOutputLayer
from src.models.dynamic_weighting import DynamicWeightingEngine
from src.models.meta_decision_layer import MetaDecisionLayer
from src.models.position_sizing import PositionSizingEngine
from src.models.governance_monitor import GovernanceMonitor

# Pipeline execution
regime = detector.get_current_regime(gold_df)
signals = normalizer.normalize_all_models(model_outputs)
weights = weighting_engine.calculate_dynamic_weights(regime, signals)
decision = meta_layer.make_meta_decision(signals, weights, regime)
position = sizing_engine.get_recommended_position_config(decision, regime)
monitor.record_model_performance(model_results)
```

### Data Pipeline Requirements
- Live OHLCV data (1-minute bars)
- Macro data: DXY, VIX, US 10Y yield, spreads, gaps
- Model inference outputs: WaveletPro, HMM Pro, LSTM, TFT
- Trade execution results (for performance tracking)

---

## ✅ Implementation Status

- ✅ RegimeDetector v3.0 (HMM Multi-Timeframe)
- ✅ StandardizedOutputLayer (4-model normalizer)
- ✅ DynamicWeightingEngine (performance-adaptive)
- ✅ MetaDecisionLayer (risk gates + sizing)
- ✅ PositionSizingEngine (Kelly + regime)
- ✅ GovernanceMonitor (health + alerts)
- ✅ React Dashboard (5 pages)
- ✅ REST API (5 endpoints)
- 🔄 Live Model Integration (in progress)
- 🔄 Real-time Macro Data Pipeline (in progress)
- 🔄 Performance Tracking System (in progress)

---

## 📚 File Structure

```
src/
  models/
    ├── hmm_regime.py              # RegimeDetector v3.0
    ├── hmm_pro.py                 # HMM Pro trading expert
    ├── lstm_predictor.py           # LSTM model
    ├── tft_predictor.py            # Temporal Fusion Transformer
    ├── wavelet_denoiser.py         # WaveletPro signal denoiser
    ├── standardized_output.py      # Output normalization layer
    ├── dynamic_weighting.py        # Adaptive weighting engine
    ├── meta_decision_layer.py      # Final decision logic
    ├── position_sizing.py          # Kelly-based sizing
    └── governance_monitor.py       # Health monitoring
  api/
    └── app.py                      # FastAPI with 5 ensemble endpoints
dashboard/
  src/
    pages/
      ├── EnsembleMetrics.jsx       # Weights + confidence
      ├── RegimeMonitor.jsx         # Regime classification
      ├── ModelPerformance.jsx      # Hit rates by regime
      ├── PositionSizing.jsx        # Sizing parameters
      └── Governance.jsx            # Health + alerts
```

---

## 🎓 Design Principles

1. **Regime Awareness**: All decisions adapt to market state
2. **Adaptive Weighting**: Performance history drives allocation
3. **Multi-Layer Risk Management**: Gates prevent over-leverage
4. **Transparency**: Full reasoning and contributing factors logged
5. **Modularity**: Each component (regime/normalize/weight/decide/size) is independent
6. **Governance**: Real-time monitoring prevents model degradation
7. **Simplicity**: Professional but understandable rules, not black-box ML

---

**Last Updated**: 2026-06-03 | **Graph Build Time**: ~5 sec | **Nodes**: 11,572 | **Status**: ✅ Architecture Complete
