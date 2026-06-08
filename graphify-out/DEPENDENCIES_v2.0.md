# System Dependencies & Integration Map - v2.0
**Date**: June 3, 2026  
**Status**: Complete Integration

---

## Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────┐  │
│  │ React Dashboard    │  │ Swagger API Docs   │  │ HTTP CLI   │  │
│  │ (5 pages)          │  │ (Uvicorn)          │  │ (curl)     │  │
│  └────────┬───────────┘  └────────┬───────────┘  └─────┬──────┘  │
└───────────┼────────────────────────┼──────────────────┼────────────┘
            │                        │                  │
            └────────────┬───────────┴──────────────────┘
                         │
                    ↓ JSON/HTTP ↓
┌────────────────────────────────────────────────────────────────────┐
│                    src/api/app.py (FastAPI)                        │
│                                                                    │
│  Endpoints (8):                                                   │
│  ├─ GET  /api/ensemble/metrics           → DynamicWeightingEngine │
│  ├─ GET  /api/ensemble/regime            → RegimeDetector v3.0    │
│  ├─ GET  /api/ensemble/live-prediction   → EnsembleOrchestrator   │
│  ├─ GET  /api/ensemble/trade-history     → TradeHistoryTracker    │
│  ├─ GET  /api/ensemble/macro-data        → MacroDataFeed          │
│  ├─ POST /api/retraining/schedule        → ModelRetrainingSchedul.│
│  ├─ GET  /api/retraining/jobs            → ModelRetrainingSchedul.│
│  └─ GET  /api/retraining/models/*/versions → ModelRetrainingSchedul.│
└───────────┬──────────────────────────────────────────────────────┘
            │
            ↓ Function calls (same process)
            │
┌────────────────────────────────────────────────────────────────────┐
│                  NEW INFRASTRUCTURE LAYER (v2.0)                   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  src/utils/ensemble_orchestrator.py                         │  │
│  │  ├─ Calls: RegimeDetector.classify_regime()               │  │
│  │  ├─ Calls: MacroDataFeed.fetch_macro_data()              │  │
│  │  ├─ Calls: Layer 1 models (async)                        │  │
│  │  ├─ Calls: DynamicWeightingEngine.calculate_weights()    │  │
│  │  ├─ Calls: PositionSizingEngine.size_position()          │  │
│  │  └─ Calls: GovernanceMonitor.record_metric()             │  │
│  └────────┬────────────────────────────────────────────────────┘  │
│           │                                                        │
│  ┌────────┴────────────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  src/utils/macro_data_feed.py                              │   │
│  │  ├─ Imports: yfinance                                      │   │
│  │ └─ Provides: DXY, VIX, yields, spreads (60s cache)        │   │
│  │                                                             │   │
│  │  src/utils/trade_history_tracker.py                        │   │
│  │  ├─ Imports: pandas, pathlib                              │   │
│  │  ├─ Provides: Trade logging, P&L calculation             │   │
│  │  └─ Feeds back: Performance multipliers                   │   │
│  │                                                             │   │
│  │  src/utils/model_retraining_scheduler.py                   │   │
│  │  ├─ Imports: json, asyncio, uuid                          │   │
│  │  ├─ Provides: Job scheduling, version management          │   │
│  │  └─ Persists: models/registry.json                        │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
             │
             ↓ Internal function calls (same imports)
             │
┌────────────────────────────────────────────────────────────────────┐
│                    EXISTING MODELS LAYER (Layer 1)                 │
│                                                                    │
│  src/paper_trading/live_inference.py                              │
│  ├─ run_wavelet(df)           → WaveletPro v2.0                  │
│  ├─ run_wavelet_basic(df)     → WaveletBasic (legacy)            │
│  ├─ run_hmm(df)               → HMM v3.0                         │
│  ├─ run_lstm(df)              → LSTM CNN-Attention               │
│  ├─ run_tft_basic(df)         → TFT Forecaster                   │
│  ├─ run_genetic(df)           → Genetic Algorithm                │
│  ├─ run_hmm_pro_gpu(df)       → HMM Pro GPU (CUDA)              │
│  └─ run_ensemble()            → Meta-Learner voting              │
│                                                                    │
│  Individual model files:                                          │
│  ├─ src/models/wavelet_pro.py (6-level DWT + CWT)               │
│  ├─ src/models/hmm_detector.py (5-regime primary)               │
│  ├─ src/models/lstm_temporal.py (CNN-LSTM-Attention)            │
│  ├─ src/models/tft_forecaster.py (Multi-horizon TFT)            │
│  ├─ src/models/genetic.py (Evolutionary optimizer)              │
│  └─ src/models/hmm_pro_gpu.py (GPU-accelerated HMM)             │
│                                                                    │
│  Supporting components:                                           │
│  ├─ src/models/dynamic_weighting.py (weight calculation)         │
│  ├─ src/models/position_sizing.py (Kelly Criterion)             │
│  ├─ src/models/meta_decision_layer.py (voting logic)            │
│  └─ src/models/governance_monitor.py (health tracking)          │
└────────────────────────────────────────────────────────────────────┘
             │
             ↓ Data consumption
             │
┌────────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES & STORAGE                        │
│                                                                    │
│  Real-time Feeds:                                                 │
│  ├─ yfinance (DXY, VIX, yields, commodities)                     │
│  ├─ Gold OHLCV (1m, 5m, 15m bars)                                │
│  ├─ Economic calendar (Forex Factory)                            │
│  └─ Market microstructure (ticks, spreads)                       │
│                                                                    │
│  Persistent Storage:                                              │
│  ├─ QuestDB (9 tables: gold_*, macro_*, fred_*, etc.)            │
│  ├─ Parquet files (data/raw/, data/processed/)                   │
│  ├─ CSV logs (logs/trades.csv)                                   │
│  ├─ JSON config (models/registry.json)                           │
│  └─ Model weights (models/*.pt, models/*.joblib)                 │
│                                                                    │
│  Cache:                                                           │
│  ├─ MacroDataFeed: 60-second TTL                                 │
│  ├─ Regime detector: Session cache                               │
│  └─ Model weights: GPU memory resident                           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Call Chain Example: Live Inference Tick

```
START (60-second tick)
  ↓
  ├─ Fetch gold 1m bar from QuestDB
  ↓
  └─ scripts/live_trader.py: run_all_models()
     └─ Calls all 7 models (parallel)
        ├─ WaveletPro(df) ─ CPU, DWT + CWT denoising
        ├─ HMM(df) ───── Multi-regime classification
        ├─ LSTM(df) ──── Temporal prediction (GPU if available)
        ├─ TFT(df) ───── Multi-horizon forecast
        ├─ Genetic(df) ─ Evolutionary optimization
        ├─ HMM_Pro(df) ─ GPU-accelerated (CUDA)
        └─ Meta-Learner ─ Consensus voting
  ↓
  └─ results = {
       wavelet: {signal: LONG, confidence: 0.75},
       hmm: {signal: SHORT, confidence: 0.65},
       ...
     }
  ↓
  └─ EnsembleOrchestrator.run_inference(df):
     │
     ├─ Step 1: RegimeDetector.classify_regime(df)
     │   └─ Analyzes: volatility, macro shocks, persistence
     │   └─ Output: regime ∈ {GROWTH, NORMAL, CRISIS}, confidence
     │
     ├─ Step 2: MacroDataFeed.fetch_macro_data()
     │   └─ Calls yfinance for: DXY, VIX, yields, spreads
     │   └─ Uses 60-second cache (yfinance rate limit)
     │   └─ Output: {dxy, vix, us10y, ...}
     │
     ├─ Step 3: Parallel model execution (results from above)
     │   └─ Already completed above
     │
     ├─ Step 4: StandardizedOutputLayer.normalize()
     │   └─ Input: {wavelet: {...}, hmm: {...}, ...}
     │   └─ Output: {wavelet: {norm_sig: 0.8, norm_conf: 0.75}, ...}
     │
     ├─ Step 5: DynamicWeightingEngine.calculate_weights()
     │   │
     │   ├─ Fetch model performance from TradeHistoryTracker
     │   │  └─ get_model_performance(model, regime)
     │   │     └─ Returns: {hit_rate, sharpe, sample_count}
     │   │
     │   ├─ Calculate: weight = base_weight × perf_mult × conf_mult × suit_mult
     │   ├─ Apply regime multipliers: Growth(+20%), Normal(0%), Crisis(-40%)
     │   └─ Output: {wavelet: 0.25, hmm: 0.22, lstm: 0.28, ...}
     │
     ├─ Step 6: MetaDecisionLayer.make_decision()
     │   │
     │   ├─ Weighted sum: score = Σ(weight[i] × signal[i])
     │   ├─ Calculate agreement: disagreement = max(SHORT%, LONG%) - 50%
     │   ├─ Apply disagreement penalty: final_score = score × (1 - disagreement)
     │   ├─ Risk gates:
     │   │  ├─ Quality gate: reject if quality < threshold
     │   │  ├─ Confidence gate: reject if confidence < min_conf
     │   │  └─ Circuit breaker: halt if daily_loss > limit
     │   │
     │   └─ Output: {side: LONG, quality: MODERATE, confidence: 0.71}
     │
     ├─ Step 7: PositionSizingEngine.size_position()
     │   │
     │   ├─ Kelly Criterion: f* = (hit_rate × avg_win - (1-hit_rate) × avg_loss) / avg_win
     │   ├─ Regime adjustment: size = size_base × regime_multiplier
     │   │   ├─ Growth: ×1.2 (aggressive)
     │   │   ├─ Normal: ×1.0 (balanced)
     │   │   └─ Crisis: ×0.6 (defensive)
     │   │
     │   ├─ Volatility adjustment: size = size × (baseline_vol / current_vol)
     │   │   └─ Uses ATR for volatility estimation
     │   │
     │   ├─ Max position limit: size = min(size, max_position_pct)
     │   │
     │   ├─ Stop-loss placement: stop = entry - (ATR × stop_mult)
     │   ├─ Take-profit placement: target = entry + (size / (1 - kelly_fraction) × stop_loss)
     │   │
     │   └─ Output: {
     │       recommended_size: 0.035 (3.5% of account),
     │       stop_loss_offset: 50 pips,
     │       take_profit_offset: 75 pips,
     │       kelly_fraction: 0.12
     │     }
     │
     └─ Step 8: GovernanceMonitor.record_metric()
        ├─ Track: system_health, signal_quality, model_disagreement
        ├─ Check: daily P&L, consecutive losses, max drawdown
        ├─ Alert if:
        │  ├─ Model degradation detected (hit_rate drop >10%)
        │  ├─ Drawdown exceeds limit
        │  ├─ Consecutive losses > 3
        │  └─ High model disagreement
        │
        └─ Output: active_alerts = [...]
  ↓
  Final Output: EnsemblePrediction = {
    regime: "normal",
    regime_confidence: 0.68,
    ensemble_signal: "LONG",
    ensemble_score: 0.42,
    ensemble_confidence: 0.71,
    signal_quality: "MODERATE",
    recommended_size: 0.035,
    stop_loss_offset: 50,
    take_profit_offset: 75,
    active_alerts: [],
    system_health: "HEALTHY",
    macro_data: {dxy: 104.5, vix: 18.2, ...}
  }
  ↓
  Trade Decision:
  IF signal_quality >= MINIMUM_THRESHOLD:
    └─ TradeHistoryTracker.log_trade_entry({
        trade_id: "trade_001",
        entry_price: 2050.00,
        side: "LONG",
        size: 0.035,
        ensemble_score: 0.42,
        regime: "normal",
        ...
      })
      └─ PaperTradingEngine.open_position(LONG, 3.5% size, stop, target)
  ELSE:
    └─ Skip trade (quality gate blocked entry)
  ↓
  Return to REST API:
  └─ Store prediction in cache
  └─ Expose via GET /api/ensemble/live-prediction
  ↓
  Dashboard receives:
  └─ Fetch from /api/ensemble/live-prediction
  └─ Update charts: weights, regime, signals, P&L
  ↓
  Wait 60 seconds for next tick...
```

---

## Bidirectional Feedback Loop

```
Live Prediction Output
        ↓
    Trade Execution
        ↓
    Trade Management
        ├─ Price moves
        ├─ P&L changes
        ├─ Stops/targets trigger
        └─ Exit conditions met
        ↓
    Trade Close
        ↓
    TradeHistoryTracker.close_trade()
        ├─ Calculate P&L (pips, USD, %)
        ├─ Mark winner/loser
        ├─ Update model performance:
        │  └─ For each model in the ensemble:
        │     ├─ Per-regime performance tracking
        │     ├─ Hit rate calculation
        │     ├─ Sharpe ratio calculation
        │     └─ Update by_regime[regime] = {hit_rate, count, ...}
        │
        └─ Persist to: logs/trades.csv
        ↓
    Performance Data Available
        ↓
    Next Inference Cycle (60s later)
        ├─ DynamicWeightingEngine.calculate_weights()
        │  └─ Calls: TradeHistoryTracker.get_model_performance()
        │     └─ Returns updated metrics
        │
        └─ Apply performance multipliers:
           └─ Models that performed well → Higher weights
           └─ Models that underperformed → Lower weights
           └─ Per-regime adjustment (regime-specific hit rates)
        ↓
    Weighted signals used in next Meta-Decision Layer
        ↓
    Next trade benefits from past performance
        ↓
    Cycle repeats...
```

---

## Component Communication Patterns

### Pattern 1: Direct Function Calls (Same Process)
```python
# EnsembleOrchestrator calls infrastructure components
regime = regime_detector.classify_regime(df)
macro = macro_feed.fetch_macro_data()
weights = weighting_engine.calculate_weights()
size = sizing_engine.get_recommended_position(...)
health = governance.record_metric(...)
```

### Pattern 2: Singleton Pattern (Global Access)
```python
# Components retrieved via module-level functions
orchestrator = get_orchestrator()
tracker = get_trade_tracker()
scheduler = get_retraining_scheduler()
```

### Pattern 3: REST API Exposure
```python
# Internal Python functions exposed via HTTP
GET /api/ensemble/live-prediction
  └─ Calls: orchestrator.run_inference()
  └─ Returns: JSON EnsemblePrediction
```

### Pattern 4: Feedback Loops
```python
# Trade results update model performance
trade_tracker.close_trade(trade_id, exit_price)
  └─ Updates: model_performance[model][regime]
  └─ Next cycle: weighting_engine uses updated metrics
```

---

## Error Handling & Resilience

| Component | Failure Mode | Handling |
|-----------|--------------|----------|
| **yfinance** | Network timeout | Fallback to neutral values (DXY=104) |
| **Model inference** | GPU OOM | Fall back to CPU |
| **LSTM loading** | CUDA incompatible | Use CPU (slower but works) |
| **Database** | QuestDB down | Use parquet cache |
| **API endpoint** | Model error | Return 500 with error details |
| **Retraining** | Training failure | Mark job as FAILED, keep previous version |

---

## Type Safety & Validation

| Layer | Type System | Validation |
|-------|-------------|-----------|
| **API Input** | Pydantic models | RetrainingRequest validation |
| **API Output** | Pydantic models | RetrainingResponse, EnsemblePrediction |
| **Internal** | Python type hints | Type annotations throughout |
| **Database** | QuestDB schema | Strict column types |
| **Serialization** | JSON | Automatic via FastAPI |

---

## Scalability Considerations

### Horizontal Scaling
- ✅ API is stateless (can run multiple instances)
- ✅ Load balance via reverse proxy (nginx)
- ✅ Shared QuestDB for data

### Vertical Scaling
- ✅ Models can be GPU-accelerated (CUDA)
- ✅ Macro data caching reduces API calls
- ✅ Async inference for parallel model execution

### Performance Optimization
- ✅ 60-second macro data cache (yfinance rate limiting)
- ✅ Model weights cached in GPU memory
- ✅ Vectorized numpy operations
- ✅ Async/await for non-blocking I/O

---

## Summary

**Total Components**: 35+
- New: 7 (infrastructure + models)
- Existing: 28+

**Dependency Depth**: 4 layers
- Layer 3: User interfaces
- Layer 2: REST API + orchestration
- Layer 1: Model inference
- Layer 0: Data sources

**Integration Quality**: ✅ PRODUCTION READY
- Type safety: Full (Pydantic models)
- Error handling: Comprehensive
- Documentation: Complete
- Testing: Verified

**Status**: 🚀 Ready for deployment
