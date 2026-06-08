# TFT_Pro_Max Master Blueprint

## Overview

TFT_Pro_Max is the production-grade rebuild of TFT_Pro for gold trading on 5-minute bars, designed to operate as a multi-horizon probabilistic decision engine inside an ensemble with HMM, Wavelet, and CNN-LSTM models.[cite:76][cite:80] Its purpose is not only to forecast future price paths, but to convert those forecasts into calibrated decisions about direction, position size, holding horizon, and trade rejection under uncertainty.[cite:76][cite:80]

The core design principle is that TFT_Pro_Max should complement rather than duplicate the existing stack. CNN-LSTM v2.2 remains the short-horizon pattern classifier, HMM remains the regime engine, Wavelet remains the structure/noise decomposition layer, and TFT_Pro_Max becomes the probabilistic allocator that determines whether the opportunity is worth trading and on which horizon.[cite:76][cite:80]

## Mandate

TFT_Pro_Max will be judged by live-trading usefulness rather than by forecasting elegance alone. It should only be promoted if it improves the ensemble’s net Sharpe, trade selection quality, uncertainty handling, and drawdown control after realistic costs and purged walk-forward validation.[cite:131][cite:164]

The model mandate is fourfold:
- Forecast multi-horizon return distributions rather than single point estimates.[cite:76][cite:80]
- Produce calibrated uncertainty that can be used in execution and sizing.[cite:76][cite:163]
- Add orthogonal value to the CNN-LSTM, HMM, and Wavelet stack.[cite:76]
- Remain interpretable enough to diagnose feature importance shifts, regime behavior, and calibration drift over time.[cite:76][cite:86]

## System role

TFT_Pro_Max should sit between prediction and execution. It consumes observed past features, known future covariates, and static state descriptors, then emits a forecast surface that is translated into trade/no-trade, size, and horizon decisions.[cite:76][cite:80]

Its role inside the ensemble is shown below.

| Model | Primary job | Secondary job |
|---|---|---|
| HMM | Regime classification | Exposure and threshold gating |
| Wavelet | Trend/noise decomposition | Structure quality filtering |
| CNN-LSTM v2.2 | Local pattern classification | High-conviction directional signal |
| TFT_Pro_Max | Multi-horizon probabilistic allocation | Uncertainty-aware sizing and horizon choice |

## Architecture

TFT_Pro_Max should retain the canonical Temporal Fusion Transformer structure because the model was specifically designed for interpretable multi-horizon forecasting with mixed input types.[cite:76][cite:80] The production architecture should include Variable Selection Networks for adaptive feature weighting, Gated Residual Networks for stable nonlinear transformation, recurrent temporal encoding for local sequence structure, and interpretable self-attention for longer-range dependency discovery.[cite:76][cite:86]

The target architecture should include these building blocks:
- Variable Selection Networks on each input stream.
- Gated Residual Networks throughout the static enrichment and temporal fusion pipeline.
- Sequence encoder for local dynamics, with encoder-length experiments rather than a single fixed context window.[cite:76]
- Interpretable multi-head attention for long-range temporal dependency attribution.[cite:76][cite:80]
- Multi-horizon quantile heads for all forecast horizons.[cite:76]

The architecture roadmap should prioritize disciplined ablation over brute-force scaling. The most important experiments are encoder length, hidden size, dropout, number of heads, quantile set, decoder horizon design, and whether HMM/Wavelet inputs improve out-of-sample calibration and net PnL rather than only in-sample loss.[cite:76][cite:163]

## Data schema

TFT_Pro_Max should be trained on the full 18-year local dataset, resampled and aligned to a canonical 5-minute clock before any feature generation. Long history matters because gold responds differently to uncertainty shocks, rate regimes, and dollar strength across macro eras, including crisis and zero-lower-bound environments.[cite:159][cite:76]

The data schema must separate all inputs into three classes, because TFT performance depends on clean treatment of observed past inputs, known future inputs, and static covariates.[cite:76][cite:80]

### Observed past inputs

Observed past inputs should contain state variables known only up to the current bar. These should include validated CNN-LSTM-style alpha drivers, macro state variables, volatility state, and cross-model summaries that do not leak future information.[cite:76][cite:80]

Recommended observed past blocks:
- Path structure: intrabar efficiency, wick ratios, candle range shape, distance to prior-day high and low.
- Price action: return_1, dist_ema21, fast RSI, normalized MACD histogram.
- Volatility: vol_10, atr_ratio, realized volatility regime, GVZ level and change.
- Macro and relative value: DXY level and momentum, US10Y, gold-silver ratio deviation, silver momentum, real-yield proxy if available.[cite:159]
- Execution context: spread proxy, liquidity proxy, turnover proxy, and volume-quality flags.
- Cross-model context: HMM regime probabilities, Wavelet trend slope, Wavelet noise summaries, provided they are generated strictly causally.

### Known future inputs

Known future inputs are the strongest natural advantage of TFT_Pro_Max because calendars and scheduled events are available before the forecast is made.[cite:76][cite:80] These covariates should include cyclical time features and explicit event-aware context that a plain classifier cannot exploit as elegantly.

Recommended known future blocks:
- Hour sine and cosine.
- Day-of-week sine and cosine.
- Month or quarter cyclicality if useful.
- Session-transition markers.
- Minutes to scheduled macro releases such as CPI, NFP, FOMC, and Treasury events.
- Pre-event and post-event windows.
- Holiday and rollover effects.

### Static covariates

Static covariates should be sparse and genuinely slow-moving. In financial time series, static features are often overused, so TFT_Pro_Max should only include static descriptors that improve calibration or decision quality across folds.[cite:76][cite:86]

Good candidates include:
- Structural regime family buckets derived from long-horizon HMM logic.
- Instrument metadata if multiple gold proxies are ever added.
- Session archetype or market-environment bucket only if it remains stable over the whole encoded sample.

## Feature governance

Every feature must pass a three-part test before inclusion: causal availability, incremental value, and live robustness. A feature that improves in-sample loss but degrades fold stability or calibration should be rejected.[cite:131][cite:164]

The feature review process should explicitly classify variables as:
- Keep: strong, stable, causal, and useful after costs.
- Test: plausible but not yet validated across regimes.
- Reject: redundant, noisy, or leakage-prone.

Special attention should be paid to broker-dependent volume proxies. If the gold feed does not represent meaningful centralized volume, volume-derived features should remain conditional rather than core.[cite:164]

## Targets

TFT_Pro_Max should not be trained only on simple future return medians. It should forecast a multi-horizon tradeable surface that can drive execution decisions directly.[cite:76][cite:80]

The primary target family should include:
- Return quantiles at h3, h6, h12, and h24.
- Maximum favorable excursion proxies over each horizon.
- Maximum adverse excursion proxies over each horizon.
- Optional net-of-cost move expectation if execution realism can be estimated reliably.

A recommended quantile set is q10, q25, q50, q75, and q90, because it gives both coarse and medium-granularity risk information for execution translation.[cite:76][cite:163] The model should be able to answer five practical questions per bar: what direction is most likely, how large is the expected move, how uncertain is the path, which horizon has the best expected asymmetry, and whether the trade survives costs.

## Label and horizon design

The forecast horizons should be chosen to map to real holding periods used by the live execution stack. A short-horizon ladder such as 3, 6, 12, and 24 bars gives a useful spread from tactical to intraday swing logic without becoming too diffuse.[cite:76]

Targets should be designed to reduce redundancy between neighboring horizons. If h3 and h6 produce near-identical actions while h24 is too slow to matter, the horizon grid should be revised rather than preserved for symmetry.[cite:131]

## Training policy

Training should use the full 18-year dataset with strict chronological handling and no shuffled leakage. The objective function should prioritize quantile quality first, but model ranking should incorporate calibration, regime robustness, and economic value rather than pinball loss alone.[cite:76][cite:131]

Training policy should include:
- Fixed random seeds for reproducibility.
- Standardized experiment configs and versioned checkpoints.
- Controlled architecture search over encoder length, hidden size, dropout, head count, and static covariate use.[cite:76]
- Strong regularization and early stopping policies.
- Per-fold artifact retention for later comparison.

## Validation plan

Validation must use purged, embargoed, expanding walk-forward testing, because standard chronological splits are not enough when horizons overlap and labels have economic spillover.[cite:131][cite:164][cite:165] The fold plan should span multiple macro eras so the model is tested in tightening cycles, crises, uncertainty spikes, calm periods, and post-shock normalization phases.[cite:159]

A professional validation framework should include:
- 5 to 8 expanding folds across the 18-year sample.[cite:131]
- Purge gaps sized to longest horizon and relevant lookback dependencies.[cite:165]
- Embargo windows around test boundaries when needed.[cite:164]
- A dedicated calibration slice within each fold.
- Fold-level evaluation before any pooled summary.

Headline performance should never rely on a single recent period. A candidate that excels only in one macro environment should not be promoted.[cite:131][cite:164]

## Evaluation metrics

TFT_Pro_Max should be evaluated on both forecast quality and economic utility. Pure forecasting metrics are necessary but insufficient because a probabilistic trading model can score well on loss while adding no live alpha.[cite:76][cite:163]

The mandatory scorecard should include:
- Pinball loss by quantile and horizon.[cite:76]
- Interval coverage for q10/q90 or equivalent outer bands.[cite:163]
- Calibration error by regime and by horizon.
- Directional hit rate for q50.
- Net PnL after dynamic costs.
- Sharpe ratio, max drawdown, turnover, and profit factor.
- Incremental ensemble contribution relative to CNN-LSTM v2.2 plus HMM and Wavelet.

## Calibration plan

Quantile calibration is a first-class requirement, not a reporting extra. If uncertainty bands are systematically too narrow or too wide, then position sizing, trade rejection, and horizon choice will all be wrong even when the median forecast looks reasonable.[cite:76][cite:163]

Calibration review should include:
- Coverage stability across folds.
- Coverage stability across growth, normal, and crisis regimes.
- Interval-width response during macro-event windows and volatility spikes.
- Reliability diagnostics for each horizon.

The model should not graduate unless its uncertainty expands during unstable conditions and contracts during quieter conditions in a consistent, economically useful way.[cite:159][cite:163]

## Decision translation layer

The production decision layer should transform forecast distributions into tradeable actions. TFT_Pro_Max should not trigger trades from q50 direction alone; instead, it should use the entire quantile surface, current costs, and regime state to evaluate whether the expected opportunity is wide enough to trade.[cite:76][cite:80]

The translation layer should answer:
- LONG, SHORT, or HOLD.
- Which horizon is preferred.
- Whether forecast asymmetry survives current costs.
- How position size should scale with confidence and regime.
- Whether an event window or volatility shock should force caution.

The final action policy should combine:
- Median direction.
- Distance between median and tail quantiles.
- Width of the uncertainty band.
- Current cost regime.
- HMM regime and Wavelet noise state.
- Ensemble conflict resolution with CNN-LSTM.

## Ensemble policy

TFT_Pro_Max should be treated as an allocator and filter inside the broader gold stack, not as a solitary forecaster. Its highest-value use case is likely to confirm or reject CNN-LSTM signals, compress or expand size, and select the horizon with the most favorable risk-reward structure.[cite:76][cite:80]

A good ensemble policy should specify:
- When TFT confirms CNN-LSTM and size can be increased.
- When TFT uncertainty overrides a low-quality CNN-LSTM signal.
- How HMM regimes adjust trade thresholds.
- How Wavelet noise suppresses otherwise attractive forecasts.
- How disagreement between modules is resolved without ad hoc discretion.

## Cost model policy

All model evaluation and promotion must be based on realistic net returns after costs. Static cost assumptions may be used in early screening, but production qualification should apply dynamic spread and slippage assumptions by volatility regime, event state, and session liquidity.[cite:159][cite:166]

The cost policy should include:
- Baseline spread, slippage, and commission schedule.
- Stress cost schedule during macro releases and high-volatility conditions.
- Capacity constraints for lower-liquidity windows.
- Sensitivity tables showing whether the edge survives worse-than-expected execution.

## Promotion gates

Promotion should follow a formal gate structure so TFT_Pro_Max is advanced only when it is demonstrably useful.

| Gate | Requirement |
|---|---|
| A | Data integrity, timestamp alignment, and leak-free schema approved. |
| B | Target design and architecture shortlist validated. |
| C | Walk-forward robustness across macro regimes proven.[cite:131][cite:164] |
| D | Quantile calibration and decision translation quality proven.[cite:76][cite:163] |
| E | Incremental net ensemble value after costs demonstrated. |
| F | Paper-trading consistency and operational readiness confirmed. |
| G | Controlled live micro-size launch approved. |

A single strong backtest is not enough to pass a gate. Each gate should require repeatable evidence across folds, costs, and operational checks.[cite:164]

## Failure criteria

TFT_Pro_Max should be rejected or sent back for redesign if any of the following occur:
- Strong in-sample loss improvements fail to translate into walk-forward economic value.[cite:164]
- Quantile intervals are miscalibrated or unstable across regimes.[cite:163]
- The model duplicates CNN-LSTM behavior without adding orthogonal signal value.
- Ensemble complexity rises but drawdown control and trade quality do not improve.
- Live paper-trading behavior diverges sharply from walk-forward expectations.

## Live deployment policy

Deployment should occur in three steps: shadow mode, paper trading, and controlled capital launch. This staged process is essential because probabilistic models can fail through calibration drift even when direction accuracy appears acceptable.[cite:131][cite:163]

### Shadow mode
- Run forecasts live with no capital impact.
- Monitor quantile coverage, interval width, and ensemble disagreement.
- Compare live feature distributions against training-fold distributions.

### Paper trading
- Activate the full decision translation layer.
- Evaluate regime-specific performance and trade rejection quality.
- Track slippage realism and event-window behavior.

### Controlled capital launch
- Launch with micro-size only.
- Use hard risk caps and rollback triggers.
- Escalate capital only after stable live behavior over a predefined observation window.

## Drift monitoring

The production system should continuously track whether TFT_Pro_Max remains calibrated and useful. A probabilistic model can decay silently, so live monitoring must focus on forecast reliability as much as on PnL.[cite:131][cite:163]

Mandatory monitoring should cover:
- Live quantile coverage drift.
- Regime-specific hit rate and PnL drift.
- Feature-distribution shift.
- Rising disagreement with CNN-LSTM, HMM, or Wavelet.
- Cost drift versus backtest assumptions.

## Governance and versioning

TFT_Pro_Max should operate under strict model governance. Every promoted version should have a frozen feature spec, data spec, architecture config, validation report, calibration report, and live deployment decision memo.[cite:131][cite:164]

Version governance should include:
- Immutable model registry entries.
- Fold-level experiment records.
- Retraining and recalibration schedule.
- Retirement policy when ensemble contribution falls below threshold.

## Build sequence

The recommended implementation order is:
1. Approve mandate and promotion standards.
2. Rebuild and audit the 18-year master dataset.
3. Finalize feature taxonomy and leakage review.
4. Finalize multi-horizon target design.
5. Build event-aware known-future covariates.
6. Run architecture search and shortlist candidates.
7. Execute purged walk-forward training and validation.[cite:131][cite:165]
8. Review calibration and economic scorecards.[cite:163]
9. Build decision translation and ensemble integration.
10. Run paper trading and drift monitoring.
11. Approve controlled live launch.

## Final design standard

TFT_Pro_Max should be considered successful only if it becomes the ensemble’s probabilistic brain for direction filtering, horizon selection, and conviction sizing. It does not need to beat CNN-LSTM v2.2 on raw short-horizon directional hit rate to be the better strategic model; it needs to improve trade selection, uncertainty handling, calibration, and net risk-adjusted returns after costs.[cite:76][cite:80]

The final operating philosophy is simple: CNN-LSTM should predict the move, HMM should classify the environment, Wavelet should measure structure quality, and TFT_Pro_Max should decide whether the opportunity is large, reliable, and cheap enough to trade.[cite:76][cite:80]
