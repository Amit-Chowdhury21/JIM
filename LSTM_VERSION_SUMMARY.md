Here is the updated architecture table reflecting the new dual-model setup currently running in your system:

| Model | PyTorch File Used | Inference Strategy applied to the model |
| :--- | :--- | :--- |
| **v1.0 (Raw LSTM)** | `lstm_cnn_attention_1m.pt` *(Phase 1 Baseline)* | Runs a standard, single forward pass (`predict`) and trades raw probabilities. Serves as the historical benchmark. |
| **v2.0 (Ensemble)** | `lstm_v2.0_triple_barrier.pt` *(Phase 2 Upgraded)* | Runs a single forward pass on the new model weights, but uses the ensemble logic (Wavelet + HMM) to gate trades. |
| **v2.1 (Execution Engine)** | `lstm_v2.0_triple_barrier.pt` *(Phase 2 Upgraded)* | Runs **30 independent passes** via Monte Carlo Dropout (`predict_mcdropout`) to calculate epistemic uncertainty, plus base gating rules. |
| **v2.3 (Alpha Engine)** | `lstm_v2.0_triple_barrier.pt` *(Phase 2 Upgraded)* | Runs **30 independent passes** via Monte Carlo Dropout, plus the new `AlphaSignalPolicy` (Macro events, Regime thresholds, Target Volatility sizing). |