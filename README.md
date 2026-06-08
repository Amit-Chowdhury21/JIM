# 🏆 Mini-Medallion: Gold (XAU) Trading Engine

> *"We are right 50.75% of the time, but we are 100% right 50.75% of the time."*
> 
> — Inspired by Jim Simons' Renaissance Technologies

**Mini-Medallion** is a production-grade, highly autonomous algorithmic trading engine designed specifically to trade Gold (XAU/USD). Built on a "Radical Empiricism" philosophy (trusting data over narrative and finding non-obvious invariants), the system utilizes a multi-model machine learning ensemble architecture operating 24/7.

---

## 🏗️ Architecture Overview

The system runs on a 7-model machine learning ensemble aggregated via a Stacking Meta-Learner to produce `LONG/SHORT/HOLD` signals. 

### Core Components
- **Data Ingestion & Features:** Automates fetching of tick data, macro indicators (DXY, US10Y, Silver), and sentiment analysis.
- **Ensemble Engine:** Incorporates 7 diverse models including **WaveletPro** (frequency-domain analysis), **HMM Pro** (temporal regime detection), LSTM, Temporal Fusion Transformers (TFT), Genetic Algorithms, and NLP.
- **Risk Management (The "Shield"):** Validates trades using a Meta-Label Critic (must be > 65% confident), sizes positions via Dynamic Kelly / Half-Kelly Criterion, and enforces circuit breakers (e.g., Max Drawdown 10%). GPU Monte Carlo VaR simulations run continuously.
- **Execution:** Paper Trading Engine with realistic slippage/commission models, and a low-latency C++ order router slated for Interactive Brokers.

### Advanced Modeling
1. **WaveletPro**: Uses a 6-level DWT (db4 wavelet) for decomposition, extracting a Wavelet Oscillator for mid-term cycles. Applies soft thresholding for denoising and Morlet CWT for volatility. Generates 36 features with ~10-15ms inference latency.
2. **HMM Pro**: GMMHMM tracking 4 market regimes (Bullish, Neutral, Bearish, Reversal). Integrates macro data with automatic feature dimension padding. ~19ms inference latency.

---

## 🛠️ Technology Stack

- **Compute & Deep Learning:** PyTorch (CUDA), NVIDIA RAPIDS (cuDF, cuML), CuPy, cuSignal.
- **Data Storage & Pipeline:** QuestDB (Tick Data), Redis (Feature Store), MinIO (Data Lake), Kafka.
- **MLOps & Monitoring:** MLflow (Model Registry), Prometheus, Grafana.
- **Execution:** Python (Research/Paper Trading), C++ (Live Order Router).
- **Deployment:** Docker Compose stack.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose
- NVIDIA GPU (Recommended) with CUDA toolkit installed

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd jim_new

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
```

### 3. Launching the Infrastructure
Copy `.env.example` to `.env` and populate any required secret values. Then start the infrastructure stack:
```bash
docker compose up -d
```
You can run the infrastructure health check to verify the services:
```bash
python scripts/check_infrastructure.py
```

### 4. Running the Pipeline
To run a demo of the current trading pipeline:
```bash
python main.py --mode demo
```

---

Because this is an aggressive high-frequency scalping engine running on 1-minute (1m) data, the system doesn't use fixed time limits. Instead, it dynamically calculates the exit time based on market volatility.

However, based on the backend architecture and mathematical parameters, the average trade duration is typically between 3 to 15 minutes.

Here is exactly how the system decides when to exit:

Prediction Horizons: The TFT_Pro AI model is explicitly trained to forecast exactly 3 minutes, 12 minutes, and 24 minutes into the future (h3, h12, h24). It doesn't care about anything beyond 24 minutes.
Minimum Holding Period: The Reinforcement Learning Execution Agent (rl_execution_agent.py) calculates a min_holding_bars constraint. Depending on how confident the ensemble is, it forces the system to hold the trade for a strict minimum of 1 to 5 minutes to prevent getting chopped up by immediate fake-outs.
Trailing Stop Phases: After the minimum hold time, the exit is entirely dictated by a dynamic trailing stop.
If the trade goes into profit by >0.5x ATR, it moves the stop to breakeven.
If it goes into profit by >1.0x ATR, it trails the price aggressively to lock in profits.
As soon as the live price hits that dynamic trailing line, the engine closes the trade instantly. In highly volatile regimes (CRISIS), trades might last just 1-2 minutes. In trending regimes (GROWTH), they might ride the trailing stop for 15-20 minutes.

---

## 📚 Documentation
All extensive project documentation has been consolidated and can be explored via the [Graphify Report](./graphify-out/GRAPH_REPORT.md). The `GRAPH_REPORT.md` file contains a detailed topology of the codebase and includes a full system architecture and technical consolidation summary at the end of the file. 

---
You should stop thinking in terms of single models and start building a multi-layer research, portfolio, execution, and risk machine. Renaissance-style trading was not one HMM or one Wavelet model; it was a data-first, ensemble, market-neutral, continuously validated system that exploited many small edges with strict execution and risk discipline.

What changes next
Your HMM and Wavelet modules are only the signal-discovery layer, not the full strategy. Renaissance’s playbook emphasized repeatable small edges, broad data coverage, market-neutral or risk-balanced positioning, automation, ongoing validation, and infrastructure strong enough to execute without emotion or downtime.

So the next step is to turn your gold system into a stack of cooperating layers:
| Layer           | What you already have                | What you need next                                                                      |
| --------------- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| Signal research | HMM, Wavelet                         | Add orthogonal alpha families, meta-labeling, ensemble weighting journals.stmjournals+1 |
| Data engine     | Likely price-focused                 | Add macro, microstructure, cross-asset, event, and regime data journals.stmjournals+2   |
| Portfolio logic | Likely single-instrument directional | Add spread, hedge, regime-conditioned sizing, neutralization journals.stmjournals+1     |
| Execution       | Probably broker order logic          | Add slippage model, order slicing, queue logic, kill switches journals.stmjournals+1    |
| Risk            | Basic stop-loss likely               | Add portfolio risk, exposure caps, model decay monitoring journals.stmjournals+1        |
| Ops             | Local/live runtime                   | Add production monitoring, failover, rollback, audit trail github                       |




## ⚠️ Disclaimer
**For Research Purposes Only.** This software is provided as-is, and the creators are not responsible for any financial losses incurred from using this trading engine. Always backtest strategies thoroughly and use paper trading before committing live capital.
