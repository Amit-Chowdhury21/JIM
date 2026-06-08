// Mock data aligned with JIM backend — Updated for Phases 1-6

function generatePriceData() {
  const data = [];
  let price = 1820;
  const start = new Date('2023-01-01');
  for (let i = 0; i < 500; i++) {
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    if (date.getDay() === 0 || date.getDay() === 6) continue;
    const change = (Math.random() - 0.48) * 18;
    price = Math.max(1600, Math.min(2500, price + change));
    data.push({
      date: date.toISOString().split('T')[0],
      open: +(price + (Math.random() - 0.5) * 6).toFixed(2),
      high: +(price + Math.random() * 12).toFixed(2),
      low: +(price - Math.random() * 12).toFixed(2),
      close: +price.toFixed(2),
      volume: Math.floor(Math.random() * 80000 + 20000),
      sma20: +(price - (Math.random() - 0.5) * 20).toFixed(2),
      sma50: +(price - (Math.random() - 0.5) * 40).toFixed(2),
      ema12: +(price - (Math.random() - 0.5) * 10).toFixed(2),
      rsi: +(Math.random() * 40 + 30).toFixed(1),
      volatility: +(Math.random() * 8 + 2).toFixed(2),
      regime: Math.random() > 0.6 ? 'GROWTH' : Math.random() > 0.4 ? 'NORMAL' : 'CRISIS',
    });
  }
  return data;
}

export const priceData = generatePriceData();
export const recentPrices = priceData.slice(-90);
export const latestPrice = priceData[priceData.length - 1];

export const portfolioKPIs = {
  portfolioValue: 127453.82, dailyPnL: 1243.56, dailyPnLPct: 0.98,
  totalReturn: 27.45, sharpeRatio: 2.34, winRate: 54.2, maxDrawdown: -6.8,
  totalTrades: 1247, openPositions: 3,
  goldPrice: latestPrice.close,
  goldChange: ((latestPrice.close - priceData[priceData.length - 2].close) / priceData[priceData.length - 2].close * 100),
  profitFactor: 1.67, commissionPerTrade: 2.50, slippageBps: 1.0,
};

export const regimeData = {
  current: 'GROWTH', confidence: 0.73,
  probabilities: { GROWTH: 0.73, NORMAL: 0.19, CRISIS: 0.08 },
  history: priceData.slice(-30).map(d => ({
    date: d.date,
    GROWTH: +(Math.random() * 0.4 + 0.4).toFixed(2),
    NORMAL: +(Math.random() * 0.3 + 0.1).toFixed(2),
    CRISIS: +(Math.random() * 0.2 + 0.05).toFixed(2),
  })),
};

export const riskMetrics = {
  kellyFraction: 0.0312, kellyPositionSize: 3120, halfKelly: 1560, quarterKelly: 780,
  kellyConfig: { fraction: 0.5, maxPositionPct: 0.05, crisisFraction: 0.25 },
  dailyVaR95: -1842.50, dailyVaR99: -2891.30, cvar99: -3456.78,
  currentDrawdown: -2.3, maxDrawdownLimit: 10, dailyLossLimit: 2.0, currentDailyLoss: -0.45,
  circuitBreakers: {
    dailyLoss: { status: 'ok', value: -0.45, limit: -2.0 },
    drawdown: { status: 'ok', value: -2.3, limit: -10.0 },
    drawdownReduce: { status: 'ok', value: -2.3, limit: -5.0 },
    modelDisagreement: { status: 'warning', value: 0.62, limit: 0.70 },
    latency: { status: 'ok', value: 45, limit: 500 },
  },
  monteCarlo: { nSimulations: 100000, runFrequency: 'hourly', varConfidence: 0.95, cvarConfidence: 0.99 },
  monteCarloResults: Array.from({ length: 50 }, (_, i) => ({
    scenario: i, return: +(Math.random() * 60 - 15).toFixed(2),
    maxDD: +(Math.random() * -20).toFixed(2), sharpe: +(Math.random() * 3 + 0.5).toFixed(2),
  })),
};

// Advanced risk metrics (Phase 4)
export const advancedRiskMetrics = {
  omegaRatio: 1.87, ulcerIndex: 3.42, conditionalVaR: -4.82,
  expectedShortfall: -0.0234, tailRatio: 1.15, recoveryFactor: 4.04,
  stressAdjustedSharpe: 2.18,
  stressTests: [
    { name: 'USD Flash Rally', impact: -4462.50, pctImpact: -3.5 },
    { name: 'Liquidity Crisis', impact: -2550.00, pctImpact: -2.0 },
    { name: 'Flash Crash', impact: -6375.00, pctImpact: -5.0 },
    { name: 'Rate Surprise', impact: -1912.50, pctImpact: -1.5 },
    { name: 'Geopolitical Event', impact: 5737.50, pctImpact: +4.5 },
  ],
};

// Meta-Labeler (Phase 4)
export const metaLabeler = {
  threshold: 0.65, isTrainedOn: 1247, trainAccuracy: 0.72, valAccuracy: 0.68,
  recentDecisions: [
    { signal: 'LONG', traderConf: 0.82, criticConf: 0.78, execute: true, regime: 'GROWTH' },
    { signal: 'HOLD', traderConf: 0.65, criticConf: 0.52, execute: false, regime: 'NORMAL' },
    { signal: 'LONG', traderConf: 0.71, criticConf: 0.69, execute: true, regime: 'GROWTH' },
    { signal: 'SHORT', traderConf: 0.58, criticConf: 0.41, execute: false, regime: 'GROWTH' },
    { signal: 'LONG', traderConf: 0.77, criticConf: 0.74, execute: true, regime: 'NORMAL' },
  ],
  featureImportance: [
    { name: 'trader_confidence', value: 0.22 }, { name: 'regime_prob', value: 0.18 },
    { name: 'recent_accuracy', value: 0.15 }, { name: 'volatility', value: 0.12 },
    { name: 'profit_factor', value: 0.10 }, { name: 'liquidity', value: 0.08 },
  ],
};

// GPU VaR (Phase 4)
export const gpuVaR = {
  var95: 1842.50, var99: 2891.30, cvar95: 2234.67, cvar99: 3456.78,
  maxDrawdownPct: 6.8, scenariosRan: 100000, computeTimeMs: 23.4, usedGPU: false,
};

// Position Manager (Phase 4)
export const positionManager = {
  maxPositions: 5, trailingStopPct: 2.0, profitTargetPct: 5.0, timeStopHours: 24,
  openPositions: [
    { id: 'POS_00045', direction: 1, size: 2, entryPrice: 2335.80, currentPnl: 11.40, pnlPct: 0.49, status: 'OPEN' },
    { id: 'POS_00044', direction: 1, size: 1, entryPrice: 2328.50, currentPnl: 18.30, pnlPct: 0.79, status: 'OPEN' },
    { id: 'POS_00043', direction: -1, size: 2, entryPrice: 2352.80, currentPnl: 14.40, pnlPct: 0.31, status: 'OPEN' },
  ],
  stats: { totalClosed: 1244, winRate: 0.542, avgPnl: 8.45, totalPnl: 10511.80, profitFactor: 1.67 },
};

export const modelMetrics = {
  wavelet_pro: {
    noiseRemoved: 32.6, bands: 5, snrImprovement: 9.0, snrDb: 9.0, family: 'db4',
    status: 'production', lastTrained: '2026-05-20T12:00',
    architecture: 'DWT → BayesShrink → IDWT → Trend Analysis',
    config: {
      family: 'db4', levels: 5, thresholdMethod: 'bayesshrink', thresholdMode: 'soft',
      denoiseRemoveLevels: [1, 2], trendSlopeWindow: 5, energyWindow: 20, minSamples: 64,
      retrainFrequency: 'hourly',
    },
    capabilities: ['Multi-wavelet family (db4/sym8/coif5)', 'BayesShrink / VisuShrink / SURE thresholding',
      'Multi-scale frequency band analysis', 'Wavelet coherence (cross-asset)', 'Energy ratio trend strength'],
    energyRatios: { trend: 0.42, slowCycle: 0.18, mediumCycle: 0.12, fastCycle: 0.08, noise: 0.20 },
  },
  hmm_pro: {
    accuracy: 0.681, lastTrained: '2026-05-20T08:00', regimeChanges: 14, logLikelihood: -234.5,
    status: 'production',
    architecture: '7-Feature Observation → GaussianHMM(3) → Vol-Ordered Regimes → Transition Matrix',
    config: {
      nRegimes: 3, covarianceType: 'diag', nIter: 1000, retrainFrequency: 'daily',
      minRegimeDuration: 5, regimePersistenceWeight: 0.3, volWindows: [10, 20, 50],
    },
    capabilities: ['Multi-feature observations (returns, vol, range, momentum, accel)',
      'Volatility-ordered regime labeling (Growth/Normal/Crisis)', 'Transition probability analysis',
      'Regime duration modeling', 'Persistence scoring (anti-whipsaw)', 'Regime-specific Kelly fractions'],
    regimeStrategies: {
      GROWTH: { kelly: 0.65, approach: 'Mean Reversion', bias: 'LONG' },
      NORMAL: { kelly: 0.50, approach: 'Balanced', bias: 'NEUTRAL' },
      CRISIS: { kelly: 0.25, approach: 'Trend Following', bias: 'DEFENSIVE' },
    },
    transitionMatrix: [[0.92, 0.06, 0.02], [0.08, 0.85, 0.07], [0.03, 0.12, 0.85]],
  },
  lstm: {
    valLoss: 0.0023, epochs: 87, lr: 0.001, hiddenSize: 128,
    status: 'production', lastTrained: '2026-05-19T22:00',
    architecture: 'BiLSTM(128×3) → Temporal Self-Attention → FC(256→128→64→3)',
    config: {
      hiddenSize: 128, numLayers: 3, bidirectional: true, dropout: 0.2,
      seqLength: 100, batchSize: 256, epochs: 100, patience: 10, gradClip: 1.0,
      retrainFrequency: 'weekly',
    },
    capabilities: ['Bidirectional LSTM (captures forward + backward patterns)',
      'Temporal self-attention (identifies key timesteps)', '15 input features per timestep',
      'Class-weighted loss (handles imbalanced labels)', 'Gradient clipping + LR scheduling',
      'Statistical proxy fallback (no PyTorch required)'],
    features: ['returns', 'volatility_10', 'volatility_20', 'momentum_5', 'momentum_10',
      'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
      'volume_zscore', 'return_acceleration', 'dxy_momentum', 'yield_momentum', 'spread_proxy'],
    parameters: 847363,
  },
  tft_pro: {
    valLoss: 0.0019, epochs: 65, attentionHeads: 4,
    status: 'production', lastTrained: '2026-05-19T22:00',
    architecture: 'Variable Selection → GRN → Multi-Head Attention(4h×2L) → Quantile Output',
    config: {
      dModel: 64, attentionHeads: 4, nLayers: 2, dropout: 0.1,
      seqLength: 60, batchSize: 128, lr: 0.0005, epochs: 80, patience: 8,
      forecastHorizons: [1, 5, 10, 20], quantiles: [0.1, 0.25, 0.5, 0.75, 0.9],
      retrainFrequency: 'weekly',
    },
    capabilities: ['Variable Selection Networks (learned feature importance)',
      'Gated Residual Networks (non-linear processing)', 'Multi-horizon forecasting (1/5/10/20 bars)',
      'Quantile regression (full predictive distribution)', 'Interpretable attention weights',
      'Statistical proxy fallback (multi-scale RSI/ATR)'],
    variableImportance: { returns: 0.18, vol_10: 0.14, rsi_14: 0.12, momentum_5: 0.11,
      bb_pos: 0.09, atr_mom: 0.08, vol_20: 0.07, rsi_7: 0.06, dxy: 0.05, yield: 0.05, rsi_21: 0.03, mom_20: 0.02 },
  },

  ensemble: {
    sharpe: 2.56, winRate: 0.572, profitFactor: 1.81, avgReturn: 0.15,
    status: 'production', lastTrained: '2026-05-20T08:00',
    architecture: 'Model Signals → Feature Vector → XGBoost Meta-Learner → Calibrated Output',
    config: {
      method: 'stacking', metaLearner: 'xgboost', nEstimators: 100, maxDepth: 5,
      disagreementPenalty: 0.3, confidenceThreshold: 0.25, ewmaAlpha: 0.1,
    },
    capabilities: ['XGBoost/RandomForest meta-learner (Level 1)', 'Regime-conditioned dynamic weighting',
      'Performance-adaptive EWMA accuracy tracking', 'Model disagreement detection + confidence penalty',
      'Signed confidence feature engineering', 'Macro overlay integration'],
    regimeWeights: {
      GROWTH: { wavelet_pro: 0.10, hmm_pro: 0.10, lstm: 0.20, tft_pro: 0.25,  },
      NORMAL: { wavelet_pro: 0.15, hmm_pro: 0.15, lstm: 0.15, tft_pro: 0.20,  },
      CRISIS: { wavelet_pro: 0.25, hmm_pro: 0.20, lstm: 0.05, tft_pro: 0.05,  },
    },
  },
};

export const signals = [
  { id: 1, time: '14:32:05', type: 'LONG', source: 'HMM+Wavelet', confidence: 0.82, price: 2341.50, status: 'active' },
  { id: 2, time: '13:45:22', type: 'HOLD', source: 'Ensemble', confidence: 0.65, price: 2338.20, status: 'active' },
  { id: 3, time: '12:10:18', type: 'LONG', source: 'HMM', confidence: 0.71, price: 2335.80, status: 'filled' },
  { id: 4, time: '11:05:44', type: 'SHORT', source: 'Genetic', confidence: 0.58, price: 2340.10, status: 'expired' },
  { id: 5, time: '09:30:00', type: 'LONG', source: 'LSTM', confidence: 0.77, price: 2332.60, status: 'filled' },
];

export const tradeHistory = [
  { id: 'T-1247', date: '2026-05-14', side: 'BUY', qty: 2, entry: 2335.80, exit: 2341.50, pnl: 11.40, pnlPct: 0.49, model: 'HMM' },
  { id: 'T-1246', date: '2026-05-13', side: 'SELL', qty: 1, entry: 2348.20, exit: 2340.10, pnl: 8.10, pnlPct: 0.35, model: 'Ensemble' },
  { id: 'T-1245', date: '2026-05-13', side: 'BUY', qty: 3, entry: 2328.50, exit: 2342.30, pnl: 41.40, pnlPct: 0.59, model: 'Wavelet' },
  { id: 'T-1244', date: '2026-05-12', side: 'BUY', qty: 1, entry: 2320.00, exit: 2315.80, pnl: -4.20, pnlPct: -0.18, model: 'LSTM' },
  { id: 'T-1243', date: '2026-05-12', side: 'SELL', qty: 2, entry: 2345.60, exit: 2338.90, pnl: 13.40, pnlPct: 0.29, model: 'Genetic' },
  { id: 'T-1242', date: '2026-05-11', side: 'BUY', qty: 1, entry: 2310.20, exit: 2328.50, pnl: 18.30, pnlPct: 0.79, model: 'HMM' },
  { id: 'T-1241', date: '2026-05-11', side: 'SELL', qty: 2, entry: 2352.80, exit: 2345.60, pnl: 14.40, pnlPct: 0.31, model: 'Ensemble' },
  { id: 'T-1240', date: '2026-05-10', side: 'BUY', qty: 1, entry: 2305.00, exit: 2298.40, pnl: -6.60, pnlPct: -0.29, model: 'LSTM' },
];

export const equityCurve = Array.from({ length: 180 }, (_, i) => {
  const date = new Date('2025-11-01');
  date.setDate(date.getDate() + i);
  return {
    date: date.toISOString().split('T')[0],
    equity: +(100000 + i * 150 + Math.sin(i / 10) * 2000 + Math.random() * 1500).toFixed(2),
    benchmark: +(100000 + i * 80 + Math.random() * 800).toFixed(2),
  };
});

export const featureImportance = [
  { name: 'rsi_14', importance: 0.142 }, { name: 'macd_signal', importance: 0.128 },
  { name: 'wavelet_d1', importance: 0.115 }, { name: 'volatility_20', importance: 0.098 },
  { name: 'sma_dist_20', importance: 0.087 }, { name: 'corr_dxy_20', importance: 0.076 },
  { name: 'parkinson_vol_20', importance: 0.065 }, { name: 'regime_prob', importance: 0.058 },
  { name: 'ema_dist_50', importance: 0.052 }, { name: 'roc_50', importance: 0.045 },
  { name: 'position_in_range_20', importance: 0.041 }, { name: 'atr_pct_14', importance: 0.038 },
];

export const featureConfig = {
  lookbackWindows: [5, 10, 20, 50, 100, 200], totalFeatures: 200,
  categories: [
    { name: 'Returns', count: 24 }, { name: 'Volatility', count: 18 }, { name: 'Momentum', count: 42 },
    { name: 'Price Levels', count: 24 }, { name: 'Temporal', count: 6 }, { name: 'Cross-Asset', count: 32 },
    { name: 'Wavelet', count: 12 },
  ],
};

export const infraServices = [
  { name: 'QuestDB', port: '9000/9009', status: 'online', uptime: '99.9%', icon: '🗄️', bg: 'var(--blue-dim)', memory: '512 MB', detail: 'Time-series DB', version: '7.4.0' },
  { name: 'Redis', port: '6379', status: 'online', uptime: '99.9%', icon: '⚡', bg: 'var(--red-dim)', memory: '128 MB', detail: 'Feature Cache', version: '7.x' },
  { name: 'MinIO', port: '9100/9101', status: 'online', uptime: '99.8%', icon: '📦', bg: 'var(--orange-dim)', memory: '256 MB', detail: 'Object Storage (S3)', version: 'latest' },
  { name: 'MLflow', port: '5000', status: 'online', uptime: '99.7%', icon: '🧪', bg: 'var(--purple-dim)', memory: '384 MB', detail: 'Model Registry', version: '2.11.1' },
  { name: 'Prometheus', port: '9090', status: 'online', uptime: '99.9%', icon: '📊', bg: 'var(--green-dim)', memory: '192 MB', detail: 'Metrics (15s scrape)', version: '2.50.0' },
  { name: 'Grafana', port: '3000', status: 'online', uptime: '99.5%', icon: '📈', bg: 'var(--cyan)', memory: '256 MB', detail: 'Dashboards', version: '10.3.1' },
];

export const gpuInfo = {
  name: 'RTX 3050 Laptop', vram: '4 GB', cudaVersion: '12.1', driver: '592.27',
  utilization: 34, temperature: 62, computeCapability: '8.6',
  rapids: false, cupy: false, cusignal: false, pytorch: true, cudnn: true,
};

export const executionEngine = {
  status: 'connected', broker: 'ibkr', mode: 'paper', orderType: 'limit', algo: 'twap', maxSlippageBps: 3.0,
  ibkr: { host: '127.0.0.1', port: 7497, clientId: 1 },
  cppEngine: {
    compiled: true,
    components: [
      { name: 'OrderRouter', status: 'active', desc: 'Smart venue routing + order management' },
      { name: 'LatencyMonitor', status: 'active', desc: 'p50/p95/p99 percentile tracking' },
      { name: 'OrderBook', status: 'active', desc: 'L3 order book with bid/ask levels' },
    ],
    latency: { p50: 12, p95: 45, p99: 89, unit: 'µs' },
  },
  recentOrders: [
    { id: 'ORD-4521', time: '14:32:05', symbol: 'XAU/USD', side: 'BUY', qty: 2, type: 'limit', price: 2341.50, status: 'filled', latency: 23 },
    { id: 'ORD-4520', time: '13:45:22', symbol: 'XAU/USD', side: 'SELL', qty: 1, type: 'limit', price: 2338.20, status: 'filled', latency: 18 },
    { id: 'ORD-4519', time: '12:10:18', symbol: 'XAU/USD', side: 'BUY', qty: 3, type: 'limit', price: 2335.80, status: 'filled', latency: 31 },
    { id: 'ORD-4518', time: '11:05:44', symbol: 'XAU/USD', side: 'SELL', qty: 2, type: 'limit', price: 2340.10, status: 'cancelled', latency: 15 },
  ],
};

// Updated phase progress — Phases 1-7 COMPLETE
export const phaseProgress = [
  { phase: 1, name: 'Infrastructure & Compute', status: 'complete', progress: 100, weeks: '1-3' },
  { phase: 2, name: 'Data Acquisition & Pipeline', status: 'complete', progress: 100, weeks: '2-5' },
  { phase: 3, name: 'Mathematical Modeling', status: 'complete', progress: 100, weeks: '4-10' },
  { phase: 4, name: 'Risk Management & Meta-Label', status: 'complete', progress: 100, weeks: '8-12' },
  { phase: 5, name: 'Backtesting & Validation', status: 'complete', progress: 100, weeks: '10-14' },
  { phase: 6, name: 'Paper Trading & Deployment', status: 'complete', progress: 100, weeks: '14-18' },
  { phase: 7, name: 'Team Culture & Operations', status: 'complete', progress: 100, weeks: 'Ongoing' },
];

export const backtestConfig = {
  initialCapital: 100000, commissionPerTrade: 2.50, slippageBps: 1.0,
  walkForward: { trainYears: 3, testYears: 1 },
  targets: { minSharpe: 2.0, maxDrawdown: 0.10, minWinRate: 0.51, minProfitFactor: 1.5, dsrPvalue: 0.05 },
};

export const waveletBands = priceData.slice(-60).map(d => ({
  date: d.date, original: d.close,
  denoised: +(d.close + (Math.random() - 0.5) * 3).toFixed(2),
  d1: +(Math.random() * 4 - 2).toFixed(2), d2: +(Math.random() * 8 - 4).toFixed(2),
}));

// Backtesting results (Phase 5)
export const backtestResults = {
  models: [
    { name: 'Wavelet', sharpe: 2.12, sortino: 2.89, calmar: 3.21, winRate: 0.53, profitFactor: 1.58, maxDD: -5.4, totalReturn: 24.1, dsrPvalue: 0.032, dsrPass: true },
    { name: 'HMM', sharpe: 1.98, sortino: 2.67, calmar: 2.95, winRate: 0.51, profitFactor: 1.45, maxDD: -6.1, totalReturn: 21.8, dsrPvalue: 0.041, dsrPass: true },
    { name: 'LSTM', sharpe: 2.41, sortino: 3.12, calmar: 3.58, winRate: 0.55, profitFactor: 1.72, maxDD: -4.8, totalReturn: 28.3, dsrPvalue: 0.018, dsrPass: true },
    { name: 'TFT', sharpe: 2.28, sortino: 3.01, calmar: 3.42, winRate: 0.54, profitFactor: 1.65, maxDD: -5.1, totalReturn: 26.7, dsrPvalue: 0.024, dsrPass: true },
    { name: 'Genetic', sharpe: 1.87, sortino: 2.45, calmar: 2.78, winRate: 0.52, profitFactor: 1.41, maxDD: -6.5, totalReturn: 19.5, dsrPvalue: 0.048, dsrPass: true },
    { name: 'Ensemble', sharpe: 2.56, sortino: 3.34, calmar: 3.81, winRate: 0.56, profitFactor: 1.78, maxDD: -4.2, totalReturn: 30.2, dsrPvalue: 0.012, dsrPass: true },
  ],
  walkForward: { periods: 4, avgISSharpe: 2.65, avgOOSSharpe: 2.18, overfitRatio: 1.22, numOverfit: 0 },
  cpcv: { folds: 6, avgSharpe: 2.31, stdSharpe: 0.28, pboProb: 0.08 },
  testsPassing: { total: 43, phase5: 43, passRate: 100 },
};

// Health Monitor (Phase 6)
export const healthMonitor = {
  overallStatus: 'healthy', uptimePercent: 99.92, slaCompliant: true,
  totalChecks: 1248, failedChecks: 1, successRate: 99.92,
  system: { cpuPercent: 24.3, memoryPercent: 61.5, diskPercent: 42.8 },
  services: [
    { name: 'QuestDB', status: 'healthy', latencyMs: 2.3 },
    { name: 'Redis', status: 'healthy', latencyMs: 0.8 },
    { name: 'Network', status: 'healthy', latencyMs: 12.4 },
  ],
  latencyEndpoints: [
    { endpoint: '/health', p50: 1.2, p95: 3.8, p99: 8.1, samples: 500 },
    { endpoint: '/signal', p50: 15.4, p95: 42.1, p99: 78.3, samples: 350 },
    { endpoint: '/regime', p50: 8.7, p95: 22.5, p99: 45.6, samples: 420 },
  ],
};

// ================================================================
// Phase 6B — Paper Trading Engine
// ================================================================
export const paperTrading = {
  status: 'RUNNING',
  startedAt: '2026-05-14T08:00:00Z',
  config: {
    initialCapital: 100000, symbol: 'XAUUSD', kellyFraction: 0.25,
    maxPositionPct: 0.10, maxDailyLossPct: 0.02, maxDrawdownPct: 0.15,
    commissionPerTrade: 5.0, slippageModel: 'spread', slippagePct: 0.3,
    minConfidence: 0.6,
  },
  portfolio: {
    totalValue: 103842.67, cash: 91245.30, positionQuantity: 5.12,
    positionValue: 12597.37, pnlRealized: 2918.45, pnlUnrealized: 924.22,
    pnlTotal: 3842.67, dailyPnl: 412.30, returnPct: 3.84,
    sharpeRatio: 1.92, maxDrawdown: -2.1, winRate: 0.58, numTrades: 47,
  },
  modelSignals: {
    wavelet_pro: { lastSignal: 'LONG', confidence: 0.78, signalCount: 142 },
    hmm_pro: { lastSignal: 'LONG', confidence: 0.72, signalCount: 138 },
    lstm: { lastSignal: 'HOLD', confidence: 0.65, signalCount: 155 },
    tft_pro: { lastSignal: 'LONG', confidence: 0.81, signalCount: 148 },
    ensemble: { lastSignal: 'LONG', confidence: 0.76, signalCount: 160 },
  },
  recentTrades: [
    { tradeId: 'PT-0047', model: 'TFT', signal: 'LONG', entry: 2341.50, exit: 2348.20, qty: 1.2, pnl: 8.04, pnlPct: 0.29, status: 'CLOSED', time: '14:32' },
    { tradeId: 'PT-0046', model: 'Ensemble', signal: 'LONG', entry: 2335.80, exit: 2341.50, qty: 1.5, pnl: 8.55, pnlPct: 0.24, status: 'CLOSED', time: '13:15' },
    { tradeId: 'PT-0045', model: 'HMM', signal: 'SHORT', entry: 2352.10, exit: 2348.20, qty: 0.8, pnl: 3.12, pnlPct: 0.17, status: 'CLOSED', time: '11:45' },
    { tradeId: 'PT-0044', model: 'LSTM', signal: 'LONG', entry: 2328.50, exit: 2335.80, qty: 2.0, pnl: 14.60, pnlPct: 0.31, status: 'CLOSED', time: '10:20' },
    { tradeId: 'PT-0043', model: 'Wavelet', signal: 'LONG', entry: 2320.00, exit: 2315.40, qty: 1.0, pnl: -4.60, pnlPct: -0.20, status: 'CLOSED', time: '09:30' },
    { tradeId: 'PT-0042', model: 'Genetic', signal: 'SHORT', entry: 2345.60, exit: 2348.20, qty: 0.5, pnl: -1.30, pnlPct: -0.11, status: 'CLOSED', time: '08:50' },
  ],
  equityCurve: Array.from({ length: 30 }, (_, i) => {
    const d = new Date('2026-04-15');
    d.setDate(d.getDate() + i);
    return {
      date: d.toISOString().split('T')[0],
      equity: +(100000 + i * 128 + Math.sin(i / 4) * 800 + Math.random() * 500).toFixed(2),
    };
  }),
  dailyPnLHistory: Array.from({ length: 14 }, (_, i) => {
    const d = new Date('2026-05-01');
    d.setDate(d.getDate() + i);
    return { date: d.toISOString().split('T')[0], pnl: +((Math.random() - 0.35) * 800).toFixed(2) };
  }),
};

// ================================================================
// Phase 6C — Stress Testing
// ================================================================
export const stressTestData = {
  resilienceScore: 78.4,
  riskLevel: 'moderate',
  scenariosPassed: 7,
  scenariosFailed: 2,
  avgMaxDrawdown: 8.42,
  avgPnlChange: -3.18,
  scenarios: [
    { name: '2008 Financial Crisis', type: 'historical', pnlChange: 2.1, maxDD: 3.2, recovery: 8, status: 'passed', prob: 0.02 },
    { name: '2020 COVID-19 Crash', type: 'historical', pnlChange: 1.4, maxDD: 4.1, recovery: 12, status: 'passed', prob: 0.01 },
    { name: 'Flash Crash (2011)', type: 'historical', pnlChange: 0.8, maxDD: 2.5, recovery: 3, status: 'passed', prob: 0.005 },
    { name: 'Unexpected Rate Spike', type: 'hypothetical', pnlChange: -4.2, maxDD: 6.8, recovery: 14, status: 'passed', prob: 0.03 },
    { name: 'Geopolitical Escalation', type: 'hypothetical', pnlChange: 5.8, maxDD: 3.5, recovery: 7, status: 'passed', prob: 0.02 },
    { name: 'Correlation Breakdown', type: 'correlation', pnlChange: -12.4, maxDD: 15.8, recovery: 22, status: 'failed', prob: 0.01 },
    { name: '5-Sigma Fat Tail', type: 'fat_tail', pnlChange: -8.6, maxDD: 12.1, recovery: 18, status: 'passed', prob: 0.002 },
    { name: '6-Sigma Black Swan', type: 'fat_tail', pnlChange: -15.2, maxDD: 22.4, recovery: 28, status: 'failed', prob: 0.0005 },
    { name: 'Cascade Failure', type: 'cascade', pnlChange: -9.1, maxDD: 14.2, recovery: 20, status: 'passed', prob: 0.015 },
  ],
  reverseStress: {
    maxLossPct: 18.7, maxLossConfidence: 0.85,
    breachScenarios: ['Correlation Breakdown', '6-Sigma Black Swan'],
    recoveryEstimateDays: 24, minimumBufferPct: 3.7,
  },
};

// ================================================================
// Phase 6C — Dynamic Risk Adjustment
// ================================================================
export const dynamicRisk = {
  baseKelly: 0.50,
  adjustment: {
    volatilityMultiplier: 1.0, drawdownMultiplier: 0.92, consensusMultiplier: 0.85,
    correlationMultiplier: 1.0, finalKelly: 0.039, riskScore: 3.4,
    reason: 'Low model consensus (85%) | Drawdown stress (8%)',
  },
  volatility: {
    vixLevel: 16.8, regime: 'normal', vol30d: 14.2, vol60d: 13.8,
    trend: 'stable', volOfVol: 2.3,
  },
  correlations: { avgCorrelation: 0.48, status: 'healthy', spikeDetected: false, spikeMagnitude: 0.0 },
  modelConsensus: { consensusStrength: 0.85, disagreementCount: 2, totalModels: 6 },
  drawdown: { currentDrawdownPct: 2.1, stressLevel: 0.08, recoveryEstimateDays: 8 },
  adjustmentHistory: Array.from({ length: 24 }, (_, i) => ({
    hour: `${String(i).padStart(2, '0')}:00`,
    kelly: +(0.035 + Math.random() * 0.015).toFixed(4),
    riskScore: +(2.5 + Math.random() * 3.0).toFixed(1),
  })),
};

// ================================================================
// Phase 6C — Feature Drift Detection
// ================================================================
export const featureDrift = {
  totalFeatures: 12,
  driftAlerts: 2,
  features: [
    { name: 'rsi_14', driftScore: 0.12, status: 'stable', samples: 5000 },
    { name: 'macd_signal', driftScore: 0.08, status: 'stable', samples: 5000 },
    { name: 'volatility_20', driftScore: 0.62, status: 'warning', samples: 4800 },
    { name: 'sma_dist_20', driftScore: 0.15, status: 'stable', samples: 5000 },
    { name: 'wavelet_d1', driftScore: 0.82, status: 'critical', samples: 4500 },
    { name: 'regime_prob', driftScore: 0.21, status: 'stable', samples: 5000 },
    { name: 'corr_dxy_20', driftScore: 0.18, status: 'stable', samples: 4900 },
    { name: 'parkinson_vol', driftScore: 0.09, status: 'stable', samples: 5000 },
  ],
  recentAlerts: [
    { feature: 'wavelet_d1', severity: 'critical', score: 0.82, time: '14:22:05', recommendation: 'Consider model retraining' },
    { feature: 'volatility_20', severity: 'warning', score: 0.62, time: '12:45:30', recommendation: 'Monitor distribution changes' },
  ],
};

// ================================================================
// Phase 6C — Logging & Observability
// ================================================================
export const observability = {
  structuredLogs: { totalEntries: 24580, errorRate: 0.12, avgLatencyMs: 4.2 },
  tracing: { enabled: true, serviceName: 'mini-medallion', totalSpans: 8924, avgSpanMs: 12.8 },
  performance: {
    operations: [
      { name: 'signal_generation', count: 1420, avgMs: 15.4, p99Ms: 42.1 },
      { name: 'trade_execution', count: 47, avgMs: 23.8, p99Ms: 89.3 },
      { name: 'risk_calculation', count: 2840, avgMs: 8.7, p99Ms: 22.5 },
      { name: 'health_check', count: 1248, avgMs: 1.2, p99Ms: 8.1 },
    ],
  },
  metrics: {
    counters: { trades_executed: 47, signals_generated: 1420, risk_checks: 2840 },
    gauges: { portfolio_value: 103842.67, active_positions: 1, risk_score: 3.4 },
  },
};

// ================================================================
// Phase 7 — Team & Operations Management
// ================================================================
export const teamOperations = {
  team: {
    totalMembers: 6,
    avgTenureDays: 180,
    members: [
      { id: 'TM-001', name: 'Dr. S. Patel', role: 'Quant Researcher', focus: 'Signal Discovery', status: 'active', expertise: ['HMM', 'Wavelets', 'TFT'] },
      { id: 'TM-002', name: 'A. Chowdhury', role: 'MLOps Engineer', focus: 'Model Deployment', status: 'active', expertise: ['Docker', 'MLflow', 'CI/CD'] },
      { id: 'TM-003', name: 'R. Kumar', role: 'Data Engineer', focus: 'Pipeline & Ingestion', status: 'active', expertise: ['QuestDB', 'Redis', 'Streaming'] },
      { id: 'TM-004', name: 'M. Singh', role: 'Risk Manager', focus: 'Position Sizing', status: 'active', expertise: ['Kelly', 'VaR', 'Stress Testing'] },
      { id: 'TM-005', name: 'K. Verma', role: 'Execution Engineer', focus: 'C++ Engine', status: 'active', expertise: ['OrderRouter', 'Latency', 'IBKR'] },
      { id: 'TM-006', name: 'N. Arjun', role: 'Operations Lead', focus: 'System Oversight', status: 'active', expertise: ['Monitoring', 'Governance', 'Planning'] },
    ],
  },
  operations: {
    totalOperations: 14,
    avgSuccessRate: 97.2,
    byFrequency: { daily: 6, weekly: 4, monthly: 3, quarterly: 1 },
    scheduled: [
      { name: 'Model Performance Review', frequency: 'daily', responsible: 'Quant Researcher', duration: 30, successRate: 98.5, nextRun: '2026-05-15T08:00' },
      { name: 'Data Quality Check', frequency: 'daily', responsible: 'Data Engineer', duration: 15, successRate: 99.2, nextRun: '2026-05-15T07:00' },
      { name: 'Risk Limit Validation', frequency: 'daily', responsible: 'Risk Manager', duration: 20, successRate: 100, nextRun: '2026-05-15T07:30' },
      { name: 'Infrastructure Health Scan', frequency: 'daily', responsible: 'MLOps Engineer', duration: 10, successRate: 99.8, nextRun: '2026-05-15T06:00' },
      { name: 'Signal Catalog Update', frequency: 'weekly', responsible: 'Quant Researcher', duration: 60, successRate: 95.0, nextRun: '2026-05-19T10:00' },
      { name: 'Backtest Regression Suite', frequency: 'weekly', responsible: 'MLOps Engineer', duration: 120, successRate: 92.0, nextRun: '2026-05-18T22:00' },
      { name: 'Portfolio Stress Test', frequency: 'monthly', responsible: 'Risk Manager', duration: 90, successRate: 97.5, nextRun: '2026-06-01T08:00' },
    ],
  },
  governance: {
    totalChanges: 8,
    pipeline: { proposed: 1, review: 1, backtest: 1, paper_trade: 2, staging: 0, production: 3, rejected: 0 },
    recentChanges: [
      { id: 'MCR-008', model: 'TFT-v2', proposedBy: 'S. Patel', status: 'paper_trade', sharpe: 2.45, approvals: 2, date: '2026-05-12' },
      { id: 'MCR-007', model: 'HMM-regime-v3', proposedBy: 'S. Patel', status: 'paper_trade', sharpe: 2.12, approvals: 2, date: '2026-05-10' },
      { id: 'MCR-006', model: 'Ensemble-stack-v2', proposedBy: 'A. Chowdhury', status: 'production', sharpe: 2.56, approvals: 3, date: '2026-05-05' },
      { id: 'MCR-005', model: 'LSTM-bidir-v4', proposedBy: 'S. Patel', status: 'production', sharpe: 2.41, approvals: 3, date: '2026-04-28' },
      { id: 'MCR-004', model: 'Genetic-opt-v2', proposedBy: 'A. Chowdhury', status: 'production', sharpe: 1.98, approvals: 3, date: '2026-04-20' },
    ],
  },
  incidents: {
    total: 5, unresolved: 1,
    bySeverity: { low: 2, medium: 2, high: 1, critical: 0 },
    recent: [
      { id: 'INC-005', title: 'Redis cache miss spike', severity: 'medium', status: 'open', affected: ['Redis', 'Feature Pipeline'], time: '2026-05-14T10:20' },
      { id: 'INC-004', title: 'QuestDB slow query', severity: 'low', status: 'resolved', affected: ['QuestDB'], time: '2026-05-13T14:15', resolution: 'Index optimization' },
      { id: 'INC-003', title: 'Signal latency >100ms', severity: 'high', status: 'resolved', affected: ['Signal Engine', 'Network'], time: '2026-05-12T09:30', resolution: 'Network route fix' },
      { id: 'INC-002', title: 'MLflow disk usage 85%', severity: 'medium', status: 'resolved', affected: ['MLflow', 'MinIO'], time: '2026-05-10T16:00', resolution: 'Artifact cleanup' },
      { id: 'INC-001', title: 'Docker memory leak', severity: 'low', status: 'resolved', affected: ['Docker'], time: '2026-05-08T11:45', resolution: 'Container restart' },
    ],
  },
  research: {
    totalSignals: 18, activeSignals: 14, retiredSignals: 4,
    totalSeminars: 12, avgDiscoverySharpe: 2.15,
    upcomingSeminars: [
      { topic: 'Attention-weighted regime transition', presenter: 'S. Patel', date: '2026-05-19' },
      { topic: 'Cross-asset correlation alpha', presenter: 'R. Kumar', date: '2026-05-26' },
    ],
  },
  performanceReports: {
    daily: { trades: 8, winRate: 0.625, pnl: 412.30, sharpe: 1.92, maxDD: -0.8 },
    weekly: { trades: 47, winRate: 0.58, pnl: 2918.45, sharpe: 1.85, maxDD: -2.1 },
    monthly: { trades: 186, winRate: 0.55, pnl: 8245.60, sharpe: 1.78, maxDD: -4.2 },
  },
};
