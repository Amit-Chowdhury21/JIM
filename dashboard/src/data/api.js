/**
 * API Service — connects the React dashboard to the FastAPI backend.
 *
 * Backend endpoints (from src/api/app.py + paper_trading_routes.py):
 *   GET  /health                     – system health
 *   GET  /signal                     – current trading signal
 *   GET  /ensemble                   – ensemble prediction
 *   GET  /regime                     – market regime
 *   GET  /features                   – engineered features
 *   GET  /models                     – model status
 *   GET  /models/performance         – model performance tracking
 *   GET  /data-quality               – data quality report
 *   GET  /gold-price                 – OHLC candlestick data
 *   POST /backtest/{strategy}        – run backtest
 *   POST /paper-trading/start        – start engine
 *   POST /paper-trading/stop         – stop engine
 *   GET  /paper-trading/status       – engine status
 *   GET  /paper-trading/performance  – perf metrics
 *   GET  /paper-trading/portfolio    – portfolio snapshot
 *   GET  /paper-trading/trades       – trade history
 *   GET  /paper-trading/risk-report  – risk analysis
 *   GET  /paper-trading/live-signals – live model signals
 *   GET  /paper-trading/inference-status – inference loop health
 *   POST /paper-trading/signal       – inject signal
 *   POST /paper-trading/config       – update config
 *   POST /paper-trading/reset-daily  – reset daily counters
 */

const API_BASE = 'http://localhost:8000';

// ============================================================================
// CORE FETCH HELPERS
// ============================================================================

const API_KEY = import.meta.env.VITE_API_ACCESS_KEY || 'medallion_secret_key';

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: { 
      'Content-Type': 'application/json', 
      'X-API-Key': API_KEY,
      ...options.headers 
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    let errMsg = err.detail || err.error || res.statusText;
    if (Array.isArray(errMsg)) {
      errMsg = errMsg.map(e => e.msg || JSON.stringify(e)).join(', ');
    } else if (typeof errMsg === 'object') {
      errMsg = JSON.stringify(errMsg);
    }
    throw new Error(errMsg);
  }
  return res.json();
}

async function apiPost(path, body) {
  return apiFetch(path, { method: 'POST', body: JSON.stringify(body) });
}

// ============================================================================
// SYSTEM ENDPOINTS
// ============================================================================

export async function fetchHealth() {
  return apiFetch('/health');
}

export async function fetchSignal() {
  return apiFetch('/signal');
}

export async function fetchEnsemble() {
  return apiFetch('/ensemble');
}

export async function fetchRegime() {
  return apiFetch('/regime');
}

export async function fetchFeatures(symbol = 'XAUUSD=X', lookbackDays = 100) {
  return apiFetch(`/features?symbol=${symbol}&lookback_days=${lookbackDays}`);
}

export async function fetchModels() {
  return apiFetch('/models');
}

export async function fetchModelPerformance() {
  return apiFetch('/models/performance');
}

export async function fetchDataQuality() {
  return apiFetch('/data-quality');
}

export async function fetchGoldPrice(interval = '15m', period = '5d') {
  return apiFetch(`/gold-price?interval=${interval}&period=${period}`);
}

export async function fetchFearGreed() {
  return apiFetch('/fear-greed');
}

export async function fetchGsRatio(period = '2y', interval = '1mo') {
  return apiFetch(`/gs-ratio?period=${period}&interval=${interval}`);
}

export async function runBacktest(strategy, params) {
  return apiPost(`/backtest/${strategy}`, params);
}

// ============================================================================
// PAPER TRADING ENDPOINTS
// ============================================================================

export async function startPaperTrading(config) {
  return apiPost('/paper-trading/start', config);
}

export async function stopPaperTrading() {
  return apiPost('/paper-trading/stop', {});
}

export async function fetchPaperTradingStatus() {
  return apiFetch('/paper-trading/status');
}

export async function fetchPaperTradingPerformance() {
  return apiFetch('/paper-trading/performance');
}

export async function fetchPaperTradingPortfolio() {
  return apiFetch('/paper-trading/portfolio');
}

export async function fetchPaperTradingTrades(limit = 50, offset = 0, statusFilter = null) {
  let url = `/paper-trading/trades?limit=${limit}&offset=${offset}`;
  if (statusFilter && statusFilter !== 'ALL') url += `&status_filter=${statusFilter}`;
  return apiFetch(url);
}

export async function fetchRiskReport() {
  return apiFetch('/paper-trading/risk-report');
}

export async function fetchLiveSignals() {
  return apiFetch('/paper-trading/live-signals');
}

export async function fetchInferenceStatus() {
  return apiFetch('/paper-trading/inference-status');
}

export async function enableAutoTrading() {
  return apiPost('/paper-trading/enable-auto-trade', {});
}

export async function fetchLSTMLogs(lines = 20) {
  return apiFetch(`/paper-trading/lstm-logs?lines=${lines}`);
}

export async function injectSignal(signalData) {
  return apiPost('/paper-trading/signal', signalData);
}

export async function updatePaperTradingConfig(config) {
  return apiPost('/paper-trading/config', config);
}

export async function resetDailyCounters() {
  return apiPost('/paper-trading/reset-daily', {});
}

export async function resetCircuitBreakers() {
  return apiPost('/paper-trading/reset-circuit-breakers', {});
}

export async function fetchModelWeights() {
  return apiFetch('/paper-trading/model-weights');
}

export async function fetchCorrelationMatrix(limit = 500) {
  return apiFetch(`/paper-trading/correlation-matrix?limit=${limit}`);
}

// ============================================================================
// WEBSOCKET
// ============================================================================

export function createWebSocket(onMessage, onOpen, onClose) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//localhost:8000/paper-trading/ws`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected');
    onOpen?.();
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      onMessage?.(msg);
    } catch { /* ignore */ }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected');
    onClose?.();
  };

  ws.onerror = () => {
    onClose?.();
  };

  return ws;
}

// ============================================================================
// HOOK HELPER — useApi
// ============================================================================

/**
 * Call an async API function, returning { data, loading, error }.
 * Automatically falls back to mockFallback when API is unreachable.
 */
export async function tryApi(apiFn, mockFallback = null) {
  try {
    return await apiFn();
  } catch (err) {
    if (mockFallback !== null) return mockFallback;
    throw err;
  }
}
