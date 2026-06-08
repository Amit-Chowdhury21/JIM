import React, { useState, useEffect, useRef } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Wifi, WifiOff, Radio, ArrowUpRight, ArrowDownRight, Minus, Zap } from 'lucide-react';
import { fetchLiveSignals, fetchInferenceStatus, fetchRegime, fetchModelWeights } from '../data/api';
import { regimeData, modelMetrics, featureImportance, featureConfig } from '../data/mockData';

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="label">{label}</div>
      {payload.map((p, i) => <div key={i} className="value" style={{ color: p.color }}>{p.name}: {p.value}</div>)}
    </div>
  );
};

function ModelConfigCard({ title, badge, configs, architecture, capabilities }) {
  return (
    <div className="card animate-in">
      <div className="card-header">
        <span className="card-title">{title}</span>
        <span className="card-badge badge-purple">{badge}</span>
      </div>
      {architecture && (
        <div style={{ fontSize: 10, color: 'var(--gold-primary)', padding: '6px 10px', marginBottom: 8, background: 'rgba(240,185,11,0.06)', borderRadius: 6, fontFamily: 'var(--font-mono)', letterSpacing: 0.3, lineHeight: 1.5 }}>
          {architecture}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {configs.map(([k, v], i) => (
          <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{k}</span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-bright)' }}>{typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}</span>
          </div>
        ))}
      </div>
      {capabilities && capabilities.length > 0 && (
        <div style={{ marginTop: 10, borderTop: '1px solid var(--border-color)', paddingTop: 8 }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Capabilities</div>
          {capabilities.slice(0, 4).map((c, i) => (
            <div key={i} style={{ fontSize: 10, color: 'var(--text-secondary)', padding: '2px 0', display: 'flex', gap: 4 }}>
              <span style={{ color: 'var(--green)', fontSize: 8 }}>●</span> {c}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const modelIcons = { wavelet_pro: '🌊', hmm_pro: '📊', lstm: '🧠', tft_pro: '⚡', ensemble: '🎯' };
const modelColors = { wavelet_pro: '#3b82f6', hmm_pro: '#a855f7', lstm: '#06b6d4', tft_pro: '#f59e0b', ensemble: '#ef4444' };
const signalColor = (s) => s === 'LONG' ? 'var(--green)' : s === 'SHORT' ? 'var(--red)' : 'var(--text-muted)';
const signalBg = (s) => s === 'LONG' ? 'var(--green-dim)' : s === 'SHORT' ? 'var(--red-dim)' : 'var(--bg-input)';

// Static fallback weights (mirrors dynamic_weights.py REGIME_BASE_WEIGHTS)
const FALLBACK_REGIME_WEIGHTS = {
  GROWTH:  { wavelet_pro:0.18, hmm_pro:0.15, lstm:0.22, tft_pro:0.22, ensemble:0.23 },
  NORMAL:  { wavelet_pro:0.24, hmm_pro:0.24, lstm:0.14, tft_pro:0.14, ensemble:0.24 },
  CRISIS:  { wavelet_pro:0.18, hmm_pro:0.33, lstm:0.07, tft_pro:0.07, ensemble:0.35 },
};
const REGIMES = ['GROWTH', 'NORMAL', 'CRISIS'];
const regimeColors = { GROWTH: '#00c48c', NORMAL: '#ff9f43', CRISIS: '#ff4d6a' };

function weightColor(w) {
  if (w >= 0.25) return 'rgba(240,185,11,0.9)';
  if (w >= 0.15) return 'rgba(240,185,11,0.55)';
  if (w >= 0.10) return 'rgba(168,85,247,0.5)';
  return 'rgba(107,114,128,0.4)';
}

function DynamicWeightsCard({ modelWeights, modelColors, modelIcons }) {
  const regimeBaseWeights = modelWeights?.regime_base_weights || FALLBACK_REGIME_WEIGHTS;
  const currentRegime = modelWeights?.regime || 'NORMAL';
  const liveWeights = modelWeights?.weights || null;
  const modelStats = modelWeights?.model_stats || {};
  const adaptationActive = modelWeights?.adaptation_active || false;
  const modelNames = Object.keys(FALLBACK_REGIME_WEIGHTS.NORMAL);

  return (
    <div className="card animate-in" style={{ marginBottom: 16 }}>
      <div className="card-header">
        <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Zap size={16} style={{ color: 'var(--gold-primary)' }} /> Dynamic Model Weights
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className={`card-badge ${adaptationActive ? 'badge-green' : 'badge-orange'}`}>
            {adaptationActive ? '● ADAPTIVE' : '○ BASE WEIGHTS'}
          </span>
          <span className="card-badge" style={{
            background: `${regimeColors[currentRegime]}20`,
            color: regimeColors[currentRegime],
            border: `1px solid ${regimeColors[currentRegime]}40`,
          }}>
            REGIME: {currentRegime}
          </span>
        </div>
      </div>

      {/* Description */}
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Real-world regime-conditional weighting: each model receives different signal weight based on market regime.
        {adaptationActive ? ' Weights are further adapted by rolling Sharpe ratio.' : ' Weights will adapt after ≥10 trades.'}
      </div>

      {/* Heatmap Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: `100px repeat(${REGIMES.length}, 1fr)`, gap: 0 }}>
        {/* Header row */}
        <div style={{ padding: '6px 0', fontSize: 10, color: 'var(--text-muted)', fontWeight: 700 }}>MODEL</div>
        {REGIMES.map(r => (
          <div key={r} style={{
            padding: '6px 8px', textAlign: 'center', fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
            color: r === currentRegime ? regimeColors[r] : 'var(--text-muted)',
            borderBottom: r === currentRegime ? `2px solid ${regimeColors[r]}` : '2px solid transparent',
          }}>{r}</div>
        ))}

        {/* Model rows */}
        {modelNames.map(model => (
          <React.Fragment key={model}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '8px 0',
              borderTop: '1px solid var(--border-color)',
            }}>
              <span style={{ fontSize: 13 }}>{modelIcons[model]}</span>
              <span style={{
                fontSize: 11, fontWeight: 600, color: 'var(--text-bright)',
                textTransform: 'capitalize',
              }}>{model}</span>
            </div>
            {REGIMES.map(regime => {
              const w = regime === currentRegime && liveWeights
                ? (liveWeights[model] || 0)
                : (regimeBaseWeights[regime]?.[model] || 0);
              const isActive = regime === currentRegime;
              const pct = (w * 100).toFixed(0);
              return (
                <div key={regime} style={{
                  padding: '8px', borderTop: '1px solid var(--border-color)',
                  display: 'flex', alignItems: 'center', gap: 6,
                  background: isActive ? 'rgba(240,185,11,0.04)' : 'transparent',
                }}>
                  <div style={{
                    flex: 1, height: 18, borderRadius: 4, background: 'var(--bg-input)',
                    overflow: 'hidden', position: 'relative',
                  }}>
                    <div style={{
                      height: '100%', width: `${Math.min(w * 280, 100)}%`,
                      background: isActive ? weightColor(w) : 'rgba(107,114,128,0.25)',
                      borderRadius: 4, transition: 'width 0.5s ease',
                    }} />
                  </div>
                  <span style={{
                    fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: isActive ? 700 : 400,
                    color: isActive ? 'var(--text-bright)' : 'var(--text-muted)',
                    minWidth: 32, textAlign: 'right',
                  }}>{pct}%</span>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>

      {/* Model Stats Footer */}
      {Object.keys(modelStats).length > 0 && (
        <div style={{ marginTop: 14, borderTop: '1px solid var(--border-color)', paddingTop: 10 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
            Model Performance (Rolling)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
            {modelNames.map(model => {
              const stats = modelStats[model] || {};
              return (
                <div key={model} style={{
                  padding: '8px 10px', borderRadius: 6,
                  background: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: modelColors[model], marginBottom: 4, textTransform: 'capitalize' }}>
                    {modelIcons[model]} {model}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                    <span>Win: <span style={{ color: 'var(--text-bright)', fontFamily: 'var(--font-mono)' }}>{((stats.win_rate || 0) * 100).toFixed(0)}%</span></span>
                    <span>Sharpe: <span style={{ color: (stats.rolling_sharpe || 0) > 1 ? 'var(--green)' : 'var(--text-bright)', fontFamily: 'var(--font-mono)' }}>{(stats.rolling_sharpe || 0).toFixed(2)}</span></span>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    Trades: <span className="mono">{stats.trade_count || 0}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Models() {
  const [live, setLive] = useState(false);
  const [liveSignals, setLiveSignals] = useState(null);
  const [inferenceStatus, setInferenceStatus] = useState(null);
  const [liveRegime, setLiveRegime] = useState(null);
  const [modelWeights, setModelWeights] = useState(null);

  const refreshRef = useRef(null);
  refreshRef.current = async () => {
    try {
      const [signals, infStatus] = await Promise.all([
        fetchLiveSignals(),
        fetchInferenceStatus(),
      ]);
      setLiveSignals(signals);
      setInferenceStatus(infStatus);
      setLive(true);
    } catch { /* offline */ }

    try { setLiveRegime(await fetchRegime()); } catch { /* offline */ }
    try { setModelWeights(await fetchModelWeights()); } catch { /* offline */ }
  };

  useEffect(() => {
    refreshRef.current?.();
    const timer = setInterval(() => refreshRef.current?.(), 10000);
    return () => clearInterval(timer);
  }, []);

  // Build model data
  const models = liveSignals?.models || null;
  const goldPrice = liveSignals?.current_price || 0;
  const inferenceRunning = liveSignals?.inference_running || false;
  const iteration = liveSignals?.iteration || 0;

  // Radar chart data
  const activeModelsList = ['wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro', 'ensemble'];
  const radarData = models
    ? Object.entries(models)
        .filter(([name]) => activeModelsList.includes(name))
        .map(([name, d]) => ({ model: name, confidence: (d.confidence || 0) * 100 }))
    : Object.entries(modelMetrics)
        .filter(([name]) => activeModelsList.includes(name))
        .map(([name, d]) => ({
      model: name, confidence: name === 'ensemble' ? (d.winRate || 0) * 100 : (d.accuracy || d.winRate || 0.6) * 100,
    }));

  // Signal log — from live signals or mock
  const signalLog = models
    ? Object.entries(models)
        .filter(([name]) => activeModelsList.includes(name))
        .map(([name, d], i) => {
          const displayName = name.charAt(0).toUpperCase() + name.slice(1);
          return {
            id: i + 1,
            time: d.last_updated ? new Date(d.last_updated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '--:--:--',
            type: d.signal || 'HOLD',
            source: displayName,
            confidence: d.confidence || 0,
            status: d.signal && d.signal !== 'HOLD' ? 'active' : 'idle',
          };
        })
    : [];

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>Models & Signals</h2>
            <p>Production-Grade — 5 Models (Wavelet Pro, HMM Regime, LSTM, TFT Pro, Ensemble) • {live ? 'Live Inference Active' : 'Offline (Static Data)'}</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {inferenceRunning && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px',
                borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
                background: 'var(--green-dim)', color: 'var(--green)',
                border: '1px solid rgba(0,196,140,0.3)', animation: 'pulse 2s infinite',
              }}>
                <Radio size={12} /> LIVE — iter #{iteration}
              </div>
            )}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px',
              borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
              background: live ? 'var(--green-dim)' : 'var(--red-dim)',
              color: live ? 'var(--green)' : 'var(--red)',
              border: `1px solid ${live ? 'rgba(0,196,140,0.3)' : 'rgba(255,77,106,0.3)'}`,
            }}>
              {live ? <Wifi size={12} /> : <WifiOff size={12} />}
              {live ? 'CONNECTED' : 'OFFLINE'}
            </div>
          </div>
        </div>
      </div>
      <div className="page-body">
        {/* Model Stats */}
        <div className="kpi-grid" style={{ marginBottom: 20 }}>
          {[
            { label: 'HMM Regime', value: `${(modelMetrics.hmm_pro.accuracy * 100).toFixed(1)}%`, sub: liveRegime ? `Regime: ${liveRegime.regime}` : `3 regimes • ${modelMetrics.hmm_pro.config.covarianceType} cov` },
            { label: 'Wavelet SNR', value: `+${modelMetrics.wavelet_pro.snrDb} dB`, sub: `${modelMetrics.wavelet_pro.noiseRemoved}% noise • ${modelMetrics.wavelet_pro.config.thresholdMethod}` },
            { label: 'Ensemble Sharpe', value: modelMetrics.ensemble.sharpe.toFixed(2), sub: `${modelMetrics.ensemble.config.metaLearner} • PF ${modelMetrics.ensemble.profitFactor}` },
            { label: 'LSTM Temporal', value: modelMetrics.lstm.valLoss.toFixed(4), sub: `BiLSTM ${modelMetrics.lstm.config.hiddenSize}×${modelMetrics.lstm.config.numLayers} • 15 features` },
            { label: 'TFT Pro Forecaster', value: modelMetrics.tft_pro.valLoss.toFixed(4), sub: `${modelMetrics.tft_pro.config.attentionHeads}h attn • ${modelMetrics.tft_pro.config.forecastHorizons.length} horizons` },
          ].map((m, i) => (
            <div key={i} className="kpi-card animate-in">
              <div className="kpi-label">{m.label}</div>
              <div className="kpi-value" style={{ fontSize: 22 }}>{m.value}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Live Model Signal Cards */}
        {models && (
          <div className="card animate-in" style={{ marginBottom: 16 }}>
            <div className="card-header">
              <span className="card-title">Live Model Signals</span>
              <div style={{ display: 'flex', gap: 8 }}>
                {goldPrice > 0 && <span className="card-badge badge-gold">Gold ${goldPrice.toFixed(2)}</span>}
                <span className={`card-badge ${inferenceRunning ? 'badge-green' : 'badge-orange'}`}>
                  {inferenceRunning ? '● LIVE' : '○ IDLE'}
                </span>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
              {Object.entries(models)
                .filter(([name]) => ['wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro', 'ensemble'].includes(name))
                .map(([name, data]) => {
                const sig = data.signal || 'IDLE';
                const conf = ((data.confidence || 0) * 100).toFixed(0);
                const updatedAgo = data.last_updated
                  ? new Date(data.last_updated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                  : 'never';
                const reasoning = (data.reasoning || '').slice(0, 100) || 'Awaiting inference...';
                const displayName = name.replace('_', ' ');
                return (
                  <div key={name} style={{
                    padding: 16, borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-secondary)',
                    border: `1px solid ${sig !== 'HOLD' && sig !== 'IDLE' ? (sig === 'LONG' ? 'rgba(0,196,140,0.4)' : 'rgba(255,77,106,0.4)') : 'var(--border-color)'}`,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: 14, background: `${modelColors[name]}20`, color: modelColors[name],
                        }}>{modelIcons[name]}</span>
                        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-bright)', textTransform: 'capitalize' }}>{displayName}</span>
                      </div>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '4px 10px', borderRadius: 12,
                        background: signalBg(sig), color: signalColor(sig),
                        display: 'flex', alignItems: 'center', gap: 4,
                      }}>
                        {sig === 'LONG' ? <ArrowUpRight size={10} /> : sig === 'SHORT' ? <ArrowDownRight size={10} /> : <Minus size={10} />}
                        {sig}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                      <span>Confidence: <span style={{ color: signalColor(sig), fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{conf}%</span></span>
                      <span>Regime: <span style={{ fontWeight: 600 }}>{data.regime || '—'}</span></span>
                      <span>{updatedAgo}</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{
                        width: `${conf}%`,
                        background: signalColor(sig),
                        transition: 'width 0.6s ease',
                      }} />
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8, fontStyle: 'italic', lineHeight: 1.3 }}>{reasoning}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Regime + Wavelet Charts */}
        <div className="grid-2" style={{ marginBottom: 16 }}>
          <div className="card animate-in">
            <div className="card-header">
              <span className="card-title">Regime Probabilities (30D)</span>
              <span className="card-badge badge-green">Current: {liveRegime?.regime || regimeData.current}</span>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={regimeData.history}>
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#6b7280' }} tickFormatter={v => v.slice(8)} />
                <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip content={<TT />} />
                <Area type="monotone" dataKey="GROWTH" stackId="1" stroke="#00c48c" fill="rgba(0,196,140,0.4)" name="Growth" />
                <Area type="monotone" dataKey="NORMAL" stackId="1" stroke="#ff9f43" fill="rgba(255,159,67,0.4)" name="Normal" />
                <Area type="monotone" dataKey="CRISIS" stackId="1" stroke="#ff4d6a" fill="rgba(255,77,106,0.4)" name="Crisis" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card animate-in">
            <div className="card-header">
              <span className="card-title">Model Confidence Radar</span>
              <span className={`card-badge ${live ? 'badge-green' : 'badge-purple'}`}>{live ? 'LIVE' : 'STATIC'}</span>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(42,53,80,.5)" />
                <PolarAngleAxis dataKey="model" tick={{ fontSize: 11, fill: '#9ca3af', fontWeight: 600 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: '#6b7280' }} />
                <Radar name="Confidence" dataKey="confidence" stroke="#a855f7" fill="rgba(168,85,247,.2)" strokeWidth={2} dot={{ r: 4, fill: '#a855f7' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Dynamic Model Weights Heatmap */}
        <DynamicWeightsCard modelWeights={modelWeights} modelColors={modelColors} modelIcons={modelIcons} />

        {/* Model Architecture Configs — Row 1: Signal Processing */}
        <div className="grid-3" style={{ marginBottom: 16 }}>
          <ModelConfigCard title="🌊 Wavelet Pro" badge="wavelet_pro.py" architecture={modelMetrics.wavelet_pro.architecture} capabilities={modelMetrics.wavelet_pro.capabilities} configs={[
            ['Wavelet Family', modelMetrics.wavelet_pro.config.family], ['Decomposition Levels', modelMetrics.wavelet_pro.config.levels],
            ['Threshold Method', modelMetrics.wavelet_pro.config.thresholdMethod], ['Threshold Mode', modelMetrics.wavelet_pro.config.thresholdMode],
            ['Remove Levels', modelMetrics.wavelet_pro.config.denoiseRemoveLevels.join(', ')], ['Min Samples', modelMetrics.wavelet_pro.config.minSamples],
            ['SNR Improvement', `+${modelMetrics.wavelet_pro.snrDb} dB`], ['Noise Removed', `${modelMetrics.wavelet_pro.noiseRemoved}%`],
            ['Retrain', modelMetrics.wavelet_pro.config.retrainFrequency],
          ]} />
          <ModelConfigCard title="📊 HMM Regime Detector" badge="hmm_regime.py" architecture={modelMetrics.hmm_pro.architecture} capabilities={modelMetrics.hmm_pro.capabilities} configs={[
            ['Regimes', modelMetrics.hmm_pro.config.nRegimes], ['Covariance', modelMetrics.hmm_pro.config.covarianceType],
            ['EM Iterations', modelMetrics.hmm_pro.config.nIter.toLocaleString()], ['Min Duration', `${modelMetrics.hmm_pro.config.minRegimeDuration} bars`],
            ['Persistence Wt', modelMetrics.hmm_pro.config.regimePersistenceWeight], ['Vol Windows', modelMetrics.hmm_pro.config.volWindows.join(', ')],
            ['Accuracy', `${(modelMetrics.hmm_pro.accuracy * 100).toFixed(1)}%`], ['Retrain', modelMetrics.hmm_pro.config.retrainFrequency],
          ]} />
          <ModelConfigCard title="🎛️ HMM Pro GPU Engine" badge="C++ / CUDA" architecture="Variational Gaussian Mixture + GPU Parallelism" capabilities={['GPU Acceleration', 'Scalable Inference', 'Variational Posterior']} configs={[
            ['Precision', 'FP16 Mixed'], ['Backend', 'CUDA / cuBLAS'],
            ['Components', 'Gaussian Mixtures'], ['Latent Dim', 'Variable'],
            ['Memory', 'Shared / Pinned'], ['Fallback', 'CPU Fallback enabled'],
            ['Auto-Tune', 'Active'], ['Throughput', '~1M ops/sec'],
          ]} />
        </div>

        {/* Row 2: Deep Learning */}
        <div className="grid-2" style={{ marginBottom: 16 }}>
          <ModelConfigCard title="🧠 LSTM Temporal" badge="PyTorch BiLSTM" architecture={modelMetrics.lstm.architecture} capabilities={modelMetrics.lstm.capabilities} configs={[
            ['Hidden Size', modelMetrics.lstm.config.hiddenSize], ['Layers', modelMetrics.lstm.config.numLayers],
            ['Bidirectional', modelMetrics.lstm.config.bidirectional], ['Dropout', modelMetrics.lstm.config.dropout],
            ['Seq Length', modelMetrics.lstm.config.seqLength], ['Batch Size', modelMetrics.lstm.config.batchSize],
            ['Patience', modelMetrics.lstm.config.patience], ['Grad Clip', modelMetrics.lstm.config.gradClip],
            ['Input Features', 15], ['Parameters', (modelMetrics.lstm.parameters || 0).toLocaleString()],
          ]} />
          <ModelConfigCard title="⚡ TFT Pro Forecaster" badge="Transformer" architecture={modelMetrics.tft_pro.architecture} capabilities={modelMetrics.tft_pro.capabilities} configs={[
            ['d_model', modelMetrics.tft_pro.config.dModel], ['Attention Heads', modelMetrics.tft_pro.config.attentionHeads],
            ['Layers', modelMetrics.tft_pro.config.nLayers], ['Dropout', modelMetrics.tft_pro.config.dropout],
            ['Seq Length', modelMetrics.tft_pro.config.seqLength], ['Horizons', modelMetrics.tft_pro.config.forecastHorizons.join(', ')],
            ['Quantiles', modelMetrics.tft_pro.config.quantiles.join(', ')], ['Retrain', modelMetrics.tft_pro.config.retrainFrequency],
          ]} />
        </div>

        {/* Ensemble Meta-Learner — Full Width */}
        <div style={{ marginBottom: 16 }}>
          <ModelConfigCard title="🎯 Ensemble Stacking Meta-Learner" badge="XGBoost" architecture={modelMetrics.ensemble.architecture} capabilities={modelMetrics.ensemble.capabilities} configs={[
            ['Method', modelMetrics.ensemble.config.method], ['Meta-Learner', modelMetrics.ensemble.config.metaLearner],
            ['Trees', modelMetrics.ensemble.config.nEstimators], ['Max Depth', modelMetrics.ensemble.config.maxDepth],
            ['Disagree Penalty', modelMetrics.ensemble.config.disagreementPenalty], ['Min Confidence', modelMetrics.ensemble.config.confidenceThreshold],
            ['EWMA α', modelMetrics.ensemble.config.ewmaAlpha], ['Sharpe', modelMetrics.ensemble.sharpe],
            ['Win Rate', `${(modelMetrics.ensemble.winRate * 100).toFixed(1)}%`], ['Profit Factor', modelMetrics.ensemble.profitFactor],
          ]} />
        </div>

        <div className="grid-2">
          <div className="card animate-in">
            <div className="card-header">
              <span className="card-title">Feature Importance</span>
              <span className="card-badge badge-gold">{featureConfig.totalFeatures}+ features</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={featureImportance} layout="vertical" margin={{ left: 100 }}>
                <XAxis type="number" tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#9ca3af' }} width={100} />
                <Tooltip content={<TT />} />
                <Bar dataKey="importance" name="Importance" fill="rgba(240,185,11,0.6)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card animate-in">
            <div className="card-header"><span className="card-title">Signal Log</span><span className="card-badge badge-gold">{signalLog.length} signals</span></div>
            <table className="data-table">
              <thead><tr><th>Time</th><th>Type</th><th>Source</th><th>Conf</th><th>Status</th></tr></thead>
              <tbody>
                {signalLog.map(s => (
                  <tr key={s.id}>
                    <td className="mono">{s.time}</td>
                    <td><span style={{ color: s.type === 'LONG' ? 'var(--green)' : s.type === 'SHORT' ? 'var(--red)' : 'var(--blue)', fontWeight: 600, fontSize: 12 }}>{s.type}</span></td>
                    <td style={{ fontSize: 12 }}>{s.source}</td>
                    <td className="mono">{(s.confidence * 100).toFixed(0)}%</td>
                    <td><span className={`card-badge ${s.status === 'active' ? 'badge-green' : s.status === 'filled' ? 'badge-blue' : 'badge-orange'}`}>{s.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 16, borderTop: '1px solid var(--border-color)', paddingTop: 12 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Feature Categories</div>
              {featureConfig.categories.map((c, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 11 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{c.name}</span>
                  <span className="mono" style={{ color: 'var(--gold-primary)' }}>{c.count}</span>
                </div>
              ))}
            </div>

            {/* Inference Status */}
            {inferenceStatus && (
              <div style={{ marginTop: 16, borderTop: '1px solid var(--border-color)', paddingTop: 12 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Inference Loop</div>
                {[
                  ['Status', inferenceStatus.running ? '● Running' : '○ Stopped'],
                  ['Iteration', `#${inferenceStatus.iteration}`],
                  ['Interval', `${inferenceStatus.interval_seconds}s`],
                  ['Last Run', inferenceStatus.last_run ? new Date(inferenceStatus.last_run).toLocaleTimeString() : '—'],
                  ['Last Error', inferenceStatus.last_error || 'None ✓'],
                ].map(([k, v], i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: 11 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                    <span className="mono" style={{ fontSize: 11, color: k === 'Status' && inferenceStatus.running ? 'var(--green)' : k === 'Last Error' && inferenceStatus.last_error ? 'var(--red)' : 'var(--text-bright)' }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
