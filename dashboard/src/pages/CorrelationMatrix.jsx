import React, { useState, useEffect } from 'react';
import { Grid3X3, Info, CheckCircle2, RefreshCw } from 'lucide-react';
import { fetchCorrelationMatrix } from '../data/api';

const modelIcons = { wavelet_pro: '🌊', hmm_pro: '📊', lstm: '🧠', tft_pro: '⚡',  ensemble: '🎯' };
const modelColors = { wavelet_pro: '#3b82f6', hmm_pro: '#a855f7', lstm: '#06b6d4', tft_pro: '#f59e0b',  ensemble: '#ef4444' };
const modelNames = ['wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro',  'ensemble'];

// Default mock data structure as fallback
const defaultCorrelationMatrix = {
  wavelet_pro: { wavelet_pro: 100, hmm_pro: 77, lstm: 27, tft_pro: 37,  ensemble: 22 },
  hmm_pro: { wavelet_pro: 39, hmm_pro: 100, lstm: 58, tft_pro: 28,  ensemble: 31 },
  lstm: { wavelet_pro: 69, hmm_pro: 45, lstm: 100, tft_pro: 78,  ensemble: 77 },
  tft_pro: { wavelet_pro: 46, hmm_pro: 54, lstm: 44, tft_pro: 100,  ensemble: 66 },
  ensemble: { wavelet_pro: 78, hmm_pro: 48, lstm: 28, tft_pro: 53,  ensemble: 100 },
};

const defaultAvgPairwise = {
  wavelet_pro: 43, hmm_pro: 45, lstm: 65, tft_pro: 51, };

const defaultConsensus = [
  { model: 'wavelet_pro', name: 'Wavelet', agreement: 22 },
  { model: 'hmm_pro', name: 'Hmm', agreement: 31 },
  { model: 'lstm', name: 'Lstm', agreement: 77 },
  { model: 'tft_pro', name: 'Tft', agreement: 66 },
];

function getCellColor(val) {
  if (val >= 70) return '#00c48c'; // Strong Agree (Green)
  if (val >= 50) return '#008560'; // Agree (Dark Green)
  if (val >= 40) return '#a68214'; // Neutral (Yellow/Brown)
  if (val >= 30) return '#7a5a04'; // Weak Conflict (Brown)
  return '#3b2f15'; // Strong Conflict (Dark Brown/Grey)
}

function getCellTextColor(val) {
  if (val >= 70) return '#0a0e1a'; // Dark text for bright green
  return '#e2e8f0'; // Light text for dark backgrounds
}

export default function CorrelationMatrix() {
  const [data, setData] = useState({
    correlationMatrix: defaultCorrelationMatrix,
    avgPairwiseAgreement: defaultAvgPairwise,
    consensusAnalysis: defaultConsensus,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchCorrelationMatrix(1000);
      if (res && Object.keys(res.correlationMatrix || {}).length > 0) {
        setData(res);
      }
    } catch (err) {
      setError(err.message || "Failed to load data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    // Poll every 10 seconds to keep live
    const timer = setInterval(loadData, 10000);
    return () => clearInterval(timer);
  }, []);

  const { correlationMatrix, avgPairwiseAgreement, consensusAnalysis } = data;
  
  // Calculate total avg agreement for the top badge
  const totalAvgAgreement = Object.values(avgPairwiseAgreement).length > 0 
    ? Math.round(Object.values(avgPairwiseAgreement).reduce((a, b) => a + b, 0) / Object.values(avgPairwiseAgreement).length) 
    : 0;

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2>Model Correlation Matrix</h2>
            <p>Signal agreement heatmap — ensemble diversity & consensus strength</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button 
              onClick={loadData}
              disabled={loading}
              style={{ 
                background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', 
                color: 'var(--text-muted)', width: 36, height: 36, borderRadius: 'var(--radius-sm)', 
                display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                opacity: loading ? 0.5 : 1
              }}>
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
            </button>
          </div>
        </div>
      </div>
      
      <div className="page-body">
        
        {/* Top Grid: Matrix + Consensus */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '20px' }}>
          
          {/* Signal Correlation Heatmap */}
          <div className="card animate-in" style={{ padding: '24px' }}>
            <div className="card-header" style={{ marginBottom: '20px' }}>
              <span className="card-title" style={{ fontSize: '13px', letterSpacing: '0.5px' }}>SIGNAL CORRELATION</span>
              <span className="card-badge" style={{ background: '#00856030', color: '#00c48c', border: '1px solid #00c48c50' }}>AVG AGREEMENT: {totalAvgAgreement}%</span>
            </div>
            
            {/* Heatmap Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(80px, 100px) repeat(6, 1fr)', gap: '8px' }}>
              
              {/* Header Row */}
              <div />
              {modelNames.map(model => (
                <div key={`header-${model}`} style={{ textAlign: 'center', paddingBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '12px' }}>{modelIcons[model]}</span>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: modelColors[model], textTransform: 'capitalize' }}>{model}</span>
                </div>
              ))}

              {/* Matrix Rows */}
              {modelNames.map(rowModel => (
                <React.Fragment key={`row-${rowModel}`}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '14px' }}>{modelIcons[rowModel]}</span>
                    <span style={{ fontSize: '13px', fontWeight: 600, color: modelColors[rowModel], textTransform: 'capitalize' }}>{rowModel}</span>
                  </div>
                  
                  {modelNames.map(colModel => {
                    const isSelf = rowModel === colModel;
                    // For self we just show empty or blank cell style according to the screenshot? 
                    // Screenshot shows values for self as well sometimes (Wait, self is always 100%, but in screenshot they seem to hide it or show different values. Let's just use the mock data.)
                    const val = correlationMatrix[rowModel]?.[colModel] || 0;
                    const bg = getCellColor(val);
                    const color = getCellTextColor(val);
                    
                    return (
                      <div key={`cell-${rowModel}-${colModel}`} style={{
                        background: bg,
                        color: color,
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '42px',
                        fontSize: '13px',
                        fontWeight: 600,
                        fontFamily: 'var(--font-mono)'
                      }}>
                        {val}%
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>

            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '16px', marginTop: '24px', fontSize: '11px', color: 'var(--text-muted)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: 12, height: 12, borderRadius: 3, background: getCellColor(80) }} /> Strong Agree</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: 12, height: 12, borderRadius: 3, background: getCellColor(60) }} /> Weak Agree</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: 12, height: 12, borderRadius: 3, background: getCellColor(45) }} /> Neutral</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: 12, height: 12, borderRadius: 3, background: getCellColor(35) }} /> Weak Conflict</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: 12, height: 12, borderRadius: 3, background: getCellColor(20) }} /> Strong Conflict</div>
            </div>
          </div>

          {/* Consensus Analysis */}
          <div className="card animate-in" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
            <div className="card-header" style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Info size={16} style={{ color: 'var(--text-muted)' }} />
                <span className="card-title" style={{ fontSize: '13px', letterSpacing: '0.5px' }}>CONSENSUS ANALYSIS</span>
              </div>
              <span className="card-badge" style={{ background: '#00c48c20', color: '#00c48c', border: '1px solid #00c48c40', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <CheckCircle2 size={12} /> HEALTHY
              </span>
            </div>
            
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: '32px' }}>
              How each model's signal aligns with the final ensemble decision. Higher values = more agreement. Low agreement signals model diversity (good for robustness).
            </p>

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {consensusAnalysis.map(item => (
                <div key={item.name} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ width: '80px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '14px' }}>{modelIcons[item.model]}</span>
                    <span style={{ fontSize: '13px', color: 'var(--text-bright)' }}>{item.name}</span>
                  </div>
                  <div style={{ flex: 1, height: '6px', background: 'var(--bg-input)', borderRadius: '3px', position: 'relative' }}>
                    <div style={{ 
                      position: 'absolute', top: 0, left: 0, height: '100%', 
                      width: `${item.agreement}%`, 
                      background: '#00c48c', borderRadius: '3px' 
                    }} />
                  </div>
                  <span style={{ width: '30px', textAlign: 'right', fontSize: '13px', color: '#00c48c', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {item.agreement}%
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: '32px', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5, fontStyle: 'italic', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
              <strong style={{ color: 'var(--text-secondary)' }}>Interpretation:</strong> Model disagreement is a feature, not a bug. When models disagree, the ensemble weights each vote by its conviction + regime-conditional trust, creating a robust consensus.
            </div>
          </div>
        </div>

        {/* Bottom Cards: Avg Pairwise Agreement */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '20px' }}>
          {['wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro'].map(model => (
            <div key={model} className="card animate-in" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <span style={{ fontSize: '14px' }}>{modelIcons[model]}</span>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>{model}</span>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 700, color: '#00c48c', fontFamily: 'var(--font-mono)', marginBottom: '8px' }}>
                {avgPairwiseAgreement[model] || 0}%
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Avg Pairwise Agreement</div>
            </div>
          ))}
        </div>

      </div>
    </>
  );
}
