import { useState, useEffect, useRef, useCallback } from 'react';
import { Database, Download, RefreshCw, AlertCircle, Save } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function PredictionLog() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [lastSaveTime, setLastSaveTime] = useState(null);
  const [saveStatus, setSaveStatus] = useState('idle'); // idle, saving, success, error
  const saveIntervalRef = useRef(null);
  const fetchIntervalRef = useRef(null);
  const logsRef = useRef([]);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/paper-trading/prediction-log?limit=200`);
      if (!response.ok) throw new Error('Failed to fetch prediction logs');
      const data = await response.json();
      const fetchedLogs = data.logs || [];
      setLogs(fetchedLogs);
      logsRef.current = fetchedLogs;
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const saveLogs = useCallback(async () => {
    if (logsRef.current.length === 0) return;
    
    setSaveStatus('saving');
    try {
      const response = await fetch(`${API_BASE}/paper-trading/save-prediction-logs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logsRef.current)
      });
      
      if (!response.ok) throw new Error('Failed to save logs');
      const data = await response.json();
      
      setSessionId(data.session_id);
      setLastSaveTime(new Date().toLocaleString());
      setSaveStatus('success');
      
      // Reset status after 3 seconds
      setTimeout(() => setSaveStatus('idle'), 3000);
      
      console.log('Logs saved successfully:', data);
    } catch (err) {
      console.error('Error saving logs:', err);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchLogs();
    
    // Set up fetch interval (15 seconds)
    fetchIntervalRef.current = setInterval(fetchLogs, 15000);
    
    // Set up save interval (5 minutes = 300,000 ms)
    saveIntervalRef.current = setInterval(() => {
      saveLogs();
    }, 300000);
    
    // Save immediately on component mount (after 1 second delay to let first fetch complete)
    const timer = setTimeout(() => {
      saveLogs();
    }, 1000);
    
    return () => {
      clearInterval(fetchIntervalRef.current);
      clearInterval(saveIntervalRef.current);
      clearTimeout(timer);
    };
  }, [saveLogs]);

  const downloadCSV = () => {
    if (logs.length === 0) return;
    
    const headers = Object.keys(logs[0]);
    const csvContent = [
      headers.join(','),
      ...logs.map(row => headers.map(header => row[header] || '').join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `prediction_log_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
  };

  const getSignalColor = (signal) => {
    if (signal === 'LONG') return 'var(--green)';
    if (signal === 'SHORT') return 'var(--red)';
    return 'var(--text-muted)';
  };

  const getPnlColor = (pnl) => {
    const val = parseFloat(pnl);
    if (isNaN(val)) return 'var(--text-muted)';
    return val > 0 ? 'var(--green)' : val < 0 ? 'var(--red)' : 'var(--text-primary)';
  };

  return (
    <div className="page-container animate-in">
      <header className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="header-icon">
            <Database size={24} />
          </div>
          <div>
            <h1 className="page-title">Prediction Log</h1>
            <p className="page-subtitle">Raw model outputs and ensemble consensus per cycle</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn" onClick={fetchLogs} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
            Refresh
          </button>
          <button 
            className="btn" 
            onClick={saveLogs} 
            disabled={logs.length === 0 || saveStatus === 'saving'}
            style={{ 
              opacity: saveStatus === 'saving' ? 0.6 : 1,
              background: saveStatus === 'success' ? 'rgba(34, 197, 94, 0.2)' : saveStatus === 'error' ? 'rgba(239, 68, 68, 0.2)' : undefined
            }}
          >
            <Save size={16} className={saveStatus === 'saving' ? 'spin-icon' : ''} />
            {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : saveStatus === 'error' ? 'Save Error' : 'Save to Disk'}
          </button>
          <button className="btn btn-primary" onClick={downloadCSV} disabled={logs.length === 0}>
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </header>

      {error && (
        <div className="alert-box error" style={{ marginBottom: '24px' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {sessionId && (
        <div className="glass-panel" style={{ marginBottom: '24px', padding: '12px 16px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
            <div>
              <span style={{ color: 'rgba(34, 197, 94, 0.9)', fontWeight: 600 }}>📁 Local Storage Active</span>
              <span style={{ marginLeft: '16px', color: 'var(--text-muted)' }}>Session: {sessionId}</span>
              {lastSaveTime && <span style={{ marginLeft: '16px', color: 'var(--text-muted)' }}>Last saved: {lastSaveTime}</span>}
            </div>
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Auto-saving every 5 minutes</span>
          </div>
        </div>
      )}

      <div className="glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)' }}>
          <h3 className="panel-title" style={{ margin: 0, fontSize: '14px' }}>Cycle Log ({logs.length} records)</h3>
        </div>
        
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table" style={{ width: '100%', minWidth: '1600px' }}>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Price</th>
                <th>Regime</th>
                <th>Wavelet Pro</th>
                <th>Hmm Pro</th>
                <th>LSTM</th>
                <th>TFT Pro</th>
                <th style={{ background: 'rgba(255,184,0,0.05)' }}>Ensemble</th>
                <th>Kelly</th>
                <th>Trade</th>
                <th>P&L</th>
              </tr>
            </thead>
            <tbody>
              {logs.length === 0 && !loading ? (
                <tr>
                  <td colSpan="14" style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                    No prediction logs found. Wait for the live inference loop to run.
                  </td>
                </tr>
              ) : (
                [...logs].reverse().map((log, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {log.timestamp ? log.timestamp.replace('T', ' ') : '-'}
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                      ${parseFloat(log.price).toFixed(2)}
                    </td>
                    <td>
                      <span className={`badge regime-${log.regime?.toLowerCase() || 'normal'}`}>
                        {log.regime}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontWeight: 600, color: getSignalColor(log.wavelet_pro_signal), fontSize: '12px' }}>{log.wavelet_pro_signal}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{log.wavelet_pro_conf ? (parseFloat(log.wavelet_pro_conf) * 100).toFixed(0) : '0'}%</span>
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontWeight: 600, color: getSignalColor(log.hmm_pro_signal), fontSize: '12px' }}>{log.hmm_pro_signal}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{log.hmm_pro_conf ? (parseFloat(log.hmm_pro_conf) * 100).toFixed(0) : '0'}%</span>
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontWeight: 600, color: getSignalColor(log.lstm_signal), fontSize: '12px' }}>{log.lstm_signal}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{log.lstm_conf ? (parseFloat(log.lstm_conf) * 100).toFixed(0) : '0'}%</span>
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontWeight: 600, color: getSignalColor(log.tft_pro_signal), fontSize: '12px' }}>{log.tft_pro_signal}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{log.tft_pro_conf ? (parseFloat(log.tft_pro_conf) * 100).toFixed(0) : '0'}%</span>
                      </div>
                    </td>
                    <td style={{ background: 'rgba(255,184,0,0.05)' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontWeight: 700, color: getSignalColor(log.ensemble_signal), fontSize: '13px' }}>{log.ensemble_signal}</span>
                        <span style={{ fontSize: '11px', color: 'var(--gold-primary)' }}>{log.ensemble_conf ? (parseFloat(log.ensemble_conf) * 100).toFixed(0) : '0'}%</span>
                      </div>
                    </td>
                    <td style={{ fontSize: '12px' }}>
                      {log.kelly_fraction && !isNaN(parseFloat(log.kelly_fraction)) ? `${(parseFloat(log.kelly_fraction) * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td>
                      {log.trade_taken === 'YES' ? (
                        <span className="badge" style={{ background: 'rgba(255,255,255,0.1)', color: '#fff' }}>EXEC</span>
                      ) : (
                        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>PASS</span>
                      )}
                    </td>
                    <td style={{ fontWeight: 600, color: getPnlColor(log.pnl), fontSize: '13px', fontFamily: 'monospace' }}>
                      {log.pnl ? (parseFloat(log.pnl) > 0 ? `+$${parseFloat(log.pnl).toFixed(2)}` : `-$${Math.abs(parseFloat(log.pnl)).toFixed(2)}`) : '-'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
