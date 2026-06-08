import { useState, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid } from 'recharts';
import { Play, Loader } from 'lucide-react';
import { runBacktest } from '../data/api';

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (<div className="custom-tooltip"><div className="label">{label}</div>
    {payload.map((p, i) => <div key={i} className="value" style={{ color: p.color }}>{p.name}: {p.value}</div>)}
  </div>);
};

export default function Backtesting() {
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [strategy, setStrategy] = useState('ensemble');
  const [config, setConfig] = useState({ initial_capital: 100000, commission: 2.5, slippage_bps: 1.0, train_years: 3, test_years: 1 });

  const handleRun = useCallback(async () => {
    setRunning(true); setError(null);
    try {
      const today = new Date();
      const startDate = new Date();
      startDate.setFullYear(today.getFullYear() - (config.train_years + config.test_years));
      
      const payload = {
          ...config,
          start_date: startDate.toISOString().split('T')[0],
          end_date: today.toISOString().split('T')[0]
      };
      
      const res = await runBacktest(strategy, payload);
      setResults(res);
    } catch (e) { setError(e.message); }
    setRunning(false);
  }, [strategy, config]);

  const metrics = results?.metrics || {};
  const regimeDrawdowns = metrics.regime_drawdowns || {};
  
  // Format regime data for bar chart
  const regimeData = Object.entries(regimeDrawdowns).map(([regime, dd]) => ({
      name: regime,
      drawdown: dd,
      fill: regime === 'CRISIS' ? '#ef4444' : regime === 'GROWTH' ? '#10b981' : '#3b82f6'
  }));

  // Format equity curve for line chart
  const equityData = (results?.equity_curve || []).map((eq, i) => ({
      step: i,
      equity: eq
  }));

  return (<>
    <div className="page-header"><h2>Backtesting & Walk-Forward Validation</h2><p>Run event-driven backtests with out-of-sample validation and regime drawdown analysis</p></div>
    <div className="page-body">
      {/* Run Controls */}
      <div className="card animate-in" style={{ marginBottom: 16 }}>
        <div className="card-header"><span className="card-title">Run Backtest</span><span className="card-badge badge-purple">LIVE</span></div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))', gap: 10, marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>Strategy</div>
            <select value={strategy} onChange={e => setStrategy(e.target.value)} style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-input)', color: 'var(--text-bright)', fontSize: 12 }}>
              {['ensemble', 'wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          {[['Capital ($)', config.initial_capital, 'initial_capital'], ['Train (yr)', config.train_years, 'train_years'], ['Test (yr)', config.test_years, 'test_years']].map(([label, val, key]) => (
            <div key={key}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
              <input type="number" step="any" value={val} onChange={e => setConfig(p => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-input)', color: 'var(--text-bright)', fontFamily: 'var(--font-mono)', fontSize: 12 }} />
            </div>
          ))}
        </div>
        <button onClick={handleRun} disabled={running} style={{ padding: '8px 24px', borderRadius: 6, border: 'none', background: running ? 'var(--bg-input)' : 'var(--gold-primary)', color: running ? 'var(--text-muted)' : '#000', fontWeight: 600, cursor: running ? 'not-allowed' : 'pointer', fontSize: 12 }}>
          {running ? <><Loader size={12} style={{ marginRight: 4, verticalAlign: 'middle', animation: 'spin 1s linear infinite' }} /> Simulating Events...</> : <><Play size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Run Full Validation</>}
        </button>
        {error && <div style={{ marginTop: 10, padding: 10, borderRadius: 'var(--radius-sm)', background: 'var(--red-dim)', color: 'var(--red)', fontSize: 12 }}>⚠ {error}</div>}
      </div>

      {/* Results */}
      {results && (<>
        <div className="kpi-grid" style={{ marginBottom: 20 }}>
          {[
            { label: 'Total Return', value: `${(metrics.total_return || 0).toFixed(1)}%`, sub: `Cost: $${(metrics.transaction_costs || 0).toFixed(0)}` },
            { label: 'Sharpe Ratio', value: (metrics.sharpe_ratio || 0).toFixed(2), sub: 'Annualized' },
            { label: 'OOS Degradation', value: (metrics.oos_degradation || 0).toFixed(2), sub: 'Walk-Forward IS vs OOS Sharpe', color: (metrics.oos_degradation || 0) < 0.5 ? 'var(--green)' : 'var(--red)' },
            { label: 'Max Drawdown', value: `${(metrics.max_drawdown || 0).toFixed(1)}%`, sub: 'Peak-to-trough' },
            { label: 'Win Rate', value: `${(metrics.win_rate || 0).toFixed(1)}%`, sub: `${metrics.num_trades} Trades` },
          ].map((m, i) => (<div key={i} className="kpi-card animate-in">
            <div className="kpi-label">{m.label}</div><div className="kpi-value" style={{ fontSize: 22, color: m.color }}>{m.value}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{m.sub}</div>
          </div>))}
        </div>

        <div className="grid-2">
            <div className="card animate-in">
                <div className="card-header"><span className="card-title">Regime-Specific Drawdowns</span><span className="card-badge badge-blue">PHASE 6</span></div>
                <div style={{ padding: '0 20px 20px', fontSize: 12, color: 'var(--text-muted)' }}>
                    Maximum drawdown experienced during specific market regimes (HMM Classification).
                </div>
                <ResponsiveContainer width="100%" height={260}>
                <BarChart data={regimeData}><XAxis dataKey="name" tick={{ fontSize: 11, fill: '#9ca3af' }} /><YAxis tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(val) => `${val}%`} /><Tooltip content={<TT />} />
                    <Bar dataKey="drawdown" name="Max DD %" radius={[4, 4, 0, 0]}>{regimeData.map((d, i) => <Cell key={i} fill={d.fill} />)}</Bar>
                </BarChart>
                </ResponsiveContainer>
            </div>
            <div className="card animate-in">
                <div className="card-header"><span className="card-title">Equity Curve</span></div>
                <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={equityData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="step" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
                        <Tooltip content={<TT />} />
                        <Line type="monotone" dataKey="equity" stroke="var(--gold-primary)" strokeWidth={2} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
        
        <div className="card animate-in" style={{ marginTop: 16 }}>
            <div className="card-header"><span className="card-title">Recent Trades</span></div>
            <table className="data-table">
                <thead><tr><th>Entry Date</th><th>Exit Date</th><th>Direction</th><th>Size</th><th>PnL</th></tr></thead>
                <tbody>
                {(results.trades || []).slice(0, 10).map((t, i) => (<tr key={i}>
                    <td className="mono">{t.entry_date.split('T')[0]}</td>
                    <td className="mono">{t.exit_date.split('T')[0]}</td>
                    <td style={{ fontWeight: 600, color: t.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{t.direction}</td>
                    <td className="mono">{t.size}</td>
                    <td className="mono" style={{ color: t.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {t.pnl >= 0 ? '+' : '-'}${Math.abs(t.pnl).toFixed(2)}
                    </td>
                </tr>))}
                </tbody>
            </table>
        </div>
      </>)}

      {!results && !running && (
        <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>📈</div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>Run Live Validation</div>
          <div style={{ fontSize: 12, marginTop: 8 }}>Tests the strategy across historical regimes with real execution slippage.</div>
        </div>
      )}
    </div>
  </>);
}
