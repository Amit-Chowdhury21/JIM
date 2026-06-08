import { useState, useEffect, useRef } from 'react';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { Play, Pause, Square, DollarSign, TrendingUp, Activity, Target, ArrowUpRight, ArrowDownRight, Minus, Wifi, WifiOff, RefreshCw, Zap } from 'lucide-react';
import { startPaperTrading, stopPaperTrading, fetchPaperTradingStatus, fetchPaperTradingPerformance, fetchPaperTradingTrades, fetchLiveSignals, fetchRiskReport, resetDailyCounters, fetchModelWeights, enableAutoTrading, fetchLSTMLogs } from '../data/api';


const signalColor = (s) => s === 'LONG' ? 'var(--green)' : s === 'SHORT' ? 'var(--red)' : 'var(--text-muted)';
const signalBg = (s) => s === 'LONG' ? 'var(--green-dim)' : s === 'SHORT' ? 'var(--red-dim)' : 'var(--bg-input)';
const signalIcon = (s) => s === 'LONG' ? <ArrowUpRight size={12}/> : s === 'SHORT' ? <ArrowDownRight size={12}/> : <Minus size={12}/>;
const CTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (<div className="custom-tooltip"><div className="label">{label}</div>
    {payload.map((p, i) => (<div key={i} className="value" style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toLocaleString(undefined,{maximumFractionDigits:2}) : p.value}</div>))}
  </div>);
};

export default function PaperTrading() {
  const [live, setLive] = useState(false);
  const [status, setStatus] = useState(null);
  const [perf, setPerf] = useState(null);
  const [trades, setTrades] = useState(null);
  const [signals, setSignals] = useState(null);
  const [risk, setRisk] = useState(null);
  const [eqHistory, setEqHistory] = useState([]);
  const [starting, setStarting] = useState(false);
  const [config, setConfig] = useState({ initial_capital:100000, kelly_fraction:0.25, max_position_pct:0.10, max_daily_loss_pct:0.02, max_drawdown_pct:0.15, min_confidence:0.60 });
  const [weights, setWeights] = useState(null);
  const [lstmLogs, setLstmLogs] = useState([]);
  const [autoTrading, setAutoTrading] = useState(false);
  const refreshRef = useRef(null);
  refreshRef.current = async () => {
    try {
      const s = await fetchPaperTradingStatus();
      setStatus(s); setLive(true);
      if (s?.engines && s.engines['v2.1']?.portfolio?.total_value) {
        setEqHistory(prev => {
          const newEntry = { 
            date: new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}), 
            v1: s.engines['v1.0']?.portfolio?.total_value || 100000,
            v2: s.engines['v2.0']?.portfolio?.total_value || 100000,
            v21: s.engines['v2.1']?.portfolio?.total_value || 100000,
            v23: s.engines['v2.3']?.portfolio?.total_value || 100000
          };
          const n = [...prev, newEntry];
          return n.length > 300 ? n.slice(-300) : n;
        });
      }
    } catch { /* offline */ }
    try { setPerf(await fetchPaperTradingPerformance()); } catch { /* offline */ }
    try { setTrades(await fetchPaperTradingTrades(20)); } catch { /* offline */ }
    try { setSignals(await fetchLiveSignals()); } catch { /* offline */ }
    try { setRisk(await fetchRiskReport()); } catch { /* offline */ }
    try { setWeights(await fetchModelWeights()); } catch { /* offline */ }
    try { 
      const logsResp = await fetchLSTMLogs(15); 
      setLstmLogs(logsResp?.logs || []); 
    } catch { /* offline */ }
  };

  useEffect(() => { refreshRef.current?.(); const t = setInterval(() => refreshRef.current?.(), 5000); return () => clearInterval(t); }, []);

  const handleStart = async () => {
    setStarting(true);
    try { await startPaperTrading(config); refreshRef.current?.(); } catch(e) { alert('Start failed: ' + e.message); }
    setStarting(false);
  };
  const handleStop = async () => { if(!confirm('Stop engine?')) return; try { await stopPaperTrading(); refreshRef.current?.(); } catch(e) { alert(e.message); } };
  const handleReset = async () => { try { await resetDailyCounters(); refreshRef.current?.(); } catch(e) { alert(e.message); } };
  const handleAutoTrade = async () => { 
    if(!confirm('WARNING: This will allow LSTM models to execute real paper trades automatically. Proceed?')) return;
    try { await enableAutoTrading(); setAutoTrading(true); } catch(e) { alert(e.message); }
  };

  const engineStatus = status?.status || 'NOT STARTED';
  const isRunning = engineStatus === 'RUNNING';
  
  const engines = ['v1.0', 'v2.0', 'v2.1', 'v2.3'];
  const equityData = eqHistory.length > 3 ? eqHistory : [];

  return (<>
    <div className="page-header">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start', flexWrap:'wrap', gap:16}}>
        <div>
          <h2>📄 Paper Trading Engine (Multi-Model)</h2>
          <p>Live side-by-side comparison of v1.0, v2.0, v2.1, and v2.3 Execution Engines</p>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:10, flexWrap:'wrap'}}>
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'5px 12px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:'var(--gold-dim)',color:'var(--gold)', border:'1px solid rgba(240,185,11,0.3)'}}>
            DXY: {signals?.macro?.dxy?.toFixed(2) || '---'} | US10Y: {signals?.macro?.us10y?.toFixed(2) || '---'}% | GSR: {signals?.macro?.gold_silver_ratio?.toFixed(1) || '---'}
          </div>
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'5px 12px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:live?'var(--green-dim)':'var(--red-dim)',color:live?'var(--green)':'var(--red)',
            border:`1px solid ${live?'rgba(0,196,140,0.3)':'rgba(255,77,106,0.3)'}`}}>
            {live?<Wifi size={12}/>:<WifiOff size={12}/>} {live?'LIVE':'OFFLINE'}
          </div>
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'6px 14px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:isRunning?'var(--green-dim)':'var(--red-dim)',color:isRunning?'var(--green)':'var(--red)',
            border:`1px solid ${isRunning?'rgba(0,196,140,0.3)':'rgba(255,77,106,0.3)'}`}}>
            {isRunning?<Play size={12}/>:<Pause size={12}/>} {engineStatus}
          </div>
        </div>
      </div>
    </div>
    <div className="page-body">
      {/* Engine Controls */}
      <div className="card animate-in" style={{marginBottom:16}}>
          <div className="card-header">
            <span className="card-title">Orchestrator Controls</span>
            <span className="card-badge badge-gold">LIVE API</span>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(140px,1fr))',gap:10,marginBottom:12}}>
            {[['Capital ($)',config.initial_capital,'initial_capital'],['Kelly Frac (v2+)',config.kelly_fraction,'kelly_fraction'],['Max Pos % (v2+)',config.max_position_pct,'max_position_pct'],
              ['Max Daily Loss %',config.max_daily_loss_pct,'max_daily_loss_pct'],['Max DD %',config.max_drawdown_pct,'max_drawdown_pct']
            ].map(([label,val,key])=>(
              <div key={key}>
                <div style={{fontSize:10,color:'var(--text-muted)',marginBottom:4}}>{label}</div>
                <input type="number" step="any" value={val} onChange={e=>setConfig(p=>({...p,[key]:parseFloat(e.target.value)||0}))}
                  style={{width:'100%',padding:'6px 10px',borderRadius:6,border:'1px solid var(--border-color)',background:'var(--bg-input)',color:'var(--text-bright)',fontFamily:'var(--font-mono)',fontSize:12}} />
              </div>
            ))}
          </div>
          <div style={{display:'flex',gap:10, marginBottom: 16}}>
            <button onClick={handleStart} disabled={starting||isRunning} style={{padding:'8px 20px',borderRadius:6,border:'none',background:isRunning?'var(--bg-input)':'var(--green)',color:isRunning?'var(--text-muted)':'#000',fontWeight:600,cursor:isRunning?'not-allowed':'pointer',fontSize:12}}>
              <Play size={12} style={{marginRight:4,verticalAlign:'middle'}}/> {starting?'Starting...':'Start All Engines'}
            </button>
            <button onClick={handleStop} disabled={!isRunning} style={{padding:'8px 20px',borderRadius:6,border:'none',background:isRunning?'var(--red)':'var(--bg-input)',color:isRunning?'#fff':'var(--text-muted)',fontWeight:600,cursor:isRunning?'pointer':'not-allowed',fontSize:12}}>
              <Square size={12} style={{marginRight:4,verticalAlign:'middle'}}/> Stop All
            </button>
            <button onClick={handleReset} disabled={!isRunning} style={{padding:'8px 20px',borderRadius:6,border:'1px solid var(--border-color)',background:'var(--bg-secondary)',color:'var(--text-secondary)',fontWeight:600,cursor:isRunning?'pointer':'not-allowed',fontSize:12}}>
              <RefreshCw size={12} style={{marginRight:4,verticalAlign:'middle'}}/> Reset Daily
            </button>
            <button onClick={handleAutoTrade} disabled={!isRunning || autoTrading} style={{padding:'8px 20px',borderRadius:6,border:'1px solid var(--border-color)',background:autoTrading?'var(--green-dim)':'var(--bg-secondary)',color:autoTrading?'var(--green)':'var(--text-bright)',fontWeight:600,cursor:(!isRunning || autoTrading)?'not-allowed':'pointer',fontSize:12}}>
              <Zap size={12} style={{marginRight:4,verticalAlign:'middle'}}/> {autoTrading ? 'Auto Trade: ON' : 'Trade all lstm'}
            </button>
          </div>
        </div>

      {/* 4-Column Layout for Models */}
      <div style={{display:'grid', gridTemplateColumns:'repeat(4, 1fr)', gap: 16, marginBottom:20}}>
        {engines.map((eng) => {
          const st = status?.engines?.[eng] || {};
          const p = perf?.[eng] || {};
          const pf = st.portfolio || {};
          const totalValue = pf.total_value ?? 100000;
          const pnlTotal = pf.pnl_total ?? 0;
          const returnPct = pf.return_pct ?? 0;
          const winRate = p.win_rate ?? 0;
          const numTrades = p.num_trades ?? pf.num_trades ?? 0;
          const engTrades = trades?.[eng] || [];

          let color = eng === 'v1.0' ? 'var(--blue)' : eng === 'v2.0' ? 'var(--gold-primary)' : eng === 'v2.1' ? 'var(--green)' : 'var(--purple)';
          let dataKey = eng === 'v1.0' ? 'v1' : eng === 'v2.0' ? 'v2' : eng === 'v2.1' ? 'v21' : 'v23';

          return (
            <div key={eng} style={{display:'flex', flexDirection:'column', gap:16}}>
              
              {/* Header */}
              <div style={{background: 'var(--bg-secondary)', border: `1px solid ${color}`, borderRadius: 8, padding: 12, textAlign:'center'}}>
                <h3 style={{margin:0, color}}>{eng}</h3>
                <div style={{fontSize: 12, color: 'var(--text-muted)'}}>
                    {eng === 'v1.0' ? 'Raw LSTM' : eng === 'v2.0' ? 'CNN-LSTM Ensemble' : eng === 'v2.1' ? 'Execution Engine' : 'Alpha Engine'}
                </div>
              </div>

              {/* KPIs */}
              <div className="card animate-in" style={{padding: 12}}>
                <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:10}}>
                  <div className="kpi-card" style={{margin:0, padding: 10}}>
                    <div className="kpi-label">P&L</div>
                    <div className="kpi-value" style={{fontSize: 20}}>${Math.abs(pnlTotal).toLocaleString(undefined,{maximumFractionDigits:0})}</div>
                    <div className={`kpi-change ${pnlTotal>=0?'positive':'negative'}`}>{pnlTotal>=0?'+':'-'}{Math.abs(returnPct).toFixed(2)}%</div>
                  </div>
                  <div className="kpi-card" style={{margin:0, padding: 10}}>
                    <div className="kpi-label">Win Rate</div>
                    <div className="kpi-value" style={{fontSize: 20}}>{(winRate*100).toFixed(1)}%</div>
                    <div className="kpi-change positive">{numTrades} trades</div>
                  </div>
                </div>
              </div>

              {/* Equity Curve */}
              <div className="card animate-in" style={{padding: 12}}>
                  <div className="card-header" style={{padding:0, marginBottom:10}}><span className="card-title" style={{fontSize: 14}}>Equity</span></div>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={equityData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                      <XAxis dataKey="date" tick={{fill:'#6b7280',fontSize:10}} tickFormatter={v=>typeof v==='string'&&v.length>5?v.slice(5):v}/>
                      <YAxis tick={{fill:'#6b7280',fontSize:10}} tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} domain={['dataMin-500','dataMax+500']} width={40}/>
                      <Tooltip content={<CTooltip/>}/>
                      <Area type="monotone" dataKey={dataKey} stroke={color} fill="transparent" strokeWidth={2} name="Equity" />
                    </AreaChart>
                  </ResponsiveContainer>
              </div>

              {/* Trade History */}
              <div className="card animate-in" style={{flex: 1, padding: 12}}>
                <div className="card-header" style={{padding:0, marginBottom:10}}><span className="card-title" style={{fontSize: 14}}>Trades</span></div>
                {engTrades.length === 0 ? (
                  <div className="empty-state" style={{minHeight:150}}>No trades</div>
                ) : (
                  <div className="table-responsive">
                    <table className="table">
                      <thead><tr><th>Signal</th><th>Entry</th><th>P&L</th></tr></thead>
                      <tbody>
                        {engTrades.slice(0, 10).map(t => (
                          <tr key={t.trade_id}>
                            <td><span className="signal-badge" style={{background:signalBg(t.signal_type),color:signalColor(t.signal_type)}}>{signalIcon(t.signal_type)}</span></td>
                            <td>${t.entry_price.toFixed(1)}</td>
                            <td style={{color:t.pnl>=0?'var(--green)':'var(--red)'}}>{t.pnl>=0?'+':''}${t.pnl.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

            </div>
          )
        })}
      </div>

      {/* LSTM LOGS SECTION */}
      <div className="card animate-in" style={{padding: 16}}>
        <div className="card-header" style={{padding:0, marginBottom:10}}>
          <span className="card-title">LSTM LOGS</span>
          <span className="card-badge badge-blue">CSV Output (IST)</span>
        </div>
        <div style={{
          background: '#0a0a0a', 
          border: '1px solid #333', 
          borderRadius: 6, 
          padding: 12, 
          height: 250, 
          overflowY: 'auto',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--green)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {lstmLogs.length === 0 ? (
            <div style={{color: '#666'}}>Waiting for logs...</div>
          ) : (
            lstmLogs.map((line, idx) => (
              <div key={idx} style={{
                padding: '4px 0', 
                borderBottom: idx === 0 ? '1px solid #333' : 'none',
                color: idx === 0 ? '#888' : 'var(--green)',
                fontWeight: idx === 0 ? 'bold' : 'normal',
                whiteSpace: 'pre-wrap'
              }}>
                {line}
              </div>
            ))
          )}
        </div>
      </div>

    </div>
  </>);
}
