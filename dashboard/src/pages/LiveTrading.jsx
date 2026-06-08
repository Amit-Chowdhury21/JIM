import { useState, useEffect, useRef } from 'react';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Play, Square, DollarSign, TrendingUp, Activity, Target, ArrowUpRight, ArrowDownRight, Minus, Wifi, WifiOff, RefreshCw, Zap, ShieldCheck, AlertTriangle, Clock } from 'lucide-react';
import { fetchPaperTradingStatus, fetchPaperTradingPerformance, fetchPaperTradingTrades, fetchLiveSignals, fetchRiskReport, fetchModelWeights } from '../data/api';

const signalColor = (s) => s === 'LONG' ? 'var(--green)' : s === 'SHORT' ? 'var(--red)' : 'var(--text-muted)';
const signalBg = (s) => s === 'LONG' ? 'var(--green-dim)' : s === 'SHORT' ? 'var(--red-dim)' : 'var(--bg-input)';
const signalIcon = (s) => s === 'LONG' ? <ArrowUpRight size={12}/> : s === 'SHORT' ? <ArrowDownRight size={12}/> : <Minus size={12}/>;

const CTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (<div className="custom-tooltip"><div className="label">{label}</div>
    {payload.map((p, i) => (<div key={i} className="value" style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toLocaleString(undefined,{maximumFractionDigits:2}) : p.value}</div>))}
  </div>);
};

export default function LiveTrading() {
  const [live, setLive] = useState(false);
  const [status, setStatus] = useState(null);
  const [perf, setPerf] = useState(null);
  const [trades, setTrades] = useState([]);
  const [signals, setSignals] = useState(null);
  const [risk, setRisk] = useState(null);
  const [eqHistory, setEqHistory] = useState([]);
  const [weights, setWeights] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [tickCount, setTickCount] = useState(0);

  const refreshRef = useRef(null);
  refreshRef.current = async () => {
    try {
      const s = await fetchPaperTradingStatus();
      setStatus(s); setLive(true); setLastUpdate(new Date());
      setTickCount(prev => prev + 1);
      if (s?.portfolio?.total_value) {
        setEqHistory(prev => {
          const n = [...prev, { date: new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}), equity: s.portfolio.total_value }];
          return n.length > 500 ? n.slice(-500) : n;
        });
      }
    } catch { setLive(false); }
    try { setPerf(await fetchPaperTradingPerformance()); } catch {}
    try { setTrades(await fetchPaperTradingTrades(30)); } catch {}
    try { setSignals(await fetchLiveSignals()); } catch {}
    try { setRisk(await fetchRiskReport()); } catch {}
    try { setWeights(await fetchModelWeights()); } catch {}
  };

  useEffect(() => { refreshRef.current?.(); const t = setInterval(() => refreshRef.current?.(), 5000); return () => clearInterval(t); }, []);

  const engineStatus = status?.status || 'OFFLINE';
  const isRunning = engineStatus === 'RUNNING';
  const pf = status?.portfolio || {};
  const totalValue = pf.total_value ?? 100000;
  const pnlTotal = pf.pnl_total ?? 0;
  const dailyPnl = pf.pnl_daily ?? 0;
  const returnPct = pf.return_pct ?? 0;
  const winRate = perf?.win_rate ?? 0;
  const sharpe = perf?.sharpe_ratio ?? 0;
  const maxDD = perf?.max_drawdown ?? 0;
  const numTrades = perf?.num_trades ?? pf.num_trades ?? 0;
  const modelSigs = signals?.models || {};
  const silverData = status?.silver || {};
  const agBeta = silverData?.beta ?? 1.0;
  const agFeed = silverData?.feed_status || {current_xag: 0, current_ratio: 0, fetch_count: 0, error_count: 0};
  const equityData = eqHistory.length > 3 ? eqHistory : [];
  const tradeList = Array.isArray(trades) && trades.length > 0 ? trades : [];

  // Risk data
  const rr = risk?.risk_report || {};
  const breakers = [
    { name: 'Drawdown', value: rr.drawdown_pct ?? 0, limit: 15, unit: '%', ok: (rr.drawdown_pct ?? 0) < 10 },
    { name: 'Daily P&L', value: rr.daily_pnl ?? 0, limit: -2000, unit: '$', ok: (rr.daily_pnl ?? 0) > -2000 },
    { name: 'Consec. Losses', value: rr.consecutive_losses ?? 0, limit: 3, unit: '', ok: (rr.consecutive_losses ?? 0) < 3 },
  ];

  return (<>
    <div className="page-header">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start', flexWrap:'wrap', gap:16}}>
        <div>
          <h2>Live Trading Engine</h2>
          <p>Real-time autonomous gold trading — 7-model ensemble with Kelly sizing &amp; circuit breakers</p>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:10, flexWrap:'wrap'}}>
          {/* Macro bar */}
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'5px 12px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:'var(--gold-dim)',color:'var(--gold)', border:'1px solid rgba(240,185,11,0.3)'}}>
            DXY: {signals?.macro?.dxy?.toFixed(2) || '---'} | US10Y: {signals?.macro?.us10y?.toFixed(2) || '---'}% | GSR: {signals?.macro?.gold_silver_ratio?.toFixed(1) || '---'}
          </div>
          {/* Regime */}
          {modelSigs['hmm_pro']?.regime && <div style={{display:'flex',alignItems:'center',gap:6,padding:'5px 12px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:modelSigs['hmm_pro']?.regime==='GROWTH'?'var(--green-dim)':modelSigs['hmm_pro']?.regime==='CRISIS'?'var(--red-dim)':'var(--blue-dim)',
            color:modelSigs['hmm_pro']?.regime==='GROWTH'?'var(--green)':modelSigs['hmm_pro']?.regime==='CRISIS'?'var(--red)':'var(--blue)',
            border:'1px solid rgba(59,130,246,0.3)'}}>
            <Activity size={12}/> {modelSigs['hmm_pro']?.regime}
          </div>}
          {/* Live status */}
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'5px 12px',borderRadius:'var(--radius-sm)',fontSize:12,fontWeight:600,
            background:isRunning?'var(--green-dim)':'var(--red-dim)',color:isRunning?'var(--green)':'var(--red)',
            border:`1px solid ${isRunning?'rgba(0,196,140,0.3)':'rgba(255,77,106,0.3)'}`}}>
            {isRunning?<Wifi size={12}/>:<WifiOff size={12}/>} {isRunning?'LIVE':'OFFLINE'}
          </div>
        </div>
      </div>
    </div>
    <div className="page-body">

      {/* Status banner */}
      <div className="card animate-in" style={{marginBottom:16, background: isRunning
        ? 'linear-gradient(135deg, rgba(0,196,140,0.08) 0%, rgba(240,185,11,0.06) 100%)'
        : 'linear-gradient(135deg, rgba(255,77,106,0.08) 0%, rgba(139,92,246,0.06) 100%)'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'4px 0',flexWrap:'wrap',gap:12}}>
          <div style={{display:'flex',alignItems:'center',gap:12}}>
            <div style={{width:10,height:10,borderRadius:'50%',
              background:isRunning?'var(--green)':'var(--red)',
              boxShadow:isRunning?'0 0 12px var(--green)':'0 0 12px var(--red)',
              animation:isRunning?'pulse 2s ease-in-out infinite':'none'}}/>
            <div>
              <div style={{fontSize:14,fontWeight:700,color:'var(--text-bright)'}}>
                {isRunning ? 'Live Trader Running' : 'Live Trader Offline'}
              </div>
              <div style={{fontSize:11,color:'var(--text-muted)'}}>
                {lastUpdate ? `Last update: ${lastUpdate.toLocaleTimeString()} • Tick #${tickCount}` : 'Waiting for connection...'}
              </div>
            </div>
          </div>
          <div style={{display:'flex',gap:20,flexWrap:'wrap',fontSize:12}}>
            <div><span style={{color:'var(--text-muted)'}}>Interval:</span> <span style={{color:'var(--gold)',fontWeight:600,fontFamily:'var(--font-mono)'}}>60s</span></div>
            <div><span style={{color:'var(--text-muted)'}}>Silver Hedge:</span> <span style={{color:'var(--blue)',fontWeight:600}}>Ag=${agFeed.current_xag > 0 ? agFeed.current_xag.toFixed(2) : 'WAIT'} | β={agBeta.toFixed(3)}</span></div>
            <div><span style={{color:'var(--text-muted)'}}>Models:</span> <span style={{color:'var(--green)',fontWeight:600}}>5 Active</span></div>
            <div><span style={{color:'var(--text-muted)'}}>Risk:</span> <span style={{color:breakers.every(b=>b.ok)?'var(--green)':'var(--red)',fontWeight:600}}>{breakers.every(b=>b.ok)?'ALL CLEAR':'ALERT'}</span></div>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="kpi-grid" style={{gridTemplateColumns:'repeat(auto-fit,minmax(150px,1fr))'}}>
        {[
          {label:'Portfolio',value:`$${totalValue.toLocaleString(undefined,{maximumFractionDigits:0})}`,change:`${returnPct>=0?'+':''}${returnPct.toFixed(2)}%`,positive:returnPct>=0,icon:<DollarSign size={16}/>},
          {label:'Total P&L',value:`${pnlTotal>=0?'+':''}$${Math.abs(pnlTotal).toLocaleString(undefined,{maximumFractionDigits:2})}`,change:`Daily: $${dailyPnl.toFixed(2)}`,positive:pnlTotal>=0,icon:<TrendingUp size={16}/>},
          {label:'Win Rate',value:`${(winRate*100).toFixed(1)}%`,change:`${numTrades} trades`,positive:winRate>0.5,icon:<Target size={16}/>},
          {label:'Sharpe',value:sharpe.toFixed(2),change:'Annualized',positive:sharpe>1.5,icon:<Activity size={16}/>},
          {label:'Max DD',value:`${maxDD.toFixed(2)}%`,change:'Limit: 15%',positive:Math.abs(maxDD)<5,icon:<ArrowDownRight size={16}/>},
        ].map((kpi,i)=>(<div key={i} className="kpi-card animate-in">
          <div className="kpi-label"><span style={{width:28,height:28,borderRadius:8,display:'flex',alignItems:'center',justifyContent:'center',background:'var(--gold-glow)',color:'var(--gold-primary)'}}>{kpi.icon}</span> {kpi.label}</div>
          <div className="kpi-value">{kpi.value}</div>
          <div className={`kpi-change ${kpi.positive?'positive':'negative'}`}>{kpi.change}</div>
        </div>))}
      </div>

      {/* Equity + Circuit Breakers */}
      <div className="grid-2" style={{marginBottom:20}}>
        <div className="card animate-in">
          <div className="card-header"><span className="card-title">Equity Curve</span><span className={`card-badge ${isRunning?'badge-green':'badge-orange'}`}>{isRunning?'LIVE':'WAITING'}</span></div>
          {equityData.length > 2 ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={equityData}>
                <defs><linearGradient id="eqLive" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00c48c" stopOpacity={0.3}/><stop offset="100%" stopColor="#00c48c" stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                <XAxis dataKey="date" tick={{fill:'#6b7280',fontSize:9}} interval={Math.max(1,Math.floor(equityData.length/6))}/>
                <YAxis tick={{fill:'#6b7280',fontSize:10}} tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} domain={['dataMin-500','dataMax+500']}/>
                <Tooltip content={<CTooltip/>}/><Area type="monotone" dataKey="equity" stroke="#00c48c" fill="url(#eqLive)" strokeWidth={2} name="Equity"/>
              </AreaChart>
            </ResponsiveContainer>
          ) : (<div style={{height:260,display:'flex',alignItems:'center',justifyContent:'center',color:'var(--text-muted)',fontSize:13}}>Awaiting equity data...</div>)}
        </div>
        <div className="card animate-in">
          <div className="card-header"><span className="card-title">Circuit Breakers</span>
            <span className={`card-badge ${breakers.every(b=>b.ok)?'badge-green':'badge-red'}`}>
              {breakers.every(b=>b.ok)?'ALL CLEAR':'WARNING'}
            </span>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:14,padding:'8px 0'}}>
            {breakers.map((b,i) => (
              <div key={i} style={{padding:'12px 16px',borderRadius:'var(--radius-sm)',background:'var(--bg-secondary)',border:`1px solid ${b.ok?'var(--border-color)':'rgba(255,77,106,0.3)'}`}}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
                  <span style={{fontSize:12,fontWeight:600,color:'var(--text-bright)',display:'flex',alignItems:'center',gap:6}}>
                    {b.ok ? <ShieldCheck size={14} style={{color:'var(--green)'}}/> : <AlertTriangle size={14} style={{color:'var(--red)'}}/>}
                    {b.name}
                  </span>
                  <span style={{fontSize:12,fontWeight:700,fontFamily:'var(--font-mono)',color:b.ok?'var(--green)':'var(--red)'}}>
                    {b.unit==='$'?`$${b.value.toFixed(0)}`:`${typeof b.value === 'number' ? b.value.toFixed(1) : b.value}${b.unit}`}
                  </span>
                </div>
                <div className="progress-bar" style={{height:6}}>
                  <div className="progress-fill" style={{
                    width:`${Math.min(100,Math.abs(b.value/b.limit)*100)}%`,
                    background:b.ok?'var(--green)':'var(--red)'
                  }}/>
                </div>
                <div style={{fontSize:10,color:'var(--text-muted)',marginTop:4,textAlign:'right'}}>Limit: {b.unit==='$'?`$${b.limit}`:b.limit}{b.unit}</div>
              </div>
            ))}
            {/* Risk summary */}
            {rr.violations && rr.violations.length > 0 && (
              <div style={{padding:'10px 14px',borderRadius:'var(--radius-sm)',background:'rgba(255,77,106,0.08)',border:'1px solid rgba(255,77,106,0.3)',fontSize:11,color:'var(--red)'}}>
                <AlertTriangle size={12} style={{verticalAlign:'middle',marginRight:4}}/> Violations: {rr.violations.join(', ')}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Signals */}
      <div className="card animate-in" style={{marginBottom:20}}>
        <div className="card-header"><span className="card-title">Model Signals (Live)</span>
          <div style={{display:'flex',gap:8}}>
            <span className="card-badge badge-gold">5 MODELS</span>
            {weights && <span className={`card-badge ${weights.adaptation_active?'badge-green':'badge-orange'}`} style={{fontSize:10}}>
              <Zap size={10} style={{marginRight:3}}/>{weights.adaptation_active?'ADAPTIVE':'BASE'}
            </span>}
          </div>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(200px, 1fr))',gap:10}}>
          {Object.entries(modelSigs)
            .filter(([m]) => ['wavelet_pro', 'hmm_pro', 'lstm', 'tft_pro', 'ensemble'].includes(m))
            .map(([model,sig])=>{
            const s = sig.signal || sig.lastSignal || 'HOLD';
            const c = sig.confidence ?? 0;
            const w = weights?.weights?.[model];
            const displayName = model.replace('_', ' ');
            return (<div key={model} style={{padding:12,borderRadius:'var(--radius-sm)',background:'var(--bg-secondary)',border:'1px solid var(--border-color)'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
                <span style={{fontSize:12,fontWeight:600,color:'var(--text-bright)',textTransform:'capitalize'}}>{displayName}</span>
                <span style={{fontSize:10,fontWeight:700,padding:'2px 8px',borderRadius:12,background:signalBg(s),color:signalColor(s),display:'flex',alignItems:'center',gap:3}}>
                  {signalIcon(s)} {s}
                </span>
              </div>
              <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'var(--text-muted)'}}>
                <span>Conf: <span style={{color:c>=0.7?'var(--green)':c>=0.5?'var(--orange)':'var(--red)',fontWeight:600,fontFamily:'var(--font-mono)'}}>{(c*100).toFixed(0)}%</span></span>
                {w != null && <span>Wt: <span style={{color:'var(--gold)',fontWeight:600,fontFamily:'var(--font-mono)'}}>{(w*100).toFixed(0)}%</span></span>}
              </div>
              <div style={{marginTop:5}}><div className="progress-bar" style={{height:4}}><div className="progress-fill" style={{width:`${c*100}%`,background:c>=0.7?'var(--green)':c>=0.5?'var(--orange)':'var(--red)'}}/></div></div>
            </div>);
          })}
        </div>
      </div>

      {/* Trade History */}
      <div className="card animate-in" style={{marginBottom:20}}>
        <div className="card-header"><span className="card-title">Trade History</span><span className="card-badge badge-purple">{tradeList.length} TRADES</span></div>
        <div className="table-responsive">
          <table className="data-table"><thead><tr><th>ID</th><th>Model</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Status</th><th>Time</th></tr></thead>
            <tbody>{tradeList.map((t,i)=>{
              const id = t.trade_id||t.tradeId||`#${i}`;
              const model = t.model_name||t.model||'—';
              const sig = t.signal_type||t.signal||'—';
              const entry = t.entry_price||t.entry||0;
              const exit = t.exit_price||t.exit||null;
              const qty = t.quantity||t.qty||0;
              const pnl = t.pnl||0;
              const st = t.status||'OPEN';
              const time = t.entry_time ? new Date(t.entry_time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : t.time||'—';
              return (<tr key={i}>
                <td className="mono">{typeof id==='string'?id.slice(0,8):id}</td>
                <td style={{fontWeight:600}}>{model}</td>
                <td><span style={{color:signalColor(sig),fontWeight:600,fontSize:11}}>{sig}</span></td>
                <td className="mono">${Number(entry).toFixed(2)}</td>
                <td className="mono">{exit?`$${Number(exit).toFixed(2)}`:'—'}</td>
                <td className="mono">
                  {Number(qty).toFixed(2)}
                  {t.silver_quantity > 0 && <span style={{display:'block', fontSize:10, color:'var(--blue)'}}>+ {t.silver_quantity.toFixed(1)}oz Ag</span>}
                </td>
                <td className="mono" style={{color:pnl>=0?'var(--green)':'var(--red)',fontWeight:600}}>{pnl>=0?'+':''}${Number(pnl).toFixed(2)}</td>
                <td><span className={`card-badge ${st==='CLOSED'?'badge-green':'badge-blue'}`}>{st}</span></td>
                <td style={{fontSize:11,color:'var(--text-muted)'}}>{time}</td>
              </tr>);
            })}</tbody>
          </table>
        </div>
      </div>

      {/* Risk Report */}
      {rr.current_equity && (
        <div className="card animate-in">
          <div className="card-header"><span className="card-title">Risk Report</span><span className="card-badge badge-red">REAL-TIME</span></div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))',gap:12}}>
            {[
              {label:'Current Equity',value:`$${rr.current_equity?.toLocaleString(undefined,{maximumFractionDigits:0})}`},
              {label:'Peak Equity',value:`$${rr.peak_equity?.toLocaleString(undefined,{maximumFractionDigits:0})}`},
              {label:'Drawdown',value:`${rr.drawdown_pct?.toFixed(2)}%`},
              {label:'Daily P&L',value:`$${rr.daily_pnl?.toFixed(2)}`},
              {label:'Consec. Losses',value:rr.consecutive_losses},
              {label:'Violations',value:rr.violations?.length?rr.violations.join(', '):'None'},
            ].map((r,i)=>(<div key={i} style={{padding:'10px 14px',borderRadius:'var(--radius-sm)',background:'var(--bg-secondary)',border:'1px solid var(--border-color)'}}>
              <div style={{fontSize:10,color:'var(--text-muted)',textTransform:'uppercase',letterSpacing:0.5}}>{r.label}</div>
              <div style={{fontSize:14,fontWeight:600,fontFamily:'var(--font-mono)',color:'var(--text-bright)',marginTop:4}}>{r.value}</div>
            </div>))}
          </div>
        </div>
      )}
    </div>
  </>);
}
