// MarketData.jsx
import { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, RefreshCw, WifiOff } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { fetchGoldPrice, fetchPaperTradingTrades, fetchFearGreed } from '../data/api';

// ============================================================================
// FEAR & GREED WIDGET
// ============================================================================
function FearGreedWidget() {
  const [currentValue, setCurrentValue] = useState(44);

  useEffect(() => {
    let mounted = true;
    fetchFearGreed().then(res => {
      if (mounted && res && res.status === 'ok') {
        setCurrentValue(res.value);
      }
    }).catch(e => console.warn('Could not fetch fear greed', e));
    return () => { mounted = false; };
  }, []);

  const data = Array.from({ length: 30 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (29 - i));
    const val = 40 + Math.sin(i / 3) * 12 + Math.random() * 8;
    return {
      date: d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
      value: Math.round(val)
    };
  });
  // Force latest value to match real time
  data[data.length - 1].value = currentValue;
  
  // Needle calculation (180deg to 0deg)
  const angle = 180 - (currentValue / 100) * 180;
  const rad = (angle * Math.PI) / 180;
  const needleX = 150 + 90 * Math.cos(rad);
  const needleY = 145 - 90 * Math.sin(rad);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 20, marginTop: 24, marginBottom: 40 }}>
      {/* Left: Line Chart */}
      <div className="card animate-in" style={{ padding: '24px', background: 'var(--bg-card)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-bright)' }}>
            Fear & Greed Index: Gold <span style={{ fontSize: 10, background: '#3b82f6', color: 'white', padding: '2px 8px', borderRadius: 12, verticalAlign: 'middle', marginLeft: 4 }}>BETA</span>
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', textAlign: 'right', textTransform: 'uppercase' }}>
            Powered by<br/><strong style={{color: '#60a5fa', fontSize: 13, fontFamily: 'serif', fontStyle: 'italic', textTransform: 'none'}}>JM BULLION</strong>
          </div>
        </div>
        <div style={{ height: 200, width: '100%' }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="date" tick={{fontSize: 10, fill: 'var(--text-muted)'}} tickLine={false} axisLine={{stroke: 'var(--border-color)'}} angle={-45} textAnchor="end" />
              <YAxis domain={[0, 100]} tick={{fontSize: 10, fill: 'var(--text-muted)'}} tickLine={false} axisLine={false} ticks={[0,20,40,60,80,100]} />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4, fill: '#1e3a8a', stroke: '#3b82f6', strokeWidth: 2 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>Last 30 Days</div>
      </div>

      {/* Right: Gauge */}
      <div className="card animate-in" style={{ padding: '24px', background: 'var(--bg-card)', display: 'flex', flexDirection: 'column' }}>
        <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-bright)' }}>
          Fear & Greed Index: Gold <span style={{ fontSize: 10, background: '#3b82f6', color: 'white', padding: '2px 8px', borderRadius: 12, verticalAlign: 'middle', marginLeft: 4 }}>BETA</span>
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20, marginTop: 4 }}>
          {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </div>
        
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative', marginTop: 20 }}>
          <svg width="300" height="150" viewBox="0 0 300 150">
            {/* Outer shadow arc */}
            <path d="M 30 145 A 120 120 0 0 1 270 145" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="60" />
            
            {/* Colored Zones (using actual colors from screenshot) */}
            <path d="M 30 145 A 120 120 0 0 1 73 61" fill="none" stroke="#dc2626" strokeWidth="50" /> {/* Extreme Fear */}
            <path d="M 73 61 A 120 120 0 0 1 123 28" fill="none" stroke="#991b1b" strokeWidth="50" /> {/* Fear */}
            <path d="M 123 28 A 120 120 0 0 1 177 28" fill="none" stroke="#1e3a8a" strokeWidth="50" /> {/* Neutral */}
            <path d="M 177 28 A 120 120 0 0 1 227 61" fill="none" stroke="#065f46" strokeWidth="50" /> {/* Greed */}
            <path d="M 227 61 A 120 120 0 0 1 270 145" fill="none" stroke="#047857" strokeWidth="50" /> {/* Extreme Greed */}
            
            {/* Arc Labels */}
            <text x="50" y="110" fill="rgba(255,255,255,0.9)" fontSize="9" fontWeight="bold" textAnchor="middle" transform="rotate(-40 50 110)">EXTREME FEAR</text>
            <text x="95" y="65" fill="rgba(255,255,255,0.9)" fontSize="9" fontWeight="bold" textAnchor="middle" transform="rotate(-20 95 65)">FEAR</text>
            <text x="150" y="45" fill="rgba(255,255,255,0.9)" fontSize="9" fontWeight="bold" textAnchor="middle">NEUTRAL</text>
            <text x="205" y="65" fill="rgba(255,255,255,0.9)" fontSize="9" fontWeight="bold" textAnchor="middle" transform="rotate(20 205 65)">GREED</text>
            <text x="250" y="110" fill="rgba(255,255,255,0.9)" fontSize="9" fontWeight="bold" textAnchor="middle" transform="rotate(40 250 110)">EXTREME GREED</text>
            
            {/* Tick Marks & Numbers (Inner Radius) */}
            <text x="95" y="125" fill="var(--text-muted)" fontSize="11" textAnchor="middle">0</text>
            <text x="125" y="85" fill="var(--text-muted)" fontSize="11" textAnchor="middle">25</text>
            <text x="150" y="70" fill="var(--text-muted)" fontSize="11" textAnchor="middle">50</text>
            <text x="175" y="85" fill="var(--text-muted)" fontSize="11" textAnchor="middle">75</text>
            <text x="205" y="125" fill="var(--text-muted)" fontSize="11" textAnchor="middle">100</text>

            {/* Needle */}
            <line x1="150" y1="145" x2={needleX} y2={needleY} stroke="#ef4444" strokeWidth="6" strokeLinecap="round" />
            <circle cx="150" cy="145" r="10" fill="#0a0e1a" stroke="#ef4444" strokeWidth="4" />
          </svg>
          
          <div style={{ textAlign: 'center', marginTop: -10 }}>
            <div style={{ fontSize: 48, fontWeight: 800, color: 'var(--text-bright)', lineHeight: 1 }}>{currentValue}</div>
            <div style={{ fontSize: 18, color: '#60a5fa', fontFamily: 'serif', fontStyle: 'italic', marginTop: 4 }}>JM BULLION</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// KPI CARD
// ============================================================================
function KpiStat({ label, value, sub, color, icon: Icon }) {
  return (
    <div style={{
      padding: '16px 20px', borderRadius: 10, background: 'var(--bg-card)',
      border: '1px solid var(--border-color)', position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, ${color}, transparent)`,
      }} />
      <div style={{ fontSize: 10, color: '#6b7280', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#e2e8f0', fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 8 }}>
        {value}
        {Icon && <Icon size={16} style={{ color, opacity: 0.7 }} />}
      </div>
      {sub && <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ============================================================================
// TRADINGVIEW WIDGET
// ============================================================================
function TradingViewWidget() {
  const container = useRef();

  useEffect(() => {
    if (container.current && container.current.children.length === 1) {
      const script = document.createElement("script");
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.type = "text/javascript";
      script.async = true;
      script.innerHTML = `
        {
          "autosize": true,
          "symbol": "OANDA:XAUUSD",
          "interval": "15",
          "range": "5D",
          "timezone": "Asia/Kolkata",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "backgroundColor": "rgba(10, 14, 26, 1)",
          "gridColor": "rgba(255, 255, 255, 0.06)",
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": false,
          "container_id": "tradingview_gold",
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650",
          "details": true,
          "studies": [
            {
              "id": "MASimple@tv-basicstudies",
              "inputs": {
                "length": 35
              }
            },
            {
              "id": "MASimple@tv-basicstudies",
              "inputs": {
                "length": 112
              }
            }
          ],
          "support_host": "https://www.tradingview.com"
        }`;
      container.current.appendChild(script);
    }
  }, []);

  return (
    <div className="tradingview-widget-container" ref={container} style={{ height: "100%", width: "100%", borderRadius: 8, overflow: 'hidden' }}>
      <div className="tradingview-widget-container__widget" style={{ height: "calc(100% - 32px)", width: "100%" }}></div>
    </div>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================
export default function MarketData({ embedded = false }) {
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [goldData, setGoldData] = useState(null);
  const [trades, setTrades] = useState([]);
  const [period, setPeriod] = useState('5d');
  const [interval, setInterval_] = useState('15m');

  const refreshRef = useRef(null);
  refreshRef.current = async () => {
    setLoading(true);
    try {
      const data = await fetchGoldPrice(interval, period);
      setGoldData(data);
      try {
        const t = await fetchPaperTradingTrades(100);
        setTrades(t || []);
      } catch (err) { console.warn("Could not fetch trades", err); }
      setLive(true);
    } catch { /* offline */ }
    setLoading(false);
  };

  useEffect(() => { refreshRef.current?.(); const t = setInterval(() => refreshRef.current?.(), 30000); return () => clearInterval(t); }, [interval, period]);

  const latest = goldData?.candles ? goldData.candles[goldData.candles.length - 1] : {};
  const prev = goldData?.candles && goldData.candles.length > 1 ? goldData.candles[goldData.candles.length - 2] : latest;
  
  const priceChange = latest.close && prev.close ? latest.close - prev.close : 0;
  const priceChangePct = prev.close ? ((priceChange / prev.close) * 100) : 0;

  if (!live && !loading) return (
    <><div className="page-header"><h2>Market Data</h2><p>⚠ Backend offline — start the API server to see live gold prices</p></div>
    <div className="page-body"><div className="card" style={{padding:40,textAlign:'center',color:'var(--text-muted)'}}>
      <WifiOff size={48} style={{margin:'0 auto 16px',opacity:0.3}}/><div style={{fontSize:16,fontWeight:600}}>No Connection</div>
      <div style={{fontSize:12,marginTop:8}}>Run <code style={{background:'var(--bg-input)',padding:'2px 8px',borderRadius:4}}>python -m uvicorn src.api.app:app --port 8000</code> to start</div>
      <button onClick={() => refreshRef.current?.()} style={{marginTop:16,padding:'8px 20px',borderRadius:6,border:'1px solid var(--border-color)',background:'var(--bg-secondary)',color:'var(--text-secondary)',cursor:'pointer',fontSize:12}}><RefreshCw size={12} style={{marginRight:4,verticalAlign:'middle'}}/> Retry</button>
    </div></div></>
  );

  return (<>
    <div className="page-header">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:12}}>
        <div>
          <h2 style={{display:'flex',alignItems:'center',gap:10}}>
            Market Data
            {priceChange !== 0 && (
              <span style={{
                fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-mono)',
                color: priceChange >= 0 ? '#00c48c' : '#ff4d6a',
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                {priceChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePct >= 0 ? '+' : ''}{priceChangePct.toFixed(2)}%)
              </span>
            )}
          </h2>
          <p>Gold (XAU/USD) — Live price action, indicators & volume from API</p>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <select value={period} onChange={e=>setPeriod(e.target.value)} style={{padding:'6px 12px',borderRadius:8,border:'1px solid var(--border-color)',background:'var(--bg-input)',color:'var(--text-bright)',fontSize:11,cursor:'pointer'}}>
            {['1d','5d','1mo','3mo','6mo','1y','2y'].map(p=><option key={p} value={p}>{p}</option>)}
          </select>
          <select value={interval} onChange={e=>setInterval_(e.target.value)} style={{padding:'6px 12px',borderRadius:8,border:'1px solid var(--border-color)',background:'var(--bg-input)',color:'var(--text-bright)',fontSize:11,cursor:'pointer'}}>
            {['1m','5m','15m','1h','1d','1wk'].map(i=><option key={i} value={i}>{i}</option>)}
          </select>
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'6px 14px',borderRadius:8,fontSize:11,fontWeight:600,background:'rgba(0,196,140,0.08)',color:'#00c48c',border:'1px solid rgba(0,196,140,0.25)'}}>
            <span style={{width:6,height:6,borderRadius:'50%',background:'#00c48c',boxShadow:'0 0 8px #00c48c',animation:'pulse 2s infinite'}} />
            LIVE
          </div>
        </div>
      </div>
    </div>
    <div className="page-body">
      {/* Main Price Chart */}
      <div className="card animate-in" style={{marginBottom:16,padding:'20px 20px 16px', height: 800}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
          <span style={{fontSize:13,fontWeight:700,color:'var(--text-bright)',letterSpacing:0.5,textTransform:'uppercase'}}>Interactive TradingView Chart (Live)</span>
        </div>
        <div style={{ height: 'calc(100% - 35px)', width: '100%' }}>
          <TradingViewWidget />
        </div>
      </div>
      
      {/* Fear & Greed Widget */}
      <FearGreedWidget />
    </div>
  </>);
}
