import { useState, useMemo, useEffect, useRef } from 'react';
import { ArrowRightLeft, TrendingUp, TrendingDown, Info, Bell } from 'lucide-react';
import { fetchGsRatio } from '../data/api';

function TradingViewChart() {
  const container = useRef();

  useEffect(() => {
    // Only inject if it hasn't been injected yet
    if (container.current && container.current.children.length === 1) {
      const script = document.createElement("script");
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.type = "text/javascript";
      script.async = true;
      script.innerHTML = `
        {
          "autosize": true,
          "symbol": "TVC:GOLDSILVER",
          "interval": "D",
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
          "container_id": "tradingview_gs",
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650",
          "details": true,
          "calendar": false,
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

export default function GoldSilverRatio() {
  const [goldPrice, setGoldPrice] = useState(0);
  const [silverPrice, setSilverPrice] = useState(0);

  const [tradeAmount, setTradeAmount] = useState(2);
  const [tradeAsset, setTradeAsset] = useState('gold'); // 'gold' or 'silver'

  useEffect(() => {
    async function loadData() {
      try {
        const data = await fetchGsRatio('2y', '1mo');
        setGoldPrice(data.current_gold);
        setSilverPrice(data.current_silver);
      } catch (err) {
        console.error('Failed to load GS Ratio data', err);
      }
    }
    loadData();
  }, []);

  const currentRatio = silverPrice > 0 ? (goldPrice / silverPrice) : 0;
  
  const ratioColor = currentRatio > 80 ? 'var(--red)' : currentRatio < 60 ? 'var(--green)' : 'var(--gold)';
  
  const signal = useMemo(() => {
    if (currentRatio > 80) return { text: "Buy Silver / Sell Gold — Silver is historically cheap", type: "sell-gold", icon: <TrendingDown size={20}/>, color: "var(--red)", bg: "var(--red-dim)" };
    if (currentRatio < 60) return { text: "Buy Gold / Sell Silver — Gold is relatively cheap", type: "buy-gold", icon: <TrendingUp size={20}/>, color: "var(--green)", bg: "var(--green-dim)" };
    return { text: "Neutral Zone — Hold your position", type: "neutral", icon: <ArrowRightLeft size={20}/>, color: "var(--text-bright)", bg: "var(--bg-input)" };
  }, [currentRatio]);

  return (
    <>
      <div className="page-header">
        <div>
          <h2>⚖️ Gold/Silver Ratio Dashboard</h2>
          <p>Track the relative value of Gold to Silver and generated rotation signals.</p>
        </div>
      </div>
      
      <div className="page-body">
        
        {/* Signal Banner */}
        <div style={{ 
          padding: '16px 24px', 
          borderRadius: 8, 
          background: signal.bg, 
          border: `1px solid ${signal.color}`, 
          display: 'flex', 
          alignItems: 'center', 
          gap: 16,
          marginBottom: 24,
          boxShadow: `0 4px 20px ${signal.color}20`
        }} className="animate-in">
          <div style={{ padding: 12, background: signal.color, color: '#000', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Bell size={24} />
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', color: signal.color, letterSpacing: 1 }}>Active Signal</div>
            <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-bright)' }}>{signal.text}</div>
          </div>
        </div>

        <div className="grid-2" style={{ marginBottom: 24 }}>
          {/* Hero Display */}
          <div className="card animate-in" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px 20px' }}>
            <div style={{ fontSize: 14, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 2, marginBottom: 8 }}>Current Ratio</div>
            <div style={{ fontSize: 72, fontWeight: 800, color: ratioColor, fontFamily: 'var(--font-mono)', lineHeight: 1, textShadow: `0 0 30px ${ratioColor}40` }}>
              {currentRatio.toFixed(1)}<span style={{ fontSize: 36, color: 'var(--text-muted)' }}>:1</span>
            </div>
            <div style={{ fontSize: 16, color: 'var(--text-secondary)', marginTop: 16, fontWeight: 500 }}>
              {currentRatio.toFixed(1)} oz of Silver = 1 oz of Gold
            </div>
            
            <div style={{ display: 'flex', gap: 24, marginTop: 32, width: '100%', justifyContent: 'center' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Gold Price</div>
                <div style={{ fontSize: 20, color: '#F5C842', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>${goldPrice.toFixed(2)}</div>
              </div>
              <div style={{ width: 1, background: 'var(--border-color)' }}></div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Silver Price</div>
                <div style={{ fontSize: 20, color: '#C0C0C0', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>${silverPrice.toFixed(2)}</div>
              </div>
            </div>
          </div>
          
          {/* Explainer Card & Simulator */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <div className="card animate-in">
              <div className="card-header">
                <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}><Info size={16}/> Ratio Strategy</span>
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6, marginBottom: 16 }}>
                The Gold/Silver ratio measures the relative value of the two metals. Traders use it to identify when one metal is historically cheap compared to the other.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ padding: 12, background: 'var(--red-dim)', borderRadius: 8, border: '1px solid var(--red)' }}>
                  <div style={{ color: 'var(--red)', fontWeight: 600, fontSize: 14 }}>Ratio {'>'} 80</div>
                  <div style={{ color: 'var(--text-bright)', fontSize: 12, marginTop: 4 }}>Gold is expensive. Sell Gold, Buy Silver.</div>
                </div>
                <div style={{ padding: 12, background: 'var(--green-dim)', borderRadius: 8, border: '1px solid var(--green)' }}>
                  <div style={{ color: 'var(--green)', fontWeight: 600, fontSize: 14 }}>Ratio {'<'} 60</div>
                  <div style={{ color: 'var(--text-bright)', fontSize: 12, marginTop: 4 }}>Silver is expensive. Sell Silver, Buy Gold.</div>
                </div>
              </div>
              <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Historical Average: ~65</span>
                <span>All-time High: 125 (Mar 2020)</span>
              </div>
            </div>
            
            <div className="card animate-in" style={{ flex: 1 }}>
              <div className="card-header">
                <span className="card-title">Trade Simulator</span>
              </div>
              
              <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                <input 
                  type="number" 
                  value={tradeAmount} 
                  onChange={e => setTradeAmount(Number(e.target.value) || 0)}
                  style={{ flex: 1, padding: '10px 14px', borderRadius: 8, border: '1px solid var(--border-color)', background: 'var(--bg-input)', color: 'var(--text-bright)', fontSize: 16, fontFamily: 'var(--font-mono)' }}
                />
                <select 
                  value={tradeAsset} 
                  onChange={e => setTradeAsset(e.target.value)}
                  style={{ padding: '10px 14px', borderRadius: 8, border: '1px solid var(--border-color)', background: 'var(--bg-input)', color: 'var(--text-bright)', fontSize: 16 }}
                >
                  <option value="gold">oz Gold</option>
                  <option value="silver">oz Silver</option>
                </select>
              </div>
              
              <div style={{ padding: 16, background: 'var(--bg-secondary)', borderRadius: 8, border: '1px dashed var(--border-color)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>You would receive approximately</div>
                <div style={{ fontSize: 28, fontWeight: 700, color: tradeAsset === 'gold' ? '#C0C0C0' : '#F5C842', fontFamily: 'var(--font-mono)' }}>
                  {tradeAsset === 'gold' 
                    ? (tradeAmount * currentRatio).toFixed(2) 
                    : (tradeAmount / currentRatio).toFixed(2)
                  } <span style={{ fontSize: 16 }}>oz of {tradeAsset === 'gold' ? 'Silver' : 'Gold'}</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                  If you swap {tradeAmount} oz of {tradeAsset === 'gold' ? 'gold' : 'silver'} at the {currentRatio.toFixed(1)} ratio.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Interactive TradingView Chart (Live)</span>
          </div>
          <div style={{ height: 600, width: '100%', marginTop: 20 }}>
            <TradingViewChart />
          </div>
        </div>
        
      </div>
    </>
  );
}
