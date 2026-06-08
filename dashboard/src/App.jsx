import { useState } from 'react';
import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { LayoutDashboard, BrainCircuit, ShieldCheck, Wallet, Server, Zap, GitBranch, FlaskConical, Activity, FileText, Database, Menu, X, ArrowRightLeft, TrendingUp, Radio, Grid3X3, GitMerge, AlertCircle } from 'lucide-react';
import Overview from './pages/Overview';
import Models from './pages/Models';
import RiskManagement from './pages/RiskManagement';
import Portfolio from './pages/Portfolio';
import Infrastructure from './pages/Infrastructure';
import Execution from './pages/Execution';
import Backtesting from './pages/Backtesting';
import PaperTrading from './pages/PaperTrading';
import PredictionLog from './pages/PredictionLog';
import GoldSilverRatio from './pages/GoldSilverRatio';
import CorrelationMatrix from './pages/CorrelationMatrix';
import LiveTrading from './pages/LiveTrading';
import EnsembleMetrics from './pages/EnsembleMetrics';
import RegimeMonitor from './pages/RegimeMonitor';
import ModelPerformance from './pages/ModelPerformance';
import PositionSizing from './pages/PositionSizing';
import Governance from './pages/Governance';

const navItems = [
  { section: 'Trading' },
  { path: '/dashboard', icon: LayoutDashboard, label: 'Overview' },
  { path: '/models', icon: BrainCircuit, label: 'Models & Signals' },
  { path: '/paper-trading', icon: FileText, label: 'Paper Trading' },
  { path: '/live-trading', icon: Radio, label: 'Live Trading' },
  { path: '/correlation', icon: Grid3X3, label: 'Correlation Matrix' },
  { path: '/gs-ratio', icon: ArrowRightLeft, label: 'G/S Ratio' },
  { path: '/prediction-log', icon: Database, label: 'Prediction Log' },
  { section: 'Ensemble Pipeline' },
  { path: '/ensemble/metrics', icon: GitMerge, label: 'Ensemble Metrics' },
  { path: '/ensemble/regime', icon: TrendingUp, label: 'Regime Monitor' },
  { path: '/ensemble/performance', icon: BrainCircuit, label: 'Model Performance' },
  { path: '/ensemble/sizing', icon: Wallet, label: 'Position Sizing' },
  { path: '/ensemble/governance', icon: AlertCircle, label: 'Governance' },
  { section: 'Management' },
  { path: '/risk', icon: ShieldCheck, label: 'Risk Management' },
  { path: '/portfolio', icon: Wallet, label: 'Portfolio' },
  { path: '/backtest', icon: FlaskConical, label: 'Backtesting' },
  { section: 'System' },
  { path: '/execution', icon: Zap, label: 'Execution Engine' },
  { path: '/infra', icon: Server, label: 'Infrastructure' },
];

function DashboardShell() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="app-layout">
      {/* Mobile Header */}
      <div className="mobile-header">
        <div className="mobile-logo-wrap">
          <div className="logo-icon w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold bg-gradient-to-br from-[#f0b90b] to-[#d4a20a] text-[#0a0e1a] shadow-[0_0_15px_rgba(240,185,11,0.2)]">M</div>
          <h1>Mini-Medallion</h1>
        </div>
        <button className="mobile-menu-btn" onClick={() => setIsSidebarOpen(true)}>
          <Menu size={24} />
        </button>
      </div>

      {/* Sidebar Overlay */}
      <div 
        className={`sidebar-overlay ${isSidebarOpen ? 'open' : ''}`}
        onClick={() => setIsSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-logo" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="logo-icon">M</div>
            <div>
              <h1>Mini-Medallion</h1>
              <span>Gold Trading Engine</span>
            </div>
          </div>
          <button 
            className="mobile-menu-btn mobile-close-btn" 
            style={{ display: 'none' }} 
            onClick={() => setIsSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item, i) =>
            item.section ? (
              <div key={i} className="nav-section-title">{item.section}</div>
            ) : (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                end={item.path === '/dashboard'}
                onClick={() => setIsSidebarOpen(false)}
              >
                <item.icon size={18} />
                {item.label}
              </NavLink>
            )
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="system-status">
            <div className="status-item"><span className="status-dot green"></span>System Online — v3.0.0</div>
            <div className="status-item"><Zap size={14} style={{color:'var(--gold)'}}/> <span>RTX 5070 Ti • CUDA 12.1</span></div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Overview />} />
          <Route path="/models" element={<Models />} />
          <Route path="/gs-ratio" element={<GoldSilverRatio />} />
          <Route path="/correlation" element={<CorrelationMatrix />} />
          <Route path="/risk" element={<RiskManagement />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/backtest" element={<Backtesting />} />
          <Route path="/paper-trading" element={<PaperTrading />} />
          <Route path="/live-trading" element={<LiveTrading />} />
          <Route path="/prediction-log" element={<PredictionLog />} />
          <Route path="/execution" element={<Execution />} />
          <Route path="/infra" element={<Infrastructure />} />
          {/* Ensemble Pipeline Routes */}
          <Route path="/ensemble/metrics" element={<EnsembleMetrics />} />
          <Route path="/ensemble/regime" element={<RegimeMonitor />} />
          <Route path="/ensemble/performance" element={<ModelPerformance />} />
          <Route path="/ensemble/sizing" element={<PositionSizing />} />
          <Route path="/ensemble/governance" element={<Governance />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return <DashboardShell />;
}

