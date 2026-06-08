import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

const PositionSizing = () => {
  const [sizing, setSizing] = useState({
    regime: 'normal',
    account_size: 100000,
    max_position_pct: 0.05,
    recommended_size_pct: 0.035,
    kelly_fraction: 0.12,
    risk_per_trade: 3500,
    growth_multiplier: 1.2,
    normal_multiplier: 1.0,
    crisis_multiplier: 0.6,
  });

  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchSizing();
    const interval = setInterval(fetchSizing, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchSizing = async () => {
    try {
      // Fetch live prediction data which includes position sizing info
      const response = await fetch('/api/ensemble/live-prediction');
      if (response.ok) {
        const data = await response.json();
        
        // Extract sizing information from orchestrator output
        setSizing({
          regime: data.regime || 'normal',
          account_size: 100000,
          max_position_pct: 0.05,
          recommended_size_pct: data.recommended_size || 0.035,
          kelly_fraction: (data.recommended_size || 0.035) * 2.5,
          risk_per_trade: 100000 * (data.recommended_size || 0.035),
          growth_multiplier: 1.2,
          normal_multiplier: 1.0,
          crisis_multiplier: 0.6,
          stop_loss_pips: data.stop_loss_offset || 50,
          take_profit_pips: data.take_profit_offset || 75,
          signal_quality: data.signal_quality || 'MODERATE'
        });
        
        // Track history
        setHistory(prev => [...prev.slice(-19), {
          timestamp: new Date().toLocaleTimeString(),
          size: (data.recommended_size || 0.035) * 100,
          regime: data.regime || 'normal'
        }]);
      }
    } catch (error) {
      console.error('Failed to fetch position sizing:', error);
    }
  };

  const regimeMultipliers = [
    { regime: 'Growth', multiplier: sizing.growth_multiplier || 1.2 },
    { regime: 'Normal', multiplier: sizing.normal_multiplier || 1.0 },
    { regime: 'Crisis', multiplier: sizing.crisis_multiplier || 0.6 },
  ];

  const radarData = [
    {
      regime: 'Growth',
      Aggression: (sizing.growth_multiplier || 1.2) * 50,
    },
    {
      regime: 'Normal',
      Aggression: (sizing.normal_multiplier || 1.0) * 50,
    },
    {
      regime: 'Crisis',
      Aggression: (sizing.crisis_multiplier || 0.6) * 50,
    },
  ];

  const getRegimeColor = (regime) => {
    switch (regime) {
      case 'growth':
        return 'from-green-900 to-green-700';
      case 'crisis':
        return 'from-red-900 to-red-700';
      default:
        return 'from-yellow-900 to-yellow-700';
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg shadow-xl p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Position Sizing</h1>

      {/* Position Summary */}
      <div className={`bg-gradient-to-r ${getRegimeColor(sizing.regime)} rounded-lg p-6 mb-6`}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-slate-200">Account Size</p>
            <p className="text-2xl font-bold text-white">${(sizing.account_size / 1000).toFixed(0)}k</p>
          </div>
          <div>
            <p className="text-sm text-slate-200">Max Position %</p>
            <p className="text-2xl font-bold text-blue-300">{((sizing.max_position_pct || 0) * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-sm text-slate-200">Recommended Size</p>
            <p className="text-2xl font-bold text-green-300">{((sizing.recommended_size_pct || 0) * 100).toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-sm text-slate-200">Risk per Trade</p>
            <p className="text-2xl font-bold text-purple-300">${(sizing.risk_per_trade || 0).toFixed(0)}</p>
          </div>
        </div>
      </div>

      {/* Kelly Criterion */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Kelly Criterion & Risk Management</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-slate-300 text-sm mb-2">Kelly Fraction</p>
            <div className="flex items-end gap-2">
              <div className="flex-1 bg-slate-600 rounded p-2">
                <div className="bg-gradient-to-r from-blue-500 to-purple-500 h-8 rounded" 
                     style={{ width: `${(sizing.kelly_fraction || 0.1) * 100}%` }}></div>
              </div>
              <p className="text-white font-bold">{((sizing.kelly_fraction || 0) * 100).toFixed(1)}%</p>
            </div>
            <p className="text-xs text-slate-400 mt-2">Half-Kelly = {((sizing.kelly_fraction || 0) / 2 * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-slate-300 text-sm mb-2">Current Regime Adjustment</p>
            <div className="flex items-end gap-2">
              <div className="flex-1 bg-slate-600 rounded p-2">
                <div className="bg-gradient-to-r from-amber-500 to-orange-500 h-8 rounded"
                     style={{ width: `${(sizing.normal_multiplier || 1.0) * 50}%` }}></div>
              </div>
              <p className="text-white font-bold">{(sizing.normal_multiplier || 1.0).toFixed(2)}x</p>
            </div>
            <p className="text-xs text-slate-400 mt-2">{sizing.regime.toUpperCase()} regime</p>
          </div>
        </div>
      </div>

      {/* Regime-Based Multipliers */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Position Size by Regime</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regimeMultipliers}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="regime" stroke="#aaa" />
            <YAxis stroke="#aaa" domain={[0, 1.5]} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #444' }} />
            <Bar dataKey="multiplier" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Guidelines */}
      <div className="bg-slate-700 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">Risk Management Guidelines</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-600 rounded p-4 border-l-4 border-green-500">
            <p className="font-semibold text-green-400">GROWTH REGIME</p>
            <p className="text-sm text-slate-300 mt-2">Normal to moderately aggressive positioning</p>
            <p className="text-xs text-slate-400 mt-2">• Allow full size if 2+ models strong</p>
            <p className="text-xs text-slate-400">• Focus on trend continuation</p>
            <p className="text-xs text-slate-400">• Higher hit rate expectations</p>
          </div>
          <div className="bg-slate-600 rounded p-4 border-l-4 border-yellow-500">
            <p className="font-semibold text-yellow-400">NORMAL REGIME</p>
            <p className="text-sm text-slate-300 mt-2">Balanced and confirmation-driven</p>
            <p className="text-xs text-slate-400 mt-2">• Require at least 2 strong models</p>
            <p className="text-xs text-slate-400">• Check HMM stability before entry</p>
            <p className="text-xs text-slate-400">• Standard position sizing</p>
          </div>
          <div className="bg-slate-600 rounded p-4 border-l-4 border-red-500">
            <p className="font-semibold text-red-400">CRISIS REGIME</p>
            <p className="text-sm text-slate-300 mt-2">Defensive & strict filters</p>
            <p className="text-xs text-slate-400 mt-2">• Smaller position sizes (40-60% reduction)</p>
            <p className="text-xs text-slate-400">• Require HMM + TFT agreement</p>
            <p className="text-xs text-slate-400">• Monitor uncertainty levels</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PositionSizing;
