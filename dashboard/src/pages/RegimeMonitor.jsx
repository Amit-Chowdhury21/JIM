import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

const RegimeMonitor = () => {
  const [regime, setRegime] = useState({
    regime: 'normal',
    probabilities: { growth: 0.33, normal: 0.34, crisis: 0.33 },
    metrics: {
      realized_vol_pctl: 50,
      vix_level: 20.0,
      vix_change: 0.0,
      dxy_shock: 0.0,
      rates_shock: 0.0,
      gold_gap_freq: 0.0,
      spread_widening: 0.0,
    },
    model_inputs: {
      hmm_bullish_prob: 0.33,
      wavelet_noise_ratio: 0.2,
    },
    confidence: 0.34,
    persistence: 0.34,
  });

  const [regimeHistory, setRegimeHistory] = useState([]);

  useEffect(() => {
    fetchRegimeData();
    const interval = setInterval(fetchRegimeData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchRegimeData = async () => {
    try {
      const response = await fetch('/api/ensemble/regime');
      if (response.ok) {
        const data = await response.json();
        setRegime(data);
      }
    } catch (error) {
      console.error('Failed to fetch regime:', error);
    }
  };

  const regimeProbabilities = [
    {
      name: 'Regime Probabilities',
      growth: (regime.probabilities?.growth || 0) * 100,
      normal: (regime.probabilities?.normal || 0) * 100,
      crisis: (regime.probabilities?.crisis || 0) * 100,
    },
  ];

  const getRegimeColor = (r) => {
    switch (r) {
      case 'growth':
        return 'text-green-400 bg-green-900/30';
      case 'crisis':
        return 'text-red-400 bg-red-900/30';
      default:
        return 'text-yellow-400 bg-yellow-900/30';
    }
  };

  const metricsData = [
    { name: 'Vol %ile', value: regime.metrics?.realized_vol_pctl || 0, max: 100 },
    { name: 'VIX', value: regime.metrics?.vix_level || 0, max: 50 },
    { name: 'DXY Shock', value: Math.abs(regime.metrics?.dxy_shock || 0), max: 1 },
    { name: 'Gap Freq', value: (regime.metrics?.gold_gap_freq || 0) * 100, max: 100 },
  ];

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg shadow-xl p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Regime Monitor</h1>

      {/* Current Regime Display */}
      <div className={`rounded-lg p-6 mb-6 ${getRegimeColor(regime.regime)}`}>
        <p className="text-sm font-semibold mb-2">Current Regime</p>
        <p className="text-4xl font-bold uppercase">{regime.regime}</p>
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div>
            <p className="text-xs opacity-75">Confidence</p>
            <p className="text-xl font-bold">{(regime.confidence * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-xs opacity-75">Persistence</p>
            <p className="text-xl font-bold">{(regime.persistence * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Regime Probabilities */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Regime Probabilities</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regimeProbabilities}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="name" stroke="#aaa" />
            <YAxis stroke="#aaa" domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #444' }} />
            <Legend />
            <Bar dataKey="growth" fill="#10b981" />
            <Bar dataKey="normal" fill="#f59e0b" />
            <Bar dataKey="crisis" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Macro Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {metricsData.map((metric, idx) => (
          <div key={idx} className="bg-slate-700 rounded-lg p-4">
            <p className="text-slate-300 text-sm">{metric.name}</p>
            <p className="text-2xl font-bold text-blue-400">{metric.value.toFixed(1)}</p>
            <div className="w-full bg-slate-600 rounded-full h-1 mt-2">
              <div
                className="bg-blue-500 h-1 rounded-full"
                style={{ width: `${Math.min(100, (metric.value / metric.max) * 100)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Model Inputs */}
      <div className="bg-slate-700 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">Model Inputs</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-slate-300 text-sm">HMM Bullish Probability</p>
            <div className="flex items-center mt-2">
              <div className="w-full bg-slate-600 rounded-full h-2 mr-3">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${(regime.model_inputs?.hmm_bullish_prob || 0) * 100}%` }}
                ></div>
              </div>
              <p className="text-white font-bold">{((regime.model_inputs?.hmm_bullish_prob || 0) * 100).toFixed(1)}%</p>
            </div>
          </div>
          <div>
            <p className="text-slate-300 text-sm">Wavelet Noise Ratio</p>
            <div className="flex items-center mt-2">
              <div className="w-full bg-slate-600 rounded-full h-2 mr-3">
                <div
                  className="bg-orange-500 h-2 rounded-full"
                  style={{ width: `${(regime.model_inputs?.wavelet_noise_ratio || 0) * 100}%` }}
                ></div>
              </div>
              <p className="text-white font-bold">{((regime.model_inputs?.wavelet_noise_ratio || 0) * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegimeMonitor;
