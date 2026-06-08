import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const EnsembleMetrics = () => {
  const [metrics, setMetrics] = useState({
    weights: { wavelet_pro: 0.25, hmm_pro: 0.25, lstm: 0.25, tft_pro: 0.25 },
    confidence_multipliers: { wavelet_pro: 1.0, hmm_pro: 1.0, lstm: 1.0, tft_pro: 1.0 },
    disagreement_penalty: 0.95,
    signal_disagreement: 0.35,
    ensemble_score: 0.42,
    ensemble_confidence: 0.68,
    current_position: 'LONG',
    recommended_size: 0.045,
  });

  const [weightHistory, setWeightHistory] = useState([]);

  useEffect(() => {
    // Fetch current metrics
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/ensemble/metrics');
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const weightsData = [
    { name: 'Wavelet Pro', value: metrics.weights.wavelet || 0, color: '#8884d8' },
    { name: 'HMM Pro', value: metrics.weights.hmm || 0, color: '#82ca9d' },
    { name: 'LSTM', value: metrics.weights.lstm || 0, color: '#ffc658' },
    { name: 'TFT Pro', value: metrics.weights.tft_pro || 0, color: '#ff7c7c' },
  ];

  const confidenceData = [
    { name: 'Wavelet Pro', confidence: metrics.confidence_multipliers.wavelet || 0 },
    { name: 'HMM Pro', confidence: metrics.confidence_multipliers.hmm || 0 },
    { name: 'LSTM', confidence: metrics.confidence_multipliers.lstm || 0 },
    { name: 'TFT Pro', confidence: metrics.confidence_multipliers.tft_pro || 0 },
  ];

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg shadow-xl p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Ensemble Metrics</h1>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-700 rounded-lg p-4">
          <p className="text-slate-300 text-sm">Ensemble Score</p>
          <p className={`text-2xl font-bold ${metrics.ensemble_score > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {(metrics.ensemble_score || 0).toFixed(2)}
          </p>
        </div>
        <div className="bg-slate-700 rounded-lg p-4">
          <p className="text-slate-300 text-sm">Ensemble Confidence</p>
          <p className="text-2xl font-bold text-blue-400">{(metrics.ensemble_confidence || 0).toFixed(2)}</p>
        </div>
        <div className="bg-slate-700 rounded-lg p-4">
          <p className="text-slate-300 text-sm">Disagreement</p>
          <p className={`text-2xl font-bold ${metrics.signal_disagreement < 0.6 ? 'text-green-400' : 'text-orange-400'}`}>
            {(metrics.signal_disagreement || 0).toFixed(2)}
          </p>
        </div>
        <div className="bg-slate-700 rounded-lg p-4">
          <p className="text-slate-300 text-sm">Recommended Size</p>
          <p className="text-2xl font-bold text-purple-400">{((metrics.recommended_size || 0) * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Model Weights Pie Chart */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Model Weights Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={weightsData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {weightsData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence Multipliers */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Confidence Multipliers</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={confidenceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="name" stroke="#aaa" />
            <YAxis stroke="#aaa" domain={[0, 1.5]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #444' }}
              formatter={(value) => value.toFixed(3)}
            />
            <Bar dataKey="confidence" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Disagreement Penalty */}
      <div className="bg-slate-700 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">Risk Metrics</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-slate-300 text-sm">Disagreement Penalty</p>
            <p className={`text-lg font-bold ${metrics.disagreement_penalty > 0.9 ? 'text-green-400' : 'text-orange-400'}`}>
              {(metrics.disagreement_penalty || 0).toFixed(3)}
            </p>
            <p className="text-slate-400 text-xs mt-2">1.0 = No penalty, &lt;0.9 = High disagreement</p>
          </div>
          <div>
            <p className="text-slate-300 text-sm">Current Position</p>
            <p className={`text-lg font-bold ${metrics.current_position === 'LONG' ? 'text-green-400' : metrics.current_position === 'SHORT' ? 'text-red-400' : 'text-gray-400'}`}>
              {metrics.current_position || 'FLAT'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnsembleMetrics;
