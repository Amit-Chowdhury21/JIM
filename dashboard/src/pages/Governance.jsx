import React, { useState, useEffect } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';

const Governance = () => {
  const [systemHealth, setSystemHealth] = useState({
    overall_status: 'healthy',
    critical_count: 0,
    warning_count: 0,
    info_count: 0,
    active_alerts: 0,
    metrics: [],
  });

  const [alerts, setAlerts] = useState([]);
  const [scorecards, setScorecards] = useState({});

  const [retrainingJobs, setRetrainingJobs] = useState([]);

  useEffect(() => {
    fetchGovernanceData();
    const interval = setInterval(fetchGovernanceData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchGovernanceData = async () => {
    try {
      // Fetch live prediction for governance/health data
      const predResponse = await fetch('/api/ensemble/live-prediction');
      const jobsResponse = await fetch('/api/retraining/jobs');
      
      if (predResponse.ok) {
        const predData = predResponse.json();
        
        // Extract governance information
        setSystemHealth({
          overall_status: predData.system_health || 'healthy',
          critical_count: (predData.active_alerts || []).filter(a => a.level === 'critical').length,
          warning_count: (predData.active_alerts || []).filter(a => a.level === 'warning').length,
          info_count: (predData.active_alerts || []).filter(a => a.level === 'info').length,
          active_alerts: (predData.active_alerts || []).length,
          metrics: [
            { name: 'Signal Quality', value: predData.signal_quality || 'N/A' },
            { name: 'Model Agreement', value: `${((1 - (predData.disagreement || 0)) * 100).toFixed(1)}%` },
            { name: 'Ensemble Confidence', value: `${((predData.ensemble_confidence || 0) * 100).toFixed(1)}%` },
            { name: 'Regime', value: predData.regime || 'N/A' }
          ]
        });
        
        setAlerts(predData.active_alerts || []);
      }
      
      if (jobsResponse.ok) {
        const jobsData = await jobsResponse.json();
        setRetrainingJobs(jobsData.jobs || []);
      }
    } catch (error) {
      console.error('Failed to fetch governance data:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'critical':
        return 'from-red-900 to-red-700';
      case 'warning':
        return 'from-orange-900 to-orange-700';
      default:
        return 'from-green-900 to-green-700';
    }
  };

  const getAlertIcon = (level) => {
    switch (level) {
      case 'critical':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-orange-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  const getAlertColor = (level) => {
    switch (level) {
      case 'critical':
        return 'bg-red-900/30 border-red-500 text-red-400';
      case 'warning':
        return 'bg-orange-900/30 border-orange-500 text-orange-400';
      default:
        return 'bg-blue-900/30 border-blue-500 text-blue-400';
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg shadow-xl p-6">
      <h1 className="text-3xl font-bold text-white mb-6">System Governance</h1>

      {/* Overall Health Status */}
      <div className={`bg-gradient-to-r ${getStatusColor(systemHealth.overall_status)} rounded-lg p-6 mb-6`}>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-8 h-8 text-white" />
            <div>
              <p className="text-xs text-slate-200">Status</p>
              <p className="text-xl font-bold text-white uppercase">{systemHealth.overall_status}</p>
            </div>
          </div>
          <div>
            <p className="text-xs text-slate-200">Critical</p>
            <p className="text-2xl font-bold text-red-300">{systemHealth.critical_count || 0}</p>
          </div>
          <div>
            <p className="text-xs text-slate-200">Warning</p>
            <p className="text-2xl font-bold text-orange-300">{systemHealth.warning_count || 0}</p>
          </div>
          <div>
            <p className="text-xs text-slate-200">Info</p>
            <p className="text-2xl font-bold text-blue-300">{systemHealth.info_count || 0}</p>
          </div>
          <div>
            <p className="text-xs text-slate-200">Active Alerts</p>
            <p className="text-2xl font-bold text-white">{systemHealth.active_alerts || 0}</p>
          </div>
        </div>
      </div>

      {/* Active Alerts */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Active Alerts</h2>
        {alerts && alerts.length > 0 ? (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {alerts.map((alert, idx) => (
              <div key={idx} className={`${getAlertColor(alert.level)} border-l-4 rounded p-3 flex gap-3`}>
                {getAlertIcon(alert.level)}
                <div className="flex-1">
                  <p className="font-semibold text-sm">{alert.title}</p>
                  <p className="text-xs opacity-75">{alert.message}</p>
                  <p className="text-xs opacity-50 mt-1">{new Date(alert.created_at).toLocaleString()}</p>
                </div>
                <span className="text-xs bg-black/30 px-2 py-1 rounded">{alert.component}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400 text-center py-4">No active alerts</p>
        )}
      </div>

      {/* Model Health Scorecards */}
      <div className="bg-slate-700 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Model Health Scorecards</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(scorecards).map(([modelName, scorecard]) => (
            <div key={modelName} className="bg-slate-600 rounded p-4 border-l-4 border-purple-500">
              <p className="font-semibold text-purple-300 capitalize">{modelName.replace('_', ' ')}</p>
              <div className="grid grid-cols-3 gap-2 mt-3 text-sm">
                <div>
                  <p className="text-xs text-slate-400">Hit Rate</p>
                  <p className={`font-bold ${(scorecard.avg_hit_rate || 0) > 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                    {((scorecard.avg_hit_rate || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-400">Sharpe</p>
                  <p className={`font-bold ${(scorecard.avg_sharpe || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(scorecard.avg_sharpe || 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-400">Samples</p>
                  <p className="font-bold text-blue-400">{scorecard.total_samples || 0}</p>
                </div>
              </div>
              
              {/* Regime breakdown */}
              {scorecard.by_regime && Object.keys(scorecard.by_regime).length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-500">
                  <p className="text-xs font-semibold text-slate-300 mb-2">By Regime</p>
                  <div className="space-y-1 text-xs">
                    {Object.entries(scorecard.by_regime).map(([regime, data]) => (
                      <div key={regime} className="flex justify-between">
                        <span className="capitalize text-slate-400">{regime}</span>
                        <span className="font-mono text-slate-300">
                          {((data.hit_rate || 0) * 100).toFixed(0)}% ({data.count} trades)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="bg-slate-700 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">System Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {systemHealth.metrics && systemHealth.metrics.slice(0, 6).map((metric, idx) => (
            <div key={idx} className="bg-slate-600 rounded p-3">
              <p className="text-sm text-slate-300 font-semibold">{metric.name}</p>
              <div className="flex items-center justify-between mt-2">
                <p className={`text-lg font-bold ${
                  metric.level === 'critical' ? 'text-red-400' :
                  metric.level === 'warning' ? 'text-orange-400' :
                  'text-green-400'
                }`}>
                  {metric.value}{metric.unit}
                </p>
                <span className={`text-xs px-2 py-1 rounded ${
                  metric.level === 'critical' ? 'bg-red-900 text-red-300' :
                  metric.level === 'warning' ? 'bg-orange-900 text-orange-300' :
                  'bg-green-900 text-green-300'
                }`}>
                  {metric.level}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Retraining Jobs */}
      <div className="bg-slate-700 rounded-lg p-4 mt-6">
        <h2 className="text-xl font-bold text-white mb-4">Model Retraining Jobs</h2>
        {retrainingJobs && retrainingJobs.length > 0 ? (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {retrainingJobs.map((job, idx) => (
              <div key={idx} className="bg-slate-600 rounded p-4 border-l-4 border-blue-500">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <p className="font-semibold text-blue-300">{job.job_id}</p>
                    <p className="text-xs text-slate-400 mt-1">
                      {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded font-semibold ${
                    job.status === 'completed' ? 'bg-green-900 text-green-300' :
                    job.status === 'failed' ? 'bg-red-900 text-red-300' :
                    job.status === 'in_progress' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-slate-500 text-slate-300'
                  }`}>
                    {job.status.toUpperCase()}
                  </span>
                </div>
                <div className="text-sm text-slate-300 mb-2">
                  <p><strong>Models:</strong> {(job.models_to_retrain || []).join(', ') || 'N/A'}</p>
                  <p><strong>Trigger:</strong> {job.trigger_reason || 'N/A'}</p>
                </div>
                {job.versions_created && job.versions_created.length > 0 && (
                  <div className="text-xs text-slate-400 mt-2">
                    <p className="font-semibold mb-1">Versions Created:</p>
                    <div className="space-y-1">
                      {job.versions_created.map((ver, vidx) => (
                        <div key={vidx} className="text-slate-300">
                          • {ver.model_name} v{ver.version_id}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-400 text-center py-4">No retraining jobs yet</p>
        )}
      </div>
    </div>
  );
};

export default Governance;
