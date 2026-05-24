import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, RefreshCw, Activity, Waves, Shield, Download } from 'lucide-react';

const StreamBar = ({ label, fakeScore, isFake, color }) => {
  const [width, setWidth] = useState(0);
  const displayScore = isFake ? fakeScore : 1.0 - fakeScore;
  const displayPct   = (displayScore * 100).toFixed(1);

  useEffect(() => {
    const t = setTimeout(() => setWidth(displayScore * 100), 200);
    return () => clearTimeout(t);
  }, [displayScore]);

  return (
    <div className="stream-item">
      <div className="stream-header">
        <div className="stream-label">{label}</div>
        <div className="stream-pct">{displayPct}%</div>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${width}%`,
            background: color,
            boxShadow: `0 0 10px ${color}44`,
          }}
        />
      </div>
    </div>
  );
};

const AnalysisDashboard = ({ videoUrl, results, onReset }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, []);

  if (!results) return null;

  const { is_fake, fake_percentage, real_percentage, scores } = results;
  const verdictLabel = is_fake ? 'Deepfake Detected' : 'Authentic Media';
  const pctValue     = is_fake ? fake_percentage.toFixed(1) : real_percentage.toFixed(1);
  const VerdictIcon  = is_fake ? AlertTriangle : CheckCircle;

  return (
    <div className="dashboard-root" style={{ opacity: visible ? 1 : 0, transition: 'opacity 0.8s ease' }}>
      <div className="dashboard-grid">
        {/* Left Side: Video */}
        <div className="video-preview glass-panel" style={{ padding: 0 }}>
          <video src={videoUrl} controls autoPlay loop muted />
        </div>

        {/* Right Side: Analysis */}
        <div className="analysis-panel glass-panel">
          <h3 className="streams-title">Multi-Modal Stream Analysis</h3>
          
          <div className="streams-container">
            <StreamBar 
              label="Spatial-Temporal (RGB)" 
              fakeScore={scores.spatial_temporal} 
              isFake={is_fake} 
              color="#00fbff" 
            />
            <StreamBar 
              label="Frequency (FFT)" 
              fakeScore={scores.frequency} 
              isFake={is_fake} 
              color="#9d50ff" 
            />
            <StreamBar 
              label="Audio (MFCC)" 
              fakeScore={scores.audio} 
              isFake={is_fake} 
              color="#00ff9d" 
            />
          </div>

          {/* Verdict Banner (Bottom Alignment) */}
          <div className={`result-banner ${is_fake ? 'fake' : 'real'}`}>
            <div className="verdict-content">
              <div className="verdict-main">{verdictLabel}</div>
              <div className="verdict-sub">This video is {pctValue}% {is_fake ? 'FAKE' : 'REAL'}</div>
              <div className="verdict-info">Based on Multi-Modal Analysis</div>
            </div>
            <div className="verdict-icon-right">
              <VerdictIcon size={48} color={is_fake ? 'var(--accent-red)' : 'var(--accent-green)'} />
            </div>
          </div>

          <div className="action-buttons" style={{ marginTop: '2rem' }}>
            <button className="btn btn-primary" onClick={onReset}>
              <RefreshCw size={18} /> New Analysis
            </button>
            <button className="btn" style={{ background: 'rgba(255,255,255,0.05)', color: '#fff' }}>
              <Download size={18} /> Export
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDashboard;
