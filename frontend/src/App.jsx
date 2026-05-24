import React, { useState } from 'react';
import Header from './components/Header';
import VideoUploader from './components/VideoUploader';
import AnalysisDashboard from './components/AnalysisDashboard';
import { Shield, Activity, Globe } from 'lucide-react';

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalysisComplete = (results, url) => {
    setAnalysisResult(results);
    setVideoUrl(url);
    setIsAnalyzing(false);
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setVideoUrl(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="app-container">
      {/* Premium Header */}
      <div className="header-container">
        <div className="logo-group">
          <div className="logo-icon">
            <Shield size={24} color="#000" />
          </div>
          <div>
            <h1 className="logo-text">SENTINEL AI</h1>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', letterSpacing: '2px', fontWeight: 600 }}>
              DEEPFAKE DEFENSE SYSTEM
            </p>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <Activity size={16} color="var(--accent-green)" />
            <span>Core Active</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <Globe size={16} color="var(--accent-blue)" />
            <span>Global Node</span>
          </div>
        </div>
      </div>

      <main>
        {!analysisResult ? (
          <div className="glass-panel" style={{ animation: 'fadeInUp 0.8s ease-out' }}>
            <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '2.5rem', marginBottom: '1rem' }}>
                Verify Authenticity
              </h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
                Upload any video for a multi-modal analysis. Our neural networks examine spatial-temporal 
                consistencies and frequency domain anomalies to verify media integrity.
              </p>
            </div>
            
            <VideoUploader 
              onAnalysisStart={() => setIsAnalyzing(true)}
              onAnalysisComplete={handleAnalysisComplete}
              isAnalyzing={isAnalyzing}
            />
          </div>
        ) : (
          <AnalysisDashboard 
            videoUrl={videoUrl} 
            results={analysisResult} 
            onReset={handleReset} 
          />
        )}
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
        <p>&copy; 2026 Sentinel AI Neural Research Lab. All rights reserved.</p>
        <p style={{ marginTop: '4px' }}>Securing digital truth through multi-modal deep learning.</p>
      </footer>
    </div>
  );
}

export default App;
