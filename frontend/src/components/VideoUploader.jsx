import React, { useRef, useState } from 'react';
import axios from 'axios';
import { Upload, FileVideo, AlertCircle, Loader2 } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

const VideoUploader = ({ onAnalysisStart, onAnalysisComplete, isAnalyzing }) => {
  const fileInputRef = useRef(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setError('Please upload a valid video file.');
      return;
    }

    setError(null);
    onAnalysisStart();

    const formData = new FormData();
    formData.append('file', file);
    const videoUrl = URL.createObjectURL(file);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData);
      onAnalysisComplete(response.data, videoUrl);
    } catch (err) {
      setError('Connection failed. Ensure the Sentinel AI backend is running.');
      onAnalysisComplete(null, null);
    }
  };

  const onDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="uploader-container">
      <input
        type="file"
        id="video-input"
        ref={fileInputRef}
        accept="video/*"
        onChange={(e) => handleFile(e.target.files[0])}
      />

      <div
        className={`uploader-box ${dragActive ? 'active' : ''}`}
        onDragEnter={onDrag}
        onDragLeave={onDrag}
        onDragOver={onDrag}
        onDrop={onDrop}
        onClick={() => !isAnalyzing && fileInputRef.current.click()}
        style={{ opacity: isAnalyzing ? 0.6 : 1, pointerEvents: isAnalyzing ? 'none' : 'auto' }}
      >
        {isAnalyzing ? (
          <div className="loading-state">
            <Loader2 className="spinner" size={48} style={{ color: 'var(--accent-green)', marginBottom: '1.5rem' }} />
            <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Analyzing Neural Streams...</h3>
            <p style={{ color: 'var(--text-secondary)' }}>Extracting frames and running frequency domain scans.</p>
          </div>
        ) : (
          <>
            <div className="uploader-icon">
              <Upload size={48} />
            </div>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Drop video to analyze</h3>
            <p style={{ color: 'var(--text-secondary)' }}>MP4, MOV, or AVI up to 50MB</p>
          </>
        )}
      </div>

      {error && (
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,51,102,0.1)', border: '1px solid var(--accent-red)', borderRadius: '12px', color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default VideoUploader;
