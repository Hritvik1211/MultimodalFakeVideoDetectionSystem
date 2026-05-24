import React from 'react';
import { ShieldCheck } from 'lucide-react';

const Header = ({ backendStatus }) => {
  return (
    <header className="header">
      <div className="header-brand">
        <ShieldCheck size={32} color="var(--accent-cyan)" />
        <h1>Sentinel AI</h1>
      </div>
      
      <div className="backend-status">
        <div className={`status-dot ${backendStatus}`}></div>
        <span>Backend {backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking...'}</span>
      </div>
    </header>
  );
};

export default Header;
