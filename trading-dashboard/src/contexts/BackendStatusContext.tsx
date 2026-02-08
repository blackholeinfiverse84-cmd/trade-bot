import React, { createContext, useContext, useState, useEffect } from 'react';
import { config } from '../config';

interface BackendStatusState {
  isOnline: boolean;
  isOffline: boolean;
  status: 'ONLINE' | 'OFFLINE' | 'CHECKING';
}

interface BackendStatusContextType extends BackendStatusState {
  checkStatus: () => Promise<void>;
}

const BackendStatusContext = createContext<BackendStatusContextType | undefined>(undefined);

export const BackendStatusProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<BackendStatusState>({
    isOnline: false,
    isOffline: false,
    status: 'CHECKING'
  });

  const checkStatus = async () => {
    setState(prev => ({ ...prev, status: 'CHECKING' }));
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000); // 20s when backend may be busy

    try {
      const response = await fetch(`${config.API_BASE_URL}/tools/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      const isOnline = response.ok;
      setState({
        isOnline,
        isOffline: !isOnline,
        status: isOnline ? 'ONLINE' : 'OFFLINE'
      });
    } catch (e: any) {
      clearTimeout(timeoutId);
      // Timeout/abort = backend likely busy (e.g. long prediction), don't mark offline
      const isTimeout = e?.name === 'AbortError';
      setState({
        isOnline: isTimeout,
        isOffline: !isTimeout,
        status: isTimeout ? 'CHECKING' : 'OFFLINE'
      });
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BackendStatusContext.Provider value={{ ...state, checkStatus }}>
      {children}
    </BackendStatusContext.Provider>
  );
};

export const useBackendStatus = (): BackendStatusContextType => {
  const context = useContext(BackendStatusContext);
  if (!context) {
    throw new Error('useBackendStatus must be used within BackendStatusProvider');
  }
  return context;
};