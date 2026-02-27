import { useState, useEffect, useCallback } from 'react';
import { councilApi } from '@/lib/api';

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

export interface WiseMan {
  id: string;
  name: string;
  title: string;
  description: string;
  icon: string;
  is_active: boolean;
}

export interface CouncilMessage {
  id: string;
  wise_man: string;
  message: string;
  timestamp: string;
  type: 'user' | 'wise_man' | 'system';
}

export interface CouncilStatus {
  is_active: boolean;
  connected_wise_men: number;
  total_wise_men: number;
  session_id?: string;
}

// ═══════════════════════════════════════════════════════════════
// Council Status Hook
// ═══════════════════════════════════════════════════════════════

export const useCouncilStatus = () => {
  const [status, setStatus] = useState<CouncilStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await councilApi.getStatus();
      setStatus(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch council status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return { status, loading, error, refresh: fetchStatus };
};

// ═══════════════════════════════════════════════════════════════
// Council History Hook
// ═══════════════════════════════════════════════════════════════

export const useCouncilHistory = () => {
  const [messages, setMessages] = useState<CouncilMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await councilApi.getHistory();
      setMessages(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const sendMessage = async (message: string, wiseMan?: string): Promise<boolean> => {
    try {
      await councilApi.sendMessage(message, wiseMan);
      await fetchHistory();
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to send message');
      return false;
    }
  };

  return { messages, loading, error, sendMessage, refresh: fetchHistory };
};

// ═══════════════════════════════════════════════════════════════
// Wise Men Hook
// ═══════════════════════════════════════════════════════════════

export const useWiseMen = () => {
  const [wiseMen, setWiseMen] = useState<WiseMan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWiseMen = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await councilApi.getWiseMen();
      setWiseMen(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch wise men');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWiseMen();
  }, [fetchWiseMen]);

  return { wiseMen, loading, error, refresh: fetchWiseMen };
};

// ═══════════════════════════════════════════════════════════════
// Combined Council Hook
// ═══════════════════════════════════════════════════════════════

export const useCouncil = () => {
  const { status, loading: statusLoading, error: statusError, refresh: refreshStatus } = useCouncilStatus();
  const { messages, loading: historyLoading, error: historyError, sendMessage, refresh: refreshHistory } = useCouncilHistory();
  const { wiseMen, loading: wiseMenLoading, error: wiseMenError, refresh: refreshWiseMen } = useWiseMen();

  const loading = statusLoading || historyLoading || wiseMenLoading;
  const error = statusError || historyError || wiseMenError;

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshStatus(), refreshHistory(), refreshWiseMen()]);
  }, [refreshStatus, refreshHistory, refreshWiseMen]);

  return {
    status,
    messages,
    wiseMen,
    loading,
    error,
    sendMessage,
    refresh: refreshAll,
  };
};

export default useCouncil;
