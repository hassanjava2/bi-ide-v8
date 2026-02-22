import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';
import { api } from '../services/api';
import type { SystemStatus } from '../types';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const DEFAULT_FALLBACK_POLL_INTERVAL = 30000; // 30 seconds fallback polling

export interface LiveMetrics {
  council_responses: number;
  fallback_rate_pct: number;
  latency_ms: {
    avg: number;
    last: number;
    min: number;
    max: number;
  };
  quality: {
    evidence_backed_rate_pct: number;
    evidence_backed_total: number;
    guard_total: number;
    daily_trend: Array<{
      day: string;
      evidence_rate_pct: number;
      evidence_backed: number;
      responses: number;
    }>;
  };
  top_wise_men: Array<{
    name: string;
    responses: number;
    avg_quality: number;
  }>;
}

export interface HierarchyMetrics {
  layers: Record<string, number>;
  active_agents: number;
  total_tasks: number;
  performance: {
    cpu_percent: number;
    memory_percent: number;
    gpu_utilization?: number;
  };
}

export interface CouncilUpdate {
  type: 'council_response' | 'meeting_started' | 'meeting_ended' | 'wise_man_joined' | 'wise_man_left';
  data: any;
  timestamp: string;
}

export interface LiveDataState {
  systemStatus: SystemStatus | null;
  liveMetrics: LiveMetrics | null;
  hierarchyMetrics: HierarchyMetrics | null;
  councilUpdates: CouncilUpdate[];
  lastUpdated: string | null;
  isConnected: boolean;
  isFallback: boolean;
  error: string | null;
}

export function useLiveData(refreshMs?: number) {
  const [state, setState] = useState<LiveDataState>({
    systemStatus: null,
    liveMetrics: null,
    hierarchyMetrics: null,
    councilUpdates: [],
    lastUpdated: null,
    isConnected: false,
    isFallback: false,
    error: null
  });

  const fallbackTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Handle WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    const timestamp = new Date().toISOString();

    setState(prev => {
      const newState = { ...prev, lastUpdated: timestamp, error: null };

      switch (message.type) {
        case 'system_status':
          newState.systemStatus = message.data;
          break;
        case 'live_metrics':
          newState.liveMetrics = message.data;
          break;
        case 'hierarchy_metrics':
          newState.hierarchyMetrics = message.data;
          break;
        case 'council_update':
          newState.councilUpdates = [...prev.councilUpdates, message.data].slice(-50); // Keep last 50
          break;
        case 'initial_state':
          // Initial state contains all data
          if (message.data.systemStatus) newState.systemStatus = message.data.systemStatus;
          if (message.data.liveMetrics) newState.liveMetrics = message.data.liveMetrics;
          if (message.data.hierarchyMetrics) newState.hierarchyMetrics = message.data.hierarchyMetrics;
          break;
        case 'error':
          newState.error = message.data?.message || 'Unknown error';
          break;
      }

      return newState;
    });
  }, []);

  const ws = useWebSocket({
    url: WS_URL,
    onMessage: handleMessage,
    reconnect: true,
    reconnectInterval: 1000,
    maxReconnectInterval: 30000,
    maxReconnectAttempts: 10
  });

  // Fallback polling when WebSocket is disconnected
  useEffect(() => {
    const enableFallback = () => {
      if (!fallbackTimerRef.current) {
        setState(prev => ({ ...prev, isFallback: true }));
        
        // Initial fetch
        fetchFallbackData();
        
        // Set up polling interval - use custom refreshMs if provided
        const pollInterval = refreshMs || DEFAULT_FALLBACK_POLL_INTERVAL;
        fallbackTimerRef.current = setInterval(fetchFallbackData, pollInterval);
      }
    };

    const disableFallback = () => {
      if (fallbackTimerRef.current) {
        clearInterval(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
        setState(prev => ({ ...prev, isFallback: false }));
      }
    };

    const fetchFallbackData = async () => {
      try {
        const timestamp = new Date().toISOString();
        
        // Fetch data in parallel
        const [systemStatus, liveMetrics, hierarchyMetrics] = await Promise.all([
          api.getSystemStatus().catch(() => null),
          api.getCouncilMetrics().catch(() => null),
          api.getHierarchyMetrics().catch(() => null)
        ]);

        setState(prev => ({
          ...prev,
          systemStatus: systemStatus || prev.systemStatus,
          liveMetrics: liveMetrics || prev.liveMetrics,
          hierarchyMetrics: hierarchyMetrics || prev.hierarchyMetrics,
          lastUpdated: timestamp,
          error: null
        }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to fetch fallback data'
        }));
      }
    };

    if (ws.isConnected) {
      disableFallback();
    } else if (!ws.isConnecting) {
      enableFallback();
    }

    return () => {
      disableFallback();
    };
  }, [ws.isConnected, ws.isConnecting]);

  // Update connection status
  useEffect(() => {
    setState(prev => ({
      ...prev,
      isConnected: ws.isConnected
    }));
  }, [ws.isConnected]);

  const refresh = useCallback(async () => {
    if (ws.isConnected) {
      // Request fresh data via WebSocket
      ws.send({ type: 'request_refresh' });
    } else {
      // Manual refresh via API
      setState(prev => ({ ...prev, isFallback: true }));
      
      try {
        const timestamp = new Date().toISOString();
        const [systemStatus, liveMetrics, hierarchyMetrics] = await Promise.all([
          api.getSystemStatus().catch(() => null),
          api.getCouncilMetrics().catch(() => null),
          api.getHierarchyMetrics().catch(() => null)
        ]);

        setState(prev => ({
          ...prev,
          systemStatus: systemStatus || prev.systemStatus,
          liveMetrics: liveMetrics || prev.liveMetrics,
          hierarchyMetrics: hierarchyMetrics || prev.hierarchyMetrics,
          lastUpdated: timestamp,
          error: null
        }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to refresh data'
        }));
      }
    }
  }, [ws]);

  return {
    ...state,
    refresh,
    wsStatus: ws.status,
    reconnect: ws.connect
  };
}

export default useLiveData;
