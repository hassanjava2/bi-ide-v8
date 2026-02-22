import { useEffect, useRef, useState, useCallback } from 'react';
import { getAccessToken } from '../services/api';

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
}

export interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export function useWebSocket(options: UseWebSocketOptions) {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnect = true,
    reconnectInterval = 1000,
    maxReconnectInterval = 30000,
    maxReconnectAttempts = 10
  } = options;

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isManualCloseRef = useRef(false);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    clearReconnectTimer();
    isManualCloseRef.current = false;
    setStatus('connecting');

    try {
      // Add auth token to URL if available
      const token = getAccessToken();
      const wsUrl = token ? `${url}?token=${token}` : url;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        reconnectAttemptsRef.current = 0;
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.warn('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        onDisconnect?.();
        wsRef.current = null;

        // Attempt reconnection if enabled and not manually closed
        if (reconnect && !isManualCloseRef.current) {
          const shouldReconnect = reconnectAttemptsRef.current < maxReconnectAttempts;
          
          if (shouldReconnect) {
            const delay = Math.min(
              reconnectInterval * Math.pow(2, reconnectAttemptsRef.current),
              maxReconnectInterval
            );
            reconnectAttemptsRef.current++;
            
            reconnectTimerRef.current = setTimeout(() => {
              connect();
            }, delay);
          }
        }
      };

      ws.onerror = (error) => {
        setStatus('error');
        onError?.(error);
      };
    } catch (error) {
      setStatus('error');
      console.error('WebSocket connection error:', error);
    }
  }, [url, reconnect, reconnectInterval, maxReconnectInterval, maxReconnectAttempts, onConnect, onDisconnect, onError, onMessage, clearReconnectTimer]);

  const disconnect = useCallback(() => {
    isManualCloseRef.current = true;
    clearReconnectTimer();
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('disconnected');
  }, [clearReconnectTimer]);

  const send = useCallback((message: WebSocketMessage | string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const data = typeof message === 'string' ? message : JSON.stringify(message);
      wsRef.current.send(data);
      return true;
    }
    return false;
  }, []);

  const subscribe = useCallback((channel: string) => {
    return send({ type: 'subscribe', data: { channel } });
  }, [send]);

  const unsubscribe = useCallback((channel: string) => {
    return send({ type: 'unsubscribe', data: { channel } });
  }, [send]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    status,
    lastMessage,
    connect,
    disconnect,
    send,
    subscribe,
    unsubscribe,
    isConnected: status === 'connected',
    isConnecting: status === 'connecting'
  };
}

export default useWebSocket;
