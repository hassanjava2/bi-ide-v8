/**
 * Tests for useOfflineMode hook
 * اختبارات هوك العمل بدون إنترنت
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useOfflineMode } from '../useOfflineMode';

describe('useOfflineMode', () => {
  let onlineEventListener: EventListener | null = null;
  let offlineEventListener: EventListener | null = null;

  beforeEach(() => {
    localStorage.clear();
    
    // Mock window.addEventListener
    window.addEventListener = jest.fn((event, callback) => {
      if (event === 'online') onlineEventListener = callback as EventListener;
      if (event === 'offline') offlineEventListener = callback as EventListener;
    });

    // Mock fetch
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should initialize with checking status', () => {
    const { result } = renderHook(() => useOfflineMode());

    expect(result.current.status).toBe('checking');
    expect(result.current.isOnline).toBe(false);
    expect(result.current.isOffline).toBe(false);
  });

  it('should detect online status', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useOfflineMode());

    await act(async () => {
      await result.current.checkConnection();
    });

    expect(result.current.status).toBe('online');
    expect(result.current.isOnline).toBe(true);
    expect(result.current.isOffline).toBe(false);
  });

  it('should detect offline status', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useOfflineMode());

    await act(async () => {
      await result.current.checkConnection();
    });

    expect(result.current.status).toBe('offline');
    expect(result.current.isOnline).toBe(false);
    expect(result.current.isOffline).toBe(true);
  });

  it('should queue actions when offline', async () => {
    const { result } = renderHook(() => useOfflineMode());

    const actionId = act(() => {
      return result.current.queueAction({
        endpoint: '/api/data',
        method: 'POST',
        body: { name: 'test' },
      });
    });

    expect(actionId).toBeDefined();
    expect(result.current.queuedActions).toHaveLength(1);
    expect(result.current.pendingCount).toBe(1);
  });

  it('should remove queued action', () => {
    const { result } = renderHook(() => useOfflineMode());

    let actionId: string;
    act(() => {
      actionId = result.current.queueAction({
        endpoint: '/api/data',
        method: 'POST',
      });
    });

    act(() => {
      result.current.removeQueuedAction(actionId!);
    });

    expect(result.current.queuedActions).toHaveLength(0);
    expect(result.current.pendingCount).toBe(0);
  });

  it('should cache data', () => {
    const { result } = renderHook(() => useOfflineMode());

    act(() => {
      result.current.cacheData('user-data', { id: 1, name: 'Test' });
    });

    const cached = result.current.getCachedData('user-data');
    expect(cached).not.toBeNull();
    expect(cached?.data).toEqual({ id: 1, name: 'Test' });
  });

  it('should return null for expired cache', () => {
    jest.useFakeTimers();
    
    const { result } = renderHook(() => useOfflineMode({ cacheExpiryHours: 1 }));

    act(() => {
      result.current.cacheData('test-data', { value: 123 });
    });

    // Advance time beyond expiry
    act(() => {
      jest.advanceTimersByTime(2 * 60 * 60 * 1000);
    });

    const cached = result.current.getCachedData('test-data');
    expect(cached).toBeNull();

    jest.useRealTimers();
  });

  it('should clear cache', () => {
    const { result } = renderHook(() => useOfflineMode());

    act(() => {
      result.current.cacheData('data1', { a: 1 });
      result.current.cacheData('data2', { b: 2 });
    });

    act(() => {
      result.current.clearCache();
    });

    expect(result.current.getCachedData('data1')).toBeNull();
    expect(result.current.getCachedData('data2')).toBeNull();
  });

  it('should sync queued actions', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useOfflineMode());

    act(() => {
      result.current.queueAction({ endpoint: '/api/1', method: 'POST' });
      result.current.queueAction({ endpoint: '/api/2', method: 'POST' });
    });

    await act(async () => {
      await result.current.sync();
    });

    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(result.current.queuedActions).toHaveLength(0);
  });

  it('should handle sync failures with retries', async () => {
    (global.fetch as jest.Mock)
      .mockRejectedValueOnce(new Error('Failed'))
      .mockRejectedValueOnce(new Error('Failed'))
      .mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useOfflineMode({ maxRetries: 2 }));

    act(() => {
      result.current.queueAction({ endpoint: '/api/test', method: 'POST' });
    });

    await act(async () => {
      await result.current.sync();
    });

    expect(global.fetch).toHaveBeenCalledTimes(3);
    expect(result.current.queuedActions).toHaveLength(0);
  });

  it('should auto-sync on reconnect', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useOfflineMode({ autoSyncOnReconnect: true }));

    act(() => {
      result.current.queueAction({ endpoint: '/api/test', method: 'POST' });
    });

    // Simulate going offline then online
    act(() => {
      offlineEventListener?.(new Event('offline'));
    });

    await act(async () => {
      onlineEventListener?.(new Event('online'));
    });

    expect(global.fetch).toHaveBeenCalled();
  });

  it('should persist queued actions to localStorage', () => {
    const { result } = renderHook(() => useOfflineMode());

    act(() => {
      result.current.queueAction({ endpoint: '/api/test', method: 'POST' });
    });

    // Reload hook
    const { result: newResult } = renderHook(() => useOfflineMode());

    expect(newResult.current.queuedActions).toHaveLength(1);
  });
});
