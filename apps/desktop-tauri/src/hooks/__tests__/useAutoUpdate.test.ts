/**
 * Tests for useAutoUpdate hook
 * اختبارات هوك التحديثات التلقائية
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useAutoUpdate, AutoUpdateOptions } from '../useAutoUpdate';
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';

// Mock Tauri APIs
jest.mock('@tauri-apps/api/updater', () => ({
  checkUpdate: jest.fn(),
  installUpdate: jest.fn(),
  onUpdaterEvent: jest.fn(),
}));

jest.mock('@tauri-apps/api/process', () => ({
  relaunch: jest.fn(),
}));

describe('useAutoUpdate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  it('should initialize with idle status', () => {
    const { result } = renderHook(() => useAutoUpdate());

    expect(result.current.status).toBe('idle');
    expect(result.current.updateInfo).toBeNull();
    expect(result.current.progress).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('should check for updates successfully', async () => {
    const mockUpdateInfo = {
      shouldUpdate: true,
      manifest: {
        version: '1.1.0',
        date: '2024-01-01',
        body: 'New features',
      },
    };

    (checkUpdate as jest.Mock).mockResolvedValue(mockUpdateInfo);

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.checkForUpdates();
    });

    expect(result.current.status).toBe('available');
    expect(result.current.updateInfo).toEqual({
      version: '1.1.0',
      date: '2024-01-01',
      body: 'New features',
      notes: 'New features',
    });
  });

  it('should handle no updates available', async () => {
    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: false,
    });

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.checkForUpdates();
    });

    expect(result.current.status).toBe('idle');
    expect(result.current.updateInfo).toBeNull();
  });

  it('should skip version if previously skipped', async () => {
    localStorage.setItem('skipped_update_version', '1.1.0');

    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: true,
      manifest: {
        version: '1.1.0',
        date: '2024-01-01',
        body: 'New features',
      },
    });

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.checkForUpdates();
    });

    expect(result.current.status).toBe('idle');
  });

  it('should download update', async () => {
    const mockUnlisten = jest.fn();
    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: true,
      manifest: {
        version: '1.1.0',
        date: '2024-01-01',
        body: 'New features',
      },
    });
    (installUpdate as jest.Mock).mockResolvedValue(undefined);

    const { onUpdaterEvent } = require('@tauri-apps/api/updater');
    (onUpdaterEvent as jest.Mock).mockResolvedValue(mockUnlisten);

    const { result } = renderHook(() => useAutoUpdate({ backgroundDownload: false }));

    await act(async () => {
      await result.current.checkForUpdates();
    });

    await act(async () => {
      await result.current.downloadUpdate();
    });

    expect(installUpdate).toHaveBeenCalled();
    expect(result.current.status).toBe('downloading');
  });

  it('should install and relaunch', async () => {
    (relaunch as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.installAndRelaunch();
    });

    expect(relaunch).toHaveBeenCalled();
  });

  it('should skip version', async () => {
    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: true,
      manifest: {
        version: '1.1.0',
        date: '2024-01-01',
        body: 'New features',
      },
    });

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.checkForUpdates();
    });

    act(() => {
      result.current.skipVersion();
    });

    expect(result.current.status).toBe('idle');
    expect(result.current.updateInfo).toBeNull();
    expect(localStorage.getItem('skipped_update_version')).toBe('1.1.0');
  });

  it('should handle errors', async () => {
    (checkUpdate as jest.Mock).mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useAutoUpdate());

    await act(async () => {
      await result.current.checkForUpdates();
    });

    expect(result.current.status).toBe('error');
    expect(result.current.error).toBe('Network error');
  });

  it('should auto-check on interval', async () => {
    jest.useFakeTimers();
    
    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: false,
    });

    renderHook(() => useAutoUpdate({ checkIntervalMinutes: 1 }));

    expect(checkUpdate).toHaveBeenCalledTimes(1);

    act(() => {
      jest.advanceTimersByTime(60000);
    });

    expect(checkUpdate).toHaveBeenCalledTimes(2);

    jest.useRealTimers();
  });

  it('should cancel download', async () => {
    (checkUpdate as jest.Mock).mockResolvedValue({
      shouldUpdate: true,
      manifest: {
        version: '1.1.0',
        date: '2024-01-01',
        body: 'New features',
      },
    });

    const { result } = renderHook(() => useAutoUpdate({ backgroundDownload: false }));

    await act(async () => {
      await result.current.checkForUpdates();
    });

    await act(async () => {
      result.current.downloadUpdate();
      result.current.cancelDownload();
    });

    expect(result.current.status).toBe('available');
    expect(result.current.progress).toBeNull();
  });
});
