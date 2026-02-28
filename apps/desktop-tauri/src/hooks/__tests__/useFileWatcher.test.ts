/**
 * Tests for useFileWatcher hook
 * اختبارات هوك مراقبة الملفات
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useFileWatcher, FileWatcherOptions } from '../useFileWatcher';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

// Mock Tauri APIs
jest.mock('@tauri-apps/api/tauri', () => ({
  invoke: jest.fn(),
}));

jest.mock('@tauri-apps/api/event', () => ({
  listen: jest.fn(),
}));

describe('useFileWatcher', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should initialize with empty watched files', () => {
    const { result } = renderHook(() => useFileWatcher());

    expect(result.current.watchedFiles.size).toBe(0);
  });

  it('should watch a file', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Initial content',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    expect(result.current.watchedFiles.has('/test/file.txt')).toBe(true);
    expect(invoke).toHaveBeenCalledWith('watch_file', { path: '/test/file.txt' });
  });

  it('should unwatch a file', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Content',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    await act(async () => {
      await result.current.unwatchFile('/test/file.txt');
    });

    expect(result.current.watchedFiles.has('/test/file.txt')).toBe(false);
    expect(invoke).toHaveBeenCalledWith('unwatch_file', { path: '/test/file.txt' });
  });

  it('should update file content', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Initial',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    act(() => {
      result.current.updateFileContent('/test/file.txt', 'Updated content');
    });

    const file = result.current.watchedFiles.get('/test/file.txt');
    expect(file?.content).toBe('Updated content');
    expect(file?.status).toBe('modified');
  });

  it('should save file', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Content',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    act(() => {
      result.current.updateFileContent('/test/file.txt', 'New content');
    });

    await act(async () => {
      await result.current.saveFile('/test/file.txt');
    });

    expect(invoke).toHaveBeenCalledWith('write_file', {
      path: '/test/file.txt',
      content: 'New content',
    });

    const file = result.current.watchedFiles.get('/test/file.txt');
    expect(file?.status).toBe('clean');
  });

  it('should reload file', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce({
        content: 'Initial',
        mtime: Date.now(),
      })
      .mockResolvedValueOnce({
        content: 'External update',
        mtime: Date.now() + 1000,
      });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    act(() => {
      result.current.updateFileContent('/test/file.txt', 'Modified');
    });

    const content = await act(async () => {
      return result.current.reloadFile('/test/file.txt');
    });

    expect(content).toBe('External update');
    expect(result.current.watchedFiles.get('/test/file.txt')?.status).toBe('clean');
  });

  it('should detect external modifications', async () => {
    const onConflictDetected = jest.fn();
    const initialMtime = Date.now();

    (invoke as jest.Mock)
      .mockResolvedValueOnce({
        content: 'Initial',
        mtime: initialMtime,
      })
      .mockResolvedValue({
        content: 'Externally modified',
        mtime: initialMtime + 5000,
      });

    const { result } = renderHook(() => useFileWatcher({
      detectConflicts: true,
      checkInterval: 1000,
      onConflictDetected,
    }));

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    act(() => {
      result.current.updateFileContent('/test/file.txt', 'Local changes');
    });

    act(() => {
      jest.advanceTimersByTime(1500);
    });

    await waitFor(() => {
      expect(onConflictDetected).toHaveBeenCalled();
    });

    const file = result.current.watchedFiles.get('/test/file.txt');
    expect(file?.status).toBe('conflict');
  });

  it('should lock and unlock file', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce({
        content: 'Content',
        mtime: Date.now(),
      })
      .mockResolvedValueOnce(true) // lock_file
      .mockResolvedValueOnce(true); // unlock_file

    const { result } = renderHook(() => useFileWatcher({ enableLocking: true }));

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    await act(async () => {
      const locked = await result.current.lockFile('/test/file.txt');
      expect(locked).toBe(true);
    });

    expect(invoke).toHaveBeenCalledWith('lock_file', { path: '/test/file.txt' });
    expect(result.current.watchedFiles.get('/test/file.txt')?.lockOwner).toBe('current');

    await act(async () => {
      const unlocked = await result.current.unlockFile('/test/file.txt');
      expect(unlocked).toBe(true);
    });

    expect(result.current.watchedFiles.get('/test/file.txt')?.lockOwner).toBeUndefined();
  });

  it('should resolve conflict with local version', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Initial',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    // Simulate conflict
    act(() => {
      result.current.updateFileContent('/test/file.txt', 'Local changes');
    });

    // Manually set conflict status for testing
    const file = result.current.watchedFiles.get('/test/file.txt');
    if (file) {
      file.status = 'conflict';
    }

    await act(async () => {
      await result.current.resolveConflict('/test/file.txt', 'local');
    });

    const resolvedFile = result.current.watchedFiles.get('/test/file.txt');
    expect(resolvedFile?.status).toBe('modified');
  });

  it('should resolve conflict with external version', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce({
        content: 'Initial',
        mtime: Date.now(),
      })
      .mockResolvedValueOnce({
        content: 'External version',
        mtime: Date.now() + 1000,
      });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    await act(async () => {
      await result.current.resolveConflict('/test/file.txt', 'external');
    });

    expect(invoke).toHaveBeenCalledWith('get_file_info', { path: '/test/file.txt' });
    expect(result.current.watchedFiles.get('/test/file.txt')?.content).toBe('External version');
  });

  it('should stop all watchers', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Content',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file1.txt');
      await result.current.watchFile('/test/file2.txt');
    });

    act(() => {
      result.current.stopAllWatchers();
    });

    expect(result.current.watchedFiles.size).toBe(0);
    expect(invoke).toHaveBeenCalledWith('unwatch_file', { path: '/test/file1.txt' });
    expect(invoke).toHaveBeenCalledWith('unwatch_file', { path: '/test/file2.txt' });
  });

  it('should auto-reload on change if enabled', async () => {
    const mockUnlisten = jest.fn();
    (listen as jest.Mock).mockResolvedValue(mockUnlisten);
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Updated',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher({
      autoReload: true,
    }));

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    // Simulate file change event
    const listenCallback = (listen as jest.Mock).mock.calls[0][1];
    await act(async () => {
      await listenCallback({
        payload: {
          path: '/test/file.txt',
          type: 'modified',
          timestamp: Date.now(),
        },
      });
    });

    expect(invoke).toHaveBeenCalledWith('get_file_info', { path: '/test/file.txt' });
  });

  it('should get file status', async () => {
    (invoke as jest.Mock).mockResolvedValue({
      content: 'Content',
      mtime: Date.now(),
    });

    const { result } = renderHook(() => useFileWatcher());

    await act(async () => {
      await result.current.watchFile('/test/file.txt');
    });

    const status = result.current.getFileStatus('/test/file.txt');
    expect(status).toBeDefined();
    expect(status?.path).toBe('/test/file.txt');
    expect(status?.status).toBe('clean');

    const nonExistent = result.current.getFileStatus('/non/existent.txt');
    expect(nonExistent).toBeUndefined();
  });
});
