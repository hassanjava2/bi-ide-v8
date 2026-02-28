/**
 * Tests for useGit hook
 * اختبارات هوك التكامل مع Git
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useGit, GitOptions } from '../useGit';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

// Mock Tauri APIs
jest.mock('@tauri-apps/api/tauri', () => ({
  invoke: jest.fn(),
}));

jest.mock('@tauri-apps/api/event', () => ({
  listen: jest.fn(),
}));

describe('useGit', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should initialize correctly', () => {
    const { result } = renderHook(() => useGit());

    expect(result.current.isRepo).toBe(false);
    expect(result.current.repoStatus).toBe('uninitialized');
    expect(result.current.currentBranch).toBe('');
    expect(result.current.branches).toEqual([]);
    expect(result.current.changedFiles).toEqual([]);
    expect(result.current.isLoading).toBe(false);
  });

  it('should check if directory is a git repo', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true) // git_is_repo
      .mockResolvedValueOnce({ // git_status
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      })
      .mockResolvedValueOnce([]) // git_branches
      .mockResolvedValueOnce([]); // git_log

    const { result } = renderHook(() => useGit());

    await waitFor(() => {
      expect(result.current.isRepo).toBe(true);
    });

    expect(invoke).toHaveBeenCalledWith('git_is_repo', { path: '.' });
  });

  it('should initialize a new repo', async () => {
    (invoke as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useGit());

    await act(async () => {
      await result.current.init();
    });

    expect(invoke).toHaveBeenCalledWith('git_init', { path: '.' });
  });

  it('should stage files', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'modified',
        branch: 'main',
        files: [{ path: 'file.txt', status: 'modified', staged: true }],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.add(['file.txt']);
    });

    expect(invoke).toHaveBeenCalledWith('git_add', {
      path: '.',
      files: ['file.txt'],
    });
  });

  it('should unstage files', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'modified',
        branch: 'main',
        files: [{ path: 'file.txt', status: 'modified', staged: false }],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.unstage(['file.txt']);
    });

    expect(invoke).toHaveBeenCalledWith('git_unstage', {
      path: '.',
      files: ['file.txt'],
    });
  });

  it('should commit changes', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 1,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.commit('Test commit', 'Description');
    });

    expect(invoke).toHaveBeenCalledWith('git_commit', {
      path: '.',
      message: 'Test commit',
      description: 'Description',
    });
  });

  it('should push changes', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit({ repoPath: '/test/repo' }));

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.push('origin', 'main', false);
    });

    expect(invoke).toHaveBeenCalledWith('git_push', {
      path: '/test/repo',
      remote: 'origin',
      branch: 'main',
      force: false,
    });
  });

  it('should pull changes', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.pull('origin', 'main', true);
    });

    expect(invoke).toHaveBeenCalledWith('git_pull', {
      path: '.',
      remote: 'origin',
      branch: 'main',
      rebase: true,
    });
  });

  it('should checkout branch', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce([]) // branches
      .mockResolvedValueOnce([]); // log

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.checkout('feature-branch', false);
    });

    expect(invoke).toHaveBeenCalledWith('git_checkout', {
      path: '.',
      branch: 'feature-branch',
      create: false,
    });
  });

  it('should create branch', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.createBranch('new-branch', true);
    });

    expect(invoke).toHaveBeenCalledWith('git_create_branch', {
      path: '.',
      name: 'new-branch',
      checkout: true,
    });
  });

  it('should delete branch', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.deleteBranch('old-branch', false);
    });

    expect(invoke).toHaveBeenCalledWith('git_delete_branch', {
      path: '.',
      name: 'old-branch',
      force: false,
    });
  });

  it('should merge branch', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.merge('feature-branch', 'default');
    });

    expect(invoke).toHaveBeenCalledWith('git_merge', {
      path: '.',
      branch: 'feature-branch',
      strategy: 'default',
    });
  });

  it('should discard changes', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.discard(['modified-file.txt']);
    });

    expect(invoke).toHaveBeenCalledWith('git_discard', {
      path: '.',
      files: ['modified-file.txt'],
    });
  });

  it('should get diff', async () => {
    const mockDiff = [
      {
        path: 'file.txt',
        status: 'modified',
        additions: 5,
        deletions: 2,
        hunks: [],
      },
    ];

    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(mockDiff);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    const diff = await act(async () => {
      return result.current.diff('file.txt', false);
    });

    expect(diff).toEqual(mockDiff);
    expect(invoke).toHaveBeenCalledWith('git_diff', {
      repoPath: '.',
      filePath: 'file.txt',
      staged: false,
    });
  });

  it('should get merge conflicts', async () => {
    const mockConflicts = [
      {
        path: 'conflict.txt',
        baseContent: 'base',
        oursContent: 'ours',
        theirsContent: 'theirs',
        markers: [],
      },
    ];

    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(mockConflicts);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    const conflicts = await act(async () => {
      return result.current.getMergeConflicts();
    });

    expect(conflicts).toEqual(mockConflicts);
  });

  it('should resolve conflict', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'modified',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      });

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.resolveConflict('conflict.txt', 'ours');
    });

    expect(invoke).toHaveBeenCalledWith('git_resolve_conflict', {
      repoPath: '.',
      filePath: 'conflict.txt',
      resolution: 'ours',
      content: undefined,
    });
  });

  it('should abort merge', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    await act(async () => {
      await result.current.abortMerge();
    });

    expect(invoke).toHaveBeenCalledWith('git_abort_merge', { path: '.' });
  });

  it('should auto-refresh status', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'clean',
        branch: 'main',
        files: [],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      })
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    renderHook(() => useGit({ autoRefresh: true, refreshInterval: 1 }));

    await waitFor(() => {
      expect(invoke).toHaveBeenCalledWith('git_status', { path: '.' });
    });

    act(() => {
      jest.advanceTimersByTime(1000);
    });

    await waitFor(() => {
      expect(invoke).toHaveBeenCalledTimes(5); // Initial + interval check
    });
  });

  it('should calculate changes count correctly', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce({
        status: 'modified',
        branch: 'main',
        files: [
          { path: 'staged.txt', status: 'modified', staged: true },
          { path: 'unstaged.txt', status: 'modified', staged: false },
          { path: 'untracked.txt', status: 'untracked', staged: false },
        ],
        ahead: 0,
        behind: 0,
        hasConflicts: false,
      })
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);

    const { result } = renderHook(() => useGit());

    await waitFor(() => expect(result.current.isRepo).toBe(true));

    expect(result.current.changesCount).toBe(1); // unstaged.txt only
    expect(result.current.stagedCount).toBe(1); // staged.txt
  });
});
