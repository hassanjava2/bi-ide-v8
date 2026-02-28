/**
 * Git Hook - هوك التكامل مع Git
 * 
 * يوفر حالة Git والتزام التغييرات والدفع/السحب وإدارة الفروع
 * مع عارض الفروقات وحل تعارضات الدمج
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

/** حالة Git */
export type GitRepoStatus = 'uninitialized' | 'clean' | 'modified' | 'ahead' | 'behind' | 'diverged' | 'conflict' | 'error';

/** حالة الملف */
export type GitFileStatus = 'unmodified' | 'modified' | 'added' | 'deleted' | 'renamed' | 'copied' | 'untracked' | 'ignored' | 'conflicted';

/** معلومات الفرع */
export interface BranchInfo {
  name: string;
  isCurrent: boolean;
  isRemote: boolean;
  upstream?: string;
  ahead: number;
  behind: number;
  lastCommit: {
    hash: string;
    message: string;
    author: string;
    date: string;
  };
}

/** معلومات الملف */
export interface GitFileInfo {
  path: string;
  status: GitFileStatus;
  staged: boolean;
  oldPath?: string;
  similarity?: number;
}

/** معلومات الـ commit */
export interface CommitInfo {
  hash: string;
  shortHash: string;
  message: string;
  author: {
    name: string;
    email: string;
  };
  date: string;
  parents: string[];
}

/** فرق الملف */
export interface FileDiff {
  path: string;
  oldPath?: string;
  status: GitFileStatus;
  additions: number;
  deletions: number;
  hunks: DiffHunk[];
}

/** جزء من الفرق */
export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: DiffLine[];
}

/** سطر الفرق */
export interface DiffLine {
  type: 'context' | 'addition' | 'deletion';
  content: string;
  oldLineNumber?: number;
  newLineNumber?: number;
}

/** تعارض الدمج */
export interface MergeConflict {
  path: string;
  baseContent: string;
  oursContent: string;
  theirsContent: string;
  markers: ConflictMarker[];
}

/** محدد التعارض */
export interface ConflictMarker {
  start: number;
  separator: number;
  end: number;
  ours: string;
  theirs: string;
}

/** إعدادات Git */
export interface GitOptions {
  /** مسار المستودع */
  repoPath?: string;
  /** التحديث التلقائي */
  autoRefresh?: boolean;
  /** فاصل التحديث (بالثواني) */
  refreshInterval?: number;
  /** رد الاتصال عند تغيير الحالة */
  onStatusChange?: (status: GitRepoStatus) => void;
  /** رد الاتصال عند تعارض الدمج */
  onMergeConflict?: (conflicts: MergeConflict[]) => void;
}

/** نتيجة هوك Git */
export interface UseGitResult {
  /** هل المسار مستودع Git */
  isRepo: boolean;
  /** حالة المستودع */
  repoStatus: GitRepoStatus;
  /** الفرع الحالي */
  currentBranch: string;
  /** قائمة الفروع */
  branches: BranchInfo[];
  /** الملفات المتغيرة */
  changedFiles: GitFileInfo[];
  /** عدد الملفات المتغيرة */
  changesCount: number;
  /** عدد الملفات المرحلة */
  stagedCount: number;
  /** عدد الـ commits غير المدفوعة */
  aheadCount: number;
  /** عدد الـ commits غير المسحوبة */
  behindCount: number;
  /** هل يوجد تعارضات */
  hasConflicts: boolean;
  /** سجل الـ commits */
  log: CommitInfo[];
  /** هل يتم التحميل */
  isLoading: boolean;
  /** رسالة الخطأ */
  error: string | null;
  /** تهيئة مستودع جديد */
  init: () => Promise<void>;
  /** فحص الحالة */
  status: () => Promise<void>;
  /** إضافة ملفات للمرحلة */
  add: (paths: string[]) => Promise<void>;
  /** إزالة ملفات من المرحلة */
  unstage: (paths: string[]) => Promise<void>;
  /** التزام التغييرات */
  commit: (message: string, description?: string) => Promise<void>;
  /** دفع التغييرات */
  push: (remote?: string, branch?: string, force?: boolean) => Promise<void>;
  /** سحب التغييرات */
  pull: (remote?: string, branch?: string, rebase?: boolean) => Promise<void>;
  '''
  /** إحضار التغييرات بدون دمج */
  fetch: (remote?: string) => Promise<void>;
  '''
  /** تغيير الفرع */
  checkout: (branch: string, create?: boolean) => Promise<void>;
  /** إنشاء فرع جديد */
  createBranch: (name: string, checkout?: boolean) => Promise<void>;
  /** حذف فرع */
  deleteBranch: (name: string, force?: boolean) => Promise<void>
  /** دمج فرع */
  merge: (branch: string, strategy?: 'default' | 'ours' | 'theirs') => Promise<void>;
  /** إعادة قاعدة فرع */
  rebase: (branch: string) => Promise<void>;
  /** إلغاء التغييرات */
  discard: (paths: string[]) => Promise<void>;
  /** إظهار الفروقات */
  diff: (path?: string, staged?: boolean) => Promise<FileDiff[]>;
  /** إظهار الفروقات بين فروع */
  diffBranches: (from: string, to: string) => Promise<FileDiff[]>;
  /** الحصول على تعارضات الدمج */
  getMergeConflicts: () => Promise<MergeConflict[]>;
  /** حل تعارض */
  resolveConflict: (path: string, resolution: 'ours' | 'theirs' | 'both', content?: string) => Promise<void>;
  /** إلغاء الدمج */
  abortMerge: () => Promise<void>;
  /** إضافة وسوم */
  tag: (name: string, message?: string, commitHash?: string) => Promise<void>;
  /** إظهار اللوم */
  blame: (path: string) => Promise<BlameLine[]>;
  /** تخزين التغييرات */
  stash: (message?: string) => Promise<void>;
  '''
  /** استعادة التخزين */
  stashPop: (index?: number) => Promise<void>;
  /** إسقاط التخزين */
  stashDrop: (index?: number) => Promise<void>;
  /** قائمة التخزين */
  stashList: () => Promise<StashInfo[]>;
  '''
  }

/** معلومات سطر اللوم */
export interface BlameLine {
  line: number;
  commit: string;
  author: string;
  date: string;
  content: string;
}

/** معلومات التخزين */
export interface StashInfo {
  index: number;
  message: string;
  commit: string;
  date: string;
}

/** الإعدادات الافتراضية */
const DEFAULT_OPTIONS: Required<Omit<GitOptions, 'repoPath' | 'onStatusChange' | 'onMergeConflict'>> = {
  autoRefresh: true,
  refreshInterval: 30,
};

/**
 * هوك التكامل مع Git
 * @param options - إعدادات Git
 * @returns نتيجة التحكم بـ Git
 */
export function useGit(options: GitOptions = {}): UseGitResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  const [isRepo, setIsRepo] = useState(false);
  const [repoStatus, setRepoStatus] = useState<GitRepoStatus>('uninitialized');
  const [currentBranch, setCurrentBranch] = useState('');
  const [branches, setBranches] = useState<BranchInfo[]>([]);
  const [changedFiles, setChangedFiles] = useState<GitFileInfo[]>([]);
  const [aheadCount, setAheadCount] = useState(0);
  const [behindCount, setBehindCount] = useState(0);
  const [hasConflicts, setHasConflicts] = useState(false);
  const [log, setLog] = useState<CommitInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const unlistenRef = useRef<(() => void) | null>(null);

  const repoPath = opts.repoPath || '.';

  /**
   * فحص ما إذا كان المسار مستودع Git
   */
  const checkIsRepo = useCallback(async (): Promise<boolean> => {
    try {
      return await invoke<boolean>('git_is_repo', { path: repoPath });
    } catch {
      return false;
    }
  }, [repoPath]);

  /**
   * تحديث الحالة
   */
  const updateStatus = useCallback(async () => {
    if (!isRepo) return;

    try {
      const status = await invoke<{
        status: GitRepoStatus;
        branch: string;
        files: GitFileInfo[];
        ahead: number;
        behind: number;
        hasConflicts: boolean;
      }>('git_status', { path: repoPath });

      setRepoStatus(status.status);
      setCurrentBranch(status.branch);
      setChangedFiles(status.files);
      setAheadCount(status.ahead);
      setBehindCount(status.behind);
      setHasConflicts(status.hasConflicts);

      if (status.status !== repoStatus) {
        opts.onStatusChange?.(status.status);
      }

      if (status.hasConflicts) {
        const conflicts = await getMergeConflicts();
        opts.onMergeConflict?.(conflicts);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تحديث الحالة');
    }
  }, [isRepo, repoPath, repoStatus, opts]);

  /**
   * تهيئة مستودع جديد
   */
  const init = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await invoke('git_init', { path: repoPath });
      setIsRepo(true);
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تهيئة المستودع');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * فحص الحالة الكامل
   */
  const status = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const repo = await checkIsRepo();
      setIsRepo(repo);

      if (repo) {
        await updateStatus();
        
        // تحديث الفروع
        const branchesList = await invoke<BranchInfo[]>('git_branches', { path: repoPath });
        setBranches(branchesList);

        // تحديث السجل
        const logList = await invoke<CommitInfo[]>('git_log', { path: repoPath, limit: 50 });
        setLog(logList);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل فحص الحالة');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, checkIsRepo, updateStatus]);

  /**
   * إضافة ملفات للمرحلة
   */
  const add = useCallback(async (paths: string[]) => {
    setIsLoading(true);
    try {
      await invoke('git_add', { path: repoPath, files: paths });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إضافة الملفات');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * إزالة ملفات من المرحلة
   */
  const unstage = useCallback(async (paths: string[]) => {
    setIsLoading(true);
    try {
      await invoke('git_unstage', { path: repoPath, files: paths });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إزالة الملفات من المرحلة');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * التزام التغييرات
   */
  const commit = useCallback(async (message: string, description?: string) => {
    setIsLoading(true);
    try {
      await invoke('git_commit', {
        path: repoPath,
        message,
        description,
      });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل التزام التغييرات');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * دفع التغييرات
   */
  const push = useCallback(async (remote = 'origin', branch?: string, force = false) => {
    setIsLoading(true);
    try {
      await invoke('git_push', {
        path: repoPath,
        remote,
        branch: branch || currentBranch,
        force,
      });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل دفع التغييرات');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, currentBranch, updateStatus]);

  /**
   * سحب التغييرات
   */
  const pull = useCallback(async (remote = 'origin', branch?: string, rebase = false) => {
    setIsLoading(true);
    try {
      await invoke('git_pull', {
        path: repoPath,
        remote,
        branch: branch || currentBranch,
        rebase,
      });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل سحب التغييرات');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, currentBranch, updateStatus]);

  /**
   * تغيير الفرع
   */
  const checkout = useCallback(async (branch: string, create = false) => {
    setIsLoading(true);
    try {
      await invoke('git_checkout', {
        path: repoPath,
        branch,
        create,
      });
      await status();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تغيير الفرع');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, status]);

  /**
   * إنشاء فرع جديد
   */
  const createBranch = useCallback(async (name: string, checkoutBranch = true) => {
    setIsLoading(true);
    try {
      await invoke('git_create_branch', {
        path: repoPath,
        name,
        checkout: checkoutBranch,
      });
      await status();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إنشاء الفرع');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, status]);

  /**
   * حذف فرع
   */
  const deleteBranch = useCallback(async (name: string, force = false) => {
    setIsLoading(true);
    try {
      await invoke('git_delete_branch', {
        path: repoPath,
        name,
        force,
      });
      await status();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل حذف الفرع');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, status]);

  /**
   * دمج فرع
   */
  const merge = useCallback(async (branch: string, strategy: 'default' | 'ours' | 'theirs' = 'default') => {
    setIsLoading(true);
    try {
      await invoke('git_merge', {
        path: repoPath,
        branch,
        strategy,
      });
      await status();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل دمج الفرع');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, status]);

  /**
   * إلغاء التغييرات
   */
  const discard = useCallback(async (paths: string[]) => {
    setIsLoading(true);
    try {
      await invoke('git_discard', {
        path: repoPath,
        files: paths,
      });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إلغاء التغييرات');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * إظهار الفروقات
   */
  const diff = useCallback(async (path?: string, staged = false): Promise<FileDiff[]> => {
    try {
      return await invoke<FileDiff[]>('git_diff', {
        repoPath,
        filePath: path,
        staged,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إظهار الفروقات');
      return [];
    }
  }, [repoPath]);

  /**
   * إظهار الفروقات بين فروع
   */
  const diffBranches = useCallback(async (from: string, to: string): Promise<FileDiff[]> => {
    try {
      return await invoke<FileDiff[]>('git_diff_branches', {
        path: repoPath,
        from,
        to,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إظهار الفروقات بين الفروع');
      return [];
    }
  }, [repoPath]);

  /**
   * الحصول على تعارضات الدمج
   */
  const getMergeConflicts = useCallback(async (): Promise<MergeConflict[]> => {
    try {
      return await invoke<MergeConflict[]>('git_get_conflicts', { path: repoPath });
    } catch {
      return [];
    }
  }, [repoPath]);

  /**
   * حل تعارض
   */
  const resolveConflict = useCallback(async (
    path: string,
    resolution: 'ours' | 'theirs' | 'both',
    content?: string
  ) => {
    setIsLoading(true);
    try {
      await invoke('git_resolve_conflict', {
        repoPath,
        filePath: path,
        resolution,
        content,
      });
      await updateStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل حل التعارض');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, updateStatus]);

  /**
   * إلغاء الدمج
   */
  const abortMerge = useCallback(async () => {
    setIsLoading(true);
    try {
      await invoke('git_abort_merge', { path: repoPath });
      await status();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل إلغاء الدمج');
    } finally {
      setIsLoading(false);
    }
  }, [repoPath, status]);

  // التحميل الأولي
  useEffect(() => {
    status();
  }, [status]);

  // إعداد التحديث التلقائي
  useEffect(() => {
    if (opts.autoRefresh && isRepo) {
      refreshIntervalRef.current = setInterval(updateStatus, opts.refreshInterval * 1000);
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [opts.autoRefresh, opts.refreshInterval, isRepo, updateStatus]);

  // الاستماع لأحداث Git
  useEffect(() => {
    const setupListener = async () => {
      const unlisten = await listen('git-changed', () => {
        updateStatus();
      });
      unlistenRef.current = unlisten;
    };

    setupListener();

    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
    };
  }, [updateStatus]);

  const changesCount = changedFiles.filter(f => !f.staged && f.status !== 'untracked').length;
  const stagedCount = changedFiles.filter(f => f.staged).length;

  return {
    isRepo,
    repoStatus,
    currentBranch,
    branches,
    changedFiles,
    changesCount,
    stagedCount,
    aheadCount,
    behindCount,
    hasConflicts,
    log,
    isLoading,
    error,
    init,
    status,
    add,
    unstage,
    commit,
    push,
    pull,
    checkout,
    createBranch,
    deleteBranch,
    merge,
    discard,
    diff,
    diffBranches,
    getMergeConflicts,
    resolveConflict,
    abortMerge,
  };
}

export default useGit;
