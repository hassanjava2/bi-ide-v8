/**
 * File Watcher Hook - هوك مراقبة الملفات
 * 
 * يراقب تغييرات الملفات ويعيد التحميل تلقائياً عند التغيير
 * مع اكتشاف التعديلات الخارجية واكتشاف التعارضات وإدارة قفل الملفات
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

/** حالة الملف */
export type FileStatus = 'clean' | 'modified' | 'external_modified' | 'conflict' | 'locked';

/** نوع تغيير الملف */
export type FileChangeType = 'created' | 'modified' | 'deleted' | 'renamed';

/** حدث تغيير الملف */
export interface FileChangeEvent {
  path: string;
  type: FileChangeType;
  timestamp: number;
}

/** معلومات الملف المراقب */
export interface WatchedFile {
  path: string;
  content: string;
  lastModified: number;
  lastSavedContent: string;
  status: FileStatus;
  lockOwner?: string;
  externalModifiedAt?: number;
}

/** إعدادات مراقبة الملفات */
export interface FileWatcherOptions {
  /** تمكين إعادة التحميل التلقائي */
  autoReload?: boolean;
  /** تمكين اكتشاف التعارضات */
  detectConflicts?: boolean;
  /** فاصل التحقق من التغييرات (بالمللي ثانية) */
  checkInterval?: number;
  /** تمكين قفل الملفات */
  enableLocking?: boolean;
  /** رد الاتصال عند تغيير الملف */
  onFileChange?: (event: FileChangeEvent) => void;
  /** رد الاتصال عند اكتشاف تعارض */
  onConflictDetected?: (path: string) => void;
}

/** نتيجة هوك مراقبة الملفات */
export interface UseFileWatcherResult {
  /** الملفات المراقبة */
  watchedFiles: Map<string, WatchedFile>;
  /** إضافة ملف للمراقبة */
  watchFile: (path: string, initialContent?: string) => Promise<void>;
  /** إزالة ملف من المراقبة */
  unwatchFile: (path: string) => Promise<void>;
  /** تحديث محتوى الملف */
  updateFileContent: (path: string, content: string) => void;
  /** حفظ الملف */
  saveFile: (path: string) => Promise<void>;
  /** إعادة تحميل الملف */
  reloadFile: (path: string) => Promise<string>;
  /** التحقق من حالة الملف */
  getFileStatus: (path: string) => WatchedFile | undefined;
  /** قفل الملف */
  lockFile: (path: string) => Promise<boolean>;
  /** فتح قفل الملف */
  unlockFile: (path: string) => Promise<boolean>;
  /** حل التعارض */
  resolveConflict: (path: string, resolution: 'local' | 'external' | 'merge', mergedContent?: string) => Promise<void>;
  /** إيقاف جميع المراقبات */
  stopAllWatchers: () => void;
}

/** الإعدادات الافتراضية */
const DEFAULT_OPTIONS: Required<FileWatcherOptions> = {
  autoReload: false,
  detectConflicts: true,
  checkInterval: 1000,
  enableLocking: true,
  onFileChange: () => {},
  onConflictDetected: () => {},
};

/**
 * هوك مراقبة الملفات
 * @param options - إعدادات المراقبة
 * @returns نتيجة التحكم بمراقبة الملفات
 */
export function useFileWatcher(options: FileWatcherOptions = {}): UseFileWatcherResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  const [watchedFiles, setWatchedFiles] = useState<Map<string, WatchedFile>>(new Map());
  const intervalsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const unlistenRef = useRef<(() => void) | null>(null);

  /**
   * الحصول على معلومات الملف من النظام
   */
  const getFileInfo = useCallback(async (path: string): Promise<{ content: string; mtime: number }> => {
    return await invoke('get_file_info', { path });
  }, []);

  /**
   * إضافة ملف للمراقبة
   */
  const watchFile = useCallback(async (path: string, initialContent?: string) => {
    try {
      const fileInfo = await getFileInfo(path);
      
      const watchedFile: WatchedFile = {
        path,
        content: initialContent ?? fileInfo.content,
        lastModified: fileInfo.mtime,
        lastSavedContent: initialContent ?? fileInfo.content,
        status: 'clean',
      };

      setWatchedFiles((prev) => {
        const next = new Map(prev);
        next.set(path, watchedFile);
        return next;
      });

      // إعداد مراقبة Tauri
      await invoke('watch_file', { path });

      // إعداد فاصل التحقق من التغييرات
      if (opts.detectConflicts) {
        const interval = setInterval(async () => {
          await checkExternalChanges(path);
        }, opts.checkInterval);
        
        intervalsRef.current.set(path, interval);
      }
    } catch (err) {
      console.error(`فشل مراقبة الملف ${path}:`, err);
      throw err;
    }
  }, [getFileInfo, opts.detectConflicts, opts.checkInterval]);

  /**
   * إزالة ملف من المراقبة
   */
  const unwatchFile = useCallback(async (path: string) => {
    try {
      // إيقاف الفاصل
      const interval = intervalsRef.current.get(path);
      if (interval) {
        clearInterval(interval);
        intervalsRef.current.delete(path);
      }

      // إلغاء المراقبة عبر Tauri
      await invoke('unwatch_file', { path });

      setWatchedFiles((prev) => {
        const next = new Map(prev);
        next.delete(path);
        return next;
      });
    } catch (err) {
      console.error(`فشل إزالة مراقبة الملف ${path}:`, err);
    }
  }, []);

  /**
   * تحديث محتوى الملف
   */
  const updateFileContent = useCallback((path: string, content: string) => {
    setWatchedFiles((prev) => {
      const file = prev.get(path);
      if (!file) return prev;

      const isModified = content !== file.lastSavedContent;
      const next = new Map(prev);
      next.set(path, {
        ...file,
        content,
        status: isModified ? 'modified' : 'clean',
      });
      return next;
    });
  }, []);

  /**
   * حفظ الملف
   */
  const saveFile = useCallback(async (path: string) => {
    const file = watchedFiles.get(path);
    if (!file) throw new Error(`الملف ${path} غير موجود`);
    
    if (file.status === 'conflict') {
      throw new Error('لا يمكن الحفظ أثناء وجود تعارض');
    }

    try {
      await invoke('write_file', {
        path,
        content: file.content,
      });

      const fileInfo = await getFileInfo(path);

      setWatchedFiles((prev) => {
        const next = new Map(prev);
        next.set(path, {
          ...file,
          lastSavedContent: file.content,
          lastModified: fileInfo.mtime,
          status: 'clean',
          externalModifiedAt: undefined,
        });
        return next;
      });
    } catch (err) {
      console.error(`فشل حفظ الملف ${path}:`, err);
      throw err;
    }
  }, [watchedFiles, getFileInfo]);

  /**
   * إعادة تحميل الملف
   */
  const reloadFile = useCallback(async (path: string): Promise<string> => {
    try {
      const fileInfo = await getFileInfo(path);

      setWatchedFiles((prev) => {
        const file = prev.get(path);
        if (!file) return prev;

        const next = new Map(prev);
        next.set(path, {
          ...file,
          content: fileInfo.content,
          lastSavedContent: fileInfo.content,
          lastModified: fileInfo.mtime,
          status: 'clean',
          externalModifiedAt: undefined,
        });
        return next;
      });

      return fileInfo.content;
    } catch (err) {
      console.error(`فشل إعادة تحميل الملف ${path}:`, err);
      throw err;
    }
  }, [getFileInfo]);

  /**
   * التحقق من التغييرات الخارجية
   */
  const checkExternalChanges = useCallback(async (path: string) => {
    const file = watchedFiles.get(path);
    if (!file) return;

    try {
      const fileInfo = await getFileInfo(path);
      
      if (fileInfo.mtime > file.lastModified) {
        // تم تعديل الملف خارجياً
        const externalContent = fileInfo.content;
        
        if (externalContent !== file.content) {
          // هناك تعارض
          if (opts.detectConflicts && file.status === 'modified') {
            setWatchedFiles((prev) => {
              const next = new Map(prev);
              next.set(path, {
                ...file,
                status: 'conflict',
                externalModifiedAt: fileInfo.mtime,
              });
              return next;
            });
            opts.onConflictDetected(path);
          } else if (opts.autoReload) {
            // إعادة التحميل التلقائي
            await reloadFile(path);
          } else {
            setWatchedFiles((prev) => {
              const next = new Map(prev);
              next.set(path, {
                ...file,
                status: 'external_modified',
                externalModifiedAt: fileInfo.mtime,
              });
              return next;
            });
          }
        }
      }
    } catch (err) {
      console.error(`فشل التحقق من التغييرات الخارجية للملف ${path}:`, err);
    }
  }, [watchedFiles, getFileInfo, opts.detectConflicts, opts.autoReload, opts.onConflictDetected, reloadFile]);

  /**
   * الحصول على حالة الملف
   */
  const getFileStatus = useCallback((path: string): WatchedFile | undefined => {
    return watchedFiles.get(path);
  }, [watchedFiles]);

  /**
   * قفل الملف
   */
  const lockFile = useCallback(async (path: string): Promise<boolean> => {
    if (!opts.enableLocking) return true;

    try {
      const success = await invoke<boolean>('lock_file', { path });
      
      if (success) {
        setWatchedFiles((prev) => {
          const file = prev.get(path);
          if (!file) return prev;

          const next = new Map(prev);
          next.set(path, {
            ...file,
            lockOwner: 'current',
          });
          return next;
        });
      }
      
      return success;
    } catch (err) {
      console.error(`فشل قفل الملف ${path}:`, err);
      return false;
    }
  }, [opts.enableLocking]);

  /**
   * فتح قفل الملف
   */
  const unlockFile = useCallback(async (path: string): Promise<boolean> => {
    if (!opts.enableLocking) return true;

    try {
      const success = await invoke<boolean>('unlock_file', { path });
      
      if (success) {
        setWatchedFiles((prev) => {
          const file = prev.get(path);
          if (!file) return prev;

          const next = new Map(prev);
          next.set(path, {
            ...file,
            lockOwner: undefined,
          });
          return next;
        });
      }
      
      return success;
    } catch (err) {
      console.error(`فشل فتح قفل الملف ${path}:`, err);
      return false;
    }
  }, [opts.enableLocking]);

  /**
   * حل التعارض
   */
  const resolveConflict = useCallback(async (
    path: string,
    resolution: 'local' | 'external' | 'merge',
    mergedContent?: string
  ) => {
    const file = watchedFiles.get(path);
    if (!file) throw new Error(`الملف ${path} غير موجود`);

    try {
      let newContent: string;

      switch (resolution) {
        case 'local':
          newContent = file.content;
          break;
        case 'external':
          const fileInfo = await getFileInfo(path);
          newContent = fileInfo.content;
          break;
        case 'merge':
          if (!mergedContent) throw new Error('محتوى الدمج مطلوب');
          newContent = mergedContent;
          break;
        default:
          throw new Error('استراتيجية حل غير صالحة');
      }

      setWatchedFiles((prev) => {
        const next = new Map(prev);
        next.set(path, {
          ...file,
          content: newContent,
          status: 'modified',
          externalModifiedAt: undefined,
        });
        return next;
      });
    } catch (err) {
      console.error(`فشل حل التعارض للملف ${path}:`, err);
      throw err;
    }
  }, [watchedFiles, getFileInfo]);

  /**
   * إيقاف جميع المراقبات
   */
  const stopAllWatchers = useCallback(() => {
    // إيقاف جميع الفواصل
    intervalsRef.current.forEach((interval) => {
      clearInterval(interval);
    });
    intervalsRef.current.clear();

    // إلغاء مراقبة جميع الملفات
    watchedFiles.forEach((_, path) => {
      invoke('unwatch_file', { path }).catch(console.error);
    });

    setWatchedFiles(new Map());
  }, [watchedFiles]);

  // الاستماع لأحداث تغيير الملف من Tauri
  useEffect(() => {
    const setupListener = async () => {
      const unlisten = await listen<FileChangeEvent>('file-changed', (event) => {
        const { path, type } = event.payload;
        
        opts.onFileChange(event.payload);

        if (type === 'deleted') {
          unwatchFile(path).catch(console.error);
        } else if (type === 'modified' && opts.autoReload) {
          reloadFile(path).catch(console.error);
        }
      });

      unlistenRef.current = unlisten;
    };

    setupListener();

    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
      stopAllWatchers();
    };
  }, [opts.onFileChange, opts.autoReload, unwatchFile, reloadFile, stopAllWatchers]);

  return {
    watchedFiles,
    watchFile,
    unwatchFile,
    updateFileContent,
    saveFile,
    reloadFile,
    getFileStatus,
    lockFile,
    unlockFile,
    resolveConflict,
    stopAllWatchers,
  };
}

export default useFileWatcher;
