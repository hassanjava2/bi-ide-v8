/**
 * Auto Update Hook - هوك فحص التحديثات التلقائي
 * 
 * يقوم بفحص التحديثات كل 5 دقائق ومقارنة النسخة المحلية بالبعيدة
 * مع تتبع التقدم وعرض سجل التغييرات وخيارات التثبيت/التخطي
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { checkUpdate, installUpdate, onUpdaterEvent } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';

/** حالة التحديث */
export type UpdateStatus = 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'error';

/** معلومات التحديث */
export interface UpdateInfo {
  version: string;
  date: string;
  body: string;
  notes?: string;
}

/** تقدم التنزيل */
export interface DownloadProgress {
  downloaded: number;
  total: number;
  percentage: number;
}

/** خيارات التحديث */
export interface AutoUpdateOptions {
  /** الفاصل الزمني للفحص (بالدقائق) - الافتراضي 5 دقائق */
  checkIntervalMinutes?: number;
  /** تفعيل التنزيل في الخلفية */
  backgroundDownload?: boolean;
  /** إصدار تجاهله */
  skippedVersion?: string | null;
}

/** نتيجة هوك التحديث */
export interface UseAutoUpdateResult {
  /** الحالة الحالية */
  status: UpdateStatus;
  /** معلومات التحديث المتاح */
  updateInfo: UpdateInfo | null;
  /** تقدم التنزيل */
  progress: DownloadProgress | null;
  /** رسالة الخطأ */
  error: string | null;
  /** فحص التحديث يدوياً */
  checkForUpdates: () => Promise<void>;
  /** تنزيل التحديث */
  downloadUpdate: () => Promise<void>;
  /** تثبيت التحديث وإعادة التشغيل */
  installAndRelaunch: () => Promise<void>;
  /** تخطي هذا الإصدار */
  skipVersion: () => void;
  /** إلغاء التنزيل */
  cancelDownload: () => void;
}

/** الفاصل الزمني الافتراضي للفحص (5 دقائق) */
const DEFAULT_CHECK_INTERVAL = 5 * 60 * 1000;

/**
 * هوك فحص التحديثات التلقائي
 * @param options - خيارات التحديث
 * @returns نتيجة التحكم بالتحديث
 */
export function useAutoUpdate(options: AutoUpdateOptions = {}): UseAutoUpdateResult {
  const {
    checkIntervalMinutes = 5,
    backgroundDownload = true,
    skippedVersion = null,
  } = options;

  const [status, setStatus] = useState<UpdateStatus>('idle');
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [skipped, setSkipped] = useState<string | null>(skippedVersion);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * فحص وجود تحديثات جديدة
   */
  const checkForUpdates = useCallback(async () => {
    try {
      setStatus('checking');
      setError(null);

      const { shouldUpdate, manifest } = await checkUpdate();

      if (shouldUpdate && manifest) {
        // التحقق من أن الإصدار لم يتم تخطيه
        if (manifest.version === skipped) {
          setStatus('idle');
          return;
        }

        setUpdateInfo({
          version: manifest.version,
          date: manifest.date || new Date().toISOString(),
          body: manifest.body || '',
          notes: manifest.body,
        });
        setStatus('available');

        // التنزيل التلقائي في الخلفية إذا مفعل
        if (backgroundDownload) {
          await downloadUpdate();
        }
      } else {
        setStatus('idle');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل فحص التحديثات');
      setStatus('error');
    }
  }, [skipped, backgroundDownload]);

  /**
   * تنزيل التحديث
   */
  const downloadUpdate = useCallback(async () => {
    if (status === 'downloading' || status === 'ready') return;

    try {
      setStatus('downloading');
      setProgress({ downloaded: 0, total: 0, percentage: 0 });

      // إنشاء AbortController للإلغاء
      abortControllerRef.current = new AbortController();

      // الاستماع لأحداث التقدم
      const unlisten = await onUpdaterEvent((event) => {
        if (event.error) {
          setError(event.error);
          setStatus('error');
          return;
        }

        if (event.status === 'DONE') {
          setStatus('ready');
          setProgress(null);
        } else if (event.status === 'PENDING' && event.data) {
          const { downloaded, total } = event.data;
          setProgress({
            downloaded,
            total,
            percentage: total > 0 ? Math.round((downloaded / total) * 100) : 0,
          });
        }
      });

      // بدء التنزيل
      await installUpdate();

      // تنظيف
      unlisten();
    } catch (err) {
      if (abortControllerRef.current?.signal.aborted) {
        setStatus('available');
      } else {
        setError(err instanceof Error ? err.message : 'فشل تنزيل التحديث');
        setStatus('error');
      }
      setProgress(null);
    }
  }, [status]);

  /**
   * تثبيت التحديث وإعادة التشغيل
   */
  const installAndRelaunch = useCallback(async () => {
    try {
      await relaunch();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تثبيت التحديث');
      setStatus('error');
    }
  }, []);

  /**
   * تخطي هذا الإصدار
   */
  const skipVersion = useCallback(() => {
    if (updateInfo) {
      setSkipped(updateInfo.version);
      setStatus('idle');
      setUpdateInfo(null);
      // حفظ في التخزين المحلي
      localStorage.setItem('skipped_update_version', updateInfo.version);
    }
  }, [updateInfo]);

  /**
   * إلغاء التنزيل
   */
  const cancelDownload = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setStatus('available');
    setProgress(null);
  }, []);

  // استعادة الإصدار المتجاهل من التخزين المحلي
  useEffect(() => {
    const saved = localStorage.getItem('skipped_update_version');
    if (saved) {
      setSkipped(saved);
    }
  }, []);

  // إعداد الفحص الدوري
  useEffect(() => {
    // الفحص الأولي
    checkForUpdates();

    // إعداد الفاصل الزمني
    const interval = checkIntervalMinutes * 60 * 1000;
    intervalRef.current = setInterval(checkForUpdates, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [checkForUpdates, checkIntervalMinutes]);

  return {
    status,
    updateInfo,
    progress,
    error,
    checkForUpdates,
    downloadUpdate,
    installAndRelaunch,
    skipVersion,
    cancelDownload,
  };
}

export default useAutoUpdate;
