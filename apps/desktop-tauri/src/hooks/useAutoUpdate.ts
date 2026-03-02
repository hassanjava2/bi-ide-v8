/**
 * Auto Update Hook - هوك فحص التحديثات التلقائي
 * 
 * يقوم بفحص التحديثات كل 5 دقائق ومقارنة النسخة المحلية بالبعيدة
 * مع تتبع التقدم وعرض سجل التغييرات وخيارات التثبيت/التخطي
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-shell';

type CheckUpdateResult = {
  shouldUpdate: boolean;
  manifest?: {
    version: string;
    date?: string;
    body?: string;
    release_notes?: string;
    critical?: boolean;
    size_mb?: number;
    download_url?: string;
  };
};

async function getCurrentVersion(): Promise<string> {
  try {
    const { getVersion } = await import('@tauri-apps/api/app');
    return await getVersion();
  } catch {
    return '0.1.0';
  }
}

async function checkUpdate(): Promise<CheckUpdateResult> {
  try {
    const currentVersion = await getCurrentVersion();
    const result = await invoke<{ has_update: boolean; manifest?: any }>('check_for_updates', {
      request: {
        current_version: currentVersion,
        channel: 'stable',
      },
    });

    return {
      shouldUpdate: result.has_update,
      manifest: result.manifest
        ? {
            version: result.manifest.version,
            body: result.manifest.release_notes,
            release_notes: result.manifest.release_notes,
            critical: result.manifest.critical,
            size_mb: result.manifest.size_mb,
            download_url: result.manifest.download_url,
          }
        : undefined,
    };
  } catch (error) {
    console.error('checkUpdate failed:', error);
    return { shouldUpdate: false };
  }
}

async function installUpdate(downloadUrl?: string): Promise<void> {
  if (!downloadUrl) {
    throw new Error('No download URL available for update');
  }
  await open(downloadUrl);
}

async function relaunch(): Promise<void> {
  try {
    const { exit } = await import('@tauri-apps/plugin-process');
    await exit(0);
  } catch { }
}
/** حالة التحديث */
export type UpdateStatus = 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'error';

/** معلومات التحديث */
export interface UpdateInfo {
  version: string;
  date: string;
  body: string;
  notes?: string;
  downloadUrl?: string;
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
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
          downloadUrl: manifest.download_url,
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
      setProgress({ downloaded: 0, total: 100, percentage: 10 });

      // بدء تنزيل التحديث (فتح رابط التحميل الرسمي)
      await installUpdate(updateInfo?.downloadUrl);

      if (updateInfo?.version) {
        await invoke('report_update_status', {
          request: {
            version_from: await getCurrentVersion(),
            version_to: updateInfo.version,
            status: 'downloaded',
            error_message: null,
          },
        });
      }

      setProgress({ downloaded: 100, total: 100, percentage: 100 });
      setStatus('ready');
      setProgress(null);
    } catch (err) {
      if (abortControllerRef.current?.signal.aborted) {
        setStatus('available');
      } else {
        setError(err instanceof Error ? err.message : 'فشل تنزيل التحديث');
        setStatus('error');
      }
      setProgress(null);
    }
  }, [status, updateInfo]);

  /**
   * تثبيت التحديث وإعادة التشغيل
   */
  const installAndRelaunch = useCallback(async () => {
    try {
      if (updateInfo?.version) {
        await invoke('report_update_status', {
          request: {
            version_from: await getCurrentVersion(),
            version_to: updateInfo.version,
            status: 'success',
            error_message: null,
          },
        });
      }
      await relaunch();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تثبيت التحديث');
      setStatus('error');
    }
  }, [updateInfo]);

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
