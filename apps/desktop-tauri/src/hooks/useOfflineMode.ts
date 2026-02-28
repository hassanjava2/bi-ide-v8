/**
 * Offline Mode Hook - هوك العمل بدون إنترنت
 * 
 * يكتشف حالة الاتصال ويخزن الاستجابات محلياً ويضع الإجراءات في قائمة الانتظار
 * للمزامنة عند إعادة الاتصال مع إدارة التخزين المحلي ومؤشر واجهة المستخدم
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/** حالة الاتصال */
export type ConnectionStatus = 'online' | 'offline' | 'checking';

/** إجراء في قائمة الانتظار */
export interface QueuedAction {
  id: string;
  timestamp: number;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
  retryCount: number;
}

/** عنصر مخبأ */
export interface CachedItem<T = unknown> {
  key: string;
  data: T;
  timestamp: number;
  expiresAt: number;
}

/** إعدادات وضع عدم الاتصال */
export interface OfflineModeOptions {
  /** مفتاح التخزين المحلي */
  storageKey?: string;
  /** مدة صلاحية الذاكرة المؤقتة (بالساعات) */
  cacheExpiryHours?: number;
  /** الحد الأقصى لإعادة المحاولة */
  maxRetries?: number;
  /** التزامن التلقائي عند إعادة الاتصال */
  autoSyncOnReconnect?: boolean;
}

/** نتيجة هوك وضع عدم الاتصال */
export interface UseOfflineModeResult {
  /** حالة الاتصال الحالية */
  status: ConnectionStatus;
  /** هل الجهاز متصل */
  isOnline: boolean;
  /** هل الجهاز غير متصل */
  isOffline: boolean;
  /** الإجراءات في قائمة الانتظار */
  queuedActions: QueuedAction[];
  /** عدد الإجراءات المعلقة */
  pendingCount: number;
  /** هل يتم التزامن حالياً */
  isSyncing: boolean;
  /** آخر خطأ في التزامن */
  lastSyncError: string | null;
  /** إضافة إجراء إلى قائمة الانتظار */
  queueAction: (action: Omit<QueuedAction, 'id' | 'timestamp' | 'retryCount'>) => string;
  /** إزالة إجراء من قائمة الانتظار */
  removeQueuedAction: (id: string) => void;
  /** تنفيذ التزامن */
  sync: () => Promise<void>;
  /** تخزين بيانات في الذاكرة المؤقتة */
  cacheData: <T>(key: string, data: T) => void;
  /** استرجاع بيانات من الذاكرة المؤقتة */
  getCachedData: <T>(key: string) => CachedItem<T> | null;
  /** مسح الذاكرة المؤقتة */
  clearCache: () => void;
  /** فحص الاتصال يدوياً */
  checkConnection: () => Promise<boolean>;
}

/** المفتاح الافتراضي للتخزين */
const DEFAULT_STORAGE_KEY = 'bi_ide_offline_queue';

/** مدة الصلاحية الافتراضية (24 ساعة) */
const DEFAULT_CACHE_EXPIRY = 24;

/** الحد الأقصى الافتراضي لإعادة المحاولة */
const DEFAULT_MAX_RETRIES = 3;

/**
 * إنشاء معرف فريد
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * هوك العمل بدون إنترنت
 * @param options - إعدادات وضع عدم الاتصال
 * @returns نتيجة التحكم بوضع عدم الاتصال
 */
export function useOfflineMode(options: OfflineModeOptions = {}): UseOfflineModeResult {
  const {
    storageKey = DEFAULT_STORAGE_KEY,
    cacheExpiryHours = DEFAULT_CACHE_EXPIRY,
    maxRetries = DEFAULT_MAX_RETRIES,
    autoSyncOnReconnect = true,
  } = options;

  const [status, setStatus] = useState<ConnectionStatus>('checking');
  const [queuedActions, setQueuedActions] = useState<QueuedAction[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncError, setLastSyncError] = useState<string | null>(null);
  const cacheRef = useRef<Map<string, CachedItem>>(new Map());
  const wasOfflineRef = useRef(false);

  /**
   * فحص حالة الاتصال
   */
  const checkConnection = useCallback(async (): Promise<boolean> => {
    setStatus('checking');
    try {
      // محاولة الوصول إلى خادم
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      await fetch('https://www.google.com/favicon.ico', {
        mode: 'no-cors',
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      setStatus('online');
      return true;
    } catch {
      setStatus('offline');
      return false;
    }
  }, []);

  /**
   * تحميل الإجراءات من التخزين المحلي
   */
  const loadQueuedActions = useCallback(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed: QueuedAction[] = JSON.parse(saved);
        setQueuedActions(parsed);
      }
    } catch (err) {
      console.error('فشل تحميل الإجراءات من قائمة الانتظار:', err);
    }
  }, [storageKey]);

  /**
   * حفظ الإجراءات في التخزين المحلي
   */
  const saveQueuedActions = useCallback((actions: QueuedAction[]) => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(actions));
    } catch (err) {
      console.error('فشل حفظ الإجراءات في قائمة الانتظار:', err);
    }
  }, [storageKey]);

  /**
   * إضافة إجراء إلى قائمة الانتظار
   */
  const queueAction = useCallback(
    (action: Omit<QueuedAction, 'id' | 'timestamp' | 'retryCount'>): string => {
      const newAction: QueuedAction = {
        ...action,
        id: generateId(),
        timestamp: Date.now(),
        retryCount: 0,
      };

      setQueuedActions((prev) => {
        const updated = [...prev, newAction];
        saveQueuedActions(updated);
        return updated;
      });

      return newAction.id;
    },
    [saveQueuedActions]
  );

  /**
   * إزالة إجراء من قائمة الانتظار
   */
  const removeQueuedAction = useCallback(
    (id: string) => {
      setQueuedActions((prev) => {
        const updated = prev.filter((action) => action.id !== id);
        saveQueuedActions(updated);
        return updated;
      });
    },
    [saveQueuedActions]
  );

  /**
   * تنفيذ الإجراء
   */
  const executeAction = useCallback(async (action: QueuedAction): Promise<void> => {
    const response = await fetch(action.endpoint, {
      method: action.method,
      headers: {
        'Content-Type': 'application/json',
        ...action.headers,
      },
      body: action.body ? JSON.stringify(action.body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`فشل تنفيذ الإجراء: ${response.statusText}`);
    }
  }, []);

  /**
   * تنفيذ التزامن
   */
  const sync = useCallback(async () => {
    if (queuedActions.length === 0 || status === 'offline') return;

    setIsSyncing(true);
    setLastSyncError(null);

    try {
      const failedActions: QueuedAction[] = [];

      for (const action of queuedActions) {
        try {
          await executeAction(action);
        } catch (err) {
          if (action.retryCount < maxRetries) {
            failedActions.push({
              ...action,
              retryCount: action.retryCount + 1,
            });
          }
        }
      }

      setQueuedActions(failedActions);
      saveQueuedActions(failedActions);

      if (failedActions.length > 0) {
        setLastSyncError(`فشل ${failedActions.length} إجراء بعد ${maxRetries} محاولات`);
      }
    } catch (err) {
      setLastSyncError(err instanceof Error ? err.message : 'فشل التزامن');
    } finally {
      setIsSyncing(false);
    }
  }, [queuedActions, status, maxRetries, executeAction, saveQueuedActions]);

  /**
   * تخزين بيانات في الذاكرة المؤقتة
   */
  const cacheData = useCallback(<T,>(key: string, data: T) => {
    const now = Date.now();
    const expiresAt = now + cacheExpiryHours * 60 * 60 * 1000;
    
    cacheRef.current.set(key, {
      key,
      data,
      timestamp: now,
      expiresAt,
    });
  }, [cacheExpiryHours]);

  /**
   * استرجاع بيانات من الذاكرة المؤقتة
   */
  const getCachedData = useCallback(<T,>(key: string): CachedItem<T> | null => {
    const item = cacheRef.current.get(key);
    
    if (!item) return null;
    
    // التحقق من انتهاء الصلاحية
    if (Date.now() > item.expiresAt) {
      cacheRef.current.delete(key);
      return null;
    }
    
    return item as CachedItem<T>;
  }, []);

  /**
   * مسح الذاكرة المؤقتة
   */
  const clearCache = useCallback(() => {
    cacheRef.current.clear();
  }, []);

  // مراقبة تغيرات حالة الاتصال
  useEffect(() => {
    const handleOnline = () => {
      setStatus('online');
      if (wasOfflineRef.current && autoSyncOnReconnect) {
        sync();
      }
      wasOfflineRef.current = false;
    };

    const handleOffline = () => {
      setStatus('offline');
      wasOfflineRef.current = true;
    };

    // التحقق من الاتصال الأولي
    checkConnection();
    loadQueuedActions();

    // الاستماع لتغيرات الاتصال
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [checkConnection, loadQueuedActions, sync, autoSyncOnReconnect]);

  // فحص دوري للاتصال
  useEffect(() => {
    const interval = setInterval(() => {
      checkConnection();
    }, 30000); // كل 30 ثانية

    return () => clearInterval(interval);
  }, [checkConnection]);

  return {
    status,
    isOnline: status === 'online',
    isOffline: status === 'offline',
    queuedActions,
    pendingCount: queuedActions.length,
    isSyncing,
    lastSyncError,
    queueAction,
    removeQueuedAction,
    sync,
    cacheData,
    getCachedData,
    clearCache,
    checkConnection,
  };
}

export default useOfflineMode;
