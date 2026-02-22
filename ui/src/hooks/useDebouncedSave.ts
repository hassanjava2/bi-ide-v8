import { useCallback, useRef, useEffect } from 'react';

export type DebouncedSaveCallbacks<T = string> = {
  save: (value: T) => void;
  cancel: () => void;
  flush: () => void;
  isPending: () => boolean;
};

/**
 * Hook for debouncing save operations with 1000ms delay.
 * Useful for localStorage saves and other non-critical persistence.
 * 
 * @param saveFn Function to call with the debounced value
 * @param delay Delay in milliseconds (default: 1000ms)
 * @returns Object with debounced save function and controls
 * 
 * @example
 * // For localStorage saves
 * const { save: savePanel, cancel: cancelPanelSave } = useDebouncedSave(
 *   (panel) => localStorage.setItem('active_panel', panel),
 *   1000
 * );
 * 
 * // On change
 * savePanel(newPanelValue);
 * 
 * // On unmount, flush pending
 * useEffect(() => () => { cancelPanelSave(); }, []);
 * 
 * @example
 * // For API saves with immediate feedback
 * const { save, flush, isPending } = useDebouncedSave(
 *   (content) => api.saveFile(content),
 *   1000
 * );
 * 
 * // Show indicator when pending
 * const pending = isPending();
 */
export function useDebouncedSave<T = string>(
  saveFn: (value: T) => void | Promise<void>,
  delay = 1000
): DebouncedSaveCallbacks<T> {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveFnRef = useRef(saveFn);
  const pendingValueRef = useRef<T | null>(null);
  const hasPendingRef = useRef(false);

  // Keep fnRef current without triggering re-renders
  useEffect(() => {
    saveFnRef.current = saveFn;
  }, [saveFn]);

  // Cleanup on unmount - flush pending saves
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      // Flush pending save on unmount
      if (hasPendingRef.current && pendingValueRef.current !== null) {
        void saveFnRef.current(pendingValueRef.current);
      }
    };
  }, []);

  const cancel = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    pendingValueRef.current = null;
    hasPendingRef.current = false;
  }, []);

  const flush = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (hasPendingRef.current && pendingValueRef.current !== null) {
      const value = pendingValueRef.current;
      pendingValueRef.current = null;
      hasPendingRef.current = false;
      void saveFnRef.current(value);
    }
  }, []);

  const isPending = useCallback(() => {
    return hasPendingRef.current;
  }, []);

  const save = useCallback(
    (value: T) => {
      // Store latest value
      pendingValueRef.current = value;
      hasPendingRef.current = true;

      // Clear existing timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      // Set new timer
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        hasPendingRef.current = false;
        const valueToSave = pendingValueRef.current;
        pendingValueRef.current = null;
        if (valueToSave !== null) {
          void saveFnRef.current(valueToSave);
        }
      }, delay);
    },
    [delay]
  );

  return { save, cancel, flush, isPending };
}

/**
 * Specialized hook for localStorage saves with error handling.
 * Automatically handles localStorage access errors gracefully.
 * 
 * @param key localStorage key
 * @param delay Delay in milliseconds (default: 1000ms)
 * @returns Object with debounced save function and controls
 * 
 * @example
 * const { save: saveSetting, cancel } = useDebouncedLocalStorage(
 *   'ide_active_tool_panel',
 *   1000
 * );
 * 
 * // On change
 * saveSetting(newPanelValue);
 */
export function useDebouncedLocalStorage(
  key: string,
  delay = 1000
): DebouncedSaveCallbacks<string> {
  const saveToStorage = useCallback((value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Silently ignore localStorage errors (private mode, quota exceeded, etc.)
    }
  }, [key]);

  return useDebouncedSave(saveToStorage, delay);
}

export default useDebouncedSave;
