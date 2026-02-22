import { useCallback, useEffect, useRef } from 'react';

export type DebouncedFunction<T extends (...args: any[]) => any> = {
  (...args: Parameters<T>): void;
  cancel: () => void;
  flush: () => void;
};

/**
 * Generic debounce hook using useRef for stable timer references.
 * Returns a debounced function that delays invoking `fn` until after `delay` milliseconds
 * have elapsed since the last time the debounced function was invoked.
 * 
 * Features:
 * - Cancelable via returned function's `cancel()` method
 * - Manual flush via `flush()` method
 * - Automatic cleanup on unmount
 * - Timer ref persists across renders without causing re-renders
 * 
 * @param fn The function to debounce
 * @param delay Delay in milliseconds (default: 500ms)
 * @returns Debounced function with `cancel` and `flush` methods
 * 
 * @example
 * const debouncedSearch = useDebouncedCallback((query: string) => {
 *   api.search(query);
 * }, 300);
 * 
 * // Cancel pending execution
 * debouncedSearch.cancel();
 * 
 * // Execute immediately if pending
 * debouncedSearch.flush();
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  fn: T,
  delay = 500
): DebouncedFunction<T> {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fnRef = useRef(fn);
  const pendingArgsRef = useRef<Parameters<T> | null>(null);

  // Keep fnRef current without triggering re-renders
  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  const cancel = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    pendingArgsRef.current = null;
  }, []);

  const flush = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (pendingArgsRef.current) {
      const args = pendingArgsRef.current;
      pendingArgsRef.current = null;
      fnRef.current(...args);
    }
  }, []);

  const debouncedFn = useCallback(
    (...args: Parameters<T>) => {
      // Store latest args for flush
      pendingArgsRef.current = args;

      // Clear existing timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      // Set new timer
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        pendingArgsRef.current = null;
        fnRef.current(...args);
      }, delay);
    },
    [delay]
  );

  // Attach cancel and flush methods
  const debouncedWithControls = debouncedFn as DebouncedFunction<T>;
  debouncedWithControls.cancel = cancel;
  debouncedWithControls.flush = flush;

  return debouncedWithControls;
}

export default useDebouncedCallback;
