import { useCallback, useRef, useEffect } from 'react';
import { getRefactorSuggestions } from '../services/api';

export type RefactorSuggestion = {
  rule: string;
  message: string;
  line: number;
  severity: string;
};

export type RefactorSummary = {
  warnings: number;
  infos: number;
  total: number;
};

export type DebouncedRefactorCallbacks = {
  runRefactor: (content: string, filePath: string) => void;
  cancel: () => void;
  flush: () => void;
};

export const DEFAULT_SUMMARY: RefactorSummary = { warnings: 0, infos: 0, total: 0 };

/**
 * Hook for debounced refactor suggestions with aggressive 2000ms delay.
 * Refactor suggestions are expensive - this prevents running too frequently.
 * 
 * @param setSuggestions State setter for refactor suggestions
 * @param setSummary State setter for refactor summary
 * @param setIsScanning State setter for scanning busy state
 * @returns Object with debounced runRefactor function and controls
 * 
 * @example
 * const [suggestions, setSuggestions] = useState([]);
 * const [summary, setSummary] = useState(DEFAULT_SUMMARY);
 * const [isScanning, setIsScanning] = useState(false);
 * 
 * const { runRefactor, cancel, flush } = useDebouncedRefactor(
 *   setSuggestions,
 *   setSummary,
 *   setIsScanning
 * );
 * 
 * // Call on editor change
 * runRefactor(code, filePath);
 * 
 * // Cancel pending scan (e.g., on unmount or file switch)
 * cancel();
 */
export function useDebouncedRefactor(
  setSuggestions: (suggestions: RefactorSuggestion[]) => void,
  setSummary: (summary: RefactorSummary) => void,
  setIsScanning: (busy: boolean) => void,
  delay = 2000
): DebouncedRefactorCallbacks {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const pendingRef = useRef<{ content: string; filePath: string } | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancel();
    };
  }, []);

  const cancel = useCallback(() => {
    // Clear timeout
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    // Abort in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    pendingRef.current = null;
  }, []);

  const flush = useCallback(async () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (pendingRef.current) {
      const { content, filePath } = pendingRef.current;
      pendingRef.current = null;
      await executeRefactor(content, filePath);
    }
  }, []);

  const executeRefactor = useCallback(async (code: string, filePath: string) => {
    // Cancel any existing request before starting new one
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsScanning(true);
    try {
      const language = detectLanguage(filePath);
      const result = await getRefactorSuggestions(code, language, filePath);
      
      // Check if aborted during request
      if (abortControllerRef.current?.signal.aborted) {
        return;
      }
      
      setSuggestions(Array.isArray(result.suggestions) ? result.suggestions : []);
      setSummary(result.summary || { warnings: 0, infos: 0, total: 0 });
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        return; // Silently ignore aborted requests
      }
      setSuggestions([{ 
        rule: 'refactor-failed', 
        message: String(error), 
        line: 1, 
        severity: 'warning' 
      }]);
      setSummary({ warnings: 1, infos: 0, total: 1 });
    } finally {
      setIsScanning(false);
    }
  }, [setSuggestions, setSummary, setIsScanning]);

  const runRefactor = useCallback(
    (content: string, filePath: string) => {
      // Don't run if missing required params
      if (!filePath) return;

      // Store latest args
      pendingRef.current = { content, filePath };

      // Clear existing timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      // Set new timer with very aggressive delay (refactor is expensive)
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        pendingRef.current = null;
        void executeRefactor(content, filePath);
      }, delay);
    },
    [delay, executeRefactor]
  );

  return { runRefactor, cancel, flush };
}

/**
 * Detect programming language from file path
 */
function detectLanguage(pathOrName: string): string {
  const lower = pathOrName.toLowerCase();
  if (lower.endsWith('.py')) return 'python';
  if (lower.endsWith('.ts') || lower.endsWith('.tsx')) return 'typescript';
  if (lower.endsWith('.js') || lower.endsWith('.jsx')) return 'javascript';
  if (lower.endsWith('.json')) return 'json';
  if (lower.endsWith('.md')) return 'markdown';
  if (lower.endsWith('.html')) return 'html';
  if (lower.endsWith('.css')) return 'css';
  return 'python';
}

export default useDebouncedRefactor;
