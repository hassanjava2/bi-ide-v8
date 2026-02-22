import { useCallback, useRef, useEffect } from 'react';
import { analyzeCode } from '../services/api';

export type DiagnosticIssue = {
  line: number;
  column: number;
  severity: string;
  message: string;
  rule: string;
};

export type DiagnosticsSummary = {
  errors: number;
  warnings: number;
  infos: number;
  total: number;
};

export type DiagnosticsResult = {
  issues: DiagnosticIssue[];
  summary: DiagnosticsSummary;
};

export type DebouncedDiagnosticsState = {
  issues: DiagnosticIssue[];
  summary: DiagnosticsSummary;
  isAnalyzing: boolean;
};

export type DebouncedDiagnosticsCallbacks = {
  runDiagnostics: (content: string, filePath: string) => void;
  cancel: () => void;
  flush: () => void;
};

const DEFAULT_SUMMARY: DiagnosticsSummary = { errors: 0, warnings: 0, infos: 0, total: 0 };

/**
 * Hook for debounced code diagnostics with aggressive 1500ms delay.
 * Prevents diagnostics from running on every keystroke.
 * 
 * @param setIssues State setter for diagnostic issues
 * @param setSummary State setter for diagnostics summary  
 * @param setIsAnalyzing State setter for analysis busy state
 * @returns Object with debounced runDiagnostics function and controls
 * 
 * @example
 * const [diagnostics, setDiagnostics] = useState([]);
 * const [summary, setSummary] = useState(DEFAULT_SUMMARY);
 * const [isAnalyzing, setIsAnalyzing] = useState(false);
 * 
 * const { runDiagnostics, cancel, flush } = useDebouncedDiagnostics(
 *   setDiagnostics,
 *   setSummary,
 *   setIsAnalyzing
 * );
 * 
 * // Call on editor change
 * runDiagnostics(code, filePath);
 * 
 * // Cancel pending analysis (e.g., on unmount or file switch)
 * cancel();
 */
export function useDebouncedDiagnostics(
  setIssues: (issues: DiagnosticIssue[]) => void,
  setSummary: (summary: DiagnosticsSummary) => void,
  setIsAnalyzing: (busy: boolean) => void,
  delay = 1500
): DebouncedDiagnosticsCallbacks {
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
      await executeDiagnostics(content, filePath);
    }
  }, []);

  const executeDiagnostics = useCallback(async (code: string, filePath: string) => {
    // Cancel any existing request before starting new one
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsAnalyzing(true);
    try {
      const language = detectLanguage(filePath);
      const result = await analyzeCode(code, language, filePath);
      
      // Check if aborted during request
      if (abortControllerRef.current?.signal.aborted) {
        return;
      }
      
      setIssues(Array.isArray(result.issues) ? result.issues : []);
      setSummary(result.summary || { errors: 0, warnings: 0, infos: 0, total: 0 });
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        return; // Silently ignore aborted requests
      }
      setIssues([{ 
        line: 1, 
        column: 1, 
        severity: 'error', 
        message: String(error), 
        rule: 'analysis-failed' 
      }]);
      setSummary({ errors: 1, warnings: 0, infos: 0, total: 1 });
    } finally {
      setIsAnalyzing(false);
    }
  }, [setIssues, setSummary, setIsAnalyzing]);

  const runDiagnostics = useCallback(
    (content: string, filePath: string) => {
      // Don't run if missing required params
      if (!filePath) return;

      // Store latest args
      pendingRef.current = { content, filePath };

      // Clear existing timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      // Set new timer with aggressive delay
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        pendingRef.current = null;
        void executeDiagnostics(content, filePath);
      }, delay);
    },
    [delay, executeDiagnostics]
  );

  return { runDiagnostics, cancel, flush };
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

export default useDebouncedDiagnostics;
