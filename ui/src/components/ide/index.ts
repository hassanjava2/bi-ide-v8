// IDE Components exports
export { FileExplorer } from './FileExplorer'
export { CodeEditor } from './CodeEditor'
export { TerminalPanel } from './TerminalPanel'
export { GitPanel } from './GitPanel'
export { DebugPanel } from './DebugPanel'
export { DiagnosticsPanel, RefactorPanel, TestsPanel, DocsPanel } from './DiagnosticsPanel'
export { ToolsSidebar } from './ToolsSidebar'

// Types
export type {
  FileTreeNode,
  GitStatusFile,
  SymbolDocsResult,
  SymbolDocsCacheEntry,
  ToolPanel,
  DiagnosticIssue,
  DiagnosticsSummary,
  RefactorSuggestion,
  RefactorSummary,
  GeneratedTestResult
} from './types'

// Constants
export {
  DOCS_CACHE_TTL_MS,
  DOCS_CACHE_MAX_ENTRIES,
  DOCS_CACHE_EVICTIONS_STORAGE_KEY,
  DOCS_SYMBOL_STORAGE_KEY
} from './types'
