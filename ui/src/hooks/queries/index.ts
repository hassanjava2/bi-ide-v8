/**
 * React Query Hooks Index
 * 
 * Central export point for all TanStack Query hooks.
 * 
 * Usage:
 * ```tsx
 * import { 
 *   useSystemStatus, 
 *   useCouncilMetrics,
 *   useSendCouncilMessage 
 * } from '../hooks/queries'
 * ```
 */

// System status hooks
export {
  useSystemStatus,
  useRefreshSystemStatus,
  prefetchSystemStatus,
  SYSTEM_STATUS_KEY,
} from './useSystemStatus'
export type { UseSystemStatusOptions } from './useSystemStatus'

// Council hooks
export {
  useCouncilMetrics,
  useCouncilHistory,
  useSendCouncilMessage,
  useCouncilDiscussion,
  COUNCIL_KEYS,
} from './useCouncilMetrics'
export type {
  CouncilMetrics,
  CouncilMessageRequest,
  CouncilMessageResponse,
  CouncilDiscussionRequest,
  CouncilDiscussionResponse,
  CouncilHistoryItem,
  CouncilHistoryResponse,
} from './useCouncilMetrics'

// Hierarchy hooks
export {
  useHierarchyStatus,
  useHierarchyMetrics,
  useExecuteCommand,
  useRefreshHierarchy,
  HIERARCHY_KEYS,
} from './useHierarchyStatus'
export type {
  LayerStatus,
  HierarchyStatus,
  HierarchyMetrics,
  ExecuteCommandRequest,
  ExecuteCommandResponse,
} from './useHierarchyStatus'

// ERP hooks
export {
  useERPDashboard,
  useInvoices,
  useInventory,
  useEmployees,
  usePayroll,
  useFinancialReport,
  useAIInsights,
  useCreateInvoice,
  useMarkInvoicePaid,
  useRefreshERP,
  ERP_KEYS,
} from './useERPDashboard'
export type {
  DashboardStats,
  ERPDashboard,
  InvoicesResponse,
  InventoryResponse,
  EmployeesResponse,
  PayrollResponse,
  FinancialReportResponse,
  AIInsightsResponse,
  CreateInvoiceRequest,
} from './useERPDashboard'

// File/IDE hooks
export {
  useFileTree,
  useFile,
  useSaveFile,
  useAnalyzeCode,
  useRefactorSuggestions,
  useGenerateTests,
  useSymbolDocumentation,
  useGitStatus,
  useGitDiff,
  useGitCommit,
  useGitPush,
  useGitPull,
  useStartTerminal,
  useExecuteTerminal,
  useStartDebug,
  useStopDebug,
  useSetBreakpoint,
  useExecuteDebugCommand,
  FILES_KEYS,
} from './useFiles'
export type {
  FileTreeNode,
  FileContent,
  CodeAnalysisResult,
  RefactorSuggestion,
  RefactorResult,
  TestGenerationResult,
  SymbolDocsResult,
  GitStatusFile,
  GitStatusResult,
  GitDiffResult,
  GitCommitResult,
  GitSyncResult,
  TerminalSession,
  TerminalExecuteResult,
  DebugSession,
  DebugCommandResult,
} from './useFiles'
