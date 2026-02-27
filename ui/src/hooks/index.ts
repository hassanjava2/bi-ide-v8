export { useDebouncedCallback, type DebouncedFunction } from './useDebouncedCallback';
export { useDebouncedDiagnostics, type DiagnosticIssue, type DiagnosticsSummary } from './useDebouncedDiagnostics';
export { useDebouncedRefactor, type RefactorSuggestion, type RefactorSummary } from './useDebouncedRefactor';
export { useDebouncedSave, useDebouncedLocalStorage, type DebouncedSaveCallbacks } from './useDebouncedSave';

// ═══════════════════════════════════════════════════════════════
// New API Hooks
// ═══════════════════════════════════════════════════════════════

export { useAuth, type User, type AuthState } from './useAuth';
export { useERP } from './useERP';
export { useCommunity } from './useCommunity';
export type { Customer, Deal, Product } from './useERP';
export type { Employee } from './useERP-legacy';
export type { Forum, Topic, Post, Article, CodeSnippet } from './useCommunity';

// Legacy exports for backward compatibility
export {
  useERPDashboard,
  useInvoices,
  useInventory,
  useEmployees,
  useAccounts,
  type DashboardData,
  type Invoice,
  type InventoryItem,
  type Account,
} from './useERP-legacy';

export {
  useCouncil,
  useCouncilStatus,
  useCouncilHistory,
  useWiseMen,
  type WiseMan,
  type CouncilMessage,
  type CouncilStatus,
} from './useCouncil';
