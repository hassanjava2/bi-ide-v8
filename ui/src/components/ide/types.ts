// Types for IDE components

export type FileTreeNode = {
  id: string
  name: string
  type: 'file' | 'folder'
  path: string
  language?: string
  children?: FileTreeNode[]
}

export type GitStatusFile = {
  path: string
  status: string
  category: string
}

export type SymbolDocsResult = {
  symbol?: string
  found?: boolean
  location?: string
  definition?: string
  documentation?: string
  related_symbols?: string[]
}

export type SymbolDocsCacheEntry = {
  result: SymbolDocsResult
  createdAt: number
}

export type ToolPanel = 'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug'

export type DiagnosticIssue = {
  line: number
  column: number
  severity: string
  message: string
  rule: string
}

export type DiagnosticsSummary = {
  errors: number
  warnings: number
  infos: number
  total: number
}

export type RefactorSuggestion = {
  rule: string
  message: string
  line: number
  severity: string
}

export type RefactorSummary = {
  warnings: number
  infos: number
  total: number
}

export type GeneratedTestResult = {
  path: string
  framework: string
  content: string
}

// Constants
export const DOCS_CACHE_TTL_MS = 60_000
export const DOCS_CACHE_MAX_ENTRIES = 100
export const DOCS_CACHE_EVICTIONS_STORAGE_KEY = 'ide_docs_cache_evictions'
export const DOCS_SYMBOL_STORAGE_KEY = 'ide_docs_symbol'
