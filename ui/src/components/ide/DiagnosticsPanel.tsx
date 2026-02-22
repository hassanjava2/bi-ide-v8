import { memo, useCallback } from 'react'
import type { DiagnosticIssue, DiagnosticsSummary, RefactorSuggestion, RefactorSummary } from './types'

interface DiagnosticsPanelProps {
  diagnostics: DiagnosticIssue[]
  summary: DiagnosticsSummary
  busy: boolean
  onAnalyze: () => void
}

export const DiagnosticsPanel = memo(function DiagnosticsPanel({
  diagnostics,
  summary,
  busy,
  onAnalyze
}: DiagnosticsPanelProps) {
  const handleAnalyze = useCallback(() => {
    onAnalyze()
  }, [onAnalyze])

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent">Diagnostics</p>
        <button
          className="text-xs text-bi-accent hover:underline"
          onClick={handleAnalyze}
          disabled={busy}
        >
          {busy ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
      <p className="text-xs text-gray-400">
        Errors: {summary.errors} | Warnings: {summary.warnings} | Total: {summary.total}
      </p>
      <div className="mt-2 space-y-2 max-h-44 overflow-auto">
        {diagnostics.length === 0 ? (
          <p className="text-xs text-gray-500">No issues found</p>
        ) : (
          diagnostics.slice(0, 20).map((issue, index) => (
            <div key={index} className="text-xs text-gray-300 border border-white/10 rounded p-2">
              <p className={issue.severity === 'error' ? 'text-red-400' : 'text-yellow-300'}>
                [{issue.severity}] {issue.rule}
              </p>
              <p>{issue.message}</p>
              <p className="text-gray-500">L{issue.line}:C{issue.column}</p>
            </div>
          ))
        )}
      </div>
    </div>
  )
})

interface RefactorPanelProps {
  suggestions: RefactorSuggestion[]
  summary: RefactorSummary
  busy: boolean
  onSuggest: () => void
}

export const RefactorPanel = memo(function RefactorPanel({
  suggestions,
  summary,
  busy,
  onSuggest
}: RefactorPanelProps) {
  const handleSuggest = useCallback(() => {
    onSuggest()
  }, [onSuggest])

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent">Refactor</p>
        <button
          className="text-xs text-bi-accent hover:underline"
          onClick={handleSuggest}
          disabled={busy}
        >
          {busy ? 'Scanning...' : 'Suggest'}
        </button>
      </div>
      <p className="text-xs text-gray-400">
        Warnings: {summary.warnings} | Infos: {summary.infos} | Total: {summary.total}
      </p>
      <div className="mt-2 space-y-2 max-h-44 overflow-auto">
        {suggestions.length === 0 ? (
          <p className="text-xs text-gray-500">No refactor suggestions</p>
        ) : (
          suggestions.slice(0, 20).map((item, index) => (
            <div key={index} className="text-xs text-gray-300 border border-white/10 rounded p-2">
              <p className={item.severity === 'warning' ? 'text-yellow-300' : 'text-blue-300'}>
                [{item.severity}] {item.rule}
              </p>
              <p>{item.message}</p>
              <p className="text-gray-500">L{item.line}</p>
            </div>
          ))
        )}
      </div>
    </div>
  )
})

interface TestsPanelProps {
  testPath: string
  framework: string
  content: string
  busy: boolean
  onGenerate: () => void
}

export const TestsPanel = memo(function TestsPanel({
  testPath,
  framework,
  content,
  busy,
  onGenerate
}: TestsPanelProps) {
  const handleGenerate = useCallback(() => {
    onGenerate()
  }, [onGenerate])

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent">Tests</p>
        <button
          className="text-xs text-bi-accent hover:underline"
          onClick={handleGenerate}
          disabled={busy}
        >
          {busy ? 'Generating...' : 'Generate'}
        </button>
      </div>

      <p className="text-xs text-gray-400">Framework: {framework || '-'}</p>
      <p className="text-xs text-gray-400 mb-2">Path: {testPath || '-'}</p>

      <div className="max-h-44 overflow-auto rounded border border-white/10 bg-black/20 p-2">
        <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
          {content || 'No generated tests yet'}
        </pre>
      </div>
    </div>
  )
})

interface DocsPanelProps {
  symbol: string
  result: { symbol?: string; found?: boolean; location?: string; definition?: string; documentation?: string; related_symbols?: string[] } | null
  cacheHit: boolean
  cacheSize: number
  cacheEvictions: number
  busy: boolean
  maxCacheEntries: number
  onSymbolChange: (symbol: string) => void
  onLookup: () => void
  onRefresh: () => void
  onClearCache: () => void
  onReset: () => void
  onOpenLocation: (location: string) => void
}

export const DocsPanel = memo(function DocsPanel({
  symbol,
  result,
  cacheHit,
  cacheSize,
  cacheEvictions,
  busy,
  maxCacheEntries,
  onSymbolChange,
  onLookup,
  onRefresh,
  onClearCache,
  onReset,
  onOpenLocation
}: DocsPanelProps) {
  const handleSymbolChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onSymbolChange(event.target.value)
  }, [onSymbolChange])

  const handleLocationClick = useCallback(() => {
    if (result?.location) {
      onOpenLocation(result.location)
    }
  }, [result?.location, onOpenLocation])

  const canReset = !busy && !result && !symbol && cacheSize === 0 && cacheEvictions === 0

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent">Docs</p>
        <div className="flex items-center gap-2">
          <button
            className="text-xs text-bi-accent hover:underline"
            onClick={onLookup}
            disabled={busy}
          >
            {busy ? 'Searching...' : 'Lookup'}
          </button>
          <button
            className="text-xs text-gray-400 hover:underline"
            onClick={onRefresh}
            disabled={busy}
          >
            Refresh
          </button>
          <button
            className="text-xs text-gray-400 hover:underline"
            onClick={onClearCache}
            disabled={busy || cacheSize === 0}
          >
            Clear Cache
          </button>
          <button
            className="text-xs text-gray-400 hover:underline"
            onClick={onReset}
            disabled={canReset}
          >
            Reset UI
          </button>
        </div>
      </div>

      <input
        type="text"
        value={symbol}
        onChange={handleSymbolChange}
        placeholder="Symbol name (optional)"
        className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
      />
      <p className="mt-1 text-[10px] text-gray-500">Shortcut: Ctrl+Shift+D or Ctrl+Click on symbol</p>
      <p className="mt-1 text-[10px] text-gray-500">Source: {cacheHit ? 'cache (ttl 60s, max 100)' : 'live'}</p>
      <p className="mt-1 text-[10px] text-gray-500">Cache: size {cacheSize}/{maxCacheEntries}, evictions {cacheEvictions}</p>

      <div className="mt-2 max-h-44 overflow-auto rounded border border-white/10 bg-black/20 p-2">
        {!result ? (
          <p className="text-xs text-gray-500">No documentation result yet</p>
        ) : (
          <div className="space-y-1 text-[10px] text-gray-300">
            <p className="text-bi-accent">Symbol: {result.symbol || '-'}</p>
            <p>Found: {result.found ? 'yes' : 'no'}</p>
            {result.location && (
              <button
                className="text-left text-bi-accent hover:underline"
                onClick={handleLocationClick}
              >
                Location: {result.location}
              </button>
            )}
            {result.definition && (
              <pre className="whitespace-pre-wrap break-words text-gray-300 border border-white/10 rounded p-1 bg-black/30">
                {result.definition}
              </pre>
            )}
            <pre className="whitespace-pre-wrap break-words text-gray-300">
              {result.documentation || 'No documentation available'}
            </pre>
            {!!result.related_symbols?.length && (
              <p className="text-gray-400">Related: {result.related_symbols.join(', ')}</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
})
