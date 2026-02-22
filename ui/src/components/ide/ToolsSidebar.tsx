import { Settings } from 'lucide-react'
import { memo, useCallback } from 'react'
import type { ToolPanel } from './types'
import { DiagnosticsPanel, RefactorPanel, TestsPanel, DocsPanel } from './DiagnosticsPanel'
import { GitPanel } from './GitPanel'
import { DebugPanel } from './DebugPanel'
import type { DiagnosticIssue, DiagnosticsSummary, RefactorSuggestion, RefactorSummary, GitStatusFile } from './types'

interface ToolsSidebarProps {
  collapsed: boolean
  activePanel: ToolPanel
  // Diagnostics
  diagnostics: DiagnosticIssue[]
  diagnosticsSummary: DiagnosticsSummary
  analyzeBusy: boolean
  onAnalyze: () => void
  // Refactor
  refactorSuggestions: RefactorSuggestion[]
  refactorSummary: RefactorSummary
  refactorBusy: boolean
  onRefactor: () => void
  // Tests
  testPath: string
  testFramework: string
  testContent: string
  testBusy: boolean
  onGenerateTests: () => void
  // Docs
  docsSymbol: string
  docsResult: { symbol?: string; found?: boolean; location?: string; definition?: string; documentation?: string; related_symbols?: string[] } | null
  docsCacheHit: boolean
  docsCacheSize: number
  docsCacheEvictions: number
  docsBusy: boolean
  maxCacheEntries: number
  onDocsSymbolChange: (symbol: string) => void
  onDocsLookup: () => void
  onDocsRefresh: () => void
  onDocsClearCache: () => void
  onDocsReset: () => void
  onOpenDocsLocation: (location: string) => void
  // Git
  gitBranch: string
  gitFiles: GitStatusFile[]
  gitError: string
  gitLoading: boolean
  gitDiffPath: string
  gitDiffText: string
  gitDiffLoading: boolean
  commitMessage: string
  commitBusy: boolean
  commitResult: string
  gitRemote: string
  gitBranchInput: string
  syncBusy: boolean
  syncResult: string
  onGitRefresh: () => void
  onGitLoadDiff: (path?: string) => void
  onGitSelectFile: (path: string) => void
  onCommitMessageChange: (message: string) => void
  onCommit: () => void
  onGitRemoteChange: (remote: string) => void
  onGitBranchInputChange: (branch: string) => void
  onPush: () => void
  onPull: () => void
  // Debug
  debugSessionId: string
  debugBusy: boolean
  debugOutput: string
  debugBreakpointLine: string
  onDebugBreakpointLineChange: (line: string) => void
  onDebugStart: () => void
  onDebugStop: () => void
  onDebugSetBreakpoint: () => void
  onDebugCommand: (command: string) => void
  // Actions
  onToggleCollapse: () => void
  onSetActivePanel: (panel: ToolPanel) => void
}

export const ToolsSidebar = memo(function ToolsSidebar(props: ToolsSidebarProps) {
  const { collapsed, activePanel, onToggleCollapse, onSetActivePanel } = props

  const handleSetDiagnostics = useCallback(() => onSetActivePanel('diagnostics'), [onSetActivePanel])
  const handleSetRefactor = useCallback(() => onSetActivePanel('refactor'), [onSetActivePanel])
  const handleSetTests = useCallback(() => onSetActivePanel('tests'), [onSetActivePanel])
  const handleSetDocs = useCallback(() => onSetActivePanel('docs'), [onSetActivePanel])
  const handleSetGit = useCallback(() => onSetActivePanel('git'), [onSetActivePanel])
  const handleSetDebug = useCallback(() => onSetActivePanel('debug'), [onSetActivePanel])

  const buttonClass = (panel: ToolPanel) =>
    `text-[10px] rounded border px-2 py-1 ${activePanel === panel ? 'border-bi-accent text-bi-accent' : 'border-white/10 text-gray-300'}`

  return (
    <div className={`${collapsed ? 'w-16' : 'w-64'} glass-panel flex flex-col transition-all duration-200`}>
      <div className="p-3 border-b border-white/10 flex items-center justify-between">
        {collapsed ? (
          <span className="text-xs font-medium">Tools</span>
        ) : (
          <span className="text-sm font-medium">Copilot</span>
        )}
        <div className="flex items-center gap-2">
          {!collapsed && <Settings className="w-4 h-4 text-gray-400" />}
          <button
            className="text-xs border border-white/10 rounded px-1 text-gray-300"
            onClick={onToggleCollapse}
            title={collapsed ? 'Expand tools panel' : 'Collapse tools panel'}
          >
            {collapsed ? '»' : '«'}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <div className="grid grid-cols-3 gap-1">
            <button className={buttonClass('diagnostics')} onClick={handleSetDiagnostics}>
              Diag
            </button>
            <button className={buttonClass('refactor')} onClick={handleSetRefactor}>
              Refactor
            </button>
            <button className={buttonClass('tests')} onClick={handleSetTests}>
              Tests
            </button>
            <button className={buttonClass('docs')} onClick={handleSetDocs}>
              Docs
            </button>
            <button className={buttonClass('git')} onClick={handleSetGit}>
              Git
            </button>
            <button className={buttonClass('debug')} onClick={handleSetDebug}>
              Debug
            </button>
          </div>

          {activePanel === 'diagnostics' && (
            <DiagnosticsPanel
              diagnostics={props.diagnostics}
              summary={props.diagnosticsSummary}
              busy={props.analyzeBusy}
              onAnalyze={props.onAnalyze}
            />
          )}

          {activePanel === 'refactor' && (
            <RefactorPanel
              suggestions={props.refactorSuggestions}
              summary={props.refactorSummary}
              busy={props.refactorBusy}
              onSuggest={props.onRefactor}
            />
          )}

          {activePanel === 'tests' && (
            <TestsPanel
              testPath={props.testPath}
              framework={props.testFramework}
              content={props.testContent}
              busy={props.testBusy}
              onGenerate={props.onGenerateTests}
            />
          )}

          {activePanel === 'docs' && (
            <DocsPanel
              symbol={props.docsSymbol}
              result={props.docsResult}
              cacheHit={props.docsCacheHit}
              cacheSize={props.docsCacheSize}
              cacheEvictions={props.docsCacheEvictions}
              busy={props.docsBusy}
              maxCacheEntries={props.maxCacheEntries}
              onSymbolChange={props.onDocsSymbolChange}
              onLookup={props.onDocsLookup}
              onRefresh={props.onDocsRefresh}
              onClearCache={props.onDocsClearCache}
              onReset={props.onDocsReset}
              onOpenLocation={props.onOpenDocsLocation}
            />
          )}

          {activePanel === 'git' && (
            <GitPanel
              branch={props.gitBranch}
              files={props.gitFiles}
              error={props.gitError}
              loading={props.gitLoading}
              diffPath={props.gitDiffPath}
              diffText={props.gitDiffText}
              diffLoading={props.gitDiffLoading}
              commitMessage={props.commitMessage}
              commitBusy={props.commitBusy}
              commitResult={props.commitResult}
              remote={props.gitRemote}
              branchInput={props.gitBranchInput}
              syncBusy={props.syncBusy}
              syncResult={props.syncResult}
              onRefresh={props.onGitRefresh}
              onLoadDiff={props.onGitLoadDiff}
              onSelectFile={props.onGitSelectFile}
              onCommitMessageChange={props.onCommitMessageChange}
              onCommit={props.onCommit}
              onRemoteChange={props.onGitRemoteChange}
              onBranchInputChange={props.onGitBranchInputChange}
              onPush={props.onPush}
              onPull={props.onPull}
            />
          )}

          {activePanel === 'debug' && (
            <DebugPanel
              sessionId={props.debugSessionId}
              busy={props.debugBusy}
              output={props.debugOutput}
              breakpointLine={props.debugBreakpointLine}
              onBreakpointLineChange={props.onDebugBreakpointLineChange}
              onStart={props.onDebugStart}
              onStop={props.onDebugStop}
              onSetBreakpoint={props.onDebugSetBreakpoint}
              onCommand={props.onDebugCommand}
            />
          )}
        </div>
      )}
    </div>
  )
})
