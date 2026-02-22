import { GitBranch, RefreshCw, Download, Upload } from 'lucide-react'
import { memo, useCallback } from 'react'
import type { GitStatusFile } from './types'

interface GitPanelProps {
  branch: string
  files: GitStatusFile[]
  error: string
  loading: boolean
  diffPath: string
  diffText: string
  diffLoading: boolean
  commitMessage: string
  commitBusy: boolean
  commitResult: string
  remote: string
  branchInput: string
  syncBusy: boolean
  syncResult: string
  onRefresh: () => void
  onLoadDiff: (path?: string) => void
  onSelectFile: (path: string) => void
  onCommitMessageChange: (message: string) => void
  onCommit: () => void
  onRemoteChange: (remote: string) => void
  onBranchInputChange: (branch: string) => void
  onPush: () => void
  onPull: () => void
}

export const GitPanel = memo(function GitPanel({
  branch,
  files,
  error,
  loading,
  diffPath,
  diffText,
  diffLoading,
  commitMessage,
  commitBusy,
  commitResult,
  remote,
  branchInput,
  syncBusy,
  syncResult,
  onRefresh,
  onLoadDiff,
  onSelectFile,
  onCommitMessageChange,
  onCommit,
  onRemoteChange,
  onBranchInputChange,
  onPush,
  onPull
}: GitPanelProps) {
  const handleCommitMessageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onCommitMessageChange(event.target.value)
  }, [onCommitMessageChange])

  const handleRemoteChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onRemoteChange(event.target.value)
  }, [onRemoteChange])

  const handleBranchInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onBranchInputChange(event.target.value)
  }, [onBranchInputChange])

  const handleSelectFile = useCallback((path: string) => {
    onSelectFile(path)
    onLoadDiff(path)
  }, [onSelectFile, onLoadDiff])

  const handleRefresh = useCallback(() => {
    onRefresh()
  }, [onRefresh])

  const handleCommit = useCallback(() => {
    onCommit()
  }, [onCommit])

  const handleLoadDiff = useCallback(() => {
    onLoadDiff(diffPath || undefined)
  }, [onLoadDiff, diffPath])

  const handlePush = useCallback(() => {
    onPush()
  }, [onPush])

  const handlePull = useCallback(() => {
    onPull()
  }, [onPull])

  if (error) {
    return (
      <div className="glass-card p-3">
        <div className="flex items-center justify-between gap-2 mb-2">
          <p className="text-xs text-bi-accent flex items-center gap-1">
            <GitBranch className="w-3 h-3" /> Git
          </p>
          <button
            className="text-xs text-bi-accent hover:underline inline-flex items-center gap-1"
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw className="w-3 h-3" /> {loading ? '...' : 'Refresh'}
          </button>
        </div>
        <p className="text-xs text-red-400">{error}</p>
      </div>
    )
  }

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent flex items-center gap-1">
          <GitBranch className="w-3 h-3" /> Git
        </p>
        <button
          className="text-xs text-bi-accent hover:underline inline-flex items-center gap-1"
          onClick={handleRefresh}
          disabled={loading}
        >
          <RefreshCw className="w-3 h-3" /> {loading ? '...' : 'Refresh'}
        </button>
      </div>

      <p className="text-xs text-gray-400 mb-2">Branch: {branch || '-'}</p>
      
      <div className="space-y-1 max-h-28 overflow-auto mb-2">
        {files.length === 0 ? (
          <p className="text-xs text-gray-500">Working tree clean</p>
        ) : (
          files.slice(0, 20).map((item, index) => (
            <button
              key={`${item.path}-${index}`}
              className={`w-full text-left text-xs border rounded p-1 ${diffPath === item.path ? 'border-bi-accent text-bi-accent' : 'border-white/10 text-gray-300'}`}
              onClick={() => handleSelectFile(item.path)}
            >
              {item.status} {item.path}
            </button>
          ))
        )}
      </div>

      <button
        className="text-xs text-bi-accent hover:underline mb-2"
        onClick={handleLoadDiff}
        disabled={diffLoading}
      >
        {diffLoading ? 'Loading diff...' : 'Load diff'}
      </button>

      <div className="max-h-28 overflow-auto rounded border border-white/10 bg-black/20 p-2">
        <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
          {diffText || 'No diff loaded'}
        </pre>
      </div>

      <div className="mt-2 space-y-1">
        <input
          type="text"
          value={commitMessage}
          onChange={handleCommitMessageChange}
          placeholder="Commit message"
          className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
        />
        <button
          className="w-full text-xs bg-bi-accent/20 border border-bi-accent/40 rounded py-1 text-bi-accent disabled:opacity-50"
          onClick={handleCommit}
          disabled={commitBusy || !commitMessage.trim()}
        >
          {commitBusy ? 'Committing...' : 'Commit (stage all)'}
        </button>
        {commitResult && <p className="text-[10px] text-gray-400 whitespace-pre-wrap">{commitResult}</p>}
      </div>

      <div className="mt-2 space-y-1 border-t border-white/10 pt-2">
        <input
          type="text"
          value={remote}
          onChange={handleRemoteChange}
          placeholder="Remote (origin)"
          className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
        />
        <input
          type="text"
          value={branchInput}
          onChange={handleBranchInputChange}
          placeholder="Branch (optional)"
          className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
        />
        <div className="grid grid-cols-2 gap-1">
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50 inline-flex items-center justify-center gap-1"
            onClick={handlePull}
            disabled={syncBusy}
          >
            <Download className="w-3 h-3" /> {syncBusy ? '...' : 'Pull'}
          </button>
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50 inline-flex items-center justify-center gap-1"
            onClick={handlePush}
            disabled={syncBusy}
          >
            <Upload className="w-3 h-3" /> {syncBusy ? '...' : 'Push'}
          </button>
        </div>
        {syncResult && <p className="text-[10px] text-gray-400 whitespace-pre-wrap">{syncResult}</p>}
      </div>
    </div>
  )
})
