import { Bug } from 'lucide-react'
import { memo, useCallback } from 'react'

interface DebugPanelProps {
  sessionId: string
  busy: boolean
  output: string
  breakpointLine: string
  onBreakpointLineChange: (line: string) => void
  onStart: () => void
  onStop: () => void
  onSetBreakpoint: () => void
  onCommand: (command: string) => void
}

export const DebugPanel = memo(function DebugPanel({
  sessionId,
  busy,
  output,
  breakpointLine,
  onBreakpointLineChange,
  onStart,
  onStop,
  onSetBreakpoint,
  onCommand
}: DebugPanelProps) {
  const handleBreakpointLineChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onBreakpointLineChange(event.target.value)
  }, [onBreakpointLineChange])

  const handleContinue = useCallback(() => {
    onCommand('continue')
  }, [onCommand])

  const handleStep = useCallback(() => {
    onCommand('step')
  }, [onCommand])

  const handleNext = useCallback(() => {
    onCommand('next')
  }, [onCommand])

  const handleStack = useCallback(() => {
    onCommand('stack')
  }, [onCommand])

  const handleLocals = useCallback(() => {
    onCommand('locals')
  }, [onCommand])

  const hasSession = !!sessionId

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-xs text-bi-accent flex items-center gap-1">
          <Bug className="w-3 h-3" /> Debug
        </p>
        <span className="text-[10px] text-gray-500">
          {sessionId ? `session ${sessionId.slice(0, 8)}` : 'idle'}
        </span>
      </div>

      <div className="space-y-1">
        <input
          type="text"
          value={breakpointLine}
          onChange={handleBreakpointLineChange}
          placeholder="Breakpoint line"
          className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
        />

        <div className="grid grid-cols-2 gap-1">
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={onStart}
            disabled={busy || hasSession}
          >
            Start
          </button>
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={onStop}
            disabled={busy || !hasSession}
          >
            Stop
          </button>
        </div>

        <button
          className="w-full text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
          onClick={onSetBreakpoint}
          disabled={busy || !hasSession}
        >
          Set Breakpoint
        </button>

        <div className="grid grid-cols-3 gap-1">
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={handleContinue}
            disabled={busy || !hasSession}
          >
            Continue
          </button>
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={handleStep}
            disabled={busy || !hasSession}
          >
            Step
          </button>
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={handleNext}
            disabled={busy || !hasSession}
          >
            Next
          </button>
        </div>

        <div className="grid grid-cols-2 gap-1">
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={handleStack}
            disabled={busy || !hasSession}
          >
            Stack
          </button>
          <button
            className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
            onClick={handleLocals}
            disabled={busy || !hasSession}
          >
            Locals
          </button>
        </div>

        <div className="max-h-28 overflow-auto rounded border border-white/10 bg-black/20 p-2">
          <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
            {output || 'Debug output will appear here'}
          </pre>
        </div>
      </div>
    </div>
  )
})
