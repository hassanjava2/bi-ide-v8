import { Terminal } from 'lucide-react'
import { memo, useCallback, useRef, useEffect, type KeyboardEvent } from 'react'

interface TerminalPanelProps {
  sessionId: string
  cwd: string
  input: string
  lines: string[]
  height: number
  busy: boolean
  onInputChange: (value: string) => void
  onExecute: () => void
  onResize: (delta: number) => void
}

export const TerminalPanel = memo(function TerminalPanel({
  sessionId,
  cwd,
  input,
  lines,
  height,
  busy,
  onInputChange,
  onExecute,
  onResize
}: TerminalPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when lines change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines])

  const handleKeyDown = useCallback((event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !busy) {
      onExecute()
    }
  }, [busy, onExecute])

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onInputChange(event.target.value)
  }, [onInputChange])

  const handleShrink = useCallback(() => {
    onResize(-24)
  }, [onResize])

  const handleGrow = useCallback(() => {
    onResize(24)
  }, [onResize])

  return (
    <div className="border-t border-white/10 flex flex-col" style={{ height: `${height}px` }}>
      <div className="p-2 border-b border-white/10 flex items-center gap-2">
        <Terminal className="w-4 h-4 text-gray-400" />
        <span className="text-xs">Terminal {sessionId ? '(live)' : '(offline)'}</span>
        <div className="ml-auto flex items-center gap-1">
          <button
            className="text-[10px] border border-white/10 rounded px-1 text-gray-300"
            onClick={handleShrink}
            title="Smaller"
          >
            -
          </button>
          <button
            className="text-[10px] border border-white/10 rounded px-1 text-gray-300"
            onClick={handleGrow}
            title="Larger"
          >
            +
          </button>
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 p-3 font-mono text-xs overflow-auto">
        {cwd && <p className="text-gray-500 mb-2">{cwd}</p>}
        {lines.map((line, index) => (
          <p key={index} className={line.startsWith('$ ') ? 'text-green-400' : 'text-gray-400'}>
            {line}
          </p>
        ))}
      </div>
      <div className="p-2 border-t border-white/10">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={sessionId ? 'type command and press Enter' : 'waiting session...'}
          disabled={!sessionId || busy}
          className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
        />
      </div>
    </div>
  )
})
