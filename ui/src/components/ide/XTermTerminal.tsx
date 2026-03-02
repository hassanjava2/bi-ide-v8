//! Real PTY Terminal using xterm.js
import { memo, useEffect, useRef, useCallback } from 'react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'
import { WebLinksAddon } from 'xterm-addon-web-links'
import 'xterm/css/xterm.css'

interface XTermTerminalProps {
  sessionId: string
  processId: number | null
  height: number
  onData: (data: string) => void
  onResize: (cols: number, rows: number) => void
}

export const XTermTerminal = memo(function XTermTerminal({
  sessionId,
  processId,
  height,
  onData,
  onResize
}: XTermTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const terminalRef = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  // Initialize terminal
  useEffect(() => {
    if (!containerRef.current || terminalRef.current) return

    const term = new Terminal({
      cursorBlink: true,
      cursorStyle: 'block',
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#1a1a2e',
        foreground: '#eaeaea',
        cursor: '#f39c12',
        selectionBackground: '#264f78',
        black: '#000000',
        red: '#cd3131',
        green: '#0dbc79',
        yellow: '#e5e510',
        blue: '#2472c8',
        magenta: '#bc3fbc',
        cyan: '#11a8cd',
        white: '#e5e5e5',
        brightBlack: '#666666',
        brightRed: '#f14c4c',
        brightGreen: '#23d18b',
        brightYellow: '#f5f543',
        brightBlue: '#3b8eea',
        brightMagenta: '#d670d6',
        brightCyan: '#29b8db',
        brightWhite: '#e5e5e5'
      },
      scrollback: 10000,
      allowProposedApi: true
    })

    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()

    term.loadAddon(fitAddon)
    term.loadAddon(webLinksAddon)

    term.open(containerRef.current)
    fitAddon.fit()

    // Handle input
    term.onData((data) => {
      onData(data)
    })

    // Handle resize
    term.onResize(({ cols, rows }) => {
      onResize(cols, rows)
    })

    terminalRef.current = term
    fitAddonRef.current = fitAddon

    // Initial resize
    const { cols, rows } = term
    onResize(cols, rows)

    // Resize observer for container
    resizeObserverRef.current = new ResizeObserver(() => {
      if (fitAddonRef.current) {
        fitAddonRef.current.fit()
      }
    })

    if (containerRef.current) {
      resizeObserverRef.current.observe(containerRef.current)
    }

    // Welcome message
    term.writeln('\r\n\x1b[1;32m╔═══════════════════════════════════════════════════════════╗\x1b[0m')
    term.writeln('\x1b[1;32m║\x1b[0m           \x1b[1;36mBI-IDE v8 Terminal\x1b[0m - \x1b[33mReady\x1b[0m              \x1b[1;32m║\x1b[0m')
    term.writeln('\x1b[1;32m╚═══════════════════════════════════════════════════════════╝\x1b[0m')
    term.writeln('')

    return () => {
      resizeObserverRef.current?.disconnect()
      term.dispose()
      terminalRef.current = null
    }
  }, [])

  // Handle session/process changes
  useEffect(() => {
    const term = terminalRef.current
    if (!term) return

    if (processId) {
      term.writeln(`\r\n\x1b[1;32m[Connected to process ${processId}]\x1b[0m\r\n`)
    } else if (sessionId) {
      term.writeln(`\r\n\x1b[1;33m[Session: ${sessionId}]\x1b[0m\r\n`)
    } else {
      term.writeln('\r\n\x1b[1;31m[Disconnected]\x1b[0m\r\n')
    }
  }, [sessionId, processId])

  // External API to write data
  const writeData = useCallback((data: string) => {
    terminalRef.current?.write(data)
  }, [])

  // Expose writeData to parent via ref
  useEffect(() => {
    if (containerRef.current) {
      (containerRef.current as any).writeData = writeData
    }
  }, [writeData])

  return (
    <div 
      ref={containerRef}
      className="w-full h-full bg-[#1a1a2e] p-2"
      style={{ height: `${height}px` }}
    />
  )
})

// Hook to use terminal output
export const useTerminalOutput = (
  containerRef: React.RefObject<HTMLDivElement>
) => {
  const writeOutput = useCallback((data: string) => {
    if (containerRef.current && (containerRef.current as any).writeData) {
      (containerRef.current as any).writeData(data)
    }
  }, [containerRef])

  return { writeOutput }
}
