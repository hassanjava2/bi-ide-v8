import { 
  memo, 
  useMemo, 
  useCallback, 
  useEffect, 
  useRef, 
  useState, 
  type KeyboardEvent, 
  type MouseEvent 
} from 'react'
import { 
  Code2, Play, Folder, FileCode, Save, Terminal, Settings, 
  GitBranch, RefreshCw, Upload, Download, Bug 
} from 'lucide-react'
import { 
  analyzeCode, createGitCommit, executeDebugCommand, executeTerminal, 
  generateTests, getFile, getFileTree, getGitDiff, getGitStatus, 
  getRefactorSuggestions, getSymbolDocumentation, pullGitChanges, 
  pushGitChanges, saveFile, setDebugBreakpoint, startDebugSession, 
  startTerminalSession, stopDebugSession 
} from '../services/api'
import { useDebouncedDiagnostics } from '../hooks/useDebouncedDiagnostics'
import { useDebouncedRefactor } from '../hooks/useDebouncedRefactor'
import { useDebouncedLocalStorage } from '../hooks/useDebouncedSave'

type FileTreeNode = {
  id: string
  name: string
  type: 'file' | 'folder'
  path: string
  language?: string
  children?: FileTreeNode[]
}

type GitStatusFile = {
  path: string
  status: string
  category: string
}

type SymbolDocsResult = {
  symbol?: string
  found?: boolean
  location?: string
  definition?: string
  documentation?: string
  related_symbols?: string[]
}

type SymbolDocsCacheEntry = {
  result: SymbolDocsResult
  createdAt: number
}

const DOCS_CACHE_TTL_MS = 60_000
const DOCS_CACHE_MAX_ENTRIES = 100
const DOCS_CACHE_EVICTIONS_STORAGE_KEY = 'ide_docs_cache_evictions'
const DOCS_SYMBOL_STORAGE_KEY = 'ide_docs_symbol'

// Memoized File Tree Node component with custom comparison
interface FileTreeNodeProps {
  node: FileTreeNode
  depth: number
  selectedFileId: string
  onFileClick: (node: FileTreeNode) => void
}

const MemoizedFileTreeNode = memo(function FileTreeNodeComponent({ 
  node, 
  depth, 
  selectedFileId, 
  onFileClick 
}: FileTreeNodeProps) {
  const isFile = node.type === 'file'
  const isSelected = selectedFileId === node.id
  
  const handleClick = useCallback(() => {
    if (isFile) {
      onFileClick(node)
    }
  }, [isFile, node, onFileClick])

  const paddingStyle = useMemo(() => ({
    paddingRight: `${8 + depth * 14}px`
  }), [depth])

  return (
    <div>
      <div
        className={`flex items-center gap-2 p-2 hover:bg-white/5 rounded cursor-pointer ${
          isSelected ? 'bg-white/10' : ''
        }`}
        style={paddingStyle}
        onClick={handleClick}
      >
        {isFile ? (
          <FileCode className="w-4 h-4 text-blue-400" />
        ) : (
          <Folder className="w-4 h-4 text-yellow-400" />
        )}
        <span className="text-sm text-gray-300 truncate">{node.name}</span>
      </div>
      {!isFile && (node.children || []).map((child) => (
        <MemoizedFileTreeNode
          key={child.id}
          node={child}
          depth={depth + 1}
          selectedFileId={selectedFileId}
          onFileClick={onFileClick}
        />
      ))}
    </div>
  )
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if node id or selection changes
  return (
    prevProps.node.id === nextProps.node.id &&
    prevProps.selectedFileId === nextProps.selectedFileId &&
    prevProps.depth === nextProps.depth
  )
})

// Memoized Diagnostics Issue component
interface DiagnosticsIssueProps {
  issue: {
    line: number
    column: number
    severity: string
    message: string
    rule: string
  }
}

const DiagnosticsIssue = memo(function DiagnosticsIssue({ issue }: DiagnosticsIssueProps) {
  return (
    <div className="text-xs text-gray-300 border border-white/10 rounded p-2">
      <p className={issue.severity === 'error' ? 'text-red-400' : 'text-yellow-300'}>
        [{issue.severity}] {issue.rule}
      </p>
      <p>{issue.message}</p>
      <p className="text-gray-500">L{issue.line}:C{issue.column}</p>
    </div>
  )
}, (prevProps, nextProps) => {
  return (
    prevProps.issue.line === nextProps.issue.line &&
    prevProps.issue.message === nextProps.issue.message
  )
})

// Memoized Refactor Suggestion component
interface RefactorSuggestionProps {
  item: {
    rule: string
    message: string
    line: number
    severity: string
  }
}

const RefactorSuggestion = memo(function RefactorSuggestion({ item }: RefactorSuggestionProps) {
  return (
    <div className="text-xs text-gray-300 border border-white/10 rounded p-2">
      <p className={item.severity === 'warning' ? 'text-yellow-300' : 'text-blue-300'}>
        [{item.severity}] {item.rule}
      </p>
      <p>{item.message}</p>
      <p className="text-gray-500">L{item.line}</p>
    </div>
  )
}, (prevProps, nextProps) => {
  return (
    prevProps.item.line === nextProps.item.line &&
    prevProps.item.message === nextProps.item.message
  )
})

// Memoized Git File Item component
interface GitFileItemProps {
  item: GitStatusFile
  isSelected: boolean
  onClick: (path: string) => void
}

const GitFileItem = memo(function GitFileItem({ item, isSelected, onClick }: GitFileItemProps) {
  const handleClick = useCallback(() => {
    onClick(item.path)
  }, [item.path, onClick])

  return (
    <button
      className={`w-full text-left text-xs border rounded p-1 ${
        isSelected ? 'border-bi-accent text-bi-accent' : 'border-white/10 text-gray-300'
      }`}
      onClick={handleClick}
    >
      {item.status} {item.path}
    </button>
  )
}, (prevProps, nextProps) => {
  return (
    prevProps.item.path === nextProps.item.path &&
    prevProps.isSelected === nextProps.isSelected
  )
})

// Memoized Tool Panel Button component
interface ToolPanelButtonProps {
  panel: 'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug'
  label: string
  isActive: boolean
  onClick: (panel: 'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug') => void
}

const ToolPanelButton = memo(function ToolPanelButton({ 
  panel, 
  label, 
  isActive, 
  onClick 
}: ToolPanelButtonProps) {
  const handleClick = useCallback(() => {
    onClick(panel)
  }, [panel, onClick])

  return (
    <button
      className={`text-[10px] rounded border px-2 py-1 ${
        isActive ? 'border-bi-accent text-bi-accent' : 'border-white/10 text-gray-300'
      }`}
      onClick={handleClick}
    >
      {label}
    </button>
  )
}, (prevProps, nextProps) => {
  return prevProps.panel === nextProps.panel && prevProps.isActive === nextProps.isActive
})

// Memoized Terminal Line component
interface TerminalLineProps {
  line: string
  index: number
}

const TerminalLine = memo(function TerminalLine({ line }: TerminalLineProps) {
  return (
    <p className={line.startsWith('$ ') ? 'text-green-400' : 'text-gray-400'}>
      {line}
    </p>
  )
}, (prevProps, nextProps) => {
  return prevProps.line === nextProps.line
})

// Memoized IDE Header component
interface IDEHeaderProps {
  selectedFileName: string
  saveMessage: string
  onSave: () => void
}

const IDEHeader = memo(function IDEHeader({ selectedFileName, saveMessage, onSave }: IDEHeaderProps) {
  return (
    <div className="glass-panel p-4 mb-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-blue-800 flex items-center justify-center">
            <Code2 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">بيئة التطوير</h1>
            <p className="text-sm text-gray-400">مع مساعد AI مدمج</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="btn-secondary flex items-center gap-2" onClick={onSave}>
            <Save className="w-4 h-4" />
            حفظ
          </button>
          <button className="btn-primary flex items-center gap-2">
            <Play className="w-4 h-4" />
            تشغيل
          </button>
        </div>
      </div>
      {saveMessage && (
        <div className="mt-2 text-xs text-gray-400">{saveMessage}</div>
      )}
    </div>
  )
})

function IDE() {
  const editorRef = useRef<HTMLTextAreaElement | null>(null)
  const [treeRoot, setTreeRoot] = useState<FileTreeNode | null>(null)
  const [selectedFileId, setSelectedFileId] = useState<string>('')
  const [selectedFileName, setSelectedFileName] = useState<string>('')
  const [selectedFilePath, setSelectedFilePath] = useState<string>('')
  const [editorContent, setEditorContent] = useState<string>('')
  const [editorSelectionStart, setEditorSelectionStart] = useState<number>(0)
  const [editorSelectionEnd, setEditorSelectionEnd] = useState<number>(0)
  const [saveMessage, setSaveMessage] = useState<string>('')
  const [diagnostics, setDiagnostics] = useState<Array<{ line: number; column: number; severity: string; message: string; rule: string }>>([])
  const [diagnosticsSummary, setDiagnosticsSummary] = useState<{ errors: number; warnings: number; infos: number; total: number }>({ errors: 0, warnings: 0, infos: 0, total: 0 })
  const [analyzeBusy, setAnalyzeBusy] = useState<boolean>(false)
  const [refactorBusy, setRefactorBusy] = useState<boolean>(false)
  const [refactorSuggestions, setRefactorSuggestions] = useState<Array<{ rule: string; message: string; line: number; severity: string }>>([])
  const [refactorSummary, setRefactorSummary] = useState<{ warnings: number; infos: number; total: number }>({ warnings: 0, infos: 0, total: 0 })
  const [testBusy, setTestBusy] = useState<boolean>(false)
  const [generatedTestPath, setGeneratedTestPath] = useState<string>('')
  const [generatedTestFramework, setGeneratedTestFramework] = useState<string>('')
  const [generatedTestContent, setGeneratedTestContent] = useState<string>('')
  const [docsBusy, setDocsBusy] = useState<boolean>(false)
  const [docsSymbol, setDocsSymbol] = useState<string>(() => {
    try {
      return localStorage.getItem(DOCS_SYMBOL_STORAGE_KEY) || ''
    } catch {
      return ''
    }
  })
  const [docsResult, setDocsResult] = useState<SymbolDocsResult | null>(null)
  const [docsCacheHit, setDocsCacheHit] = useState<boolean>(false)
  const [docsCacheSize, setDocsCacheSize] = useState<number>(0)
  const [docsCacheEvictions, setDocsCacheEvictions] = useState<number>(() => {
    try {
      const stored = Number.parseInt(localStorage.getItem(DOCS_CACHE_EVICTIONS_STORAGE_KEY) || '0', 10)
      return Number.isFinite(stored) && stored >= 0 ? stored : 0
    } catch {
      return 0
    }
  })
  const docsCacheRef = useRef<Record<string, SymbolDocsCacheEntry>>({})
  const [activeToolPanel, setActiveToolPanel] = useState<'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug'>(() => {
    try {
      const stored = localStorage.getItem('ide_active_tool_panel')
      if (stored === 'diagnostics' || stored === 'refactor' || stored === 'tests' || stored === 'docs' || stored === 'git' || stored === 'debug') {
        return stored
      }
    } catch {
      // ignore localStorage access errors
    }
    return 'diagnostics'
  })
  const [toolsCollapsed, setToolsCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem('ide_tools_collapsed') === '1'
    } catch {
      return false
    }
  })
  const [gitBranch, setGitBranch] = useState<string>('')
  const [gitFiles, setGitFiles] = useState<GitStatusFile[]>([])
  const [gitError, setGitError] = useState<string>('')
  const [gitLoading, setGitLoading] = useState<boolean>(false)
  const [gitDiffPath, setGitDiffPath] = useState<string>('')
  const [gitDiffText, setGitDiffText] = useState<string>('')
  const [gitDiffLoading, setGitDiffLoading] = useState<boolean>(false)
  const [commitMessage, setCommitMessage] = useState<string>('')
  const [commitBusy, setCommitBusy] = useState<boolean>(false)
  const [commitResult, setCommitResult] = useState<string>('')
  const [gitRemote, setGitRemote] = useState<string>('origin')
  const [gitBranchInput, setGitBranchInput] = useState<string>('')
  const [syncBusy, setSyncBusy] = useState<boolean>(false)
  const [syncResult, setSyncResult] = useState<string>('')
  const [debugSessionId, setDebugSessionId] = useState<string>('')
  const [debugBusy, setDebugBusy] = useState<boolean>(false)
  const [debugOutput, setDebugOutput] = useState<string>('')
  const [debugBreakpointLine, setDebugBreakpointLine] = useState<string>('1')

  const [terminalSessionId, setTerminalSessionId] = useState<string>('')
  const [terminalCwd, setTerminalCwd] = useState<string>('')
  const [terminalInput, setTerminalInput] = useState<string>('')
  const [terminalLines, setTerminalLines] = useState<string[]>([
    'Connecting to terminal...'
  ])
  const [terminalBusy, setTerminalBusy] = useState<boolean>(false)
  const [terminalHeight, setTerminalHeight] = useState<number>(() => {
    try {
      const stored = Number.parseInt(localStorage.getItem('ide_terminal_height') || '192', 10)
      if (Number.isFinite(stored)) return Math.min(420, Math.max(140, stored))
    } catch {
      // ignore localStorage access errors
    }
    return 192
  })

  // Debounced hooks for expensive operations
  const { runDiagnostics: runDebouncedDiagnostics, cancel: cancelDiagnostics } = useDebouncedDiagnostics(
    setDiagnostics,
    setDiagnosticsSummary,
    setAnalyzeBusy,
    1500 // Aggressive 1500ms delay
  )

  const { runRefactor: runDebouncedRefactor, cancel: cancelRefactor } = useDebouncedRefactor(
    setRefactorSuggestions,
    setRefactorSummary,
    setRefactorBusy,
    2000 // Very aggressive 2000ms delay for expensive refactor operations
  )

  // Debounced localStorage saves (1000ms delay)
  const { save: saveActivePanel } = useDebouncedLocalStorage('ide_active_tool_panel', 1000)
  const { save: saveTerminalHeight } = useDebouncedLocalStorage('ide_terminal_height', 1000)
  const { save: saveToolsCollapsed } = useDebouncedLocalStorage('ide_tools_collapsed', 1000)
  const { save: saveDocsCacheEvictions } = useDebouncedLocalStorage(DOCS_CACHE_EVICTIONS_STORAGE_KEY, 1000)
  const { save: saveDocsSymbol } = useDebouncedLocalStorage(DOCS_SYMBOL_STORAGE_KEY, 1000)

  // Memoized utility functions
  const findFirstFile = useCallback((node: FileTreeNode | null): FileTreeNode | null => {
    if (!node) return null
    if (node.type === 'file') return node
    for (const child of node.children || []) {
      const found = findFirstFile(child)
      if (found) return found
    }
    return null
  }, [])

  const normalizeWorkspacePath = useCallback((value: string): string => {
    return value.replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '')
  }, [])

  const findFileNodeByPath = useCallback((node: FileTreeNode | null, targetPath: string): FileTreeNode | null => {
    if (!node) return null
    if (node.type === 'file' && normalizeWorkspacePath(node.path || node.name) === targetPath) {
      return node
    }
    for (const child of node.children || []) {
      const found = findFileNodeByPath(child, targetPath)
      if (found) return found
    }
    return null
  }, [normalizeWorkspacePath])

  const getLineStartOffset = useCallback((content: string, lineNumber: number): number => {
    if (lineNumber <= 1) return 0
    let currentLine = 1
    for (let index = 0; index < content.length; index += 1) {
      if (content[index] === '\n') {
        currentLine += 1
        if (currentLine === lineNumber) {
          return index + 1
        }
      }
    }
    return content.length
  }, [])

  const setEditorSelectionRange = useCallback((start: number, end: number) => {
    const safeStart = Math.max(0, start)
    const safeEnd = Math.max(0, end)
    setEditorSelectionStart(safeStart)
    setEditorSelectionEnd(safeEnd)

    const editor = editorRef.current
    if (!editor) return
    editor.focus()
    editor.setSelectionRange(safeStart, safeEnd)
  }, [])

  const parseDocsLocation = useCallback((location: string): { path: string; line?: number } => {
    const normalized = normalizeWorkspacePath(String(location || '').trim())
    const separatorIndex = normalized.lastIndexOf(':')
    if (separatorIndex > 0) {
      const linePart = normalized.slice(separatorIndex + 1)
      if (/^\d+$/.test(linePart)) {
        return { path: normalized.slice(0, separatorIndex), line: Number.parseInt(linePart, 10) }
      }
    }
    return { path: normalized }
  }, [normalizeWorkspacePath])

  // Memoized file operations
  const openFile = useCallback(async (node: FileTreeNode, targetLine?: number) => {
    if (node.type !== 'file') return
    try {
      const result = await getFile(node.id)
      setSelectedFileId(node.id)
      setSelectedFileName(node.name)
      setSelectedFilePath(node.path || node.name)
      const content = String(result.content || '')
      setEditorContent(content)
      if (targetLine && targetLine > 0) {
        const lineStart = getLineStartOffset(content, targetLine)
        const nextLineStart = getLineStartOffset(content, targetLine + 1)
        const lineEnd = nextLineStart > lineStart ? Math.max(lineStart, nextLineStart - 1) : lineStart
        setTimeout(() => {
          setEditorSelectionRange(lineStart, lineEnd)
          setTimeout(() => setEditorSelectionRange(lineStart, lineStart), 900)
        }, 0)
        setSaveMessage(`Opened ${node.path || node.name} at line ${targetLine}`)
      } else {
        setTimeout(() => setEditorSelectionRange(0, 0), 0)
        setSaveMessage('')
      }
      await runDiagnostics(content, node.path || node.name)
      await runRefactorSuggestions(content, node.path || node.name)
    } catch (error) {
      setSaveMessage(`Open failed: ${String(error)}`)
    }
  }, [getLineStartOffset, setEditorSelectionRange])

  const openDocsLocation = useCallback(async (location: string) => {
    const parsed = parseDocsLocation(location)
    if (!parsed.path) return

    try {
      let root = treeRoot
      if (!root) {
        root = await getFileTree()
        setTreeRoot(root)
      }
      const fileNode = findFileNodeByPath(root, parsed.path)
      if (!fileNode) {
        setSaveMessage(`Docs location not found: ${parsed.path}`)
        return
      }
      await openFile(fileNode, parsed.line)
    } catch (error) {
      setSaveMessage(`Open location failed: ${String(error)}`)
    }
  }, [treeRoot, findFileNodeByPath, openFile, parseDocsLocation])

  const detectLanguage = useCallback((pathOrName: string): string => {
    const lower = pathOrName.toLowerCase()
    if (lower.endsWith('.py')) return 'python'
    if (lower.endsWith('.ts') || lower.endsWith('.tsx')) return 'typescript'
    if (lower.endsWith('.js') || lower.endsWith('.jsx')) return 'javascript'
    if (lower.endsWith('.json')) return 'json'
    if (lower.endsWith('.md')) return 'markdown'
    if (lower.endsWith('.html')) return 'html'
    if (lower.endsWith('.css')) return 'css'
    return 'python'
  }, [])

  const getDocsCacheKey = useCallback((filePath: string, language: string, symbol: string): string => {
    return `${filePath}::${language}::${symbol.trim().toLowerCase()}`
  }, [])

  const enforceDocsCacheLimit = useCallback(() => {
    const entries = Object.entries(docsCacheRef.current)
    if (entries.length <= DOCS_CACHE_MAX_ENTRIES) {
      setDocsCacheSize(entries.length)
      return
    }

    const keysToRemove = entries
      .sort((a, b) => a[1].createdAt - b[1].createdAt)
      .slice(0, entries.length - DOCS_CACHE_MAX_ENTRIES)
      .map(([key]) => key)

    keysToRemove.forEach((key) => {
      delete docsCacheRef.current[key]
    })

    if (keysToRemove.length > 0) {
      setDocsCacheEvictions((previous) => previous + keysToRemove.length)
    }
    setDocsCacheSize(Object.keys(docsCacheRef.current).length)
  }, [])

  const inferSymbolFromEditorSelection = useCallback((code: string, start: number, end: number): string => {
    if (!code) return ''

    const safeStart = Math.max(0, Math.min(start, code.length))
    const safeEnd = Math.max(0, Math.min(end, code.length))
    const [left, right] = safeStart <= safeEnd ? [safeStart, safeEnd] : [safeEnd, safeStart]

    if (right > left) {
      const selectedText = code.slice(left, right).trim()
      const token = selectedText.match(/[A-Za-z_][A-Za-z0-9_]*/)?.[0] || ''
      if (token) return token
    }

    let wordStart = left
    let wordEnd = left
    while (wordStart > 0 && /[A-Za-z0-9_]/.test(code[wordStart - 1])) {
      wordStart -= 1
    }
    while (wordEnd < code.length && /[A-Za-z0-9_]/.test(code[wordEnd])) {
      wordEnd += 1
    }

    const token = code.slice(wordStart, wordEnd).trim()
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(token) ? token : ''
  }, [])

  const runDiagnostics = useCallback(async (contentArg?: string, filePathArg?: string) => {
    const code = contentArg ?? editorContent
    const filePath = filePathArg ?? selectedFilePath
    if (!filePath) return

    setAnalyzeBusy(true)
    try {
      const language = detectLanguage(filePath)
      const result = await analyzeCode(code, language, filePath)
      setDiagnostics(Array.isArray(result.issues) ? result.issues : [])
      setDiagnosticsSummary(result.summary || { errors: 0, warnings: 0, infos: 0, total: 0 })
    } catch (error) {
      setDiagnostics([{ line: 1, column: 1, severity: 'error', message: String(error), rule: 'analysis-failed' }])
      setDiagnosticsSummary({ errors: 1, warnings: 0, infos: 0, total: 1 })
    } finally {
      setAnalyzeBusy(false)
    }
  }, [editorContent, selectedFilePath, detectLanguage])

  const runRefactorSuggestions = useCallback(async (contentArg?: string, filePathArg?: string) => {
    const code = contentArg ?? editorContent
    const filePath = filePathArg ?? selectedFilePath
    if (!filePath) return

    setRefactorBusy(true)
    try {
      const language = detectLanguage(filePath)
      const result = await getRefactorSuggestions(code, language, filePath)
      setRefactorSuggestions(Array.isArray(result.suggestions) ? result.suggestions : [])
      setRefactorSummary(result.summary || { warnings: 0, infos: 0, total: 0 })
    } catch (error) {
      setRefactorSuggestions([{ rule: 'refactor-failed', message: String(error), line: 1, severity: 'warning' }])
      setRefactorSummary({ warnings: 1, infos: 0, total: 1 })
    } finally {
      setRefactorBusy(false)
    }
  }, [editorContent, selectedFilePath, detectLanguage])

  const runTestGeneration = useCallback(async () => {
    if (!selectedFilePath) return

    setTestBusy(true)
    try {
      const language = detectLanguage(selectedFilePath)
      const result = await generateTests(editorContent, language, selectedFilePath)
      setGeneratedTestPath(String(result.test_path || ''))
      setGeneratedTestFramework(String(result.framework || ''))
      setGeneratedTestContent(String(result.content || ''))
    } catch (error) {
      setGeneratedTestPath('')
      setGeneratedTestFramework('')
      setGeneratedTestContent(`Test generation failed: ${String(error)}`)
    } finally {
      setTestBusy(false)
    }
  }, [selectedFilePath, editorContent, detectLanguage])

  const runSymbolDocs = useCallback(async (symbolOverride?: string, forceRefresh = false) => {
    if (!selectedFilePath) return

    const explicitSymbol = (symbolOverride ?? docsSymbol).trim()
    const inferredSymbol = explicitSymbol || inferSymbolFromEditorSelection(editorContent, editorSelectionStart, editorSelectionEnd)
    if (!explicitSymbol && inferredSymbol) {
      setDocsSymbol(inferredSymbol)
    }

    const language = detectLanguage(selectedFilePath)
    const cacheKey = getDocsCacheKey(selectedFilePath, language, inferredSymbol || '')
    if (!forceRefresh) {
      const cached = docsCacheRef.current[cacheKey]
      if (cached) {
        const isFresh = Date.now() - cached.createdAt < DOCS_CACHE_TTL_MS
        if (isFresh) {
          cached.createdAt = Date.now()
          setDocsResult(cached.result)
          setDocsCacheHit(true)
          setDocsCacheSize(Object.keys(docsCacheRef.current).length)
          return
        }
        delete docsCacheRef.current[cacheKey]
        setDocsCacheSize(Object.keys(docsCacheRef.current).length)
      }
    }

    setDocsBusy(true)
    setDocsCacheHit(false)
    try {
      const result = await getSymbolDocumentation(editorContent, language, selectedFilePath, inferredSymbol || undefined)
      const normalizedResult = (result || null) as SymbolDocsResult | null
      setDocsResult(normalizedResult)
      if (normalizedResult) {
        docsCacheRef.current[cacheKey] = {
          result: normalizedResult,
          createdAt: Date.now()
        }
        enforceDocsCacheLimit()
      } else {
        setDocsCacheSize(Object.keys(docsCacheRef.current).length)
      }
    } catch (error) {
      setDocsResult({
        symbol: inferredSymbol || undefined,
        found: false,
        documentation: `Documentation lookup failed: ${String(error)}`,
        related_symbols: []
      })
      setDocsCacheHit(false)
    } finally {
      setDocsBusy(false)
    }
  }, [selectedFilePath, docsSymbol, editorContent, editorSelectionStart, editorSelectionEnd, detectLanguage, getDocsCacheKey, enforceDocsCacheLimit, inferSymbolFromEditorSelection])

  const clearDocsCache = useCallback(() => {
    docsCacheRef.current = {}
    setDocsCacheSize(0)
    setDocsCacheEvictions(0)
    setDocsCacheHit(false)
  }, [])

  const resetDocsUi = useCallback(() => {
    clearDocsCache()
    setDocsSymbol('')
    setDocsResult(null)
    setDocsBusy(false)
  }, [clearDocsCache])

  const onEditorMouseUp = useCallback((event: MouseEvent<HTMLTextAreaElement>) => {
    const start = event.currentTarget.selectionStart || 0
    const end = event.currentTarget.selectionEnd || 0
    setEditorSelectionStart(start)
    setEditorSelectionEnd(end)

    if (!event.ctrlKey && !event.metaKey) return
    if (!selectedFilePath || docsBusy) return

    const symbol = inferSymbolFromEditorSelection(editorContent, start, end)
    if (!symbol) return

    setActiveToolPanel('docs')
    setDocsSymbol(symbol)
    void runSymbolDocs(symbol)
  }, [selectedFilePath, docsBusy, editorContent, inferSymbolFromEditorSelection, runSymbolDocs])

  const refreshTree = useCallback(async () => {
    try {
      const root = await getFileTree()
      setTreeRoot(root)

      if (!selectedFileId) {
        const firstFile = findFirstFile(root)
        if (firstFile) {
          await openFile(firstFile)
        }
      }
    } catch (error) {
      setSaveMessage(`Tree load failed: ${String(error)}`)
    }
  }, [selectedFileId, findFirstFile, openFile])

  const refreshGitStatus = useCallback(async () => {
    setGitLoading(true)
    setGitError('')
    try {
      const result = await getGitStatus()
      if (!result.ok) {
        setGitBranch('')
        setGitFiles([])
        setGitError(String(result.error || 'Git status failed'))
        return
      }
      setGitBranch(String(result.branch || ''))
      setGitFiles(Array.isArray(result.files) ? result.files : [])
      if (!gitDiffPath && Array.isArray(result.files) && result.files.length > 0) {
        setGitDiffPath(String(result.files[0].path || ''))
      }
    } catch (error) {
      setGitBranch('')
      setGitFiles([])
      setGitError(String(error))
    } finally {
      setGitLoading(false)
    }
  }, [gitDiffPath])

  const loadGitDiff = useCallback(async (pathArg?: string) => {
    const targetPath = pathArg ?? gitDiffPath
    setGitDiffLoading(true)
    try {
      const result = await getGitDiff(targetPath || undefined)
      if (!result.ok) {
        setGitDiffText(`Diff failed: ${String(result.error || 'unknown error')}`)
        return
      }
      setGitDiffText(String(result.diff || ''))
    } catch (error) {
      setGitDiffText(`Diff failed: ${String(error)}`)
    } finally {
      setGitDiffLoading(false)
    }
  }, [gitDiffPath])

  const onCommit = useCallback(async () => {
    const message = commitMessage.trim()
    if (!message || commitBusy) return

    setCommitBusy(true)
    setCommitResult('')
    try {
      const result = await createGitCommit(message, true)
      if (result.ok) {
        setCommitMessage('')
        setCommitResult(result.output ? String(result.output) : 'Commit created')
      } else {
        setCommitResult(`Commit failed: ${String(result.error || 'unknown error')}`)
      }
      await refreshGitStatus()
      await loadGitDiff(gitDiffPath || undefined)
    } catch (error) {
      setCommitResult(`Commit failed: ${String(error)}`)
    } finally {
      setCommitBusy(false)
    }
  }, [commitMessage, commitBusy, refreshGitStatus, loadGitDiff, gitDiffPath])

  const onPush = useCallback(async () => {
    if (syncBusy) return
    setSyncBusy(true)
    setSyncResult('')
    try {
      const result = await pushGitChanges(gitRemote.trim() || 'origin', gitBranchInput.trim() || undefined)
      if (result.ok) {
        setSyncResult(result.output ? String(result.output) : 'Push completed')
      } else {
        setSyncResult(`Push failed: ${String(result.error || 'unknown error')}`)
      }
      await refreshGitStatus()
    } catch (error) {
      setSyncResult(`Push failed: ${String(error)}`)
    } finally {
      setSyncBusy(false)
    }
  }, [syncBusy, gitRemote, gitBranchInput, refreshGitStatus])

  const onPull = useCallback(async () => {
    if (syncBusy) return
    setSyncBusy(true)
    setSyncResult('')
    try {
      const result = await pullGitChanges(gitRemote.trim() || 'origin', gitBranchInput.trim() || undefined)
      if (result.ok) {
        setSyncResult(result.output ? String(result.output) : 'Pull completed')
      } else {
        setSyncResult(`Pull failed: ${String(result.error || 'unknown error')}`)
      }
      await refreshGitStatus()
      await loadGitDiff(gitDiffPath || undefined)
    } catch (error) {
      setSyncResult(`Pull failed: ${String(error)}`)
    } finally {
      setSyncBusy(false)
    }
  }, [syncBusy, gitRemote, gitBranchInput, refreshGitStatus, loadGitDiff, gitDiffPath])

  const toWorkspaceDebugPath = useCallback(() => {
    const value = selectedFilePath || selectedFileName
    if (!value) return ''
    return value.startsWith('/') ? value.slice(1) : value
  }, [selectedFilePath, selectedFileName])

  const onDebugStart = useCallback(async () => {
    if (debugBusy) return
    const filePath = toWorkspaceDebugPath()
    if (!filePath) {
      setDebugOutput('Select a file first')
      return
    }

    setDebugBusy(true)
    try {
      const line = Number.parseInt(debugBreakpointLine, 10)
      const breakpoints = Number.isFinite(line) && line > 0 ? [line] : []
      const result = await startDebugSession(filePath, breakpoints)
      if (!result.ok) {
        setDebugOutput(`Debug start failed: ${String(result.error || 'unknown error')}`)
        return
      }
      setDebugSessionId(String(result.session_id || ''))
      setDebugOutput(String(result.output || 'Debug session started'))
    } catch (error) {
      setDebugOutput(`Debug start failed: ${String(error)}`)
    } finally {
      setDebugBusy(false)
    }
  }, [debugBusy, toWorkspaceDebugPath, debugBreakpointLine])

  const onDebugStop = useCallback(async () => {
    if (!debugSessionId || debugBusy) return
    setDebugBusy(true)
    try {
      const result = await stopDebugSession(debugSessionId)
      setDebugOutput(String(result.output || 'Debug session stopped'))
      setDebugSessionId('')
    } catch (error) {
      setDebugOutput(`Debug stop failed: ${String(error)}`)
    } finally {
      setDebugBusy(false)
    }
  }, [debugSessionId, debugBusy])

  const onDebugBreakpoint = useCallback(async () => {
    if (!debugSessionId || debugBusy) return
    const filePath = toWorkspaceDebugPath()
    const line = Number.parseInt(debugBreakpointLine, 10)
    if (!filePath || !Number.isFinite(line) || line <= 0) {
      setDebugOutput('Invalid breakpoint input')
      return
    }

    setDebugBusy(true)
    try {
      const result = await setDebugBreakpoint(debugSessionId, filePath, line)
      if (!result.ok) {
        setDebugOutput(`Breakpoint failed: ${String(result.error || 'unknown error')}`)
        return
      }
      setDebugOutput(String(result.output || `Breakpoint set at line ${line}`))
    } catch (error) {
      setDebugOutput(`Breakpoint failed: ${String(error)}`)
    } finally {
      setDebugBusy(false)
    }
  }, [debugSessionId, debugBusy, toWorkspaceDebugPath, debugBreakpointLine])

  const onDebugCommand = useCallback(async (command: string) => {
    if (!debugSessionId || debugBusy) return
    setDebugBusy(true)
    try {
      const result = await executeDebugCommand(debugSessionId, command)
      if (!result.ok) {
        setDebugOutput(`Debug command failed: ${String(result.error || 'unknown error')}`)
        return
      }
      setDebugOutput(String(result.output || ''))
      if (result.exit_code && Number(result.exit_code) !== 0) {
        setDebugSessionId('')
      }
    } catch (error) {
      setDebugOutput(`Debug command failed: ${String(error)}`)
    } finally {
      setDebugBusy(false)
    }
  }, [debugSessionId, debugBusy])

  const onSave = useCallback(async () => {
    if (!selectedFileId) return
    try {
      const result = await saveFile(selectedFileId, editorContent)
      setSaveMessage(result?.success ? 'Saved' : 'Save failed')
      await runDiagnostics(editorContent, selectedFilePath)
      await runRefactorSuggestions(editorContent, selectedFilePath)
    } catch (error) {
      setSaveMessage(`Save failed: ${String(error)}`)
    }
  }, [selectedFileId, editorContent, selectedFilePath, runDiagnostics, runRefactorSuggestions])

  // Memoized event handlers
  const handleToolPanelClick = useCallback((panel: 'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug') => {
    setActiveToolPanel(panel)
  }, [])

  const handleGitFileClick = useCallback((path: string) => {
    setGitDiffPath(path)
    loadGitDiff(path)
  }, [loadGitDiff])

  // Effects
  useEffect(() => {
    refreshTree()

    let mounted = true
    const initSession = async () => {
      try {
        const data = await startTerminalSession()
        if (!mounted) return
        setTerminalSessionId(data.session_id || '')
        setTerminalCwd(data.cwd || '')
        setTerminalLines([
          `Terminal connected`,
          `Session: ${data.session_id || '-'}`,
          `CWD: ${data.cwd || '-'}`
        ])
      } catch (error) {
        if (!mounted) return
        setTerminalLines([
          'Terminal connection failed',
          String(error)
        ])
      }
    }
    initSession()
    refreshGitStatus()
    return () => {
      mounted = false
    }
  }, [])

  // Debounced diagnostics and refactor on editor content change
  // Increased from 700ms to 1500ms for diagnostics, 2000ms for refactor
  useEffect(() => {
    if (!selectedFilePath) return
    
    // Use debounced versions for typing
    runDebouncedDiagnostics(editorContent, selectedFilePath)
    runDebouncedRefactor(editorContent, selectedFilePath)
    
    return () => {
      // Cleanup handled by hooks internally
    }
  }, [editorContent, selectedFilePath, runDebouncedDiagnostics, runDebouncedRefactor])

  // Cleanup debounced operations when file changes
  useEffect(() => {
    return () => {
      cancelDiagnostics()
      cancelRefactor()
    }
  }, [selectedFileId, cancelDiagnostics, cancelRefactor])

  // Debounced localStorage saves
  useEffect(() => {
    saveActivePanel(activeToolPanel)
  }, [activeToolPanel, saveActivePanel])

  useEffect(() => {
    saveTerminalHeight(String(terminalHeight))
  }, [terminalHeight, saveTerminalHeight])

  useEffect(() => {
    saveToolsCollapsed(toolsCollapsed ? '1' : '0')
  }, [toolsCollapsed, saveToolsCollapsed])

  useEffect(() => {
    saveDocsCacheEvictions(String(docsCacheEvictions))
  }, [docsCacheEvictions, saveDocsCacheEvictions])

  useEffect(() => {
    saveDocsSymbol(docsSymbol)
  }, [docsSymbol, saveDocsSymbol])

  const runTerminalCommand = useCallback(async () => {
    const command = terminalInput.trim()
    if (!command || !terminalSessionId || terminalBusy) return

    setTerminalBusy(true)
    setTerminalLines(prev => [...prev, `$ ${command}`])

    try {
      const result = await executeTerminal(terminalSessionId, command)
      const out: string[] = []
      if (result.stdout) out.push(String(result.stdout))
      if (result.stderr) out.push(String(result.stderr))
      out.push(`[exit=${result.exit_code}] [${result.elapsed_ms}ms]`)
      if (result.cwd) {
        setTerminalCwd(String(result.cwd))
      }
      setTerminalLines(prev => [...prev, ...out])
    } catch (error) {
      setTerminalLines(prev => [...prev, `ERROR: ${String(error)}`])
    } finally {
      setTerminalBusy(false)
      setTerminalInput('')
    }
  }, [terminalInput, terminalSessionId, terminalBusy])

  const onTerminalKeyDown = useCallback((event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      runTerminalCommand()
    }
  }, [runTerminalCommand])

  const onEditorKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    const isDocsShortcut = (event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'd'
    if (!isDocsShortcut) return

    event.preventDefault()
    if (!selectedFilePath || docsBusy) return

    const start = event.currentTarget.selectionStart || 0
    const end = event.currentTarget.selectionEnd || 0
    setEditorSelectionStart(start)
    setEditorSelectionEnd(end)

    const symbol = inferSymbolFromEditorSelection(editorContent, start, end)
    if (symbol) {
      setDocsSymbol(symbol)
    }

    setActiveToolPanel('docs')
    void runSymbolDocs(symbol || undefined)
  }, [selectedFilePath, docsBusy, editorContent, inferSymbolFromEditorSelection, runSymbolDocs])

  const onEditorChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditorContent(event.target.value)
  }, [])

  const onEditorSelect = useCallback((event: React.SyntheticEvent<HTMLTextAreaElement>) => {
    setEditorSelectionStart(event.currentTarget.selectionStart || 0)
    setEditorSelectionEnd(event.currentTarget.selectionEnd || 0)
  }, [])

  const onEditorKeyUp = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    setEditorSelectionStart(event.currentTarget.selectionStart || 0)
    setEditorSelectionEnd(event.currentTarget.selectionEnd || 0)
  }, [])

  // Memoized lists
  const diagnosticsList = useMemo(() => {
    return diagnostics.slice(0, 20).map((issue, index) => (
      <DiagnosticsIssue key={`${issue.line}-${issue.rule}-${index}`} issue={issue} />
    ))
  }, [diagnostics])

  const refactorSuggestionsList = useMemo(() => {
    return refactorSuggestions.slice(0, 20).map((item, index) => (
      <RefactorSuggestion key={`${item.line}-${item.rule}-${index}`} item={item} />
    ))
  }, [refactorSuggestions])

  const gitFilesList = useMemo(() => {
    return gitFiles.slice(0, 20).map((item, index) => (
      <GitFileItem
        key={`${item.path}-${index}`}
        item={item}
        isSelected={gitDiffPath === item.path}
        onClick={handleGitFileClick}
      />
    ))
  }, [gitFiles, gitDiffPath, handleGitFileClick])

  const terminalLinesList = useMemo(() => {
    return terminalLines.map((line, index) => (
      <TerminalLine key={`${index}-${line.slice(0, 20)}`} line={line} index={index} />
    ))
  }, [terminalLines])

  // Memoized tool panel buttons
  const toolPanelButtons = useMemo(() => {
    const panels: Array<{ id: 'diagnostics' | 'refactor' | 'tests' | 'docs' | 'git' | 'debug'; label: string }> = [
      { id: 'diagnostics', label: 'Diag' },
      { id: 'refactor', label: 'Refactor' },
      { id: 'tests', label: 'Tests' },
      { id: 'docs', label: 'Docs' },
      { id: 'git', label: 'Git' },
      { id: 'debug', label: 'Debug' },
    ]

    return panels.map(panel => (
      <ToolPanelButton
        key={panel.id}
        panel={panel.id}
        label={panel.label}
        isActive={activeToolPanel === panel.id}
        onClick={handleToolPanelClick}
      />
    ))
  }, [activeToolPanel, handleToolPanelClick])

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col">
      <IDEHeader 
        selectedFileName={selectedFileName} 
        saveMessage={saveMessage}
        onSave={onSave}
      />

      <div className="flex-1 flex gap-4 min-h-0">
        {/* الملفات */}
        <div className="w-64 glass-panel flex flex-col">
          <div className="p-3 border-b border-white/10 flex items-center justify-between">
            <span className="text-sm font-medium">المستكشف</span>
            <Folder className="w-4 h-4 text-gray-400" />
          </div>
          <div className="flex-1 overflow-auto p-2 space-y-1">
            {treeRoot ? (
              <MemoizedFileTreeNode
                node={treeRoot}
                depth={0}
                selectedFileId={selectedFileId}
                onFileClick={openFile}
              />
            ) : (
              <div className="text-xs text-gray-400 p-2">Loading files...</div>
            )}
          </div>
        </div>

        {/* المحرر */}
        <div className="flex-1 glass-panel flex flex-col min-w-0">
          <div className="flex items-center gap-1 p-2 border-b border-white/10">
            <div className="px-4 py-2 bg-white/10 rounded-t-lg flex items-center gap-2">
              <FileCode className="w-4 h-4 text-blue-400" />
              <span className="text-sm">{selectedFileName || 'No file selected'}</span>
            </div>
            {saveMessage && <span className="text-xs text-gray-400 px-2">{saveMessage}</span>}
          </div>
          <div className="flex-1 p-4 font-mono text-sm overflow-auto">
            <textarea
              ref={editorRef}
              value={editorContent}
              onChange={onEditorChange}
              onKeyDown={onEditorKeyDown}
              onSelect={onEditorSelect}
              onMouseUp={onEditorMouseUp}
              onKeyUp={onEditorKeyUp}
              className="w-full h-full bg-transparent text-gray-300 outline-none resize-none"
              placeholder="Select a file from explorer..."
            />
          </div>
        </div>

        {/* الأدوات */}
        <div className={`${toolsCollapsed ? 'w-16' : 'w-64'} glass-panel flex flex-col transition-all duration-200`}>
          <div className="p-3 border-b border-white/10 flex items-center justify-between">
            {toolsCollapsed ? (
              <span className="text-xs font-medium">Tools</span>
            ) : (
              <span className="text-sm font-medium">Copilot</span>
            )}
            <div className="flex items-center gap-2">
              {!toolsCollapsed && <Settings className="w-4 h-4 text-gray-400" />}
              <button
                className="text-xs border border-white/10 rounded px-1 text-gray-300"
                onClick={() => setToolsCollapsed((prev) => !prev)}
                title={toolsCollapsed ? 'Expand tools panel' : 'Collapse tools panel'}
              >
                {toolsCollapsed ? '»' : '«'}
              </button>
            </div>
          </div>
          {!toolsCollapsed && <div className="flex-1 overflow-auto p-4 space-y-4">
            <div className="grid grid-cols-3 gap-1">
              {toolPanelButtons}
            </div>

            {activeToolPanel === 'diagnostics' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent">Diagnostics</p>
                <button
                  className="text-xs text-bi-accent hover:underline"
                  onClick={() => runDiagnostics()}
                  disabled={analyzeBusy || !selectedFilePath}
                >
                  {analyzeBusy ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>
              <p className="text-xs text-gray-400">
                Errors: {diagnosticsSummary.errors} | Warnings: {diagnosticsSummary.warnings} | Total: {diagnosticsSummary.total}
              </p>
              <div className="mt-2 space-y-2 max-h-44 overflow-auto">
                {diagnostics.length === 0 ? (
                  <p className="text-xs text-gray-500">No issues found</p>
                ) : diagnosticsList}
              </div>
            </div>}

            {activeToolPanel === 'refactor' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent">Refactor</p>
                <button
                  className="text-xs text-bi-accent hover:underline"
                  onClick={() => runRefactorSuggestions()}
                  disabled={refactorBusy || !selectedFilePath}
                >
                  {refactorBusy ? 'Scanning...' : 'Suggest'}
                </button>
              </div>
              <p className="text-xs text-gray-400">
                Warnings: {refactorSummary.warnings} | Infos: {refactorSummary.infos} | Total: {refactorSummary.total}
              </p>
              <div className="mt-2 space-y-2 max-h-44 overflow-auto">
                {refactorSuggestions.length === 0 ? (
                  <p className="text-xs text-gray-500">No refactor suggestions</p>
                ) : refactorSuggestionsList}
              </div>
            </div>}

            {activeToolPanel === 'tests' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent">Tests</p>
                <button
                  className="text-xs text-bi-accent hover:underline"
                  onClick={() => runTestGeneration()}
                  disabled={testBusy || !selectedFilePath}
                >
                  {testBusy ? 'Generating...' : 'Generate'}
                </button>
              </div>

              <p className="text-xs text-gray-400">Framework: {generatedTestFramework || '-'}</p>
              <p className="text-xs text-gray-400 mb-2">Path: {generatedTestPath || '-'}</p>

              <div className="max-h-44 overflow-auto rounded border border-white/10 bg-black/20 p-2">
                <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
                  {generatedTestContent || 'No generated tests yet'}
                </pre>
              </div>
            </div>}

            {activeToolPanel === 'docs' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent">Docs</p>
                <div className="flex items-center gap-2">
                  <button
                    className="text-xs text-bi-accent hover:underline"
                    onClick={() => runSymbolDocs()}
                    disabled={docsBusy || !selectedFilePath}
                  >
                    {docsBusy ? 'Searching...' : 'Lookup'}
                  </button>
                  <button
                    className="text-xs text-gray-400 hover:underline"
                    onClick={() => runSymbolDocs(undefined, true)}
                    disabled={docsBusy || !selectedFilePath}
                  >
                    Refresh
                  </button>
                  <button
                    className="text-xs text-gray-400 hover:underline"
                    onClick={clearDocsCache}
                    disabled={docsBusy || docsCacheSize === 0}
                  >
                    Clear Cache
                  </button>
                  <button
                    className="text-xs text-gray-400 hover:underline"
                    onClick={resetDocsUi}
                    disabled={!docsBusy && !docsResult && !docsSymbol && docsCacheSize === 0 && docsCacheEvictions === 0}
                  >
                    Reset UI
                  </button>
                </div>
              </div>

              <input
                type="text"
                value={docsSymbol}
                onChange={(event) => setDocsSymbol(event.target.value)}
                placeholder="Symbol name (optional)"
                className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
              />
              <p className="mt-1 text-[10px] text-gray-500">Shortcut: Ctrl+Shift+D or Ctrl+Click on symbol</p>
              <p className="mt-1 text-[10px] text-gray-500">Source: {docsCacheHit ? 'cache (ttl 60s, max 100)' : 'live'}</p>
              <p className="mt-1 text-[10px] text-gray-500">Cache: size {docsCacheSize}/{DOCS_CACHE_MAX_ENTRIES}, evictions {docsCacheEvictions}</p>

              <div className="mt-2 max-h-44 overflow-auto rounded border border-white/10 bg-black/20 p-2">
                {!docsResult ? (
                  <p className="text-xs text-gray-500">No documentation result yet</p>
                ) : (
                  <div className="space-y-1 text-[10px] text-gray-300">
                    <p className="text-bi-accent">Symbol: {docsResult.symbol || '-'}</p>
                    <p>Found: {docsResult.found ? 'yes' : 'no'}</p>
                    {docsResult.location && (
                      <button
                        className="text-left text-bi-accent hover:underline"
                        onClick={() => {
                          const location = String(docsResult.location || '').trim()
                          if (!location) return
                          void openDocsLocation(location)
                        }}
                      >
                        Location: {docsResult.location}
                      </button>
                    )}
                    {docsResult.definition && (
                      <pre className="whitespace-pre-wrap break-words text-gray-300 border border-white/10 rounded p-1 bg-black/30">{docsResult.definition}</pre>
                    )}
                    <pre className="whitespace-pre-wrap break-words text-gray-300">{docsResult.documentation || 'No documentation available'}</pre>
                    {!!docsResult.related_symbols?.length && (
                      <p className="text-gray-400">Related: {docsResult.related_symbols.join(', ')}</p>
                    )}
                  </div>
                )}
              </div>
            </div>}

            {activeToolPanel === 'git' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent flex items-center gap-1">
                  <GitBranch className="w-3 h-3" /> Git
                </p>
                <button
                  className="text-xs text-bi-accent hover:underline inline-flex items-center gap-1"
                  onClick={() => refreshGitStatus()}
                  disabled={gitLoading}
                >
                  <RefreshCw className="w-3 h-3" /> {gitLoading ? '...' : 'Refresh'}
                </button>
              </div>

              {gitError ? (
                <p className="text-xs text-red-400">{gitError}</p>
              ) : (
                <>
                  <p className="text-xs text-gray-400 mb-2">Branch: {gitBranch || '-'}</p>
                  <div className="space-y-1 max-h-28 overflow-auto mb-2">
                    {gitFiles.length === 0 ? (
                      <p className="text-xs text-gray-500">Working tree clean</p>
                    ) : gitFilesList}
                  </div>

                  <button
                    className="text-xs text-bi-accent hover:underline mb-2"
                    onClick={() => loadGitDiff(gitDiffPath || undefined)}
                    disabled={gitDiffLoading}
                  >
                    {gitDiffLoading ? 'Loading diff...' : 'Load diff'}
                  </button>

                  <div className="max-h-28 overflow-auto rounded border border-white/10 bg-black/20 p-2">
                    <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
                      {gitDiffText || 'No diff loaded'}
                    </pre>
                  </div>

                  <div className="mt-2 space-y-1">
                    <input
                      type="text"
                      value={commitMessage}
                      onChange={(event) => setCommitMessage(event.target.value)}
                      placeholder="Commit message"
                      className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
                    />
                    <button
                      className="w-full text-xs bg-bi-accent/20 border border-bi-accent/40 rounded py-1 text-bi-accent disabled:opacity-50"
                      onClick={onCommit}
                      disabled={commitBusy || !commitMessage.trim()}
                    >
                      {commitBusy ? 'Committing...' : 'Commit (stage all)'}
                    </button>
                    {commitResult && <p className="text-[10px] text-gray-400 whitespace-pre-wrap">{commitResult}</p>}
                  </div>

                  <div className="mt-2 space-y-1 border-t border-white/10 pt-2">
                    <input
                      type="text"
                      value={gitRemote}
                      onChange={(event) => setGitRemote(event.target.value)}
                      placeholder="Remote (origin)"
                      className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
                    />
                    <input
                      type="text"
                      value={gitBranchInput}
                      onChange={(event) => setGitBranchInput(event.target.value)}
                      placeholder="Branch (optional)"
                      className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
                    />
                    <div className="grid grid-cols-2 gap-1">
                      <button
                        className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50 inline-flex items-center justify-center gap-1"
                        onClick={onPull}
                        disabled={syncBusy}
                      >
                        <Download className="w-3 h-3" /> {syncBusy ? '...' : 'Pull'}
                      </button>
                      <button
                        className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50 inline-flex items-center justify-center gap-1"
                        onClick={onPush}
                        disabled={syncBusy}
                      >
                        <Upload className="w-3 h-3" /> {syncBusy ? '...' : 'Push'}
                      </button>
                    </div>
                    {syncResult && <p className="text-[10px] text-gray-400 whitespace-pre-wrap">{syncResult}</p>}
                  </div>
                </>
              )}
            </div>}

            {activeToolPanel === 'debug' && <div className="glass-card p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <p className="text-xs text-bi-accent flex items-center gap-1">
                  <Bug className="w-3 h-3" /> Debug
                </p>
                <span className="text-[10px] text-gray-500">{debugSessionId ? `session ${debugSessionId.slice(0, 8)}` : 'idle'}</span>
              </div>

              <div className="space-y-1">
                <input
                  type="text"
                  value={debugBreakpointLine}
                  onChange={(event) => setDebugBreakpointLine(event.target.value)}
                  placeholder="Breakpoint line"
                  className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
                />

                <div className="grid grid-cols-2 gap-1">
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={onDebugStart}
                    disabled={debugBusy || !!debugSessionId}
                  >
                    Start
                  </button>
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={onDebugStop}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Stop
                  </button>
                </div>

                <button
                  className="w-full text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                  onClick={onDebugBreakpoint}
                  disabled={debugBusy || !debugSessionId}
                >
                  Set Breakpoint
                </button>

                <div className="grid grid-cols-3 gap-1">
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={() => onDebugCommand('continue')}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Continue
                  </button>
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={() => onDebugCommand('step')}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Step
                  </button>
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={() => onDebugCommand('next')}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Next
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-1">
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={() => onDebugCommand('stack')}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Stack
                  </button>
                  <button
                    className="text-xs border border-white/10 rounded py-1 text-gray-200 disabled:opacity-50"
                    onClick={() => onDebugCommand('locals')}
                    disabled={debugBusy || !debugSessionId}
                  >
                    Locals
                  </button>
                </div>

                <div className="max-h-28 overflow-auto rounded border border-white/10 bg-black/20 p-2">
                  <pre className="text-[10px] text-gray-300 whitespace-pre-wrap break-words">
                    {debugOutput || 'Debug output will appear here'}
                  </pre>
                </div>
              </div>
            </div>}
          </div>}

          {/* Terminal */}
          <div className="border-t border-white/10 flex flex-col" style={{ height: `${terminalHeight}px` }}>
            <div className="p-2 border-b border-white/10 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-400" />
              <span className="text-xs">Terminal {terminalSessionId ? '(live)' : '(offline)'}</span>
              <div className="ml-auto flex items-center gap-1">
                <button
                  className="text-[10px] border border-white/10 rounded px-1 text-gray-300"
                  onClick={() => setTerminalHeight((prev) => Math.max(140, prev - 24))}
                  title="Smaller"
                >
                  -
                </button>
                <button
                  className="text-[10px] border border-white/10 rounded px-1 text-gray-300"
                  onClick={() => setTerminalHeight((prev) => Math.min(420, prev + 24))}
                  title="Larger"
                >
                  +
                </button>
              </div>
            </div>
            <div className="flex-1 p-3 font-mono text-xs overflow-auto">
              {terminalCwd && <p className="text-gray-500 mb-2">{terminalCwd}</p>}
              {terminalLinesList}
            </div>
            <div className="p-2 border-t border-white/10">
              <input
                type="text"
                value={terminalInput}
                onChange={(event) => setTerminalInput(event.target.value)}
                onKeyDown={onTerminalKeyDown}
                placeholder={terminalSessionId ? 'type command and press Enter' : 'waiting session...'}
                disabled={!terminalSessionId || terminalBusy}
                className="w-full rounded bg-black/30 border border-white/10 px-2 py-1 text-xs text-gray-200"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default memo(IDE)
