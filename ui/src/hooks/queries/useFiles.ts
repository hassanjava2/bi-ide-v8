/**
 * useFiles Hook
 * 
 * Manages file operations for the IDE from /api/v1/ide/* endpoints
 * Features:
 * - File tree browsing
 * - File CRUD operations
 * - Git operations
 * - Terminal sessions
 * - Debug sessions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../services/api'

// Query keys
const FILES_KEYS = {
  all: ['files'] as const,
  tree: () => [...FILES_KEYS.all, 'tree'] as const,
  file: (id: string) => [...FILES_KEYS.all, 'file', id] as const,
  git: {
    all: ['git'] as const,
    status: () => [...FILES_KEYS.all, ...['git'], 'status'] as const,
    diff: (path?: string) => [...FILES_KEYS.all, ...['git'], 'diff', { path }] as const,
  },
  terminal: (sessionId?: string) => [...FILES_KEYS.all, 'terminal', { sessionId }] as const,
  debug: (sessionId?: string) => [...FILES_KEYS.all, 'debug', { sessionId }] as const,
}

// Types
interface FileTreeNode {
  id: string
  name: string
  type: 'file' | 'folder'
  path: string
  language?: string
  children?: FileTreeNode[]
}

interface FileContent {
  id: string
  name: string
  content: string
  language?: string
  path?: string
}

interface CodeAnalysisResult {
  issues: Array<{
    line: number
    column: number
    severity: 'error' | 'warning' | 'info'
    message: string
    rule: string
  }>
  summary: {
    errors: number
    warnings: number
    infos: number
    total: number
  }
}

interface RefactorSuggestion {
  rule: string
  message: string
  line: number
  severity: 'warning' | 'info'
}

interface RefactorResult {
  suggestions: RefactorSuggestion[]
  summary: {
    warnings: number
    infos: number
    total: number
  }
}

interface TestGenerationResult {
  test_path: string
  framework: string
  content: string
}

interface SymbolDocsResult {
  symbol?: string
  found?: boolean
  location?: string
  definition?: string
  documentation?: string
  related_symbols?: string[]
}

interface GitStatusFile {
  path: string
  status: string
  category: string
}

interface GitStatusResult {
  ok: boolean
  branch?: string
  files?: GitStatusFile[]
  error?: string
}

interface GitDiffResult {
  ok: boolean
  diff?: string
  error?: string
}

interface GitCommitResult {
  ok: boolean
  output?: string
  error?: string
}

interface GitSyncResult {
  ok: boolean
  output?: string
  error?: string
}

interface TerminalSession {
  session_id: string
  cwd: string
}

interface TerminalExecuteResult {
  stdout: string
  stderr: string
  exit_code: number
  elapsed_ms: number
  cwd?: string
}

interface DebugSession {
  ok: boolean
  session_id?: string
  output?: string
  error?: string
}

interface DebugCommandResult {
  ok: boolean
  output?: string
  exit_code?: number
  error?: string
}

interface SaveFileRequest {
  fileId: string
  content: string
}

interface AnalyzeCodeRequest {
  code: string
  language: string
  filePath: string
}

interface GitCommitRequest {
  message: string
  stageAll?: boolean
}

interface GitSyncRequest {
  remote?: string
  branch?: string
}

interface TerminalExecuteRequest {
  sessionId: string
  command: string
}

interface DebugStartRequest {
  filePath: string
  breakpoints?: number[]
}

interface DebugBreakpointRequest {
  sessionId: string
  filePath: string
  line: number
}

interface DebugCommandRequest {
  sessionId: string
  command: string
}

// Fetch functions
const fetchFileTree = async (): Promise<FileTreeNode> => {
  return api.getFileTree()
}

const fetchFile = async (fileId: string): Promise<FileContent> => {
  return api.getFile(fileId)
}

const fetchGitStatus = async (): Promise<GitStatusResult> => {
  return api.getGitStatus()
}

const fetchGitDiff = async (path?: string): Promise<GitDiffResult> => {
  return api.getGitDiff(path)
}

// Mutation functions
const saveFileContent = async ({ fileId, content }: SaveFileRequest): Promise<{ success: boolean }> => {
  return api.saveFile(fileId, content)
}

const analyzeCodeFn = async ({ code, language, filePath }: AnalyzeCodeRequest): Promise<CodeAnalysisResult> => {
  return api.analyzeCode(code, language, filePath)
}

const getRefactorSuggestionsFn = async ({ code, language, filePath }: AnalyzeCodeRequest): Promise<RefactorResult> => {
  return api.getRefactorSuggestions(code, language, filePath)
}

const generateTestsFn = async ({ code, language, filePath }: AnalyzeCodeRequest): Promise<TestGenerationResult> => {
  return api.generateTests(code, language, filePath)
}

const getSymbolDocsFn = async (params: AnalyzeCodeRequest & { symbol?: string }): Promise<SymbolDocsResult> => {
  return api.getSymbolDocumentation(params.code, params.language, params.filePath, params.symbol)
}

const createCommit = async ({ message, stageAll = true }: GitCommitRequest): Promise<GitCommitResult> => {
  return api.createGitCommit(message, stageAll)
}

const pushChanges = async ({ remote = 'origin', branch }: GitSyncRequest): Promise<GitSyncResult> => {
  return api.pushGitChanges(remote, branch)
}

const pullChanges = async ({ remote = 'origin', branch }: GitSyncRequest): Promise<GitSyncResult> => {
  return api.pullGitChanges(remote, branch)
}

const startTerminal = async (cwd?: string): Promise<TerminalSession> => {
  return api.startTerminalSession(cwd)
}

const executeTerminalCommand = async ({ sessionId, command }: TerminalExecuteRequest): Promise<TerminalExecuteResult> => {
  return api.executeTerminal(sessionId, command)
}

const startDebug = async ({ filePath, breakpoints = [] }: DebugStartRequest): Promise<DebugSession> => {
  return api.startDebugSession(filePath, breakpoints)
}

const stopDebug = async (sessionId: string): Promise<DebugSession> => {
  return api.stopDebugSession(sessionId)
}

const setBreakpoint = async ({ sessionId, filePath, line }: DebugBreakpointRequest): Promise<DebugSession> => {
  return api.setDebugBreakpoint(sessionId, filePath, line)
}

const executeDebugCmd = async ({ sessionId, command }: DebugCommandRequest): Promise<DebugCommandResult> => {
  return api.executeDebugCommand(sessionId, command)
}

/**
 * Hook for fetching the file tree
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useFileTree()
 * ```
 */
export function useFileTree(options: { enabled?: boolean } = {}) {
  return useQuery<FileTreeNode>({
    queryKey: FILES_KEYS.tree(),
    queryFn: fetchFileTree,
    staleTime: 1000 * 30,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for fetching a specific file
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useFile('file-123')
 * ```
 */
export function useFile(fileId: string, options: { enabled?: boolean } = {}) {
  return useQuery<FileContent>({
    queryKey: FILES_KEYS.file(fileId),
    queryFn: () => fetchFile(fileId),
    staleTime: 1000 * 10,
    enabled: options.enabled !== false && !!fileId,
  })
}

/**
 * Hook for saving file content
 * 
 * @example
 * ```tsx
 * const mutation = useSaveFile()
 * mutation.mutate({ fileId: '123', content: 'new content' })
 * ```
 */
export function useSaveFile() {
  const queryClient = useQueryClient()

  return useMutation<{ success: boolean }, Error, SaveFileRequest>({
    mutationFn: saveFileContent,
    onSuccess: (_data: { success: boolean }, variables: SaveFileRequest) => {
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.file(variables.fileId) })
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.tree() })
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.git.status() })
    },
  })
}

/**
 * Hook for analyzing code
 * 
 * @example
 * ```tsx
 * const mutation = useAnalyzeCode()
 * mutation.mutate({ code: '...', language: 'python', filePath: 'test.py' })
 * ```
 */
export function useAnalyzeCode() {
  return useMutation<CodeAnalysisResult, Error, AnalyzeCodeRequest>({
    mutationFn: analyzeCodeFn,
  })
}

/**
 * Hook for getting refactor suggestions
 * 
 * @example
 * ```tsx
 * const mutation = useRefactorSuggestions()
 * mutation.mutate({ code: '...', language: 'python', filePath: 'test.py' })
 * ```
 */
export function useRefactorSuggestions() {
  return useMutation<RefactorResult, Error, AnalyzeCodeRequest>({
    mutationFn: getRefactorSuggestionsFn,
  })
}

/**
 * Hook for generating tests
 * 
 * @example
 * ```tsx
 * const mutation = useGenerateTests()
 * mutation.mutate({ code: '...', language: 'python', filePath: 'test.py' })
 * ```
 */
export function useGenerateTests() {
  return useMutation<TestGenerationResult, Error, AnalyzeCodeRequest>({
    mutationFn: generateTestsFn,
  })
}

/**
 * Hook for getting symbol documentation
 * 
 * @example
 * ```tsx
 * const mutation = useSymbolDocumentation()
 * mutation.mutate({ code: '...', language: 'python', filePath: 'test.py', symbol: 'myFunction' })
 * ```
 */
export function useSymbolDocumentation() {
  return useMutation<SymbolDocsResult, Error, AnalyzeCodeRequest & { symbol?: string }>({
    mutationFn: getSymbolDocsFn,
  })
}

/**
 * Hook for fetching git status
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useGitStatus()
 * ```
 */
export function useGitStatus(options: { enabled?: boolean; poll?: boolean } = {}) {
  const { enabled = true, poll = false } = options

  return useQuery<GitStatusResult>({
    queryKey: FILES_KEYS.git.status(),
    queryFn: fetchGitStatus,
    staleTime: 1000 * 5,
    refetchInterval: poll ? 5000 : false,
    enabled,
  })
}

/**
 * Hook for fetching git diff
 * 
 * @example
 * ```tsx
 * const { data } = useGitDiff({ path: 'src/main.py' })
 * ```
 */
export function useGitDiff(path?: string, options: { enabled?: boolean } = {}) {
  return useQuery<GitDiffResult>({
    queryKey: FILES_KEYS.git.diff(path),
    queryFn: () => fetchGitDiff(path),
    staleTime: 1000 * 5,
    enabled: options.enabled !== false,
  })
}

/**
 * Hook for creating a git commit
 * 
 * @example
 * ```tsx
 * const mutation = useGitCommit()
 * mutation.mutate({ message: 'Fix bug', stageAll: true })
 * ```
 */
export function useGitCommit() {
  const queryClient = useQueryClient()

  return useMutation<GitCommitResult, Error, GitCommitRequest>({
    mutationFn: createCommit,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.git.status() })
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.git.diff() })
    },
  })
}

/**
 * Hook for pushing git changes
 * 
 * @example
 * ```tsx
 * const mutation = useGitPush()
 * mutation.mutate({ remote: 'origin', branch: 'main' })
 * ```
 */
export function useGitPush() {
  const queryClient = useQueryClient()

  return useMutation<GitSyncResult, Error, GitSyncRequest>({
    mutationFn: pushChanges,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.git.status() })
    },
  })
}

/**
 * Hook for pulling git changes
 * 
 * @example
 * ```tsx
 * const mutation = useGitPull()
 * mutation.mutate({ remote: 'origin', branch: 'main' })
 * ```
 */
export function useGitPull() {
  const queryClient = useQueryClient()

  return useMutation<GitSyncResult, Error, GitSyncRequest>({
    mutationFn: pullChanges,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.git.status() })
      queryClient.invalidateQueries({ queryKey: FILES_KEYS.tree() })
    },
  })
}

/**
 * Hook for starting a terminal session
 * 
 * @example
 * ```tsx
 * const mutation = useStartTerminal()
 * mutation.mutate('/workspace')
 * ```
 */
export function useStartTerminal() {
  return useMutation<TerminalSession, Error, string | undefined>({
    mutationFn: startTerminal,
  })
}

/**
 * Hook for executing terminal commands
 * 
 * @example
 * ```tsx
 * const mutation = useExecuteTerminal()
 * mutation.mutate({ sessionId: 'term-123', command: 'ls -la' })
 * ```
 */
export function useExecuteTerminal() {
  return useMutation<TerminalExecuteResult, Error, TerminalExecuteRequest>({
    mutationFn: executeTerminalCommand,
  })
}

/**
 * Hook for starting a debug session
 * 
 * @example
 * ```tsx
 * const mutation = useStartDebug()
 * mutation.mutate({ filePath: 'main.py', breakpoints: [10, 20] })
 * ```
 */
export function useStartDebug() {
  return useMutation<DebugSession, Error, DebugStartRequest>({
    mutationFn: startDebug,
  })
}

/**
 * Hook for stopping a debug session
 * 
 * @example
 * ```tsx
 * const mutation = useStopDebug()
 * mutation.mutate('debug-session-123')
 * ```
 */
export function useStopDebug() {
  return useMutation<DebugSession, Error, string>({
    mutationFn: stopDebug,
  })
}

/**
 * Hook for setting a debug breakpoint
 * 
 * @example
 * ```tsx
 * const mutation = useSetBreakpoint()
 * mutation.mutate({ sessionId: 'debug-123', filePath: 'main.py', line: 15 })
 * ```
 */
export function useSetBreakpoint() {
  return useMutation<DebugSession, Error, DebugBreakpointRequest>({
    mutationFn: setBreakpoint,
  })
}

/**
 * Hook for executing debug commands
 * 
 * @example
 * ```tsx
 * const mutation = useExecuteDebugCommand()
 * mutation.mutate({ sessionId: 'debug-123', command: 'continue' })
 * ```
 */
export function useExecuteDebugCommand() {
  return useMutation<DebugCommandResult, Error, DebugCommandRequest>({
    mutationFn: executeDebugCmd,
  })
}

export { FILES_KEYS }
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
}
export default useFileTree
