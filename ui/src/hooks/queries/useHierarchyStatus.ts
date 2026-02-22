/**
 * useHierarchyStatus Hook
 * 
 * Fetches and caches hierarchy status from /api/v1/hierarchy/status
 * Features:
 * - Layer status tracking
 * - Metrics aggregation
 * - Command execution mutations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../services/api'
import type { AlertLevel } from '../../types'

// Query keys
const HIERARCHY_KEYS = {
  all: ['hierarchy'] as const,
  status: () => [...HIERARCHY_KEYS.all, 'status'] as const,
  metrics: () => [...HIERARCHY_KEYS.all, 'metrics'] as const,
}

// Types
interface LayerStatus {
  name: string
  active: boolean
  health: number
  last_activity?: string
}

interface HierarchyStatus {
  status: string
  layers: Record<string, LayerStatus>
  president: {
    in_meeting: boolean
    veto_power: boolean
  }
  council: {
    is_meeting: boolean
    wise_men_count: number
    meeting_status: string
    president_present: boolean
  }
  scouts: {
    intel_buffer_size: number
    high_priority_queue: number
  }
  meta: {
    performance_score: number
    quality_score: number
    evolution_stage: number
    learning_progress: number
    status: string
  }
  experts: {
    total: number
    domains: string[]
  }
  execution: {
    active_forces: number
    active_sprints: number
    active_crises: number
    quality_score: number
  }
}

interface HierarchyMetrics {
  layers: Record<string, number>
  [key: string]: any
}

interface ExecuteCommandRequest {
  command: string
  alertLevel: AlertLevel
  context?: Record<string, any>
}

interface ExecuteCommandResponse {
  success: boolean
  result?: string
  error?: string
}

// Fetch functions
const fetchHierarchyStatus = async (): Promise<HierarchyStatus> => {
  return api.getHierarchyStatus()
}

const fetchHierarchyMetrics = async (): Promise<HierarchyMetrics> => {
  return api.getHierarchyMetrics()
}

const executeHierarchyCommand = async (data: ExecuteCommandRequest): Promise<ExecuteCommandResponse> => {
  return api.executeCommand(data.command, data.alertLevel, data.context)
}

interface UseHierarchyStatusOptions {
  /** Enable automatic polling (default: true) */
  poll?: boolean
  /** Polling interval in milliseconds (default: 5000) */
  pollInterval?: number
  enabled?: boolean
}

/**
 * Hook for fetching hierarchy status with caching
 * 
 * @example
 * ```tsx
 * const { data, isLoading, error } = useHierarchyStatus()
 * 
 * // Access specific layer
 * const councilLayer = data?.layers?.council
 * ```
 */
export function useHierarchyStatus(options: UseHierarchyStatusOptions = {}) {
  const { poll = true, pollInterval = 5000, enabled = true } = options

  return useQuery<HierarchyStatus>({
    queryKey: HIERARCHY_KEYS.status(),
    queryFn: fetchHierarchyStatus,
    staleTime: 1000 * 5,
    refetchInterval: poll ? pollInterval : false,
    refetchIntervalInBackground: true,
    enabled,
  })
}

/**
 * Hook for fetching hierarchy metrics
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useHierarchyMetrics()
 * ```
 */
export function useHierarchyMetrics(options: { enabled?: boolean } = {}) {
  return useQuery<HierarchyMetrics>({
    queryKey: HIERARCHY_KEYS.metrics(),
    queryFn: fetchHierarchyMetrics,
    staleTime: 1000 * 3,
    refetchInterval: 3000,
    refetchIntervalInBackground: true,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for executing commands through the hierarchy
 * 
 * @example
 * ```tsx
 * const mutation = useExecuteCommand()
 * 
 * mutation.mutate({
 *   command: 'analyze market trends',
 *   alertLevel: 'YELLOW'
 * })
 * ```
 */
export function useExecuteCommand() {
  const queryClient = useQueryClient()

  return useMutation<ExecuteCommandResponse, Error, ExecuteCommandRequest>({
    mutationFn: executeHierarchyCommand,
    onSuccess: () => {
      // Invalidate related queries after command execution
      queryClient.invalidateQueries({ queryKey: HIERARCHY_KEYS.status() })
      queryClient.invalidateQueries({ queryKey: HIERARCHY_KEYS.metrics() })
    },
  })
}

/**
 * Hook for refreshing hierarchy data
 * 
 * @example
 * ```tsx
 * const { refreshAll, isRefreshing } = useRefreshHierarchy()
 * 
 * <button onClick={refreshAll} disabled={isRefreshing}>
 *   Refresh
 * </button>
 * ```
 */
export function useRefreshHierarchy() {
  const queryClient = useQueryClient()

  const refreshAll = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: HIERARCHY_KEYS.status() }),
      queryClient.invalidateQueries({ queryKey: HIERARCHY_KEYS.metrics() }),
    ])
  }

  const isRefreshing = 
    queryClient.isFetching({ queryKey: HIERARCHY_KEYS.status() }) > 0 ||
    queryClient.isFetching({ queryKey: HIERARCHY_KEYS.metrics() }) > 0

  return { refreshAll, isRefreshing }
}

export { HIERARCHY_KEYS }
export type { 
  LayerStatus, 
  HierarchyStatus, 
  HierarchyMetrics,
  ExecuteCommandRequest,
  ExecuteCommandResponse 
}
export default useHierarchyStatus
