/**
 * useSystemStatus Hook
 * 
 * Fetches and caches system status from /api/v1/status
 * Features:
 * - Automatic polling every 10 seconds when enabled
 * - Optimistic updates
 * - Error handling with retry
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../../services/api'
import type { SystemStatus } from '../../types'

// Query key factory for system status
const SYSTEM_STATUS_KEY = ['system', 'status'] as const

// Fetch function
const fetchSystemStatus = async (): Promise<SystemStatus> => {
  return api.getSystemStatus()
}

interface UseSystemStatusOptions {
  /** Enable automatic polling (default: true) */
  poll?: boolean
  /** Polling interval in milliseconds (default: 10000) */
  pollInterval?: number
  /** Whether the query is enabled (default: true) */
  enabled?: boolean
}

/**
 * Hook for fetching system status with caching and optional polling
 * 
 * @example
 * ```tsx
 * // Basic usage
 * const { data, isLoading, error } = useSystemStatus()
 * 
 * // With polling disabled
 * const { data } = useSystemStatus({ poll: false })
 * 
 * // Custom poll interval
 * const { data } = useSystemStatus({ pollInterval: 5000 })
 * ```
 */
export function useSystemStatus(options: UseSystemStatusOptions = {}) {
  const { 
    poll = true, 
    pollInterval = 10000, 
    enabled = true 
  } = options

  return useQuery<SystemStatus>({
    queryKey: SYSTEM_STATUS_KEY,
    queryFn: fetchSystemStatus,
    // Shorter stale time for status (5 seconds)
    staleTime: 1000 * 5,
    // Poll every 10 seconds when enabled
    refetchInterval: poll ? pollInterval : false,
    // Continue polling even when tab is in background
    refetchIntervalInBackground: true,
    enabled,
  })
}

/**
 * Hook for manually refreshing system status
 * 
 * @example
 * ```tsx
 * const { refresh, isRefreshing } = useRefreshSystemStatus()
 * 
 * <button onClick={() => refresh()} disabled={isRefreshing}>
 *   Refresh
 * </button>
 * ```
 */
export function useRefreshSystemStatus() {
  const queryClient = useQueryClient()

  const refresh = async () => {
    return queryClient.invalidateQueries({ queryKey: SYSTEM_STATUS_KEY })
  }

  const isRefreshing = queryClient.isFetching({ queryKey: SYSTEM_STATUS_KEY }) > 0

  return { refresh, isRefreshing }
}

/**
 * Prefetch system status (useful for navigation)
 * 
 * @example
 * ```tsx
 * const queryClient = useQueryClient()
 * 
 * // Prefetch on hover
 * <Link 
 *   to="/dashboard" 
 *   onMouseEnter={() => prefetchSystemStatus(queryClient)}
 * >
 *   Dashboard
 * </Link>
 * ```
 */
export function prefetchSystemStatus(queryClient: ReturnType<typeof useQueryClient>) {
  return queryClient.prefetchQuery({
    queryKey: SYSTEM_STATUS_KEY,
    queryFn: fetchSystemStatus,
    staleTime: 1000 * 5,
  })
}

export { SYSTEM_STATUS_KEY }
export default useSystemStatus
