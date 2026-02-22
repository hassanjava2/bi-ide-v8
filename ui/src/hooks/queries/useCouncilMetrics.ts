/**
 * useCouncilMetrics Hook
 * 
 * Fetches and caches council metrics from /api/v1/council/metrics
 * Features:
 * - Real-time metrics with polling
 * - Metrics history tracking
 * - Mutations for council interactions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../services/api'
import type { AlertLevel } from '../../types'

// Query keys
const COUNCIL_KEYS = {
  all: ['council'] as const,
  metrics: () => [...COUNCIL_KEYS.all, 'metrics'] as const,
  history: () => [...COUNCIL_KEYS.all, 'history'] as const,
  message: (id: string) => [...COUNCIL_KEYS.all, 'message', id] as const,
}

// Types
interface CouncilMetrics {
  council_responses: number
  fallback_rate_pct: number
  latency_ms: {
    avg: number
    last: number
  }
  quality: {
    evidence_backed_rate_pct: number
    evidence_backed_total: number
    guard_total: number
    daily_trend: Array<{
      day: string
      evidence_rate_pct: number
      evidence_backed: number
      responses: number
    }>
  }
  top_wise_men: Array<{
    name: string
    responses: number
  }>
}

interface CouncilMessageRequest {
  message: string
  user_id?: string
  alert_level?: AlertLevel
}

interface CouncilMessageResponse {
  response: string
  council_member: string
  response_source?: string
  source?: string
  evidence?: Array<{
    topic?: string
    source?: string
  }>
  needs_topic_specificity?: boolean
}

interface CouncilDiscussionRequest {
  topic: string
}

interface CouncilDiscussionResponse {
  discussion: Array<{
    wise_man: string
    response: string
    response_source?: string
    evidence?: Array<{
      topic?: string
      source?: string
    }>
  }>
  filtered_out?: number
}

interface CouncilHistoryItem {
  role: 'user' | 'council'
  message: string
  timestamp: string
  council_member?: string
  response_source?: string
  source?: string
  evidence?: Array<{
    topic?: string
    source?: string
  }>
}

interface CouncilHistoryResponse {
  history: CouncilHistoryItem[]
}

// Fetch functions
const fetchCouncilMetrics = async (): Promise<CouncilMetrics> => {
  return api.getCouncilMetrics()
}

const fetchCouncilHistory = async (): Promise<CouncilHistoryResponse> => {
  const res = await fetch('/api/v1/council/history')
  if (!res.ok) throw new Error('Failed to fetch council history')
  return res.json()
}

const sendCouncilMessage = async (data: CouncilMessageRequest): Promise<CouncilMessageResponse> => {
  const res = await fetch('/api/v1/council/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to send message')
  return res.json()
}

const startCouncilDiscussion = async (data: CouncilDiscussionRequest): Promise<CouncilDiscussionResponse> => {
  const res = await fetch('/api/v1/council/discuss', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to start discussion')
  return res.json()
}

interface UseCouncilMetricsOptions {
  /** Enable automatic polling (default: true) */
  poll?: boolean
  /** Polling interval in milliseconds (default: 3000) */
  pollInterval?: number
  enabled?: boolean
}

/**
 * Hook for fetching council metrics with caching
 * 
 * @example
 * ```tsx
 * const { data, isLoading, error } = useCouncilMetrics()
 * ```
 */
export function useCouncilMetrics(options: UseCouncilMetricsOptions = {}) {
  const { poll = true, pollInterval = 3000, enabled = true } = options

  return useQuery<CouncilMetrics>({
    queryKey: COUNCIL_KEYS.metrics(),
    queryFn: fetchCouncilMetrics,
    staleTime: 1000 * 3,
    refetchInterval: poll ? pollInterval : false,
    refetchIntervalInBackground: true,
    enabled,
  })
}

/**
 * Hook for fetching council discussion history
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useCouncilHistory()
 * ```
 */
export function useCouncilHistory(options: { enabled?: boolean } = {}) {
  return useQuery<CouncilHistoryResponse>({
    queryKey: COUNCIL_KEYS.history(),
    queryFn: fetchCouncilHistory,
    staleTime: 1000 * 10,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for sending a message to the council
 * 
 * @example
 * ```tsx
 * const mutation = useSendCouncilMessage()
 * 
 * mutation.mutate({
 *   message: 'What is the status?',
 *   alert_level: 'GREEN'
 * })
 * ```
 */
export function useSendCouncilMessage() {
  const queryClient = useQueryClient()

  return useMutation<CouncilMessageResponse, Error, CouncilMessageRequest>({
    mutationFn: sendCouncilMessage,
    onSuccess: () => {
      // Invalidate metrics to show updated stats
      queryClient.invalidateQueries({ queryKey: COUNCIL_KEYS.metrics() })
      // Invalidate history to show new message
      queryClient.invalidateQueries({ queryKey: COUNCIL_KEYS.history() })
    },
  })
}

/**
 * Hook for starting a group discussion
 * 
 * @example
 * ```tsx
 * const mutation = useCouncilDiscussion()
 * 
 * mutation.mutate({ topic: 'Strategic planning' })
 * ```
 */
export function useCouncilDiscussion() {
  const queryClient = useQueryClient()

  return useMutation<CouncilDiscussionResponse, Error, CouncilDiscussionRequest>({
    mutationFn: startCouncilDiscussion,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: COUNCIL_KEYS.metrics() })
      queryClient.invalidateQueries({ queryKey: COUNCIL_KEYS.history() })
    },
  })
}

export { COUNCIL_KEYS }
export type {
  CouncilMetrics,
  CouncilMessageRequest,
  CouncilMessageResponse,
  CouncilDiscussionRequest,
  CouncilDiscussionResponse,
  CouncilHistoryItem,
  CouncilHistoryResponse,
}
export default useCouncilMetrics
