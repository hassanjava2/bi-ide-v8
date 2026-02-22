/**
 * Query Provider - React Query (TanStack Query) Configuration
 * 
 * Provides global query client with optimized defaults for the BI-IDE v8 application.
 * Features:
 * - Intelligent caching with staleTime configuration
 * - Automatic background refetching
 * - Error retry logic
 * - DevTools integration (development only)
 */

import { ReactNode, useState } from 'react'
import {
  QueryClient,
  QueryClientProvider,
  DefaultOptions,
} from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

// Default query options for optimal API performance
const defaultQueryOptions: DefaultOptions = {
  queries: {
    // Data stays fresh for 30 seconds
    staleTime: 1000 * 30,
    // Cache data for 5 minutes after last use
    gcTime: 1000 * 60 * 5,
    // Retry failed requests 2 times with exponential backoff
    retry: 2,
    retryDelay: (attemptIndex: number) => Math.min(1000 * 2 ** attemptIndex, 30000),
    // Refetch on window focus for real-time updates
    refetchOnWindowFocus: true,
    // Refetch when reconnecting
    refetchOnReconnect: true,
    // Don't refetch on mount if data is fresh
    refetchOnMount: 'always',
  },
  mutations: {
    // Retry mutations only once (they might be side-effectful)
    retry: 1,
    retryDelay: 1000,
  },
}

// Create query client factory for potential SSR compatibility
function createQueryClient() {
  return new QueryClient({
    defaultOptions: defaultQueryOptions,
  })
}

interface QueryProviderProps {
  children: ReactNode
}

/**
 * QueryProvider wraps the application with React Query functionality
 * 
 * Usage:
 * ```tsx
 * <QueryProvider>
 *   <App />
 * </QueryProvider>
 * ```
 */
export function QueryProvider({ children }: QueryProviderProps) {
  // Create query client once per component lifecycle
  const [queryClient] = useState(() => createQueryClient())

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* React Query DevTools - only rendered in development */}
      <ReactQueryDevtools 
        initialIsOpen={false} 
        position="bottom"
        buttonPosition="bottom-right"
      />
    </QueryClientProvider>
  )
}

export { createQueryClient }
export default QueryProvider
