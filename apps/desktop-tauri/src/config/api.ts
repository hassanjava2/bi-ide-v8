/**
 * BI-IDE Desktop API Configuration
 * 
 * Centralized API configuration for Tauri desktop app.
 * All API calls should use this configuration.
 */

// API Configuration
const API_CONFIG = {
  // Base URLs by environment
  baseUrls: {
    production: 'https://bi-iq.com/api/v1',
    development: 'http://localhost:8000/api/v1',
    local: 'http://127.0.0.1:8000/api/v1',
  },
  
  // RTX 4090 Server Configuration (STANDARDIZED)
  rtx: {
    host: '192.168.1.164',
    port: 8090,
    get url() {
      return `http://${this.host}:${this.port}`;
    },
  },
  
  // Request configuration
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000, // 1 second
  
  // Feature flags
  features: {
    useStreaming: true,
    useCache: true,
    offlineMode: true,
  },
};

// Determine current environment
function getEnvironment(): 'production' | 'development' | 'local' {
  // Check for explicit environment variable
  if (import.meta.env.VITE_APP_ENV) {
    return import.meta.env.VITE_APP_ENV as 'production' | 'development' | 'local';
  }
  
  // Check for production mode
  if (import.meta.env.PROD) {
    return 'production';
  }
  
  return 'development';
}

// Get API base URL
export function getApiBaseUrl(): string {
  const env = getEnvironment();
  return API_CONFIG.baseUrls[env];
}

// Get RTX server URL
export function getRtxUrl(): string {
  return API_CONFIG.rtx.url;
}

// Build full API URL
export function buildApiUrl(endpoint: string): string {
  const baseUrl = getApiBaseUrl();
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${baseUrl}${cleanEndpoint}`;
}

// API Endpoints
export const API_ENDPOINTS = {
  // Council
  council: {
    message: '/council/message',
    status: '/council/status',
    members: '/council/members',
    decisions: '/council/decisions',
  },
  
  // Auth
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    refresh: '/auth/refresh',
    logout: '/auth/logout',
    me: '/auth/me',
  },
  
  // Training
  training: {
    start: '/training/start',
    status: '/training/status',
    stop: '/training/stop',
    metrics: '/training/metrics',
    models: '/training/models',
    history: '/training/history',
  },
  
  // Orchestrator
  orchestrator: {
    health: '/orchestrator/health',
    workers: '/orchestrator/workers',
    jobs: '/orchestrator/jobs',
    autoSchedule: '/orchestrator/auto-schedule',
    brainStatus: '/orchestrator/brain/status',
  },
  
  // Monitoring
  monitoring: {
    metrics: '/monitoring/metrics',
    alerts: '/monitoring/alerts',
    logs: '/monitoring/logs',
  },
};

// Request options with defaults
export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  retryAttempts?: number;
  signal?: AbortSignal;
}

// Standard response type
export interface ApiResponse<T = unknown> {
  status: 'ok' | 'error';
  data?: T;
  error?: string;
  error_code?: string;
  timestamp: string;
}

// Council message types
export interface CouncilMessageRequest {
  message: string;
  context?: {
    session_id?: string;
    user_id?: string;
    previous_messages?: Array<{ role: string; content: string }>;
  };
}

export interface CouncilMessageResponse {
  response: string;
  source: 'rtx4090' | 'local-fallback' | 'hierarchy' | 'cloud';
  confidence: number;
  evidence: string[];
  response_source: string;
  wise_man: string;
  processing_time_ms: number;
  timestamp: string;
}

// Fetch with timeout and retry
export async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = API_CONFIG.timeout
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// Main API request function
export async function apiRequest<T = unknown>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<ApiResponse<T>> {
  const url = buildApiUrl(endpoint);
  const {
    method = 'GET',
    headers = {},
    body,
    timeout = API_CONFIG.timeout,
    retryAttempts = API_CONFIG.retryAttempts,
  } = options;
  
  const requestInit: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  };
  
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt < retryAttempts; attempt++) {
    try {
      const response = await fetchWithTimeout(url, requestInit, timeout);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || `HTTP ${response.status}: ${response.statusText}`
        );
      }
      
      const data = await response.json();
      return data as ApiResponse<T>;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      // Don't retry on client errors (4xx)
      if (error instanceof Response && error.status >= 400 && error.status < 500) {
        break;
      }
      
      // Wait before retry
      if (attempt < retryAttempts - 1) {
        await new Promise(resolve => setTimeout(resolve, API_CONFIG.retryDelay * (attempt + 1)));
      }
    }
  }
  
  throw lastError || new Error('Request failed after retries');
}

// Specialized function for council messages with fallback
export async function sendCouncilMessage(
  message: string,
  context?: CouncilMessageRequest['context']
): Promise<CouncilMessageResponse> {
  const request: CouncilMessageRequest = { message, context };
  
  try {
    // Try primary API first
    const response = await apiRequest<CouncilMessageResponse>(
      API_ENDPOINTS.council.message,
      {
        method: 'POST',
        body: request,
        timeout: 15000, // 15 seconds for council
      }
    );
    
    if (response.status === 'ok' && response.data) {
      return response.data;
    }
    
    throw new Error(response.error || 'Invalid response from council');
  } catch (error) {
    console.warn('Primary council API failed, trying fallback:', error);
    
    // Try direct RTX connection as fallback
    try {
      const rtxResponse = await fetchWithTimeout(
        `${getRtxUrl()}/council/message`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        },
        10000
      );
      
      if (rtxResponse.ok) {
        const data = await rtxResponse.json();
        return {
          response: data.response || data.message || 'No response',
          source: 'rtx4090',
          confidence: data.confidence || 0.85,
          evidence: data.evidence || [],
          response_source: 'rtx4090',
          wise_man: data.council_member || 'حكيم القرار',
          processing_time_ms: data.processing_time_ms || 0,
          timestamp: new Date().toISOString(),
        };
      }
    } catch (rtxError) {
      console.warn('RTX fallback failed:', rtxError);
    }
    
    // Final fallback: local response
    return {
      response: `عذراً، لا يمكن الاتصال بالمجلس حالياً. رسالتك: "${message}"`,
      source: 'local-fallback',
      confidence: 0.5,
      evidence: ['offline-mode'],
      response_source: 'local-fallback',
      wise_man: 'النظام',
      processing_time_ms: 0,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export configuration
export { API_CONFIG };
export default API_CONFIG;
