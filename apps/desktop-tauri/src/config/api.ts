/**
 * BI-IDE Desktop API Configuration — Smart Dual-Path Routing
 * 
 * Path 1 (Fast): RTX 5090 direct on LAN (192.168.1.164:8090)
 * Path 2 (Reliable): VPS (bi-iq.com/api/v1)
 * 
 * On startup, probes RTX. If reachable → use fast path.
 * Falls back to VPS automatically.
 */

import { invoke } from "@tauri-apps/api/core";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";

// ─── Configuration ───────────────────────────────────────────────

const API_CONFIG = {
  vps: {
    baseUrl: 'https://bi-iq.com/api/v1',
    timeout: 60000, // VPS AI takes ~30s to process
  },
  rtx: {
    host: '192.168.1.164',
    port: 8090,
    get baseUrl() {
      return `http://${this.host}:${this.port}`;
    },
    timeout: 5000,
  },
  retryAttempts: 2,
  retryDelay: 1000,
  probeIntervalMs: 60000, // re-check RTX every 60s
};

// ─── RTX Availability Detection ──────────────────────────────────

let _rtxAvailable: boolean | null = null;
let _lastProbeTime = 0;

async function probeRtx(): Promise<boolean> {
  const now = Date.now();
  if (_rtxAvailable !== null && now - _lastProbeTime < API_CONFIG.probeIntervalMs) {
    return _rtxAvailable;
  }

  try {
    const timeoutPromise = new Promise<Response>((_, reject) =>
      setTimeout(() => reject(new Error('RTX probe timeout')), 3000)
    );
    const fetchPromise = tauriFetch(`${API_CONFIG.rtx.baseUrl}/health`);
    const res = await Promise.race([fetchPromise, timeoutPromise]);
    _rtxAvailable = res.ok;
  } catch {
    _rtxAvailable = false;
  }
  _lastProbeTime = now;
  console.log(`[AI Routing] RTX 5090 ${_rtxAvailable ? '✅ reachable' : '❌ unreachable'}`);
  return _rtxAvailable;
}

// Probe on module load
probeRtx();

export function isRtxAvailable(): boolean {
  return _rtxAvailable === true;
}

export function forceReprobe() {
  _lastProbeTime = 0;
  _rtxAvailable = null;
  probeRtx();
}

// ─── API Endpoints ───────────────────────────────────────────────

export const API_ENDPOINTS = {
  council: {
    message: '/council/message',
    status: '/council/status',
    members: '/council/members',
  },
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    refresh: '/auth/refresh',
    logout: '/auth/logout',
    me: '/auth/me',
  },
  training: {
    start: '/training/start',
    status: '/training/status',
    stop: '/training/stop',
    metrics: '/training/metrics',
    models: '/training/models',
  },
  orchestrator: {
    health: '/orchestrator/health',
    workers: '/orchestrator/workers',
    jobs: '/orchestrator/jobs',
  },
  monitoring: {
    metrics: '/monitoring/metrics',
    alerts: '/monitoring/alerts',
  },
};

// ─── Types ───────────────────────────────────────────────────────

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
  source: 'rtx5090' | 'local-fallback' | 'hierarchy' | 'cloud' | 'rtx5090-direct' | 'rtx5090-error';
  confidence: number;
  evidence: string[];
  response_source: string;
  wise_man: string;
  processing_time_ms: number;
  timestamp: string;
}

export interface ApiResponse<T = unknown> {
  status: 'ok' | 'error';
  data?: T;
  error?: string;
  timestamp: string;
}

// ─── Fetch Helpers ───────────────────────────────────────────────

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = 60000
): Promise<Response> {
  // tauriFetch may not fully support AbortController, use race pattern
  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`Request timeout after ${timeoutMs}ms`)), timeoutMs)
  );
  const fetchPromise = tauriFetch(url, options);
  return Promise.race([fetchPromise, timeoutPromise]);
}

// ─── Smart Council Message (via Rust backend) ───────────────────

export async function sendCouncilMessage(
  message: string,
  context?: CouncilMessageRequest['context']
): Promise<CouncilMessageResponse> {
  console.log('[AI Routing] Using Rust invoke (bypasses WebKit)');

  try {
    // Call Rust command — HTTP done in Rust via reqwest, bypasses all frontend restrictions
    const result = await invoke<{
      response: string;
      source: string;
      confidence: number;
      wise_man: string;
      processing_time_ms: number;
    }>('send_council_message', {
      message,
      sessionId: context?.session_id || 'desktop-app',
    });

    return {
      response: result.response,
      source: result.source as CouncilMessageResponse['source'],
      confidence: result.confidence,
      evidence: [],
      response_source: result.source,
      wise_man: result.wise_man,
      processing_time_ms: result.processing_time_ms,
      timestamp: new Date().toISOString(),
    };
  } catch (err: any) {
    console.error('[AI Routing] Rust invoke failed:', err);
    return {
      response: `⚡ خطأ في الاتصال: ${err.message || err}\n\nتحقق من اتصال الإنترنت وحالة السيرفر.`,
      source: 'local-fallback',
      confidence: 0,
      evidence: ['invoke-error'],
      response_source: 'local-fallback',
      wise_man: 'النظام',
      processing_time_ms: 0,
      timestamp: new Date().toISOString(),
    };
  }
}

// ─── General API Request (always to VPS) ─────────────────────────

export async function apiRequest<T = unknown>(
  endpoint: string,
  options: { method?: string; body?: unknown; timeout?: number } = {}
): Promise<ApiResponse<T>> {
  const url = `${API_CONFIG.vps.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
  const res = await fetchWithTimeout(
    url,
    {
      method: options.method || 'GET',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: options.body ? JSON.stringify(options.body) : undefined,
    },
    options.timeout || API_CONFIG.vps.timeout
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `HTTP ${res.status}`);
  }
  return await res.json() as ApiResponse<T>;
}

// ─── URL Helpers ─────────────────────────────────────────────────

export function getApiBaseUrl(): string {
  return API_CONFIG.vps.baseUrl;
}

export function getRtxUrl(): string {
  return API_CONFIG.rtx.baseUrl;
}

export function buildApiUrl(endpoint: string): string {
  return `${API_CONFIG.vps.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
}

// ─── Export ──────────────────────────────────────────────────────

export { API_CONFIG };
export default API_CONFIG;
