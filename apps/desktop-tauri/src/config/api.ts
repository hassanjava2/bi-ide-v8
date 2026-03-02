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
    timeout: 15000,
  },
  rtx: {
    host: '192.168.1.164',
    port: 8090,
    get baseUrl() {
      return `http://${this.host}:${this.port}`;
    },
    timeout: 8000,
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
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await tauriFetch(`${API_CONFIG.rtx.baseUrl}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
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
  timeoutMs: number = 15000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await tauriFetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// ─── Smart Council Message (Dual-Path) ───────────────────────────

export async function sendCouncilMessage(
  message: string,
  context?: CouncilMessageRequest['context']
): Promise<CouncilMessageResponse> {
  const request: CouncilMessageRequest = { message, context };
  const body = JSON.stringify(request);
  const headers = { 'Content-Type': 'application/json' };

  // ── Path 1: Try RTX Direct (fast, LAN only) ──
  const rtxReachable = await probeRtx();
  if (rtxReachable) {
    try {
      console.log('[AI Routing] Using RTX direct path');
      const res = await fetchWithTimeout(
        `${API_CONFIG.rtx.baseUrl}/council/message`,
        { method: 'POST', headers, body },
        API_CONFIG.rtx.timeout
      );
      if (res.ok) {
        const data = await res.json();
        return {
          response: data.response || '',
          source: 'rtx5090',
          confidence: data.confidence || 0.85,
          evidence: data.evidence || [],
          response_source: 'rtx5090-direct',
          wise_man: data.wise_man || 'المجلس',
          processing_time_ms: data.processing_time_ms || 0,
          timestamp: data.timestamp || new Date().toISOString(),
        };
      }
    } catch (err) {
      console.warn('[AI Routing] RTX direct failed, falling back to VPS:', err);
      _rtxAvailable = false; // mark as down
    }
  }

  // ── Path 2: Try VPS (reliable, always available) ──
  try {
    console.log('[AI Routing] Using VPS path');
    const res = await fetchWithTimeout(
      `${API_CONFIG.vps.baseUrl}/council/message`,
      { method: 'POST', headers, body },
      API_CONFIG.vps.timeout
    );
    if (res.ok) {
      const data = await res.json();
      // VPS wraps in ApiResponse with status/data
      const responseData = data.data || data;
      return {
        response: responseData.response || '',
        source: responseData.source || 'hierarchy',
        confidence: responseData.confidence || 0.7,
        evidence: responseData.evidence || [],
        response_source: responseData.response_source || 'vps',
        wise_man: responseData.wise_man || 'المجلس',
        processing_time_ms: responseData.processing_time_ms || 0,
        timestamp: responseData.timestamp || new Date().toISOString(),
      };
    }
    // Non-OK response
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
  } catch (err) {
    console.warn('[AI Routing] VPS path failed:', err);
  }

  // ── Path 3: Offline fallback ──
  return {
    response: `⚡ لا يمكن الاتصال بنظام AI حالياً.\n\n🔍 تحقق من:\n• اتصال الإنترنت\n• حالة السيرفر (bi-iq.com)\n• حالة RTX 5090 (${API_CONFIG.rtx.host})\n\nرسالتك محفوظة: "${message}"`,
    source: 'local-fallback',
    confidence: 0,
    evidence: ['offline-mode'],
    response_source: 'local-fallback',
    wise_man: 'النظام',
    processing_time_ms: 0,
    timestamp: new Date().toISOString(),
  };
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
