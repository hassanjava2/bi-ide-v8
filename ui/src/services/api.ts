/**
 * API Service - خدمة الاتصال بالـ Backend
 * تدعم: IDE + ERP + AI Hierarchy + Auth
 */

const API_BASE_URL = '';

// ========== Auth Token Management ==========

let _accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  _accessToken = token;
  if (token) {
    localStorage.setItem('bi_access_token', token);
  } else {
    localStorage.removeItem('bi_access_token');
  }
}

export function getAccessToken(): string | null {
  if (_accessToken) return _accessToken;
  _accessToken = localStorage.getItem('bi_access_token');
  return _accessToken;
}

export function clearAuth() {
  _accessToken = null;
  localStorage.removeItem('bi_access_token');
}

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

// ========== Fetch Helper ==========

async function fetchApi(endpoint: string, options?: RequestInit) {
  const token = getAccessToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...(options?.headers || {}),
  };

  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  // 401 handling:
  // - For login: return a normal error (invalid credentials) without reloading.
  // - For other endpoints: clear token and notify the app to show Login.
  if (res.status === 401) {
    if (!endpoint.startsWith('/api/v1/auth/login')) {
      clearAuth();
      window.dispatchEvent(new CustomEvent('bi:auth:logout'));
    }

    const errorData = await res.json().catch(() => null);
    throw new Error(errorData?.detail || errorData?.message || 'Unauthorized');
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => null);
    throw new Error(errorData?.detail || errorData?.message || `HTTP ${res.status}`);
  }

  return res.json();
}

// ========== Auth APIs ==========

export const login = async (username: string, password: string) => {
  const data = await fetchApi('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  setAccessToken(data.access_token);
  return data;
};

export const logout = () => {
  clearAuth();
};

// ========== System Status ==========

export const getSystemStatus = () =>
  fetchApi('/api/v1/status');

export const getHierarchyStatus = () =>
  fetchApi('/api/v1/hierarchy/status');

export const getWisdom = (horizon: string = 'century') =>
  fetchApi(`/api/v1/wisdom?horizon=${horizon}`);

export const getGuardianReport = () =>
  fetchApi('/api/v1/guardian/status');

export const getCouncilMetrics = () =>
  fetchApi('/api/v1/council/metrics');

export const getHierarchyMetrics = () =>
  fetchApi('/api/v1/hierarchy/metrics');

export const executeCommand = (command: string, alertLevel: string = 'GREEN', context?: any) =>
  fetchApi('/api/v1/command', {
    method: 'POST',
    body: JSON.stringify({ command, alert_level: alertLevel, context })
  });

// ========== IDE APIs ==========

export const getFileTree = () =>
  fetchApi('/api/v1/ide/files');

export const getFile = (fileId: string) =>
  fetchApi(`/api/v1/ide/files/${fileId}`);

export const saveFile = (fileId: string, content: string) =>
  fetchApi(`/api/v1/ide/files/${fileId}`, {
    method: 'POST',
    body: JSON.stringify({ content })
  });

export const getCodeSuggestions = (code: string, cursorPosition: number, language: string, filePath: string) =>
  fetchApi('/api/v1/ide/copilot/suggest', {
    method: 'POST',
    body: JSON.stringify({ code, cursor_position: cursorPosition, language, file_path: filePath })
  });

export const analyzeCode = (code: string, language: string, filePath: string) =>
  fetchApi('/api/v1/ide/analysis', {
    method: 'POST',
    body: JSON.stringify({ code, language, file_path: filePath })
  });

export const getRefactorSuggestions = (code: string, language: string, filePath: string) =>
  fetchApi('/api/v1/ide/refactor/suggest', {
    method: 'POST',
    body: JSON.stringify({ code, language, file_path: filePath })
  });

export const generateTests = (code: string, language: string, filePath: string) =>
  fetchApi('/api/v1/ide/tests/generate', {
    method: 'POST',
    body: JSON.stringify({ code, language, file_path: filePath })
  });

export const getSymbolDocumentation = (code: string, language: string, filePath: string, symbol?: string) =>
  fetchApi('/api/v1/ide/docs/symbol', {
    method: 'POST',
    body: JSON.stringify({ code, language, file_path: filePath, symbol })
  });

export const getGitStatus = () =>
  fetchApi('/api/v1/ide/git/status');

export const getGitDiff = (path?: string) =>
  fetchApi(`/api/v1/ide/git/diff${path ? `?path=${encodeURIComponent(path)}` : ''}`);

export const createGitCommit = (message: string, stageAll: boolean = true) =>
  fetchApi('/api/v1/ide/git/commit', {
    method: 'POST',
    body: JSON.stringify({ message, stage_all: stageAll })
  });

export const pushGitChanges = (remote: string = 'origin', branch?: string) =>
  fetchApi('/api/v1/ide/git/push', {
    method: 'POST',
    body: JSON.stringify({ remote, branch })
  });

export const pullGitChanges = (remote: string = 'origin', branch?: string) =>
  fetchApi('/api/v1/ide/git/pull', {
    method: 'POST',
    body: JSON.stringify({ remote, branch })
  });

export const startDebugSession = (filePath: string, breakpoints: number[] = []) =>
  fetchApi('/api/v1/ide/debug/session/start', {
    method: 'POST',
    body: JSON.stringify({ file_path: filePath, breakpoints })
  });

export const setDebugBreakpoint = (sessionId: string, filePath: string, line: number) =>
  fetchApi('/api/v1/ide/debug/breakpoint', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, file_path: filePath, line })
  });

export const executeDebugCommand = (sessionId: string, command: string) =>
  fetchApi('/api/v1/ide/debug/command', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, command })
  });

export const stopDebugSession = (sessionId: string) =>
  fetchApi('/api/v1/ide/debug/session/stop', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId })
  });

export const executeTerminal = (sessionId: string, command: string) =>
  fetchApi('/api/v1/ide/terminal/execute', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, command })
  });

export const startTerminalSession = (cwd?: string) =>
  fetchApi('/api/v1/ide/terminal/session/start', {
    method: 'POST',
    body: JSON.stringify({ cwd })
  });

// ========== ERP APIs ==========

export const getERPDashboard = () =>
  fetchApi('/api/v1/erp/dashboard');

export const getInvoices = (status?: string) =>
  fetchApi(`/api/v1/erp/invoices${status ? `?status=${status}` : ''}`);

export const createInvoice = (data: any) =>
  fetchApi('/api/v1/erp/invoices', {
    method: 'POST',
    body: JSON.stringify(data)
  });

export const markInvoicePaid = (invoiceId: string) =>
  fetchApi(`/api/v1/erp/invoices/${invoiceId}/pay`, { method: 'POST' });

export const getInventory = () =>
  fetchApi('/api/v1/erp/inventory');

export const getEmployees = () =>
  fetchApi('/api/v1/erp/hr/employees');

export const getPayroll = () =>
  fetchApi('/api/v1/erp/hr/payroll');

export const getFinancialReport = (period: string = 'month') =>
  fetchApi(`/api/v1/erp/reports/financial?period=${period}`);

export const getERP_AI_Insights = () =>
  fetchApi('/api/v1/erp/ai-insights');

// Default export for compatibility
export const api = {
  login,
  logout,
  isAuthenticated,
  getSystemStatus,
  getHierarchyStatus,
  getWisdom,
  getGuardianReport,
  getCouncilMetrics,
  getHierarchyMetrics,
  executeCommand,
  getFileTree,
  getFile,
  saveFile,
  analyzeCode,
  getRefactorSuggestions,
  generateTests,
  getSymbolDocumentation,
  getGitStatus,
  getGitDiff,
  createGitCommit,
  pushGitChanges,
  pullGitChanges,
  startDebugSession,
  setDebugBreakpoint,
  executeDebugCommand,
  stopDebugSession,
  getCodeSuggestions,
  startTerminalSession,
  executeTerminal,
  getERPDashboard,
  getInvoices,
  createInvoice,
  markInvoicePaid,
  getInventory,
  getEmployees,
  getPayroll,
  getFinancialReport,
  getERP_AI_Insights
};

export default api;
