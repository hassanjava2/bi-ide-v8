//! Tauri Command Wrappers
import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

// File System
export interface FileInfo {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  modified_at?: number;
  created_at?: number;
}

export interface ReadFileResponse {
  content: string;
  encoding: string;
}

export const fs = {
  readFile: (path: string): Promise<ReadFileResponse> =>
    invoke("read_file", { request: { path } }),

  writeFile: (path: string, content: string): Promise<void> =>
    invoke("write_file", { request: { path, content } }),

  readDir: (path: string, recursive?: boolean): Promise<FileInfo[]> =>
    invoke("read_dir", { request: { path, recursive } }),

  createDir: (path: string, recursive?: boolean): Promise<void> =>
    invoke("create_dir", { request: { path, recursive } }),

  deleteFile: (path: string, recursive?: boolean): Promise<void> =>
    invoke("delete_file", { request: { path, recursive } }),

  renameFile: (from: string, to: string): Promise<void> =>
    invoke("rename_file", { request: { from, to } }),

  watchPath: (path: string, workspaceId: string): Promise<void> =>
    invoke("watch_path", { path, workspaceId }),

  unwatchPath: (path: string): Promise<void> =>
    invoke("unwatch_path", { path }),
};

// Git
export interface GitStatus {
  branch: string;
  ahead: number;
  behind: number;
  modified: string[];
  added: string[];
  deleted: string[];
  untracked: string[];
  conflicted: string[];
  is_clean: boolean;
}

export interface GitCommit {
  hash: string;
  short_hash: string;
  message: string;
  author: string;
  timestamp: number;
}

export interface GitBranch {
  name: string;
  is_current: boolean;
  is_remote: boolean;
  upstream?: string;
}

export const git = {
  status: (path: string): Promise<GitStatus> =>
    invoke("git_status", { request: { path } }),

  add: (path: string, files: string[]): Promise<void> =>
    invoke("git_add", { request: { path, files } }),

  commit: (path: string, message: string): Promise<string> =>
    invoke("git_commit", { request: { path, message } }),

  push: (path: string, remote?: string, branch?: string): Promise<void> =>
    invoke("git_push", { request: { path, remote, branch } }),

  pull: (path: string, remote?: string, branch?: string): Promise<string> =>
    invoke("git_pull", { request: { path, remote, branch } }),

  log: (path: string, limit?: number): Promise<GitCommit[]> =>
    invoke("git_log", { request: { path, limit } }),

  branches: (path: string): Promise<GitBranch[]> =>
    invoke("git_branches", { request: { path } }),

  checkout: (path: string, branch: string, create?: boolean): Promise<void> =>
    invoke("git_checkout", { request: { path, branch, create } }),

  clone: (url: string, path: string, depth?: number): Promise<void> =>
    invoke("git_clone", { request: { url, path, depth } }),
};

// Dialog
export const dialog = {
  open: (options: Parameters<typeof openDialog>[0]) => openDialog(options),
};

// Terminal
export interface ProcessOutput {
  stdout: string;
  stderr: string;
  exit_code?: number;
}

export interface SpawnedProcess {
  process_id: number;
  command: string;
}

export const terminal = {
  execute: (
    command: string,
    args?: string[],
    cwd?: string,
    env?: Record<string, string>,
    timeoutMs?: number
  ): Promise<ProcessOutput> =>
    invoke("execute_command", {
      request: { command, args, cwd, env, timeout_ms: timeoutMs },
    }),

  spawn: (
    command: string,
    args?: string[],
    cwd?: string,
    env?: Record<string, string>
  ): Promise<SpawnedProcess> =>
    invoke("spawn_process", {
      request: { command, args, cwd, env },
    }),

  kill: (processId: number): Promise<void> =>
    invoke("kill_process", { request: { process_id: processId } }),

  readOutput: (processId: number): Promise<ProcessOutput> =>
    invoke("read_process_output", { processId }),

  writeInput: (processId: number, input: string): Promise<void> =>
    invoke("write_process_input", { request: { process_id: processId, input } }),
};

// System
export interface SystemInfo {
  platform: string;
  arch: string;
  version: string;
  hostname: string;
  device_id: string;
  app_version: string;
}

export interface ResourceUsage {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  disk_percent: number;
}

export const system = {
  getInfo: (): Promise<SystemInfo> => invoke("get_system_info"),

  getResourceUsage: (): Promise<ResourceUsage> => invoke("get_resource_usage"),

  openPath: (path: string): Promise<void> =>
    invoke("open_path", { request: { path } }),

  showNotification: (title: string, body: string): Promise<void> =>
    invoke("show_notification", { request: { title, body } }),
};

// Auth
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_at: number;
}

export const auth = {
  getDeviceId: (): Promise<string> => invoke("get_device_id"),

  registerDevice: (
    serverUrl: string,
    deviceName: string,
    username: string,
    password: string
  ): Promise<AuthTokens> =>
    invoke("register_device", {
      request: { server_url: serverUrl, device_name: deviceName, username, password },
    }),

  getAccessToken: (): Promise<string | null> => invoke("get_access_token"),

  setAccessToken: (token: string, refreshToken?: string, expiresAt?: number): Promise<void> =>
    invoke("set_access_token", {
      request: { access_token: token, refresh_token: refreshToken, expires_at: expiresAt },
    }),
};

// Sync
export interface SyncStatus {
  enabled: boolean;
  server_url: string;
  is_connected: boolean;
  last_sync?: number;
  pending_count: number;
  conflicts_count: number;
}

export const sync = {
  getStatus: (): Promise<SyncStatus> => invoke("get_sync_status"),

  forceSync: (workspaceId?: string): Promise<void> =>
    invoke("force_sync", { request: { workspace_id: workspaceId } }),

  getPendingOperations: (): Promise<{ operations: any[] }> =>
    invoke("get_pending_operations"),
};

// Workspace
export interface FileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  modified_at?: number;
}

export interface WorkspaceInfo {
  id: string;
  path: string;
  name: string;
  files: FileEntry[];
}

export interface ActiveWorkspace {
  id?: string;
  path?: string;
  name?: string;
}

export const workspace = {
  open: (path: string): Promise<WorkspaceInfo> =>
    invoke("open_workspace", { request: { path } }),

  close: (workspaceId: string): Promise<void> =>
    invoke("close_workspace", { workspaceId }),

  getFiles: (workspaceId: string, path?: string): Promise<FileEntry[]> =>
    invoke("get_workspace_files", { request: { workspace_id: workspaceId, path } }),

  getActive: (): Promise<ActiveWorkspace> => invoke("get_active_workspace"),
};

// Training
export interface TrainingStatus {
  enabled: boolean;
  current_job?: {
    job_id: string;
    job_type: string;
    progress_percent: number;
    status: string;
    started_at: number;
    estimated_completion?: number;
  };
  metrics: {
    jobs_completed: number;
    jobs_failed: number;
    total_training_time_hours: number;
    last_training_at?: number;
  };
}

export interface TrainingMetrics {
  current?: {
    loss: number;
    accuracy: number;
    samples_processed: number;
    epoch: number;
    total_epochs: number;
  };
  history: {
    timestamp: number;
    loss: number;
    accuracy: number;
  }[];
}

export interface GPUDevice {
  id: number;
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  utilization_percent: number;
  temperature_celsius: number;
  fan_speed_percent: number;
  power_draw_watts: number;
  clock_speed_mhz: number;
  memory_clock_mhz: number;
  driver_version: string;
}

export interface GPUMetrics {
  available: boolean;
  devices: GPUDevice[];
  error?: string;
}

// AI Operations
export interface AIExplanation {
  explanation: string;
  suggestions: string[];
}

export interface AIRefactor {
  refactored_code: string;
  explanation: string;
  changes: string[];
}

export const training = {
  getStatus: (): Promise<TrainingStatus> => invoke("get_training_status"),

  startJob: (jobType: string, priority?: number): Promise<{ job_id: string; status: string }> =>
    invoke("start_training_job", { request: { job_type: jobType, priority } }),

  pauseJob: (jobId: string): Promise<void> =>
    invoke("pause_training_job", { request: { job_id: jobId } }),

  getMetrics: (): Promise<TrainingMetrics> => invoke("get_training_metrics"),

  getGpuMetrics: (): Promise<GPUMetrics> => invoke("get_gpu_metrics"),
};

// AI Assistant
export interface AiChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface AiChatResponse {
  generated_text?: string;
  text?: string;
  response?: string;
  error?: string;
  source?: string;
}

// API base URL - try cloud first, fallback to local
const API_BASE = "https://bi-iq.com";
const LOCAL_API = "http://127.0.0.1:8000";

async function apiFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = { "Content-Type": "application/json", ...options.headers };
  try {
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers, signal: AbortSignal.timeout(10000) });
    if (res.ok) return res;
  } catch { /* fallback */ }
  return fetch(`${LOCAL_API}${path}`, { ...options, headers });
}

export const ai = {
  chat: async (messages: AiChatMessage[], maxLength = 512): Promise<AiChatResponse> => {
    // Use Rust invoke — bypasses all WebKit/CSP/timeout issues
    const userMessage = messages.filter(m => m.role === "user").pop()?.content || "";
    const result = await invoke<{
      response: string;
      source: string;
      confidence: number;
      wise_man: string;
      processing_time_ms: number;
    }>("send_council_message", {
      message: userMessage,
      sessionId: "desktop-sidebar",
    });
    return {
      generated_text: result.response,
      response: result.response,
      text: result.response,
      source: result.source,
    };
  },

  getCompletion: async (context: { textBeforeCursor: string; textAfterCursor: string; language: string; filePath: string }): Promise<{ completion: string }> => {
    const response = await apiFetch("/api/v1/ai/completion", {
      method: "POST",
      body: JSON.stringify(context),
    });
    if (!response.ok) {
      return { completion: "" };
    }
    return response.json();
  },

  // Desktop AI commands via Tauri
  explain: (code: string, language?: string): Promise<AIExplanation> =>
    invoke("explain_code", { request: { code, language } }),

  refactor: (code: string, language?: string, instruction?: string): Promise<AIRefactor> =>
    invoke("refactor_code", { request: { code, language, instruction } }),

  complete: (prompt: string, maxTokens?: number): Promise<string> =>
    invoke("get_ai_completion", { prompt, max_tokens: maxTokens }),
};

export interface TrainingSamplePayload {
  input_text: string;
  output_text?: string;
  workspace_path?: string;
  file_path?: string;
  metadata?: Record<string, unknown>;
}

export const trainingData = {
  ingestChatSample: async (payload: TrainingSamplePayload): Promise<void> => {
    try {
      await apiFetch("/api/v1/training-data/ingest", {
        method: "POST",
        body: JSON.stringify({
          samples: [{
            source: "desktop-tauri",
            kind: "ai_chat_pair",
            input_text: payload.input_text,
            output_text: payload.output_text,
            workspace_path: payload.workspace_path,
            file_path: payload.file_path,
            metadata: payload.metadata || {},
            timestamp_ms: Date.now(),
          }],
          relay: true,
          store_local: true,
        }),
      });
    } catch { /* silent */ }
  },
};

// ─── Council API ───
export interface WiseMan {
  id: string;
  name: string;
  role: string;
  specialization: string;
  status: string;
}

export interface CouncilStatus {
  status: string;
  connected: boolean;
  wise_men_count: number;
  active_discussions: number;
  messages_total: number;
  last_message_at?: string;
}

export const council = {
  getStatus: async (): Promise<CouncilStatus> => {
    const res = await apiFetch("/api/v1/council/status");
    return res.json();
  },

  getWiseMen: async (): Promise<WiseMan[]> => {
    const res = await apiFetch("/api/v1/council/wise-men");
    const data = await res.json();
    return data.wise_men || data || [];
  },

  sendMessage: async (message: string): Promise<any> => {
    const res = await apiFetch("/api/v1/council/message", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    return res.json();
  },

  discuss: async (topic: string): Promise<any> => {
    const res = await apiFetch("/api/v1/council/discuss", {
      method: "POST",
      body: JSON.stringify({ topic }),
    });
    return res.json();
  },

  deliberate: async (message: string): Promise<any> => {
    const res = await apiFetch("/api/v1/council/deliberate", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    return res.json();
  },

  getMetrics: async (): Promise<any> => {
    const res = await apiFetch("/api/v1/council/metrics");
    return res.json();
  },

  getHistory: async (): Promise<any[]> => {
    const res = await apiFetch("/api/v1/council/history");
    const data = await res.json();
    return data.messages || data || [];
  },
};

// ─── Hierarchy API ───
export interface LayerStatus {
  name: string;
  specialization: string;
  epoch: number;
  loss: number;
  accuracy: number;
  samples: number;
  fetches: number;
  vram_gb: number;
}

export interface HierarchyStatus {
  is_training: boolean;
  device: string;
  mode: string;
  layers: Record<string, LayerStatus>;
  gpu?: {
    name: string;
    utilization: number;
    memory_used: number;
    memory_total: number;
  };
}

export const hierarchy = {
  getStatus: async (): Promise<HierarchyStatus> => {
    const res = await apiFetch("/api/v1/hierarchy/status");
    return res.json();
  },

  getMetrics: async (): Promise<any> => {
    const res = await apiFetch("/api/v1/hierarchy/metrics");
    return res.json();
  },

  executeCommand: async (command: string, context?: any): Promise<any> => {
    const res = await apiFetch("/api/v1/hierarchy/execute", {
      method: "POST",
      body: JSON.stringify({ command, context }),
    });
    return res.json();
  },

  getWisdom: async (horizon = "century"): Promise<any> => {
    const res = await apiFetch(`/api/v1/hierarchy/wisdom?horizon=${horizon}`);
    return res.json();
  },

  getGuardianStatus: async (): Promise<any> => {
    const res = await apiFetch("/api/v1/hierarchy/guardian");
    return res.json();
  },
};

// ─── Orchestrator API ───
export const orchestrator = {
  getHealth: async (): Promise<any> => {
    const res = await apiFetch("/api/v1/orchestrator/health");
    return res.json();
  },
};

