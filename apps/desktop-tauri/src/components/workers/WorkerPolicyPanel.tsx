//! Worker Resource Governance Panel
//! Control resource limits for connected workers from Desktop

import { useState, useEffect, useCallback } from "react";
import { 
  Cpu, HardDrive, Thermometer, Clock, Activity, 
  Server, Laptop, Monitor, Check, X, AlertCircle,
  Save, RefreshCw, Power, PowerOff
} from "lucide-react";
import { useStore } from "../../lib/store";
import { invoke } from "@tauri-apps/api/core";

interface WorkerDevice {
  device_id: string;
  device_name: string;
  device_type: "desktop" | "laptop" | "server" | "workstation" | "vps" | "embedded";
  status: "online" | "idle" | "busy" | "training" | "offline" | "error";
  capabilities: {
    cpu_cores: number;
    memory_gb: number;
    has_gpu: boolean;
    gpu_memory_gb?: number;
    gpu_model?: string;
  };
  current_usage: {
    cpu_percent: number;
    memory_percent: number;
    gpu_percent?: number;
    temperature_c?: number;
  };
  policy?: ResourcePolicy;
}

interface ResourcePolicy {
  mode: "full" | "assist" | "training_only" | "idle_only" | "disabled";
  limits: {
    cpu_max_percent: number;
    ram_max_gb: number;
    gpu_mem_max_percent?: number;
    max_duration_hours?: number;
  };
  schedule?: {
    timezone: string;
    windows: TimeWindow[];
    idle_only: boolean;
  };
  safety: {
    thermal_cutoff_c: number;
    auto_pause_on_user_activity: boolean;
    max_consecutive_hours?: number;
    required_break_minutes?: number;
  };
}

interface TimeWindow {
  start: string;
  end: string;
  days?: number[];
}

const DEVICE_ICONS: Record<string, React.ReactNode> = {
  desktop: <Monitor className="w-5 h-5" />,
  laptop: <Laptop className="w-5 h-5" />,
  server: <Server className="w-5 h-5" />,
  workstation: <Cpu className="w-5 h-5" />,
  vps: <Server className="w-5 h-5" />,
  embedded: <HardDrive className="w-5 h-5" />,
};

const STATUS_COLORS: Record<string, string> = {
  online: "text-green-400",
  idle: "text-blue-400",
  busy: "text-yellow-400",
  training: "text-primary-400",
  offline: "text-dark-500",
  error: "text-red-400",
};

const MODE_OPTIONS = [
  { value: "full", label: "Full", description: "Use all available resources" },
  { value: "assist", label: "Assist", description: "Light background tasks only" },
  { value: "training_only", label: "Training Only", description: "AI training when idle" },
  { value: "idle_only", label: "Idle Only", description: "Only when system is idle" },
  { value: "disabled", label: "Disabled", description: "No tasks assigned" },
];

export function WorkerPolicyPanel() {
  const [workers, setWorkers] = useState<WorkerDevice[]>([]);
  const [selectedWorker, setSelectedWorker] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Load workers
  const loadWorkers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await invoke<WorkerDevice[]>("get_workers", {
        request: { include_offline: true },
      });
      setWorkers(result);
      if (result.length > 0 && !selectedWorker) {
        setSelectedWorker(result[0].device_id);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load workers");
      setWorkers([]);
    } finally {
      setLoading(false);
    }
  }, [selectedWorker]);

  useEffect(() => {
    loadWorkers();
    const interval = setInterval(loadWorkers, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [loadWorkers]);

  const selectedWorkerData = workers.find(w => w.device_id === selectedWorker);

  const handleSavePolicy = async (deviceId: string, policy: ResourcePolicy) => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    
    try {
      await invoke("apply_worker_policy", {
        request: { device_id: deviceId, policy },
      });
      setSuccess("Policy applied successfully");
      setTimeout(() => setSuccess(null), 3000);
      await loadWorkers();
    } catch (err: any) {
      setError(err.message || "Failed to apply policy");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-dark-700">
        <div className="flex items-center gap-2">
          <Server className="w-5 h-5 text-primary-400" />
          <span className="font-medium">Worker Resources</span>
          <span className="text-xs text-dark-500">
            ({workers.filter(w => w.status !== "offline").length} online)
          </span>
        </div>
        <button
          onClick={loadWorkers}
          disabled={loading}
          className="p-1.5 hover:bg-dark-700 rounded"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Notifications */}
      {error && (
        <div className="mx-4 mt-2 p-2 bg-red-900/30 border border-red-700 rounded text-xs text-red-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="p-1 hover:bg-red-900/50 rounded">
            <X className="w-3 h-3" />
          </button>
        </div>
      )}
      {success && (
        <div className="mx-4 mt-2 p-2 bg-green-900/30 border border-green-700 rounded text-xs text-green-300 flex items-center gap-2">
          <Check className="w-4 h-4" />
          <span className="flex-1">{success}</span>
        </div>
      )}

      {/* Workers List */}
      <div className="flex-1 flex overflow-hidden">
        <div className="w-1/2 border-r border-dark-700 overflow-y-auto">
          {workers.map((worker) => (
            <button
              key={worker.device_id}
              onClick={() => setSelectedWorker(worker.device_id)}
              className={`w-full p-3 text-left border-b border-dark-700 hover:bg-dark-700/50 transition-colors
                ${selectedWorker === worker.device_id ? "bg-dark-700" : ""}`}
            >
              <div className="flex items-start gap-3">
                <div className="text-dark-400">
                  {DEVICE_ICONS[worker.device_type] || <Server className="w-5 h-5" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate">{worker.device_name}</span>
                    <span className={`text-xs ${STATUS_COLORS[worker.status]}`}>
                      ● {worker.status}
                    </span>
                  </div>
                  <div className="text-xs text-dark-500 mt-1">
                    {worker.capabilities.cpu_cores} cores • {worker.capabilities.memory_gb}GB RAM
                    {worker.capabilities.has_gpu && ` • ${worker.capabilities.gpu_model}`}
                  </div>
                  <div className="flex items-center gap-3 mt-2">
                    <UsageBar 
                      label="CPU" 
                      value={worker.current_usage.cpu_percent} 
                      color="bg-blue-500" 
                    />
                    <UsageBar 
                      label="RAM" 
                      value={worker.current_usage.memory_percent} 
                      color="bg-green-500" 
                    />
                    {worker.current_usage.gpu_percent !== undefined && (
                      <UsageBar 
                        label="GPU" 
                        value={worker.current_usage.gpu_percent} 
                        color="bg-purple-500" 
                      />
                    )}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Policy Editor */}
        <div className="w-1/2 overflow-y-auto">
          {selectedWorkerData ? (
            <PolicyEditor 
              worker={selectedWorkerData}
              onSave={handleSavePolicy}
              saving={saving}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-dark-500">
              <div className="text-center">
                <Server className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Select a worker to configure</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UsageBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-dark-500">{label}</span>
      <div className="w-16 h-1.5 bg-dark-700 rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
      <span className="text-xs text-dark-400">{value.toFixed(0)}%</span>
    </div>
  );
}

interface PolicyEditorProps {
  worker: WorkerDevice;
  onSave: (deviceId: string, policy: ResourcePolicy) => void;
  saving: boolean;
}

function PolicyEditor({ worker, onSave, saving }: PolicyEditorProps) {
  const [policy, setPolicy] = useState<ResourcePolicy>(worker.policy || {
    mode: "full",
    limits: {
      cpu_max_percent: 85,
      ram_max_gb: Math.floor(worker.capabilities.memory_gb * 0.8),
      gpu_mem_max_percent: 90,
    },
    safety: {
      thermal_cutoff_c: 85,
      auto_pause_on_user_activity: true,
    },
  });

  const hasChanges = JSON.stringify(policy) !== JSON.stringify(worker.policy);

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">Resource Policy</h3>
        <button
          onClick={() => onSave(worker.device_id, policy)}
          disabled={!hasChanges || saving}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 disabled:text-dark-500 text-white text-sm rounded transition-colors"
        >
          <Save className="w-4 h-4" />
          {saving ? "Saving..." : "Apply"}
        </button>
      </div>

      {/* Mode Selection */}
      <div>
        <label className="text-sm text-dark-400 block mb-2">Worker Mode</label>
        <div className="space-y-2">
          {MODE_OPTIONS.map((mode) => (
            <button
              key={mode.value}
              onClick={() => setPolicy({ ...policy, mode: mode.value as any })}
              className={`w-full p-3 rounded border text-left transition-colors
                ${policy.mode === mode.value 
                  ? "border-primary-500 bg-primary-600/10" 
                  : "border-dark-700 hover:border-dark-600"
                }`}
            >
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded-full border flex items-center justify-center
                  ${policy.mode === mode.value ? "border-primary-500" : "border-dark-500"}`}>
                  {policy.mode === mode.value && <div className="w-2 h-2 rounded-full bg-primary-500" />}
                </div>
                <span className="font-medium">{mode.label}</span>
              </div>
              <p className="text-xs text-dark-500 mt-1 ml-6">{mode.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Resource Limits */}
      <div>
        <label className="text-sm text-dark-400 block mb-2">Resource Limits</label>
        <div className="space-y-3 p-3 bg-dark-900 rounded border border-dark-700">
          {/* CPU Limit */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" />
                Max CPU
              </span>
              <span className="text-sm text-dark-400">{policy.limits.cpu_max_percent}%</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              value={policy.limits.cpu_max_percent}
              onChange={(e) => setPolicy({
                ...policy,
                limits: { ...policy.limits, cpu_max_percent: parseInt(e.target.value) }
              })}
              className="w-full"
            />
          </div>

          {/* RAM Limit */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5" />
                Max RAM
              </span>
              <span className="text-sm text-dark-400">{policy.limits.ram_max_gb} GB</span>
            </div>
            <input
              type="range"
              min="1"
              max={worker.capabilities.memory_gb}
              value={policy.limits.ram_max_gb}
              onChange={(e) => setPolicy({
                ...policy,
                limits: { ...policy.limits, ram_max_gb: parseInt(e.target.value) }
              })}
              className="w-full"
            />
          </div>

          {/* GPU Memory Limit */}
          {worker.capabilities.has_gpu && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5" />
                  Max GPU Memory
                </span>
                <span className="text-sm text-dark-400">
                  {policy.limits.gpu_mem_max_percent || 90}%
                </span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                value={policy.limits.gpu_mem_max_percent || 90}
                onChange={(e) => setPolicy({
                  ...policy,
                  limits: { ...policy.limits, gpu_mem_max_percent: parseInt(e.target.value) }
                })}
                className="w-full"
              />
            </div>
          )}
        </div>
      </div>

      {/* Safety Settings */}
      <div>
        <label className="text-sm text-dark-400 block mb-2">Safety Settings</label>
        <div className="space-y-3 p-3 bg-dark-900 rounded border border-dark-700">
          {/* Thermal Cutoff */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm flex items-center gap-1.5">
                <Thermometer className="w-3.5 h-3.5" />
                Thermal Cutoff
              </span>
              <span className="text-sm text-dark-400">{policy.safety.thermal_cutoff_c}°C</span>
            </div>
            <input
              type="range"
              min="60"
              max="95"
              value={policy.safety.thermal_cutoff_c}
              onChange={(e) => setPolicy({
                ...policy,
                safety: { ...policy.safety, thermal_cutoff_c: parseInt(e.target.value) }
              })}
              className="w-full"
            />
          </div>

          {/* Auto-pause on activity */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={policy.safety.auto_pause_on_user_activity}
              onChange={(e) => setPolicy({
                ...policy,
                safety: { ...policy.safety, auto_pause_on_user_activity: e.target.checked }
              })}
              className="rounded border-dark-600"
            />
            <span className="text-sm">Auto-pause when user is active</span>
          </label>
        </div>
      </div>

      {/* Schedule */}
      <div>
        <label className="text-sm text-dark-400 block mb-2">Schedule (Optional)</label>
        <div className="p-3 bg-dark-900 rounded border border-dark-700 space-y-2">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-dark-500" />
            <span className="text-sm">Only run during:</span>
          </div>
          {policy.schedule ? (
            <div className="space-y-2">
              {policy.schedule.windows.map((window, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <input
                    type="time"
                    value={window.start}
                    onChange={(e) => {
                      const windows = [...policy.schedule!.windows];
                      windows[idx] = { ...window, start: e.target.value };
                      setPolicy({ ...policy, schedule: { ...policy.schedule!, windows } });
                    }}
                    className="bg-dark-800 border border-dark-700 rounded px-2 py-1 text-sm"
                  />
                  <span className="text-dark-500">to</span>
                  <input
                    type="time"
                    value={window.end}
                    onChange={(e) => {
                      const windows = [...policy.schedule!.windows];
                      windows[idx] = { ...window, end: e.target.value };
                      setPolicy({ ...policy, schedule: { ...policy.schedule!, windows } });
                    }}
                    className="bg-dark-800 border border-dark-700 rounded px-2 py-1 text-sm"
                  />
                  <button
                    onClick={() => {
                      const windows = policy.schedule!.windows.filter((_, i) => i !== idx);
                      setPolicy({ ...policy, schedule: { ...policy.schedule!, windows } });
                    }}
                    className="p-1 hover:bg-dark-700 rounded"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
              <button
                onClick={() => setPolicy({
                  ...policy,
                  schedule: {
                    ...policy.schedule!,
                    windows: [...policy.schedule!.windows, { start: "22:00", end: "07:00" }]
                  }
                })}
                className="text-xs text-primary-400 hover:text-primary-300"
              >
                + Add window
              </button>
            </div>
          ) : (
            <button
              onClick={() => setPolicy({
                ...policy,
                schedule: {
                  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                  windows: [{ start: "22:00", end: "07:00" }],
                  idle_only: true,
                }
              })}
              className="text-sm text-primary-400 hover:text-primary-300"
            >
              + Add schedule
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
