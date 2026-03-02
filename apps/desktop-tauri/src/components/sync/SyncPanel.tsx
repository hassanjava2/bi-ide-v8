//! Sync Panel - Real-time synchronization status

import { useState, useEffect, useCallback } from "react";
import { RefreshCw, Cloud, CloudOff, Check, AlertTriangle, Clock } from "lucide-react";
import { useStore } from "../../lib/store";
import { invoke } from "@tauri-apps/api/core";

interface SyncDevice {
  device_id: string;
  device_name: string;
  status: "synced" | "syncing" | "conflict" | "offline";
  last_seen: number;
}

export function SyncPanel() {
  const [devices, setDevices] = useState<SyncDevice[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const { currentWorkspace, deviceId } = useStore();

  const loadDevices = useCallback(async () => {
    if (!currentWorkspace) return;
    try {
      const result = await invoke<SyncDevice[]>("get_sync_devices", {
        workspace_id: currentWorkspace.id,
      });
      setDevices(result);
    } catch (err) {
      console.error("Failed to load devices:", err);
    }
  }, [currentWorkspace]);

  const forceSync = async () => {
    if (!currentWorkspace) return;
    setLoading(true);
    try {
      await invoke("force_sync", {
        request: { workspace_id: currentWorkspace.id },
      });
    } catch (err) {
      console.error("Sync failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDevices();
    const interval = setInterval(loadDevices, 10000);
    return () => clearInterval(interval);
  }, [loadDevices]);

  if (!currentWorkspace) {
    return (
      <div className="h-full flex items-center justify-center text-dark-500">
        <Cloud className="w-12 h-12 mb-3 opacity-30" />
        <p className="text-sm">Open a workspace to sync</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-dark-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {isConnected ? <Cloud className="w-5 h-5 text-green-400" /> : <CloudOff className="w-5 h-5 text-red-400" />}
          <span className="font-medium">Sync</span>
        </div>
        <button
          onClick={forceSync}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 text-white text-sm rounded"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Sync Now
        </button>
      </div>

      <div className="space-y-2">
        {devices.map((device) => (
          <div key={device.device_id} className="flex items-center justify-between p-3 bg-dark-900 rounded">
            <div>
              <div className="text-sm font-medium">{device.device_name}</div>
              <div className="text-xs text-dark-500">
                {device.device_id === deviceId ? "This device" : "Remote"}
              </div>
            </div>
            <StatusBadge status={device.status} />
          </div>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    synced: "text-green-400",
    syncing: "text-blue-400",
    conflict: "text-red-400",
    offline: "text-dark-500",
  };
  return <span className={`text-xs ${colors[status] || "text-dark-400"}`}>● {status}</span>;
}
