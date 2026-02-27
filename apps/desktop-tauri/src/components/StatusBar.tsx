import { useEffect, useState } from "react";
import { 
  GitBranch, 
  AlertCircle, 
  CheckCircle, 
  Cloud, 
  CloudOff,
  Cpu,
  HardDrive,
  MemoryStick,
  RefreshCw
} from "lucide-react";
import { useStore } from "../lib/store";
import { system, sync } from "../lib/tauri";

interface StatusBarProps {
  deviceId: string;
}

export function StatusBar({ deviceId }: StatusBarProps) {
  const [resources, setResources] = useState({
    cpu: 0,
    memory: 0,
    disk: 0,
  });
  
  const {
    currentWorkspace,
    gitState,
    syncStatus,
    trainingStatus,
    setSyncStatus,
  } = useStore();

  // Update resource usage periodically
  useEffect(() => {
    const updateResources = async () => {
      try {
        const usage = await system.getResourceUsage();
        setResources({
          cpu: Math.round(usage.cpu_percent),
          memory: Math.round(usage.memory_percent),
          disk: Math.round(usage.disk_percent),
        });
      } catch (e) {
        // Ignore errors
      }
    };

    updateResources();
    const interval = setInterval(updateResources, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleForceSync = async () => {
    try {
      await sync.forceSync();
      const status = await sync.getStatus();
      setSyncStatus({
        isEnabled: status.enabled,
        isConnected: status.is_connected,
        lastSync: status.last_sync,
        pendingCount: status.pending_count,
      });
    } catch (e) {
      console.error("Force sync failed:", e);
    }
  };

  return (
    <div className="h-7 bg-primary-700 flex items-center px-3 text-xs text-white">
      {/* Left Section */}
      <div className="flex items-center gap-4">
        {/* Git Branch */}
        {gitState && (
          <div className="flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5" />
            <span>{gitState.branch}</span>
            {gitState.ahead > 0 && <span className="text-green-300">↑{gitState.ahead}</span>}
            {gitState.behind > 0 && <span className="text-red-300">↓{gitState.behind}</span>}
            {!gitState.isClean && (
              <span className="text-orange-300">●{gitState.modified.length + gitState.added.length}</span>
            )}
          </div>
        )}

        {/* Errors/Warnings */}
        <div className="flex items-center gap-1">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>0</span>
        </div>
      </div>

      {/* Center - Sync Status */}
      <div className="flex-1 flex justify-center">
        {syncStatus.isEnabled && (
          <button
            onClick={handleForceSync}
            className="flex items-center gap-1.5 hover:bg-primary-600 px-2 py-0.5 rounded transition-colors"
          >
            {syncStatus.isConnected ? (
              <Cloud className="w-3.5 h-3.5" />
            ) : (
              <CloudOff className="w-3.5 h-3.5" />
            )}
            <span>
              {syncStatus.isConnected 
                ? syncStatus.lastSync 
                  ? `Synced ${formatTime(syncStatus.lastSync)}`
                  : "Ready"
                : "Offline"
              }
            </span>
            {syncStatus.pendingCount > 0 && (
              <span className="bg-orange-500 text-white px-1 rounded text-xs">
                {syncStatus.pendingCount}
              </span>
            )}
          </button>
        )}
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-4">
        {/* Training Status */}
        {trainingStatus.isEnabled && trainingStatus.currentJob && (
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" />
            <span>{trainingStatus.currentJob.progress.toFixed(0)}%</span>
          </div>
        )}

        {/* Resources */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1" title="CPU Usage">
            <Cpu className="w-3.5 h-3.5" />
            <span>{resources.cpu}%</span>
          </div>
          
          <div className="flex items-center gap-1" title="Memory Usage">
            <MemoryStick className="w-3.5 h-3.5" />
            <span>{resources.memory}%</span>
          </div>
          
          <div className="flex items-center gap-1" title="Disk Usage">
            <HardDrive className="w-3.5 h-3.5" />
            <span>{resources.disk}%</span>
          </div>
        </div>

        {/* Encoding */}
        <div className="text-primary-200">UTF-8</div>

        {/* Language */}
        <div className="text-primary-200">Plain Text</div>

        {/* Line/Column */}
        <div className="text-primary-200">Ln 1, Col 1</div>

        {/* Device ID */}
        <div className="text-primary-300" title={deviceId}>
          {deviceId.slice(0, 8)}...
        </div>
      </div>
    </div>
  );
}

function formatTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  
  if (diff < 60000) {
    return "just now";
  } else if (diff < 3600000) {
    return `${Math.floor(diff / 60000)}m ago`;
  } else {
    return `${Math.floor(diff / 3600000)}h ago`;
  }
}
