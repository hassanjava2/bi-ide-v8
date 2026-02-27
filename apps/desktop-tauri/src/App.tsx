import { useEffect, useState } from "react";
import { useStore } from "./lib/store";
import { system, workspace, sync, training } from "./lib/tauri";
import { Layout } from "./components/Layout";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { listen } from "@tauri-apps/api/event";

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [deviceId, setDeviceId] = useState<string>("");
  
  const {
    currentWorkspace,
    setCurrentWorkspace,
    setSyncStatus,
    setTrainingStatus,
    updateSettings,
  } = useStore();

  // Initialize app
  useEffect(() => {
    const init = async () => {
      try {
        // Get device info
        const info = await system.getInfo();
        setDeviceId(info.device_id);
        console.log("BI-IDE Desktop v" + info.app_version);
        console.log("Device ID:", info.device_id);
        console.log("Platform:", info.platform, info.arch);

        // Check for saved workspace
        const savedWorkspace = localStorage.getItem("bi-ide-last-workspace");
        if (savedWorkspace) {
          try {
            const ws = await workspace.open(savedWorkspace);
            setCurrentWorkspace({
              id: ws.id,
              path: ws.path,
              name: ws.name,
            });
          } catch (e) {
            console.error("Failed to restore workspace:", e);
          }
        }

        // Get sync status
        const syncStatus = await sync.getStatus();
        setSyncStatus({
          isEnabled: syncStatus.enabled,
          isConnected: syncStatus.is_connected,
          lastSync: syncStatus.last_sync,
          pendingCount: syncStatus.pending_count,
        });

        // Get training status
        const trainingStatus = await training.getStatus();
        setTrainingStatus({
          isEnabled: trainingStatus.enabled,
          currentJob: trainingStatus.current_job
            ? {
                id: trainingStatus.current_job.job_id,
                type: trainingStatus.current_job.job_type,
                progress: trainingStatus.current_job.progress_percent,
                status: trainingStatus.current_job.status,
              }
            : undefined,
        });
      } catch (error) {
        console.error("Initialization error:", error);
      } finally {
        setIsLoading(false);
      }
    };

    init();
  }, [setCurrentWorkspace, setSyncStatus, setTrainingStatus]);

  // Listen for Tauri events
  useEffect(() => {
    const unlistenResource = listen("resource-usage", (event) => {
      // Could update a resource usage display
      console.log("Resource usage:", event.payload);
    });

    const unlistenSync = listen("sync-complete", (event) => {
      console.log("Sync complete:", event.payload);
      sync.getStatus().then((status) => {
        setSyncStatus({
          isEnabled: status.enabled,
          isConnected: status.is_connected,
          lastSync: status.last_sync,
          pendingCount: status.pending_count,
        });
      });
    });

    const unlistenFileChange = listen("file-changed", (event) => {
      console.log("File changed:", event.payload);
      // Could refresh file tree
    });

    return () => {
      unlistenResource.then((fn) => fn());
      unlistenSync.then((fn) => fn());
      unlistenFileChange.then((fn) => fn());
    };
  }, [setSyncStatus]);

  // Auto-save workspace
  useEffect(() => {
    if (currentWorkspace) {
      localStorage.setItem("bi-ide-last-workspace", currentWorkspace.path);
    }
  }, [currentWorkspace]);

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-dark-900">
        <div className="text-center">
          <div className="text-2xl font-bold text-primary-400 mb-2">BI-IDE Desktop</div>
          <div className="text-dark-400">Initializing...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-dark-900 text-dark-100 overflow-hidden">
      {currentWorkspace ? (
        <Layout deviceId={deviceId} />
      ) : (
        <WelcomeScreen deviceId={deviceId} />
      )}
    </div>
  );
}

export default App;
