import { lazy, Suspense, useEffect, useState } from "react";
import { useStore } from "./lib/store";
import { system, workspace, sync, training } from "./lib/tauri";
import { Layout } from "./components/Layout";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { listen } from "@tauri-apps/api/event";

const CommandPalette = lazy(() =>
  import("./components/editor/CommandPalette").then((module) => ({
    default: module.CommandPalette,
  }))
);

const QuickOpen = lazy(() =>
  import("./components/editor/QuickOpen").then((module) => ({
    default: module.QuickOpen,
  }))
);

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [deviceId, setDeviceId] = useState<string>("");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [quickOpenOpen, setQuickOpenOpen] = useState(false);

  const {
    currentWorkspace,
    setCurrentWorkspace,
    setSyncStatus,
    setTrainingStatus,
    setDeviceId: setStoreDeviceId,
  } = useStore();

  // Initialize app
  useEffect(() => {
    const init = async () => {
      try {
        // Get device info
        const info = await system.getInfo();
        setDeviceId(info.device_id);
        setStoreDeviceId(info.device_id);
        console.log("BI-IDE Desktop v" + info.app_version);
        console.log("Device ID:", info.device_id);
        console.log("Platform:", info.platform, info.arch);
      } catch (error) {
        console.error("System info error (non-fatal):", error);
        setDeviceId("local-" + Date.now());
        setStoreDeviceId("local-" + Date.now());
      }

      // Check for saved workspace (separate try/catch)
      try {
        const savedWorkspace = localStorage.getItem("bi-ide-last-workspace");
        if (savedWorkspace) {
          const ws = await workspace.open(savedWorkspace);
          setCurrentWorkspace({
            id: ws.id,
            path: ws.path,
            name: ws.name,
          });
        }
      } catch (e) {
        console.error("Workspace restore error (non-fatal):", e);
      }

      // Get sync status (separate try/catch)
      try {
        const syncStatus = await sync.getStatus();
        setSyncStatus({
          isEnabled: syncStatus.enabled,
          isConnected: syncStatus.is_connected,
          lastSync: syncStatus.last_sync,
          pendingCount: syncStatus.pending_count,
        });
      } catch (e) {
        console.error("Sync status error (non-fatal):", e);
      }

      // Get training status (separate try/catch)
      try {
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
      } catch (e) {
        console.error("Training status error (non-fatal):", e);
      }

      // Always finish loading
      setIsLoading(false);
    };

    init();
  }, [setCurrentWorkspace, setSyncStatus, setTrainingStatus, setStoreDeviceId]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd+Shift+P → Command Palette
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "p") {
        e.preventDefault();
        setCommandPaletteOpen(prev => !prev);
        setQuickOpenOpen(false);
        return;
      }
      // Ctrl/Cmd+P → Quick Open (not in inputs)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p" && !e.shiftKey) {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
        e.preventDefault();
        setQuickOpenOpen(prev => !prev);
        setCommandPaletteOpen(false);
        return;
      }
      // Escape closes modals
      if (e.key === "Escape") {
        setCommandPaletteOpen(false);
        setQuickOpenOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Listen for Tauri events
  useEffect(() => {
    const unlistenResource = listen("resource-usage", (event) => {
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
    });

    return () => {
      unlistenResource.then((fn) => fn());
      unlistenSync.then((fn) => fn());
      unlistenFileChange.then((fn) => fn());
    };
  }, [setSyncStatus]);

  // Listen for CommandPalette events
  useEffect(() => {
    // Workspace operations
    const unlistenWorkspaceOpen = listen("workspace-opened", (event: any) => {
      const ws = event.payload;
      if (ws) {
        setCurrentWorkspace({
          id: ws.id,
          path: ws.path,
          name: ws.name,
        });
      }
    });

    // Quick Open
    const unlistenQuickOpen = listen("open-quick-open", () => {
      setQuickOpenOpen(true);
      setCommandPaletteOpen(false);
    });

    return () => {
      unlistenWorkspaceOpen.then((fn) => fn());
      unlistenQuickOpen.then((fn) => fn());
    };
  }, [setCurrentWorkspace]);

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

      <Suspense fallback={null}>
        <CommandPalette isOpen={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
      </Suspense>
      <Suspense fallback={null}>
        <QuickOpen isOpen={quickOpenOpen} onClose={() => setQuickOpenOpen(false)} />
      </Suspense>
    </div>
  );
}

export default App;
