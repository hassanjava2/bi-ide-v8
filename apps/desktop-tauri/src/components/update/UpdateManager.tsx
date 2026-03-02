//! Auto-Update Manager

import { useState, useEffect } from "react";
import { Download, Check, AlertCircle, RefreshCw } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

interface UpdateManifest {
  version: string;
  critical: boolean;
  size_mb: number;
}

export function UpdateManager() {
  const [hasUpdate, setHasUpdate] = useState(false);
  const [manifest, setManifest] = useState<UpdateManifest | null>(null);
  const [loading, setLoading] = useState(false);

  const checkForUpdates = async () => {
    setLoading(true);
    try {
      const result = await invoke<{has_update: boolean; manifest?: UpdateManifest}>("check_for_updates", {
        request: { current_version: "1.0.0", channel: "stable" },
      });
      setHasUpdate(result.has_update);
      setManifest(result.manifest || null);
    } catch (err) {
      console.error("Update check failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkForUpdates();
  }, []);

  if (!hasUpdate || !manifest) {
    return (
      <div className="p-4">
        <button
          onClick={checkForUpdates}
          disabled={loading}
          className="flex items-center gap-2 text-sm text-dark-400 hover:text-dark-200"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Check for updates
        </button>
      </div>
    );
  }

  return (
    <div className="p-4 bg-primary-900/20 border border-primary-700 rounded m-4">
      <div className="flex items-start gap-3">
        <Download className="w-5 h-5 text-primary-400" />
        <div className="flex-1">
          <h3 className="font-medium">Update Available</h3>
          <p className="text-sm text-dark-400">Version {manifest.version}</p>
          {manifest.critical && (
            <div className="flex items-center gap-1.5 mt-1 text-xs text-red-400">
              <AlertCircle className="w-3.5 h-3.5" />
              Critical update
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
