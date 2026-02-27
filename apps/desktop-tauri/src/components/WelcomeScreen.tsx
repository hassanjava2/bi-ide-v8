import { useState } from "react";
import { FolderOpen, GitBranch, Settings, Sparkles } from "lucide-react";
import { useStore } from "../lib/store";
import { workspace, dialog } from "../lib/tauri";

interface WelcomeScreenProps {
  deviceId: string;
}

export function WelcomeScreen({ deviceId }: WelcomeScreenProps) {
  const [isOpening, setIsOpening] = useState(false);
  const [recentWorkspaces, setRecentWorkspaces] = useState<string[]>(() => {
    const saved = localStorage.getItem("bi-ide-recent-workspaces");
    return saved ? JSON.parse(saved) : [];
  });

  const { setCurrentWorkspace } = useStore();

  const handleOpenFolder = async () => {
    try {
      setIsOpening(true);
      
      // Open dialog to select folder
      const selected = await dialog.open({
        directory: true,
        multiple: false,
        title: "Select Workspace Folder",
      });

      if (selected && typeof selected === "string") {
        await openWorkspace(selected);
      }
    } catch (error) {
      console.error("Failed to open folder:", error);
    } finally {
      setIsOpening(false);
    }
  };

  const handleCloneRepo = async () => {
    // Would show clone dialog
    console.log("Clone repository");
  };

  const openWorkspace = async (path: string) => {
    try {
      const ws = await workspace.open(path);
      setCurrentWorkspace({
        id: ws.id,
        path: ws.path,
        name: ws.name,
      });

      // Add to recent
      const newRecent = [path, ...recentWorkspaces.filter((p) => p !== path)].slice(0, 5);
      setRecentWorkspaces(newRecent);
      localStorage.setItem("bi-ide-recent-workspaces", JSON.stringify(newRecent));
    } catch (error) {
      console.error("Failed to open workspace:", error);
    }
  };

  const formatPath = (path: string) => {
    const parts = path.split("/");
    return parts.slice(-2).join("/");
  };

  return (
    <div className="h-full flex flex-col items-center justify-center p-8 bg-gradient-to-br from-dark-900 to-dark-950">
      {/* Logo and Title */}
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="w-12 h-12 bg-primary-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white">BI-IDE Desktop</h1>
        </div>
        <p className="text-dark-400">AI-Powered Development Environment</p>
        <p className="text-dark-500 text-sm mt-2">Device: {deviceId.slice(0, 8)}...</p>
      </div>

      {/* Actions */}
      <div className="grid grid-cols-2 gap-4 max-w-2xl w-full mb-8">
        <button
          onClick={handleOpenFolder}
          disabled={isOpening}
          className="flex items-center gap-4 p-6 bg-dark-800 hover:bg-dark-700 rounded-xl transition-colors text-left group"
        >
          <div className="w-12 h-12 bg-primary-900/50 rounded-lg flex items-center justify-center group-hover:bg-primary-900">
            <FolderOpen className="w-6 h-6 text-primary-400" />
          </div>
          <div>
            <div className="font-semibold text-white">Open Folder</div>
            <div className="text-sm text-dark-400">Open an existing project</div>
          </div>
        </button>

        <button
          onClick={handleCloneRepo}
          className="flex items-center gap-4 p-6 bg-dark-800 hover:bg-dark-700 rounded-xl transition-colors text-left group"
        >
          <div className="w-12 h-12 bg-purple-900/50 rounded-lg flex items-center justify-center group-hover:bg-purple-900">
            <GitBranch className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <div className="font-semibold text-white">Clone Repository</div>
            <div className="text-sm text-dark-400">Clone from Git URL</div>
          </div>
        </button>
      </div>

      {/* Recent Workspaces */}
      {recentWorkspaces.length > 0 && (
        <div className="max-w-2xl w-full">
          <h3 className="text-sm font-medium text-dark-400 mb-3">Recent</h3>
          <div className="space-y-2">
            {recentWorkspaces.map((path) => (
              <button
                key={path}
                onClick={() => openWorkspace(path)}
                className="w-full flex items-center gap-3 p-3 bg-dark-800/50 hover:bg-dark-800 rounded-lg transition-colors text-left"
              >
                <FolderOpen className="w-5 h-5 text-dark-500" />
                <span className="text-dark-300">{formatPath(path)}</span>
                <span className="text-dark-500 text-sm ml-auto">{path}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Features */}
      <div className="mt-12 flex gap-8 text-center">
        <div>
          <div className="w-10 h-10 bg-green-900/30 rounded-lg flex items-center justify-center mx-auto mb-2">
            <Sparkles className="w-5 h-5 text-green-400" />
          </div>
          <div className="text-sm text-dark-400">AI Assistant</div>
        </div>
        <div>
          <div className="w-10 h-10 bg-blue-900/30 rounded-lg flex items-center justify-center mx-auto mb-2">
            <GitBranch className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-sm text-dark-400">Git Integration</div>
        </div>
        <div>
          <div className="w-10 h-10 bg-orange-900/30 rounded-lg flex items-center justify-center mx-auto mb-2">
            <Settings className="w-5 h-5 text-orange-400" />
          </div>
          <div className="text-sm text-dark-400">Local Training</div>
        </div>
      </div>
    </div>
  );
}
