import { 
  Menu, 
  Sidebar, 
  Terminal, 
  RefreshCw, 
  Settings,
  Minus,
  Square,
  X,
  Maximize2
} from "lucide-react";
import { useStore } from "../lib/store";
import { getCurrentWindow } from "@tauri-apps/api/window";

interface HeaderProps {
  deviceId: string;
}

export function Header({ deviceId }: HeaderProps) {
  const { 
    currentWorkspace, 
    sidebarVisible, 
    terminalVisible, 
    toggleSidebar, 
    toggleTerminal,
    gitState,
    syncStatus,
  } = useStore();

  const handleMinimize = async () => {
    await getCurrentWindow().minimize();
  };

  const handleMaximize = async () => {
    const window = getCurrentWindow();
    if (await window.isMaximized()) {
      await window.unmaximize();
    } else {
      await window.maximize();
    }
  };

  const handleClose = async () => {
    await getCurrentWindow().hide();
  };

  return (
    <div className="h-12 bg-dark-800 border-b border-dark-700 flex items-center drag-region">
      {/* Left Section */}
      <div className="flex items-center gap-2 px-3 no-drag">
        <div className="w-6 h-6 bg-primary-600 rounded flex items-center justify-center">
          <span className="text-xs font-bold text-white">BI</span>
        </div>
        
        <button className="p-1.5 hover:bg-dark-700 rounded transition-colors">
          <Menu className="w-4 h-4 text-dark-300" />
        </button>

        <div className="h-4 w-px bg-dark-600 mx-1" />

        <button 
          onClick={toggleSidebar}
          className={`p-1.5 rounded transition-colors ${sidebarVisible ? "bg-dark-700 text-primary-400" : "hover:bg-dark-700 text-dark-400"}`}
          title="Toggle Sidebar"
        >
          <Sidebar className="w-4 h-4" />
        </button>

        <button 
          onClick={toggleTerminal}
          className={`p-1.5 rounded transition-colors ${terminalVisible ? "bg-dark-700 text-primary-400" : "hover:bg-dark-700 text-dark-400"}`}
          title="Toggle Terminal"
        >
          <Terminal className="w-4 h-4" />
        </button>

        <button className="p-1.5 hover:bg-dark-700 rounded transition-colors text-dark-400">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Center - Title */}
      <div className="flex-1 text-center">
        <div className="text-sm text-dark-200 font-medium">
          {currentWorkspace?.name || "BI-IDE Desktop"}
        </div>
        {gitState && (
          <div className="text-xs text-dark-500 flex items-center justify-center gap-2">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-orange-400" />
              {gitState.branch}
            </span>
            {gitState.ahead > 0 && <span>↑{gitState.ahead}</span>}
            {gitState.behind > 0 && <span>↓{gitState.behind}</span>}
            {!gitState.isClean && <span className="text-orange-400">●</span>}
          </div>
        )}
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-1 px-3 no-drag">
        {/* Sync Status */}
        {syncStatus.isEnabled && (
          <div className="flex items-center gap-2 mr-4">
            <div className={`w-2 h-2 rounded-full ${
              syncStatus.isConnected ? "status-online" : "status-offline"
            }`} />
            <span className="text-xs text-dark-400">
              {syncStatus.isConnected ? "Synced" : "Offline"}
            </span>
          </div>
        )}

        <button className="p-1.5 hover:bg-dark-700 rounded transition-colors text-dark-400">
          <Settings className="w-4 h-4" />
        </button>

        <div className="h-4 w-px bg-dark-600 mx-1" />

        {/* Window Controls */}
        <button 
          onClick={handleMinimize}
          className="p-1.5 hover:bg-dark-700 rounded transition-colors text-dark-400"
        >
          <Minus className="w-4 h-4" />
        </button>
        
        <button 
          onClick={handleMaximize}
          className="p-1.5 hover:bg-dark-700 rounded transition-colors text-dark-400"
        >
          <Square className="w-4 h-4" />
        </button>
        
        <button 
          onClick={handleClose}
          className="p-1.5 hover:bg-red-600 rounded transition-colors text-dark-400 hover:text-white"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
