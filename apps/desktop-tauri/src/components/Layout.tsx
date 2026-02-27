import { useState } from "react";
import { useStore } from "../lib/store";
import { Sidebar } from "./Sidebar";
import { Editor } from "./Editor";
import { Terminal } from "./Terminal";
import { StatusBar } from "./StatusBar";
import { Header } from "./Header";

interface LayoutProps {
  deviceId: string;
}

export function Layout({ deviceId }: LayoutProps) {
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [isResizingTerminal, setIsResizingTerminal] = useState(false);
  
  const {
    sidebarVisible,
    sidebarWidth,
    terminalVisible,
    terminalHeight,
    setSidebarWidth,
    setTerminalHeight,
  } = useStore();

  // Handle sidebar resize
  const handleSidebarResize = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingSidebar(true);

    const startX = e.clientX;
    const startWidth = sidebarWidth;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.max(200, Math.min(500, startWidth + e.clientX - startX));
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizingSidebar(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  // Handle terminal resize
  const handleTerminalResize = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingTerminal(true);

    const startY = e.clientY;
    const startHeight = terminalHeight;

    const handleMouseMove = (e: MouseEvent) => {
      const newHeight = Math.max(100, Math.min(600, startHeight - (e.clientY - startY)));
      setTerminalHeight(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizingTerminal(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  return (
    <div className="h-full flex flex-col">
      <Header deviceId={deviceId} />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {sidebarVisible && (
          <>
            <div
              className="h-full flex-shrink-0 overflow-hidden"
              style={{ width: sidebarWidth }}
            >
              <Sidebar />
            </div>
            
            {/* Resize Handle */}
            <div
              className={`w-1 h-full cursor-col-resize transition-colors ${
                isResizingSidebar ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
              }`}
              onMouseDown={handleSidebarResize}
            />
          </>
        )}

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Editor */}
          <div className="flex-1 min-h-0">
            <Editor />
          </div>

          {/* Terminal */}
          {terminalVisible && (
            <>
              {/* Terminal Resize Handle */}
              <div
                className={`h-1 w-full cursor-row-resize transition-colors ${
                  isResizingTerminal ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
                }`}
                onMouseDown={handleTerminalResize}
              />
              
              <div
                className="flex-shrink-0 overflow-hidden"
                style={{ height: terminalHeight }}
              >
                <Terminal />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar deviceId={deviceId} />
    </div>
  );
}
