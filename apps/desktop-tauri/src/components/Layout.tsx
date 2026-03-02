//! Main Layout Component

import { lazy, Suspense, useState } from "react";
import { useStore } from "../lib/store";
import { Sidebar } from "./Sidebar";
import { StatusBar } from "./StatusBar";
import { Header } from "./Header";

const MonacoEditor = lazy(() =>
  import("./editor/MonacoEditor").then((module) => ({
    default: module.MonacoEditor,
  }))
);

const RealTerminal = lazy(() =>
  import("./terminal/RealTerminal").then((module) => ({
    default: module.RealTerminal,
  }))
);

interface LayoutProps {
  deviceId: string;
}

export function Layout({ deviceId }: LayoutProps) {
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [isResizingTerminal, setIsResizingTerminal] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const { sidebarVisible, terminalVisible, terminalHeight, setTerminalHeight } = useStore();

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
        {sidebarVisible && (
          <>
            <div className="h-full flex-shrink-0 overflow-hidden" style={{ width: sidebarWidth }}>
              <Sidebar />
            </div>
            <div
              className={`w-1 h-full cursor-col-resize transition-colors ${isResizingSidebar ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
                }`}
              onMouseDown={handleSidebarResize}
            />
          </>
        )}

        <div className="flex-1 flex flex-col min-w-0">
          {/* Editor Area */}
          <div className="flex-1 overflow-hidden">
            <Suspense fallback={<div className="h-full bg-dark-900" />}>
              <MonacoEditor />
            </Suspense>
          </div>

          {/* Terminal Panel */}
          {terminalVisible && (
            <>
              <div
                className={`h-1 cursor-row-resize transition-colors ${isResizingTerminal ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
                  }`}
                onMouseDown={handleTerminalResize}
              />
              <div className="flex-shrink-0 overflow-hidden" style={{ height: terminalHeight }}>
                <Suspense fallback={<div className="h-full bg-dark-900" />}>
                  <RealTerminal />
                </Suspense>
              </div>
            </>
          )}
        </div>
      </div>

      <StatusBar deviceId={deviceId} />
    </div>
  );
}
