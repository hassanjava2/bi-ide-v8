//! Main Layout Component

import { useState, useEffect } from "react";
import { useStore } from "../lib/store";
import { Sidebar } from "./Sidebar";
import { StatusBar } from "./StatusBar";
import { Header } from "./Header";

interface LayoutProps {
  deviceId: string;
}

export function Layout({ deviceId }: LayoutProps) {
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const { sidebarVisible } = useStore();

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
              className={`w-1 h-full cursor-col-resize transition-colors ${
                isResizingSidebar ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
              }`}
              onMouseDown={handleSidebarResize}
            />
          </>
        )}

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 flex items-center justify-center bg-dark-900">
            <div className="text-center text-dark-500">
              <div className="text-4xl mb-4">📁</div>
              <p>Select a file to start editing</p>
            </div>
          </div>
        </div>
      </div>

      <StatusBar deviceId={deviceId} />
    </div>
  );
}
