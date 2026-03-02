//! Breadcrumbs Component for Monaco Editor

import { ChevronRight, Folder, FileCode } from "lucide-react";
import { memo, useMemo } from "react";

interface BreadcrumbsProps {
  path: string;
  workspace?: string;
}

export const Breadcrumbs = memo(function Breadcrumbs({ path, workspace }: BreadcrumbsProps) {
  const segments = useMemo(() => {
    // Remove workspace prefix if present
    let relativePath = path;
    if (workspace && path.startsWith(workspace)) {
      relativePath = path.slice(workspace.length).replace(/^\/+/, "");
    }
    
    return relativePath.split("/").filter(Boolean);
  }, [path, workspace]);

  const workspaceName = useMemo(() => {
    if (!workspace) return "Workspace";
    return workspace.split("/").pop() || "Workspace";
  }, [workspace]);

  if (segments.length === 0) {
    return (
      <div className="flex items-center gap-1 px-3 py-1.5 bg-[#1e1e1e] border-b border-[#333] text-xs">
        <Folder className="w-3.5 h-3.5 text-primary-400" />
        <span className="text-dark-300">{workspaceName}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 px-3 py-1.5 bg-[#1e1e1e] border-b border-[#333] text-xs overflow-x-auto">
      <Folder className="w-3.5 h-3.5 text-primary-400 flex-shrink-0" />
      <span className="text-dark-400 flex-shrink-0">{workspaceName}</span>
      
      {segments.map((segment, index) => {
        const isLast = index === segments.length - 1;
        
        return (
          <div key={index} className="flex items-center gap-1 flex-shrink-0">
            <ChevronRight className="w-3 h-3 text-dark-500" />
            {isLast ? (
              <span className="flex items-center gap-1 text-dark-100 font-medium">
                <FileCode className="w-3.5 h-3.5 text-primary-400" />
                {segment}
              </span>
            ) : (
              <span className="text-dark-400 hover:text-dark-200 cursor-pointer transition-colors">
                {segment}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
});
