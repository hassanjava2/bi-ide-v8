//! File Tab Component for Monaco Editor

import { X, FileCode, FileJson, FileType, File } from "lucide-react";
import { memo } from "react";
import { OpenFile } from "../../types/files";

interface FileInfoProps {
  file: OpenFile;
  isActive: boolean;
  onClick: () => void;
  onClose: (e: React.MouseEvent) => void;
}

// File icon based on extension
function getFileIcon(path: string) {
  const ext = path.split(".").pop()?.toLowerCase() || "";
  
  // Code files
  if (["js", "jsx", "ts", "tsx", "vue", "svelte"].includes(ext)) {
    return <FileCode className="w-3.5 h-3.5 text-yellow-400" />;
  }
  // Config files
  if (["json", "yaml", "yml", "toml", "xml"].includes(ext)) {
    return <FileJson className="w-3.5 h-3.5 text-orange-400" />;
  }
  // Style files
  if (["css", "scss", "sass", "less", "styl"].includes(ext)) {
    return <FileType className="w-3.5 h-3.5 text-blue-400" />;
  }
  // Rust
  if (ext === "rs") {
    return <FileCode className="w-3.5 h-3.5 text-orange-500" />;
  }
  // Python
  if (ext === "py") {
    return <FileCode className="w-3.5 h-3.5 text-green-400" />;
  }
  // Go
  if (ext === "go") {
    return <FileCode className="w-3.5 h-3.5 text-cyan-400" />;
  }
  // Markdown
  if (["md", "mdx"].includes(ext)) {
    return <FileType className="w-3.5 h-3.5 text-gray-400" />;
  }
  
  return <File className="w-3.5 h-3.5 text-dark-400" />;
}

export const FileInfo = memo(function FileInfo({
  file,
  isActive,
  onClick,
  onClose,
}: FileInfoProps) {
  const filename = file.path.split("/").pop() || file.path;
  const fileIcon = getFileIcon(file.path);

  return (
    <div
      onClick={onClick}
      className={`
        group flex items-center gap-2 px-3 py-2 min-w-[120px] max-w-[200px] 
        cursor-pointer select-none text-sm border-r border-[#1e1e1e]
        transition-colors duration-150
        ${
          isActive
            ? "bg-[#1e1e1e] text-white"
            : "bg-[#2d2d2d] text-dark-300 hover:bg-[#2a2d2e]"
        }
      `}
    >
      {/* File Icon */}
      {fileIcon}

      {/* Filename */}
      <span className="truncate flex-1">{filename}</span>

      {/* Modified Indicator */}
      {file.isDirty && (
        <span className="w-2 h-2 rounded-full bg-primary-500" title="Unsaved changes" />
      )}

      {/* Close Button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClose(e);
        }}
        className={`
          p-0.5 rounded opacity-0 group-hover:opacity-100 
          hover:bg-dark-600 transition-opacity
          ${isActive ? "opacity-100" : ""}
        `}
        title="Close"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
});
