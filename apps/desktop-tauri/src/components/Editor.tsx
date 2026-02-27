import { useEffect, useRef, useCallback } from "react";
import { X } from "lucide-react";
import { useStore } from "../lib/store";
import { fs } from "../lib/tauri";
import { debounce } from "../lib/utils";

export function Editor() {
  const editorRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const {
    openFiles,
    activeFilePath,
    closeFile,
    setActiveFile,
    updateFileContent,
    markFileSaved,
    settings,
  } = useStore();

  const activeFile = openFiles.find((f) => f.path === activeFilePath);

  // Auto-save with debounce
  const debouncedSave = useCallback(
    debounce(async (path: string, content: string) => {
      try {
        await fs.writeFile(path, content);
        markFileSaved(path);
      } catch (e) {
        console.error("Auto-save failed:", e);
      }
    }, settings.autoSaveDelay),
    [settings.autoSaveDelay, markFileSaved]
  );

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (!activeFile) return;
    
    const newContent = e.target.value;
    updateFileContent(activeFile.path, newContent);
    
    if (settings.autoSave) {
      debouncedSave(activeFile.path, newContent);
    }
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      // Ctrl/Cmd + S - Save
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (activeFile && activeFile.isDirty) {
          try {
            await fs.writeFile(activeFile.path, activeFile.content);
            markFileSaved(activeFile.path);
          } catch (err) {
            console.error("Save failed:", err);
          }
        }
      }

      // Ctrl/Cmd + W - Close tab
      if ((e.ctrlKey || e.metaKey) && e.key === "w") {
        e.preventDefault();
        if (activeFilePath) {
          closeFile(activeFilePath);
        }
      }

      // Ctrl/Cmd + Tab - Next tab
      if ((e.ctrlKey || e.metaKey) && e.key === "Tab") {
        e.preventDefault();
        if (openFiles.length > 1) {
          const currentIndex = openFiles.findIndex((f) => f.path === activeFilePath);
          const nextIndex = (currentIndex + 1) % openFiles.length;
          setActiveFile(openFiles[nextIndex].path);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeFile, activeFilePath, openFiles, closeFile, setActiveFile, markFileSaved]);

  // Focus textarea when file changes
  useEffect(() => {
    if (textareaRef.current && activeFile) {
      textareaRef.current.focus();
    }
  }, [activeFilePath]);

  if (openFiles.length === 0) {
    return (
      <div className="h-full flex items-center justify-center bg-dark-900">
        <div className="text-center">
          <div className="text-4xl mb-4">📁</div>
          <div className="text-dark-400 text-sm">Select a file to start editing</div>
          <div className="text-dark-500 text-xs mt-2">
            Or press Ctrl+O to open a folder
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* Tabs */}
      <div className="flex bg-dark-800 border-b border-dark-700 overflow-x-auto">
        {openFiles.map((file) => (
          <div
            key={file.path}
            onClick={() => setActiveFile(file.path)}
            className={`tab group ${activeFilePath === file.path ? "active" : ""} ${
              file.isDirty ? "modified" : ""
            }`}
          >
            <span className="truncate max-w-[150px]">
              {file.path.split("/").pop()}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                closeFile(file.path);
              }}
              className="ml-2 p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-dark-600"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* Editor Content */}
      <div className="flex-1 relative overflow-hidden">
        {activeFile ? (
          <div className="h-full flex">
            {/* Line Numbers */}
            <div className="w-12 bg-dark-800 border-r border-dark-700 flex-shrink-0 py-4 text-right select-none">
              {activeFile.content.split("\n").map((_, i) => (
                <div
                  key={i}
                  className="px-2 text-xs text-dark-500 leading-6"
                >
                  {i + 1}
                </div>
              ))}
            </div>

            {/* Text Area */}
            <textarea
              ref={textareaRef}
              value={activeFile.content}
              onChange={handleContentChange}
              className="flex-1 bg-dark-900 text-dark-100 p-4 resize-none outline-none font-mono text-sm leading-6"
              style={{
                fontSize: settings.fontSize,
                fontFamily: settings.fontFamily,
                whiteSpace: settings.wordWrap ? "pre-wrap" : "pre",
                overflowWrap: settings.wordWrap ? "break-word" : "normal",
                tabSize: 2,
              }}
              spellCheck={false}
            />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-dark-500">Select a tab to view</div>
          </div>
        )}
      </div>
    </div>
  );
}
