//! Command Palette Component (Cmd+Shift+P)
//! Access all IDE commands from a searchable interface

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { 
  Search, X, FolderOpen, Save, GitBranch, Settings, 
  Terminal, Moon, Sun, RefreshCw, LogOut, Keyboard,
  FilePlus, FolderPlus, Copy, Scissors, Clipboard,
  Search as SearchIcon, Replace, Command, Sparkles,
  Cpu, Zap, Eye, EyeOff, Layout, Maximize, Minimize
} from "lucide-react";
import { useStore } from "../../lib/store";
import { fs, workspace } from "../../lib/tauri";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { emit } from "@tauri-apps/api/event";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

interface CommandItem {
  id: string;
  title: string;
  description?: string;
  icon: React.ReactNode;
  shortcut?: string;
  category: string;
  action: () => void;
  disabled?: boolean;
}

// Fuzzy match function
function fuzzyMatch(pattern: string, str: string): number {
  pattern = pattern.toLowerCase();
  str = str.toLowerCase();
  
  if (pattern.length === 0) return 1;
  if (str.length === 0) return 0;
  
  let patternIdx = 0;
  let strIdx = 0;
  let score = 0;
  let consecutive = 0;
  
  while (patternIdx < pattern.length && strIdx < str.length) {
    if (pattern[patternIdx] === str[strIdx]) {
      score += 1 + consecutive;
      consecutive++;
      patternIdx++;
    } else {
      consecutive = 0;
    }
    strIdx++;
  }
  
  if (patternIdx < pattern.length) return 0;
  
  return score / (pattern.length * 2 + str.length * 0.1);
}

export function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const { 
    currentWorkspace, 
    settings, 
    updateSettings,
    sidebarVisible,
    terminalVisible,
    toggleSidebar,
    toggleTerminal,
  } = useStore();

  // Define all available commands
  const commands: CommandItem[] = useMemo(() => [
    // File Commands
    {
      id: "file.openFolder",
      title: "File: Open Folder",
      description: "Open a workspace folder",
      icon: <FolderOpen className="w-4 h-4" />,
      shortcut: "Ctrl+K Ctrl+O",
      category: "File",
      action: async () => {
        try {
          const selected = await openDialog({
            directory: true,
            multiple: false,
            title: "Open Workspace Folder"
          });
          if (selected && typeof selected === "string") {
            const ws = await workspace.open(selected);
            // Emit event to notify App.tsx to update workspace
            await emit("workspace-opened", ws);
          }
        } catch (err) {
          console.error("Failed to open folder:", err);
        }
        onClose();
      },
    },
    {
      id: "file.newFile",
      title: "File: New File",
      description: "Create a new file in the current workspace",
      icon: <FilePlus className="w-4 h-4" />,
      shortcut: "Ctrl+N",
      category: "File",
      disabled: !currentWorkspace,
      action: async () => {
        if (!currentWorkspace) return;
        try {
          // Emit event to show new file input in sidebar
          await emit("new-file-requested", { workspaceId: currentWorkspace.id });
        } catch (err) {
          console.error("Failed to create new file:", err);
        }
        onClose();
      },
    },
    {
      id: "file.newFolder",
      title: "File: New Folder",
      description: "Create a new folder in the current workspace",
      icon: <FolderPlus className="w-4 h-4" />,
      shortcut: "Ctrl+Shift+N",
      category: "File",
      disabled: !currentWorkspace,
      action: async () => {
        if (!currentWorkspace) return;
        try {
          await emit("new-folder-requested", { workspaceId: currentWorkspace.id });
        } catch (err) {
          console.error("Failed to create new folder:", err);
        }
        onClose();
      },
    },
    {
      id: "file.save",
      title: "File: Save",
      description: "Save the current file",
      icon: <Save className="w-4 h-4" />,
      shortcut: "Ctrl+S",
      category: "File",
      action: async () => {
        try {
          await emit("save-active-file", {});
        } catch (err) {
          console.error("Failed to save file:", err);
        }
        onClose();
      },
    },
    {
      id: "file.saveAll",
      title: "File: Save All",
      description: "Save all open files",
      icon: <Save className="w-4 h-4" />,
      shortcut: "Ctrl+K S",
      category: "File",
      action: async () => {
        try {
          await emit("save-all-files", {});
        } catch (err) {
          console.error("Failed to save all files:", err);
        }
        onClose();
      },
    },
    
    // Edit Commands - These will be handled by Monaco Editor
    {
      id: "edit.undo",
      title: "Edit: Undo",
      icon: <RefreshCw className="w-4 h-4" />,
      shortcut: "Ctrl+Z",
      category: "Edit",
      action: async () => {
        try {
          await emit("editor-undo", {});
        } catch (err) {
          console.error("Failed to undo:", err);
        }
        onClose();
      },
    },
    {
      id: "edit.redo",
      title: "Edit: Redo",
      icon: <RefreshCw className="w-4 h-4" />,
      shortcut: "Ctrl+Shift+Z",
      category: "Edit",
      action: async () => {
        try {
          await emit("editor-redo", {});
        } catch (err) {
          console.error("Failed to redo:", err);
        }
        onClose();
      },
    },
    {
      id: "edit.cut",
      title: "Edit: Cut",
      icon: <Scissors className="w-4 h-4" />,
      shortcut: "Ctrl+X",
      category: "Edit",
      action: async () => {
        try {
          await emit("editor-cut", {});
        } catch (err) {
          console.error("Failed to cut:", err);
        }
        onClose();
      },
    },
    {
      id: "edit.copy",
      title: "Edit: Copy",
      icon: <Copy className="w-4 h-4" />,
      shortcut: "Ctrl+C",
      category: "Edit",
      action: async () => {
        try {
          await emit("editor-copy", {});
        } catch (err) {
          console.error("Failed to copy:", err);
        }
        onClose();
      },
    },
    {
      id: "edit.paste",
      title: "Edit: Paste",
      icon: <Clipboard className="w-4 h-4" />,
      shortcut: "Ctrl+V",
      category: "Edit",
      action: async () => {
        try {
          await emit("editor-paste", {});
        } catch (err) {
          console.error("Failed to paste:", err);
        }
        onClose();
      },
    },
    
    // Search Commands
    {
      id: "search.find",
      title: "Search: Find in File",
      icon: <SearchIcon className="w-4 h-4" />,
      shortcut: "Ctrl+F",
      category: "Search",
      action: async () => {
        try {
          await emit("editor-find", {});
        } catch (err) {
          console.error("Failed to open find:", err);
        }
        onClose();
      },
    },
    {
      id: "search.replace",
      title: "Search: Replace in File",
      icon: <Replace className="w-4 h-4" />,
      shortcut: "Ctrl+H",
      category: "Search",
      action: async () => {
        try {
          await emit("editor-replace", {});
        } catch (err) {
          console.error("Failed to open replace:", err);
        }
        onClose();
      },
    },
    {
      id: "search.global",
      title: "Search: Find in Files",
      description: "Search across all files in workspace",
      icon: <SearchIcon className="w-4 h-4" />,
      shortcut: "Ctrl+Shift+F",
      category: "Search",
      action: async () => {
        try {
          await emit("open-global-search", {});
        } catch (err) {
          console.error("Failed to open global search:", err);
        }
        onClose();
      },
    },
    {
      id: "search.quickOpen",
      title: "Search: Quick Open",
      description: "Quickly navigate to a file",
      icon: <FolderOpen className="w-4 h-4" />,
      shortcut: "Ctrl+P",
      category: "Search",
      action: async () => {
        try {
          await emit("open-quick-open", {});
        } catch (err) {
          console.error("Failed to open quick open:", err);
        }
        onClose();
      },
    },
    
    // View Commands
    {
      id: "view.toggleSidebar",
      title: `View: ${sidebarVisible ? "Hide" : "Show"} Sidebar`,
      icon: <Layout className="w-4 h-4" />,
      shortcut: "Ctrl+B",
      category: "View",
      action: () => {
        toggleSidebar();
        onClose();
      },
    },
    {
      id: "view.toggleTerminal",
      title: `View: ${terminalVisible ? "Hide" : "Show"} Terminal`,
      icon: <Terminal className="w-4 h-4" />,
      shortcut: "Ctrl+`",
      category: "View",
      action: () => {
        toggleTerminal();
        onClose();
      },
    },
    {
      id: "view.toggleMinimap",
      title: `View: ${settings.minimapEnabled ? "Hide" : "Show"} Minimap`,
      icon: <Eye className="w-4 h-4" />,
      category: "View",
      action: () => {
        updateSettings({ minimapEnabled: !settings.minimapEnabled });
        onClose();
      },
    },
    {
      id: "view.toggleWordWrap",
      title: `View: ${settings.wordWrap ? "Disable" : "Enable"} Word Wrap`,
      icon: <Layout className="w-4 h-4" />,
      shortcut: "Alt+Z",
      category: "View",
      action: () => {
        updateSettings({ wordWrap: !settings.wordWrap });
        onClose();
      },
    },
    {
      id: "view.toggleLineNumbers",
      title: `View: ${settings.lineNumbers ? "Hide" : "Show"} Line Numbers`,
      icon: <Layout className="w-4 h-4" />,
      category: "View",
      action: () => {
        updateSettings({ lineNumbers: !settings.lineNumbers });
        onClose();
      },
    },
    
    // Theme Commands
    {
      id: "theme.toggle",
      title: `Theme: Switch to ${settings.theme === "dark" ? "Light" : "Dark"} Mode`,
      icon: settings.theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />,
      shortcut: "Ctrl+K Ctrl+T",
      category: "Preferences",
      action: () => {
        updateSettings({ theme: settings.theme === "dark" ? "light" : "dark" });
        onClose();
      },
    },
    {
      id: "theme.dark",
      title: "Preferences: Color Theme → Dark",
      icon: <Moon className="w-4 h-4" />,
      category: "Preferences",
      action: () => {
        updateSettings({ theme: "dark" });
        onClose();
      },
    },
    {
      id: "theme.light",
      title: "Preferences: Color Theme → Light",
      icon: <Sun className="w-4 h-4" />,
      category: "Preferences",
      action: () => {
        updateSettings({ theme: "light" });
        onClose();
      },
    },
    {
      id: "preferences.openSettings",
      title: "Preferences: Open Settings",
      icon: <Settings className="w-4 h-4" />,
      shortcut: "Ctrl+,",
      category: "Preferences",
      action: () => {
        onClose();
      },
    },
    {
      id: "preferences.keyboardShortcuts",
      title: "Preferences: Keyboard Shortcuts",
      icon: <Keyboard className="w-4 h-4" />,
      shortcut: "Ctrl+K Ctrl+S",
      category: "Preferences",
      action: () => {
        onClose();
      },
    },
    
    // Git Commands
    {
      id: "git.refresh",
      title: "Git: Refresh",
      icon: <RefreshCw className="w-4 h-4" />,
      category: "Git",
      action: async () => {
        try {
          await emit("git-refresh", {});
        } catch (err) {
          console.error("Failed to refresh git:", err);
        }
        onClose();
      },
    },
    
    // AI Commands
    {
      id: "ai.council",
      title: "AI: Open Council Panel",
      description: "Open the AI Council for discussion",
      icon: <Sparkles className="w-4 h-4" />,
      shortcut: "Ctrl+Shift+A",
      category: "AI",
      action: async () => {
        try {
          await emit("open-council-panel", {});
        } catch (err) {
          console.error("Failed to open council:", err);
        }
        onClose();
      },
    },
    {
      id: "ai.explain",
      title: "AI: Explain Selected Code",
      icon: <Sparkles className="w-4 h-4" />,
      category: "AI",
      action: async () => {
        try {
          await emit("ai-explain-code", {});
        } catch (err) {
          console.error("Failed to explain code:", err);
        }
        onClose();
      },
    },
    {
      id: "ai.refactor",
      title: "AI: Refactor Selected Code",
      icon: <Sparkles className="w-4 h-4" />,
      category: "AI",
      action: async () => {
        try {
          await emit("ai-refactor-code", {});
        } catch (err) {
          console.error("Failed to refactor code:", err);
        }
        onClose();
      },
    },
    
    // Training Commands
    {
      id: "training.start",
      title: "Training: Start Training Job",
      icon: <Cpu className="w-4 h-4" />,
      category: "Training",
      action: async () => {
        try {
          await emit("training-start-job", { type: "lora" });
        } catch (err) {
          console.error("Failed to start training:", err);
        }
        onClose();
      },
    },
    {
      id: "training.status",
      title: "Training: View Status",
      icon: <Zap className="w-4 h-4" />,
      category: "Training",
      action: async () => {
        try {
          await emit("open-training-panel", {});
        } catch (err) {
          console.error("Failed to open training panel:", err);
        }
        onClose();
      },
    },
    
    // Window Commands
    {
      id: "window.reload",
      title: "Window: Reload",
      icon: <RefreshCw className="w-4 h-4" />,
      shortcut: "Ctrl+R",
      category: "Window",
      action: () => {
        window.location.reload();
      },
    },
    {
      id: "window.close",
      title: "Window: Close Editor",
      icon: <X className="w-4 h-4" />,
      shortcut: "Ctrl+W",
      category: "Window",
      action: () => {
        onClose();
      },
    },
  ], [currentWorkspace, settings, sidebarVisible, terminalVisible, toggleSidebar, toggleTerminal, updateSettings, onClose]);

  // Filter and sort commands
  const filteredCommands = useMemo(() => {
    if (!query.trim()) {
      return commands.slice(0, 15);
    }
    
    return commands
      .map(cmd => ({
        ...cmd,
        score: Math.max(
          fuzzyMatch(query, cmd.title),
          fuzzyMatch(query, cmd.description || ""),
          fuzzyMatch(query, cmd.category)
        ),
      }))
      .filter(cmd => cmd.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 20);
  }, [query, commands]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          onClose();
          break;
          
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < filteredCommands.length - 1 ? prev + 1 : prev
          );
          break;
          
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
          break;
          
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
          }
          break;
          
        case "Tab":
          e.preventDefault();
          if (e.shiftKey) {
            setSelectedIndex(prev => 
              prev > 0 ? prev - 1 : filteredCommands.length - 1
            );
          } else {
            setSelectedIndex(prev => 
              prev < filteredCommands.length - 1 ? prev + 1 : 0
            );
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  if (!isOpen) return null;

  // Group commands by category
  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) acc[cmd.category] = [];
    acc[cmd.category].push(cmd);
    return acc;
  }, {} as Record<string, CommandItem[]>);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-3xl bg-dark-800 rounded-lg shadow-2xl border border-dark-600 overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-dark-600">
          <Command className="w-5 h-5 text-primary-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent text-dark-100 placeholder-dark-500 outline-none text-lg"
            spellCheck={false}
          />
          <button
            onClick={onClose}
            className="p-1 hover:bg-dark-700 rounded"
          >
            <X className="w-5 h-5 text-dark-400" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[55vh] overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-dark-500">
              No matching commands found
            </div>
          ) : (
            <div className="py-2">
              {query.trim() ? (
                // Flat list when searching
                filteredCommands.map((command, index) => (
                  <CommandItemRow
                    key={command.id}
                    command={command}
                    isSelected={index === selectedIndex}
                    onClick={() => command.action()}
                    onHover={() => setSelectedIndex(index)}
                    showCategory
                  />
                ))
              ) : (
                // Grouped by category when not searching
                Object.entries(groupedCommands).map(([category, cmds]) => (
                  <div key={category}>
                    <div className="px-4 py-1 text-xs font-medium text-dark-500 uppercase tracking-wider">
                      {category}
                    </div>
                    {cmds.map((command) => {
                      const index = filteredCommands.indexOf(command);
                      return (
                        <CommandItemRow
                          key={command.id}
                          command={command}
                          isSelected={index === selectedIndex}
                          onClick={() => command.action()}
                          onHover={() => setSelectedIndex(index)}
                        />
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 bg-dark-900/50 border-t border-dark-600 text-xs text-dark-500">
          <div className="flex items-center gap-4">
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">↑↓</kbd> navigate</span>
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">↵</kbd> execute</span>
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">esc</kbd> close</span>
          </div>
          <span>{filteredCommands.length} commands</span>
        </div>
      </div>
    </div>
  );
}

interface CommandItemRowProps {
  command: CommandItem;
  isSelected: boolean;
  onClick: () => void;
  onHover: () => void;
  showCategory?: boolean;
}

function CommandItemRow({ command, isSelected, onClick, onHover, showCategory }: CommandItemRowProps) {
  return (
    <div
      onClick={onClick}
      onMouseEnter={onHover}
      className={`
        flex items-center gap-3 px-4 py-2 cursor-pointer
        ${isSelected ? "bg-primary-600/30" : "hover:bg-dark-700/50"}
        ${command.disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <span className="text-dark-400">{command.icon}</span>
      
      <div className="flex-1 min-w-0">
        <div className="text-sm text-dark-100">{command.title}</div>
        {command.description && (
          <div className="text-xs text-dark-500">{command.description}</div>
        )}
      </div>
      
      {showCategory && (
        <span className="text-xs text-dark-500 px-2 py-0.5 bg-dark-700 rounded">
          {command.category}
        </span>
      )}
      
      {command.shortcut && (
        <kbd className="text-xs text-dark-400 px-2 py-0.5 bg-dark-700 rounded font-mono">
          {command.shortcut}
        </kbd>
      )}
    </div>
  );
}
