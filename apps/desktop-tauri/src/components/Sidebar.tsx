import { lazy, Suspense, useState, useEffect, useCallback, useRef } from "react";
import {
  ChevronRight,
  ChevronDown,
  File,
  Folder,
  FolderOpen,
  RefreshCw,
  Bot,
  Send,
  Plus,
  FilePlus,
  FolderPlus,
  Search as SearchIcon,
  GitBranch,
  Sparkles,
  GraduationCap,
  Monitor,
  Server,
} from "lucide-react";
import { useStore, FileNode } from "../lib/store";
import { fs, workspace, git, training, ai, trainingData, AiChatMessage } from "../lib/tauri";
import { CouncilPanel } from "./CouncilPanel";
import { HierarchyPanel } from "./HierarchyPanel";
import { ProjectsPanel } from "./ProjectsPanel";
import { FactoriesPanel } from "./FactoriesPanel";
import { CapsuleTreePanel } from "./CapsuleTreePanel";
import { BrainDashboardPanel } from "./BrainDashboardPanel";
import { listen } from "@tauri-apps/api/event";

const AIChat = lazy(() =>
  import("./chat/AIChat").then((module) => ({
    default: module.AIChat,
  }))
);

const GitPanel = lazy(() =>
  import("./git/GitPanel").then((module) => ({
    default: module.GitPanel,
  }))
);

const SearchPanel = lazy(() =>
  import("./editor/SearchPanel").then((module) => ({
    default: module.SearchPanel,
  }))
);

const SyncPanel = lazy(() =>
  import("./sync/SyncPanel").then((module) => ({
    default: module.SyncPanel,
  }))
);

const TrainingDashboard = lazy(() =>
  import("./training/TrainingDashboard").then((module) => ({
    default: module.TrainingDashboard,
  }))
);

const MonitorDashboard = lazy(() =>
  import("./monitor/MonitorDashboard").then((module) => ({
    default: module.MonitorDashboard,
  }))
);

const FleetPanel = lazy(() =>
  import("./training/FleetPanel").then((module) => ({
    default: module.FleetPanel,
  }))
);

interface TreeNodeProps {
  node: FileNode;
  depth: number;
  workspaceRoot: string;
  onToggle: (path: string) => void;
  onSelect: (path: string, content: string) => void;
}

function TreeNode({ node, depth, workspaceRoot, onToggle, onSelect }: TreeNodeProps) {
  const { expandedDirs, activeFilePath } = useStore();
  const isExpanded = expandedDirs.has(node.path);
  const absolutePath = node.path.startsWith("/") ? node.path : `${workspaceRoot}/${node.path}`;
  const isActive = activeFilePath === absolutePath;

  const handleClick = async () => {
    if (node.isDir) {
      onToggle(node.path);
    } else {
      try {
        const { content } = await fs.readFile(absolutePath);
        onSelect(absolutePath, content);
      } catch (e) {
        console.error("Failed to read file:", e);
      }
    }
  };

  return (
    <div>
      <div
        onClick={handleClick}
        className={`tree-item ${isActive ? "active" : ""}`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {node.isDir ? (
          <>
            <span className="text-dark-400">
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </span>
            {isExpanded ? (
              <FolderOpen className="w-4 h-4 text-yellow-500" />
            ) : (
              <Folder className="w-4 h-4 text-yellow-500" />
            )}
          </>
        ) : (
          <>
            <span className="w-4" />
            <File className="w-4 h-4 text-dark-400" />
          </>
        )}
        <span className="truncate">{node.name}</span>
      </div>

      {node.isDir && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              workspaceRoot={workspaceRoot}
              onToggle={onToggle}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function Sidebar() {
  const [activeTab, setActiveTab] = useState<"explorer" | "search" | "git" | "ai" | "training" | "fleet" | "system">("explorer");
  const [isLoading, setIsLoading] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [isSendingChat, setIsSendingChat] = useState(false);
  const [includeFileContext, setIncludeFileContext] = useState(true);
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    {
      role: "assistant",
      content: "هلا، أني AI assistant داخل BI-IDE. كلي شتريد أسوي بالكود؟",
    },
  ]);
  const [isTrainingAction, setIsTrainingAction] = useState(false);

  // New file/folder state
  const [isCreatingNew, setIsCreatingNew] = useState<"file" | "folder" | null>(null);
  const [newItemName, setNewItemName] = useState("");
  const [newItemPath, setNewItemPath] = useState("");
  const newItemInputRef = useRef<HTMLInputElement>(null);

  const {
    currentWorkspace,
    fileTree,
    setFileTree,
    expandedDirs,
    toggleDir,
    openFile,
    gitState,
    setGitState,
    trainingStatus,
    setTrainingStatus,
    openFiles,
    activeFilePath,
  } = useStore();

  const activeFile = openFiles.find((file) => file.path === activeFilePath);

  // Load file tree
  const loadFileTree = useCallback(async () => {
    if (!currentWorkspace) return;

    setIsLoading(true);
    try {
      const buildTree = async (path: string): Promise<FileNode[]> => {
        const entries = await workspace.getFiles(currentWorkspace.id, path);

        const nodes: FileNode[] = await Promise.all(
          entries.map(async (entry) => {
            const node: FileNode = {
              path: entry.path,
              name: entry.name,
              isDir: entry.is_dir,
              size: entry.size,
              modifiedAt: entry.modified_at,
            };

            if (entry.is_dir && expandedDirs.has(entry.path)) {
              node.children = await buildTree(entry.path);
            }

            return node;
          })
        );

        return nodes;
      };

      const tree = await buildTree("");
      setFileTree(tree);
    } catch (e) {
      console.error("Failed to load file tree:", e);
    } finally {
      setIsLoading(false);
    }
  }, [currentWorkspace, expandedDirs, setFileTree]);

  // Load git status
  const loadGitStatus = useCallback(async () => {
    if (!currentWorkspace) return;

    try {
      const status = await git.status(currentWorkspace.path);
      setGitState({
        branch: status.branch,
        ahead: status.ahead,
        behind: status.behind,
        modified: status.modified,
        added: status.added,
        deleted: status.deleted,
        untracked: status.untracked,
        isClean: status.is_clean,
      });
    } catch (e) {
      // Not a git repo
      setGitState(null);
    }
  }, [currentWorkspace, setGitState]);

  useEffect(() => {
    loadFileTree();
    loadGitStatus();
  }, [loadFileTree, loadGitStatus]);

  // Listen for new file/folder events from CommandPalette
  useEffect(() => {
    const unlistenNewFile = listen("new-file-requested", (event: any) => {
      setIsCreatingNew("file");
      setNewItemName("");
      setNewItemPath(currentWorkspace?.path || "");
      setTimeout(() => newItemInputRef.current?.focus(), 100);
    });

    const unlistenNewFolder = listen("new-folder-requested", (event: any) => {
      setIsCreatingNew("folder");
      setNewItemName("");
      setNewItemPath(currentWorkspace?.path || "");
      setTimeout(() => newItemInputRef.current?.focus(), 100);
    });

    return () => {
      unlistenNewFile.then((fn) => fn());
      unlistenNewFolder.then((fn) => fn());
    };
  }, [currentWorkspace]);

  // Handle creating new file/folder
  const handleCreateNew = async () => {
    if (!newItemName || !currentWorkspace) return;

    const fullPath = newItemPath
      ? `${newItemPath}/${newItemName}`
      : `${currentWorkspace.path}/${newItemName}`;

    try {
      if (isCreatingNew === "file") {
        await fs.writeFile(fullPath, "");
      } else {
        await fs.createDir(fullPath, true);
      }
      await loadFileTree();
      setIsCreatingNew(null);
      setNewItemName("");
    } catch (err) {
      console.error(`Failed to create ${isCreatingNew}:`, err);
    }
  };

  const handleToggle = async (path: string) => {
    toggleDir(path);
    // Tree will reload due to expandedDirs change
  };

  const handleSelect = (path: string, content: string) => {
    openFile(path, content);
  };

  const handleSendChat = async () => {
    if (!chatInput.trim() || isSendingChat) return;

    const userMessage = chatInput.trim();
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsSendingChat(true);

    try {
      const contextMessages: AiChatMessage[] = [];

      if (includeFileContext && activeFile) {
        const truncatedContent = activeFile.content.length > 12000
          ? `${activeFile.content.slice(0, 12000)}\n\n...[truncated]`
          : activeFile.content;

        contextMessages.push({
          role: "system",
          content: `You are BI-IDE assistant. Current workspace: ${currentWorkspace?.path || "unknown"}. Active file: ${activeFile.path}. File content:\n\n${truncatedContent}`,
        });
      }

      const payload: AiChatMessage[] = [
        ...contextMessages,
        ...chatMessages,
        { role: "user", content: userMessage },
      ];
      const result = await ai.chat(payload);
      const assistantText = result.generated_text || result.text || result.response || "ماكو رد من المودل";
      setChatMessages((prev) => [...prev, { role: "assistant", content: assistantText }]);

      try {
        await trainingData.ingestChatSample({
          input_text: userMessage,
          output_text: assistantText,
          workspace_path: currentWorkspace?.path,
          file_path: activeFile?.path,
          metadata: {
            include_file_context: includeFileContext,
          },
        });
      } catch (ingestError) {
        console.warn("Training ingest failed:", ingestError);
      }
    } catch (error) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `AI error: ${String(error)}` },
      ]);
    } finally {
      setIsSendingChat(false);
    }
  };

  const refreshTrainingStatus = async () => {
    const status = await training.getStatus();
    setTrainingStatus({
      isEnabled: status.enabled,
      currentJob: status.current_job
        ? {
          id: status.current_job.job_id,
          type: status.current_job.job_type,
          progress: status.current_job.progress_percent,
          status: status.current_job.status,
        }
        : undefined,
    });
  };

  const handleStartTraining = async () => {
    setIsTrainingAction(true);
    try {
      await training.startJob("lora", 60);
      await refreshTrainingStatus();
    } catch (error) {
      console.error("Failed to start training:", error);
    } finally {
      setIsTrainingAction(false);
    }
  };

  const handlePauseTraining = async () => {
    if (!trainingStatus.currentJob?.id) return;
    setIsTrainingAction(true);
    try {
      await training.pauseJob(trainingStatus.currentJob.id);
      await refreshTrainingStatus();
    } catch (error) {
      console.error("Failed to pause training:", error);
    } finally {
      setIsTrainingAction(false);
    }
  };

  // Listen for tab/control events from CommandPalette
  useEffect(() => {
    const unlistenGlobalSearch = listen("open-global-search", () => {
      setActiveTab("search");
    });

    const unlistenGitRefresh = listen("git-refresh", async () => {
      setActiveTab("git");
      await loadGitStatus();
    });

    const unlistenTrainingPanel = listen("open-training-panel", async () => {
      setActiveTab("training");
      await refreshTrainingStatus();
    });

    const unlistenCouncilPanel = listen("open-council-panel", () => {
      setActiveTab("system");
    });

    const unlistenProjectsPanel = listen("open-projects-panel", () => {
      setActiveTab("ai");
    });

    const unlistenTrainingStart = listen("training-start-job", async (event: any) => {
      const requestedType = event?.payload?.type;
      const jobType = typeof requestedType === "string" && requestedType.length > 0 ? requestedType : "lora";

      setActiveTab("training");
      setIsTrainingAction(true);
      try {
        await training.startJob(jobType, 60);
        await refreshTrainingStatus();
      } catch (error) {
        console.error("Failed to start training from palette:", error);
      } finally {
        setIsTrainingAction(false);
      }
    });

    return () => {
      unlistenGlobalSearch.then((fn) => fn());
      unlistenGitRefresh.then((fn) => fn());
      unlistenTrainingPanel.then((fn) => fn());
      unlistenCouncilPanel.then((fn) => fn());
      unlistenTrainingStart.then((fn) => fn());
      unlistenProjectsPanel.then((fn) => fn());
    };
  }, [loadGitStatus, refreshTrainingStatus]);

  const panelFallback = <div className="p-3 text-xs text-dark-500">Loading...</div>;

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* Tab Bar — Clean 6-tab professional layout */}
      <div className="flex border-b border-dark-700">
        {[
          { id: "explorer" as const, label: "Explorer", icon: null },
          { id: "search" as const, label: "Search", icon: SearchIcon },
          { id: "git" as const, label: "Git", icon: GitBranch },
          { id: "ai" as const, label: "AI", icon: Sparkles },
          { id: "training" as const, label: "Train", icon: GraduationCap },
          { id: "fleet" as const, label: "Fleet", icon: Server },
          { id: "system" as const, label: "System", icon: Monitor },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 text-xs font-medium transition-colors relative ${activeTab === tab.id
              ? "text-primary-400 border-b-2 border-primary-500"
              : "text-dark-400 hover:text-dark-200"
              }`}
          >
            {tab.label}
            {tab.id === "git" && gitState && !gitState.isClean && (
              <span className="absolute top-1 right-1 w-2 h-2 bg-orange-400 rounded-full" />
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === "explorer" && (
          <div className="h-full flex flex-col">
            {/* Workspace Header */}
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs font-semibold text-dark-400 uppercase">
                {currentWorkspace?.name || "No Folder"}
              </span>
              <div className="flex items-center gap-1">
                {currentWorkspace && (
                  <>
                    <button
                      onClick={() => {
                        setIsCreatingNew("file");
                        setNewItemName("");
                        setNewItemPath(currentWorkspace.path);
                        setTimeout(() => newItemInputRef.current?.focus(), 100);
                      }}
                      className="p-1 hover:bg-dark-800 rounded transition-colors"
                      title="New File"
                    >
                      <FilePlus className="w-3 h-3 text-dark-400" />
                    </button>
                    <button
                      onClick={() => {
                        setIsCreatingNew("folder");
                        setNewItemName("");
                        setNewItemPath(currentWorkspace.path);
                        setTimeout(() => newItemInputRef.current?.focus(), 100);
                      }}
                      className="p-1 hover:bg-dark-800 rounded transition-colors"
                      title="New Folder"
                    >
                      <FolderPlus className="w-3 h-3 text-dark-400" />
                    </button>
                  </>
                )}
                <button
                  onClick={loadFileTree}
                  className="p-1 hover:bg-dark-800 rounded transition-colors"
                  disabled={isLoading}
                >
                  <RefreshCw className={`w-3 h-3 text-dark-400 ${isLoading ? "animate-spin" : ""}`} />
                </button>
              </div>
            </div>

            {/* New File/Folder Input */}
            {isCreatingNew && (
              <div className="px-3 py-2">
                <div className="flex items-center gap-2 bg-dark-800 rounded px-2 py-1">
                  {isCreatingNew === "file" ? (
                    <FilePlus className="w-4 h-4 text-dark-400" />
                  ) : (
                    <FolderPlus className="w-4 h-4 text-dark-400" />
                  )}
                  <input
                    ref={newItemInputRef}
                    type="text"
                    value={newItemName}
                    onChange={(e) => setNewItemName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleCreateNew();
                      } else if (e.key === "Escape") {
                        setIsCreatingNew(null);
                        setNewItemName("");
                      }
                    }}
                    onBlur={() => {
                      if (newItemName) {
                        handleCreateNew();
                      } else {
                        setIsCreatingNew(null);
                      }
                    }}
                    placeholder={isCreatingNew === "file" ? "filename.ts" : "foldername"}
                    className="flex-1 bg-transparent text-xs text-dark-100 outline-none"
                    autoFocus
                  />
                </div>
              </div>
            )}

            {/* File Tree */}
            <div className="flex-1 overflow-auto pb-4">
              {currentWorkspace &&
                fileTree.map((node) => (
                  <TreeNode
                    key={node.path}
                    node={node}
                    depth={0}
                    workspaceRoot={currentWorkspace.path}
                    onToggle={handleToggle}
                    onSelect={handleSelect}
                  />
                ))}
            </div>
          </div>
        )}

        {activeTab === "ai" && (
          <Suspense fallback={panelFallback}>
            <AIChat />
          </Suspense>
        )}

        {activeTab === "search" && (
          <Suspense fallback={panelFallback}>
            <SearchPanel isOpen={activeTab === "search"} onClose={() => setActiveTab("explorer")} />
          </Suspense>
        )}

        {activeTab === "git" && (
          <Suspense fallback={panelFallback}>
            <GitPanel />
          </Suspense>
        )}

        {activeTab === "training" && (
          <Suspense fallback={panelFallback}>
            <TrainingDashboard />
          </Suspense>
        )}

        {activeTab === "fleet" && (
          <Suspense fallback={panelFallback}>
            <FleetPanel />
          </Suspense>
        )}

        {activeTab === "system" && (
          <div className="h-full flex flex-col overflow-auto">
            <CouncilPanel />
            <div className="border-t border-dark-700">
              <HierarchyPanel />
            </div>
            <div className="border-t border-dark-700">
              <FactoriesPanel />
            </div>
            <div className="border-t border-dark-700">
              <CapsuleTreePanel />
            </div>
            <div className="border-t border-dark-700">
              <BrainDashboardPanel />
            </div>
            <div className="border-t border-dark-700">
              <Suspense fallback={panelFallback}>
                <MonitorDashboard />
              </Suspense>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
