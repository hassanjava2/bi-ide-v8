import { useState, useEffect, useCallback } from "react";
import {
  ChevronRight,
  ChevronDown,
  File,
  Folder,
  FolderOpen,
  RefreshCw,
  Search,
  GitBranch,
  Cpu,
  Cloud,
  Bot,
  Send,
  Pause,
  Play,
} from "lucide-react";
import { useStore, FileNode } from "../lib/store";
import { fs, workspace, git, training, ai, trainingData, AiChatMessage } from "../lib/tauri";
import { CouncilPanel } from "./CouncilPanel";
import { HierarchyPanel } from "./HierarchyPanel";

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
  const [activeTab, setActiveTab] = useState<"explorer" | "search" | "git" | "ai" | "training" | "council" | "hierarchy">("explorer");
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

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* Tab Bar */}
      <div className="flex border-b border-dark-700">
        <button
          onClick={() => setActiveTab("explorer")}
          className={`flex-1 py-2 text-xs font-medium transition-colors ${activeTab === "explorer"
              ? "text-primary-400 border-b-2 border-primary-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
        >
          Explorer
        </button>
        <button
          onClick={() => setActiveTab("search")}
          className={`flex-1 py-2 text-xs font-medium transition-colors ${activeTab === "search"
              ? "text-primary-400 border-b-2 border-primary-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
        >
          Search
        </button>
        <button
          onClick={() => setActiveTab("git")}
          className={`flex-1 py-2 text-xs font-medium transition-colors relative ${activeTab === "git"
              ? "text-primary-400 border-b-2 border-primary-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
        >
          Git
          {gitState && !gitState.isClean && (
            <span className="absolute top-1 right-1 w-2 h-2 bg-orange-400 rounded-full" />
          )}
        </button>
        <button
          onClick={() => setActiveTab("ai")}
          className={`flex-1 py-2 text-xs font-medium transition-colors relative ${activeTab === "ai"
              ? "text-primary-400 border-b-2 border-primary-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
        >
          AI
        </button>
        <button
          onClick={() => setActiveTab("council")}
          className={`flex-1 py-2 text-xs font-medium transition-colors relative ${activeTab === "council"
              ? "text-yellow-400 border-b-2 border-yellow-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
          title="مجلس الحكماء"
        >
          🏛️
        </button>
        <button
          onClick={() => setActiveTab("hierarchy")}
          className={`flex-1 py-2 text-xs font-medium transition-colors relative ${activeTab === "hierarchy"
              ? "text-cyan-400 border-b-2 border-cyan-500"
              : "text-dark-400 hover:text-dark-200"
            }`}
          title="النظام الهرمي"
        >
          📊
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === "explorer" && (
          <div>
            {/* Workspace Header */}
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs font-semibold text-dark-400 uppercase">
                {currentWorkspace?.name || "No Folder"}
              </span>
              <button
                onClick={loadFileTree}
                className="p-1 hover:bg-dark-800 rounded transition-colors"
                disabled={isLoading}
              >
                <RefreshCw className={`w-3 h-3 text-dark-400 ${isLoading ? "animate-spin" : ""}`} />
              </button>
            </div>

            {/* File Tree */}
            <div className="pb-4">
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
          <div className="h-full flex flex-col p-3 gap-3">
            <div className="flex items-center gap-2 text-dark-300 text-sm font-medium">
              <Bot className="w-4 h-4 text-primary-400" />
              AI Assistant
            </div>

            <div className="bg-dark-800 rounded-lg p-2 text-xs text-dark-400 space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeFileContext}
                  onChange={(e) => setIncludeFileContext(e.target.checked)}
                />
                Include current file context
              </label>
              <div>
                Active file: <span className="text-dark-300">{activeFile?.path || "None"}</span>
              </div>
            </div>

            <div className="flex-1 overflow-auto bg-dark-800 rounded-lg p-2 space-y-2">
              {chatMessages.map((message, index) => (
                <div
                  key={index}
                  className={`text-xs p-2 rounded ${message.role === "user"
                      ? "bg-primary-700/40 text-primary-100"
                      : "bg-dark-700 text-dark-200"
                    }`}
                >
                  {message.content}
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleSendChat();
                  }
                }}
                placeholder="اكتب سؤالك..."
                className="flex-1 px-3 py-2 bg-dark-800 border border-dark-700 rounded text-sm text-dark-100 placeholder-dark-500 focus:outline-none focus:border-primary-500"
              />
              <button
                onClick={handleSendChat}
                disabled={isSendingChat}
                className="px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded text-white"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {activeTab === "search" && (
          <div className="p-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
              <input
                type="text"
                placeholder="Search files..."
                className="w-full pl-9 pr-3 py-2 bg-dark-800 border border-dark-700 rounded text-sm text-dark-200 placeholder-dark-500 focus:outline-none focus:border-primary-500"
              />
            </div>
            <p className="text-xs text-dark-500 mt-4 text-center">
              Search across workspace
            </p>
          </div>
        )}

        {activeTab === "git" && (
          <div className="p-3">
            {!gitState ? (
              <div className="text-center text-dark-500 text-sm">
                <GitBranch className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Not a git repository</p>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-dark-200">
                    {gitState.branch}
                  </span>
                  <button className="text-xs px-2 py-1 bg-primary-600 hover:bg-primary-700 rounded text-white">
                    Sync
                  </button>
                </div>

                {gitState.modified.length > 0 && (
                  <div className="mb-3">
                    <div className="text-xs text-dark-500 mb-1">Modified</div>
                    {gitState.modified.map((file) => (
                      <div key={file} className="text-sm text-orange-400 py-1">
                        M {file}
                      </div>
                    ))}
                  </div>
                )}

                {gitState.added.length > 0 && (
                  <div className="mb-3">
                    <div className="text-xs text-dark-500 mb-1">Staged</div>
                    {gitState.added.map((file) => (
                      <div key={file} className="text-sm text-green-400 py-1">
                        A {file}
                      </div>
                    ))}
                  </div>
                )}

                {gitState.untracked.length > 0 && (
                  <div className="mb-3">
                    <div className="text-xs text-dark-500 mb-1">Untracked</div>
                    {gitState.untracked.map((file) => (
                      <div key={file} className="text-sm text-dark-400 py-1">
                        U {file}
                      </div>
                    ))}
                  </div>
                )}

                {gitState.isClean && (
                  <div className="text-center text-green-400 text-sm py-4">
                    ✓ Working tree clean
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === "training" && (
          <div className="p-3">
            <div className="flex items-center gap-2 mb-3">
              <Cloud className={`w-4 h-4 ${trainingStatus.isEnabled ? "text-green-400" : "text-dark-500"}`} />
              <span className="text-sm">
                {trainingStatus.isEnabled ? "Enabled" : "Disabled"}
              </span>
            </div>

            {trainingStatus.currentJob && (
              <div className="bg-dark-800 rounded-lg p-3 mb-3">
                <div className="text-xs text-dark-400 mb-1">Current Job</div>
                <div className="text-sm font-medium">{trainingStatus.currentJob.type}</div>
                <div className="mt-2">
                  <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500 transition-all"
                      style={{ width: `${trainingStatus.currentJob.progress}%` }}
                    />
                  </div>
                  <div className="text-xs text-dark-400 mt-1">
                    {trainingStatus.currentJob.progress.toFixed(1)}%
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={handleStartTraining}
                disabled={isTrainingAction}
                className="w-full py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded text-sm text-white transition-colors flex items-center justify-center gap-1"
              >
                <Play className="w-3 h-3" /> Start
              </button>
              <button
                onClick={handlePauseTraining}
                disabled={isTrainingAction || !trainingStatus.currentJob}
                className="w-full py-2 bg-dark-700 hover:bg-dark-600 disabled:opacity-50 rounded text-sm text-white transition-colors flex items-center justify-center gap-1"
              >
                <Pause className="w-3 h-3" /> Pause
              </button>
            </div>
          </div>
        )}

        {activeTab === "council" && <CouncilPanel />}
        {activeTab === "hierarchy" && <HierarchyPanel />}
      </div>
    </div>
  );
}
