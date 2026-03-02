//! Git Panel Component
//! Git status, staging, commits, and branch management

import { useState, useEffect, useCallback } from "react";
import { 
  GitBranch, GitCommit, RefreshCw, Plus, Check, X, 
  ArrowUp, ArrowDown, History, FolderGit2, AlertCircle
} from "lucide-react";
import { useStore } from "../../lib/store";
import { git } from "../../lib/tauri";

interface GitStatus {
  branch: string;
  ahead: number;
  behind: number;
  modified: string[];
  added: string[];
  deleted: string[];
  untracked: string[];
  conflicted: string[];
  is_clean: boolean;
}

interface GitCommit {
  hash: string;
  short_hash: string;
  message: string;
  author: string;
  timestamp: number;
}

export function GitPanel() {
  const [status, setStatus] = useState<GitStatus | null>(null);
  const [commits, setCommits] = useState<GitCommit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [commitMessage, setCommitMessage] = useState("");
  const [activeTab, setActiveTab] = useState<"changes" | "history">("changes");
  
  const { currentWorkspace } = useStore();

  const loadGitStatus = useCallback(async () => {
    if (!currentWorkspace?.path) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const status = await git.status(currentWorkspace.path);
      setStatus(status);
      
      const commits = await git.log(currentWorkspace.path, 10);
      setCommits(commits);
    } catch (err: any) {
      if (err.message?.includes("not a git repository")) {
        setStatus(null);
      } else {
        setError(err.message || "Failed to load git status");
      }
    } finally {
      setLoading(false);
    }
  }, [currentWorkspace?.path]);

  useEffect(() => {
    loadGitStatus();
    const interval = setInterval(loadGitStatus, 10000);
    return () => clearInterval(interval);
  }, [loadGitStatus]);

  const handleStage = async (file: string) => {
    if (!currentWorkspace?.path) return;
    try {
      await git.add(currentWorkspace.path, [file]);
      await loadGitStatus();
    } catch (err: any) {
      setError(err.message || "Failed to stage file");
    }
  };

  const handleStageAll = async () => {
    if (!currentWorkspace?.path || !status) return;
    const files = [...status.modified, ...status.deleted, ...status.untracked];
    if (files.length === 0) return;
    
    try {
      await git.add(currentWorkspace.path, files);
      await loadGitStatus();
    } catch (err: any) {
      setError(err.message || "Failed to stage all");
    }
  };

  const handleCommit = async () => {
    if (!currentWorkspace?.path || !commitMessage.trim()) return;
    
    try {
      await git.commit(currentWorkspace.path, commitMessage.trim());
      setCommitMessage("");
      await loadGitStatus();
    } catch (err: any) {
      setError(err.message || "Failed to commit");
    }
  };

  const handlePush = async () => {
    if (!currentWorkspace?.path) return;
    try {
      await git.push(currentWorkspace.path);
      await loadGitStatus();
    } catch (err: any) {
      setError(err.message || "Failed to push");
    }
  };

  const handlePull = async () => {
    if (!currentWorkspace?.path) return;
    try {
      await git.pull(currentWorkspace.path);
      await loadGitStatus();
    } catch (err: any) {
      setError(err.message || "Failed to pull");
    }
  };

  if (!currentWorkspace) {
    return (
      <div className="h-full flex items-center justify-center text-dark-500">
        <div className="text-center">
          <FolderGit2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Open a workspace to view Git status</p>
        </div>
      </div>
    );
  }

  if (status === null && !loading) {
    return (
      <div className="h-full flex items-center justify-center text-dark-500">
        <div className="text-center p-4">
          <GitBranch className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm mb-2">Not a Git repository</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-dark-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-dark-700">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-primary-400" />
          <span className="text-sm font-medium">{status?.branch || "main"}</span>
          {status && (
            <>
              {status.ahead > 0 && (
                <span className="text-xs text-green-400 flex items-center gap-0.5">
                  <ArrowUp className="w-3 h-3" /> {status.ahead}
                </span>
              )}
              {status.behind > 0 && (
                <span className="text-xs text-red-400 flex items-center gap-0.5">
                  <ArrowDown className="w-3 h-3" /> {status.behind}
                </span>
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={handlePull} disabled={loading} className="p-1.5 hover:bg-dark-700 rounded">
            <ArrowDown className="w-4 h-4" />
          </button>
          <button onClick={handlePush} disabled={loading} className="p-1.5 hover:bg-dark-700 rounded">
            <ArrowUp className="w-4 h-4" />
          </button>
          <button onClick={loadGitStatus} disabled={loading} className="p-1.5 hover:bg-dark-700 rounded">
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mt-2 p-2 bg-red-900/30 border border-red-700 rounded text-xs text-red-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="p-1 hover:bg-red-900/50 rounded ml-auto">
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-dark-700 mt-2">
        <button
          onClick={() => setActiveTab("changes")}
          className={`flex-1 py-2 text-sm ${activeTab === "changes" ? "text-primary-400 border-b-2 border-primary-400" : "text-dark-400"}`}
        >
          Changes {status && getChangesCount(status) > 0 && (
            <span className="ml-1 px-1.5 py-0.5 text-xs bg-primary-600 rounded-full">{getChangesCount(status)}</span>
          )}
        </button>
        <button
          onClick={() => setActiveTab("history")}
          className={`flex-1 py-2 text-sm ${activeTab === "history" ? "text-primary-400 border-b-2 border-primary-400" : "text-dark-400"}`}
        >
          History
        </button>
      </div>

      {/* Changes Tab */}
      {activeTab === "changes" && (
        <div className="flex-1 overflow-y-auto">
          {status && status.added.length > 0 && (
            <div className="p-3">
              <span className="text-xs font-medium text-green-400">STAGED</span>
              {status.added.map((file) => (
                <FileItem key={file} file={file} status="added" />
              ))}
            </div>
          )}

          {status && (
            <div className="p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-dark-400">CHANGES</span>
                <button onClick={handleStageAll} className="text-xs text-dark-400 hover:text-dark-200">
                  Stage All
                </button>
              </div>
              
              {status.modified.map((file) => (
                <FileItem key={file} file={file} status="modified" onStage={() => handleStage(file)} />
              ))}
              {status.deleted.map((file) => (
                <FileItem key={file} file={file} status="deleted" onStage={() => handleStage(file)} />
              ))}
              {status.untracked.map((file) => (
                <FileItem key={file} file={file} status="untracked" onStage={() => handleStage(file)} />
              ))}

              {status.is_clean && (
                <div className="text-center py-8 text-dark-500">
                  <Check className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p className="text-sm">No changes</p>
                </div>
              )}
            </div>
          )}

          {status && !status.is_clean && (
            <div className="p-3 border-t border-dark-700">
              <textarea
                value={commitMessage}
                onChange={(e) => setCommitMessage(e.target.value)}
                placeholder="Message (Ctrl+Enter to commit)"
                className="w-full h-20 px-3 py-2 bg-dark-900 border border-dark-600 rounded text-sm resize-none"
                onKeyDown={(e) => { if (e.ctrlKey && e.key === "Enter") handleCommit(); }}
              />
              <button
                onClick={handleCommit}
                disabled={!commitMessage.trim() || status.added.length === 0}
                className="w-full mt-2 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 text-white text-sm rounded"
              >
                <GitCommit className="w-4 h-4 inline mr-2" />
                Commit
              </button>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === "history" && (
        <div className="flex-1 overflow-y-auto p-3">
          {commits.map((commit, index) => (
            <div key={commit.hash} className={`py-3 ${index !== commits.length - 1 ? "border-b border-dark-700" : ""}`}>
              <p className="text-sm text-dark-100">{commit.message}</p>
              <div className="flex items-center gap-2 mt-1 text-xs text-dark-500">
                <span>{commit.author}</span>
                <span>•</span>
                <span>{formatTimestamp(commit.timestamp)}</span>
                <code className="font-mono">{commit.short_hash}</code>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FileItem({ file, status, onStage }: { file: string; status: string; onStage?: () => void }) {
  const colors: Record<string, string> = {
    added: "text-green-400",
    modified: "text-yellow-400",
    deleted: "text-red-400",
    untracked: "text-dark-400",
  };
  const labels: Record<string, string> = { added: "A", modified: "M", deleted: "D", untracked: "U" };

  return (
    <div className="group flex items-center gap-2 py-1 hover:bg-dark-700/30 rounded px-1">
      {onStage && (
        <button onClick={onStage} className="w-5 h-5 flex items-center justify-center rounded bg-dark-700 text-dark-400 hover:bg-dark-600">
          <Plus className="w-3 h-3" />
        </button>
      )}
      <span className={`text-xs font-mono w-5 ${colors[status]}`}>{labels[status]}</span>
      <span className="text-sm text-dark-200 truncate flex-1">{file}</span>
    </div>
  );
}

function getChangesCount(status: GitStatus): number {
  return status.modified.length + status.added.length + status.deleted.length + status.untracked.length;
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMs / 3600000);
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString();
}
