//! Git Commands
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;
use tracing::{info, error};

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct GitStatus {
    pub branch: String,
    pub ahead: u32,
    pub behind: u32,
    pub modified: Vec<String>,
    pub added: Vec<String>,
    pub deleted: Vec<String>,
    pub untracked: Vec<String>,
    pub conflicted: Vec<String>,
    pub is_clean: bool,
}

#[derive(Debug, Serialize)]
pub struct GitCommit {
    pub hash: String,
    pub short_hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize)]
pub struct GitBranch {
    pub name: String,
    pub is_current: bool,
    pub is_remote: bool,
    pub upstream: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GitStatusRequest {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct GitAddRequest {
    pub path: String,
    pub files: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct GitCommitRequest {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct GitPushRequest {
    pub path: String,
    pub remote: Option<String>,
    pub branch: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GitPullRequest {
    pub path: String,
    pub remote: Option<String>,
    pub branch: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GitLogRequest {
    pub path: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct GitBranchesRequest {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct GitCheckoutRequest {
    pub path: String,
    pub branch: String,
    pub create: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct GitCloneRequest {
    pub url: String,
    pub path: String,
    pub depth: Option<usize>,
}

#[tauri::command]
pub async fn git_status(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitStatusRequest,
) -> Result<GitStatus, String> {
    let path = PathBuf::from(&request.path);
    
    info!("Getting git status for: {:?}", path);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                let mut status = GitStatus {
                    branch: String::new(),
                    ahead: 0,
                    behind: 0,
                    modified: Vec::new(),
                    added: Vec::new(),
                    deleted: Vec::new(),
                    untracked: Vec::new(),
                    conflicted: Vec::new(),
                    is_clean: true,
                };

                // Get current branch
                if let Ok(head) = repo.head() {
                    if let Some(name) = head.shorthand() {
                        status.branch = name.to_string();
                    }
                }

                // Get status
                let mut opts = git2::StatusOptions::new();
                opts.include_untracked(true);
                
                if let Ok(statuses) = repo.statuses(Some(&mut opts)) {
                    for entry in statuses.iter() {
                        let path = entry.path().unwrap_or("").to_string();
                        let status_bits = entry.status();

                        if status_bits.contains(git2::Status::WT_MODIFIED) {
                            status.modified.push(path.clone());
                        }
                        if status_bits.contains(git2::Status::INDEX_NEW) || 
                           status_bits.contains(git2::Status::INDEX_MODIFIED) {
                            status.added.push(path.clone());
                        }
                        if status_bits.contains(git2::Status::WT_DELETED) ||
                           status_bits.contains(git2::Status::INDEX_DELETED) {
                            status.deleted.push(path.clone());
                        }
                        if status_bits.contains(git2::Status::WT_NEW) {
                            status.untracked.push(path.clone());
                        }
                        if status_bits.contains(git2::Status::CONFLICTED) {
                            status.conflicted.push(path.clone());
                        }
                    }
                }

                status.is_clean = status.modified.is_empty() 
                    && status.added.is_empty() 
                    && status.deleted.is_empty() 
                    && status.untracked.is_empty();

                Ok(status)
            }
            Err(e) => {
                error!("Failed to open repository: {}", e);
                Err(format!("Not a git repository: {}", e))
            }
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_add(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitAddRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    let files = request.files.clone();
    
    info!("Adding files to git: {:?}", files);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                let mut index = repo.index()
                    .map_err(|e| format!("Failed to get index: {}", e))?;

                for file in &files {
                    if file == "." {
                        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
                            .map_err(|e| format!("Failed to add all: {}", e))?;
                    } else {
                        index.add_path(std::path::Path::new(file))
                            .map_err(|e| format!("Failed to add {}: {}", file, e))?;
                    }
                }

                index.write()
                    .map_err(|e| format!("Failed to write index: {}", e))?;

                Ok(())
            }
            Err(e) => Err(format!("Not a git repository: {}", e))
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_commit(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitCommitRequest,
) -> Result<String, String> {
    let path = PathBuf::from(&request.path);
    let message = request.message.clone();
    
    info!("Committing to git: {}", message);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                let signature = repo.signature()
                    .map_err(|e| format!("Failed to get signature: {}", e))?;
                
                let mut index = repo.index()
                    .map_err(|e| format!("Failed to get index: {}", e))?;
                
                let tree_id = index.write_tree()
                    .map_err(|e| format!("Failed to write tree: {}", e))?;
                
                let tree = repo.find_tree(tree_id)
                    .map_err(|e| format!("Failed to find tree: {}", e))?;

                let parent_commit = repo.head()
                    .ok()
                    .and_then(|h| h.target())
                    .and_then(|oid| repo.find_commit(oid).ok());

                let parents: Vec<&git2::Commit> = parent_commit.iter().collect();

                let commit_id = repo.commit(
                    Some("HEAD"),
                    &signature,
                    &signature,
                    &message,
                    &tree,
                    &parents,
                ).map_err(|e| format!("Failed to commit: {}", e))?;

                Ok(commit_id.to_string())
            }
            Err(e) => Err(format!("Not a git repository: {}", e))
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_push(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitPushRequest,
) -> Result<(), String> {
    // Note: git2 push requires callbacks for authentication
    // For simplicity, we'll use git command
    let path = request.path;
    let remote = request.remote.unwrap_or_else(|| "origin".to_string());
    let branch = request.branch;

    info!("Pushing to remote: {}", remote);

    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("-C").arg(&path).arg("push").arg(&remote);
    
    if let Some(branch) = branch {
        cmd.arg(&branch);
    }

    let output = cmd.output().await
        .map_err(|e| format!("Failed to execute git push: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Git push failed: {}", stderr))
    }
}

#[tauri::command]
pub async fn git_pull(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitPullRequest,
) -> Result<String, String> {
    let path = request.path;
    let remote = request.remote.unwrap_or_else(|| "origin".to_string());
    let branch = request.branch;

    info!("Pulling from remote: {}", remote);

    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("-C").arg(&path).arg("pull").arg(&remote);
    
    if let Some(branch) = branch {
        cmd.arg(&branch);
    }

    let output = cmd.output().await
        .map_err(|e| format!("Failed to execute git pull: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Git pull failed: {}", stderr))
    }
}

#[tauri::command]
pub async fn git_log(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitLogRequest,
) -> Result<Vec<GitCommit>, String> {
    let path = PathBuf::from(&request.path);
    let limit = request.limit.unwrap_or(20);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                let mut commits = Vec::new();
                let mut revwalk = repo.revwalk()
                    .map_err(|e| format!("Failed to create revwalk: {}", e))?;
                
                revwalk.push_head()
                    .map_err(|e| format!("Failed to push head: {}", e))?;

                for (i, oid) in revwalk.enumerate() {
                    if i >= limit {
                        break;
                    }

                    if let Ok(oid) = oid {
                        if let Ok(commit) = repo.find_commit(oid) {
                            let message = commit.message()
                                .unwrap_or("")
                                .lines()
                                .next()
                                .unwrap_or("")
                                .to_string();

                            commits.push(GitCommit {
                                hash: oid.to_string(),
                                short_hash: oid.to_string().chars().take(7).collect(),
                                message,
                                author: commit.author().name()
                                    .unwrap_or("Unknown")
                                    .to_string(),
                                timestamp: commit.time().seconds() as u64,
                            });
                        }
                    }
                }

                Ok(commits)
            }
            Err(e) => Err(format!("Not a git repository: {}", e))
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_branches(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitBranchesRequest,
) -> Result<Vec<GitBranch>, String> {
    let path = PathBuf::from(&request.path);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                let current_branch = repo.head()
                    .ok()
                    .and_then(|h| h.shorthand().map(|s| s.to_string()));

                let mut branches = Vec::new();
                
                // Local branches
                if let Ok(branch_iter) = repo.branches(Some(git2::BranchType::Local)) {
                    for branch in branch_iter.flatten() {
                        if let Some(name) = branch.0.name().ok().flatten() {
                            branches.push(GitBranch {
                                name: name.to_string(),
                                is_current: current_branch.as_ref() == Some(&name.to_string()),
                                is_remote: false,
                                upstream: branch.0.upstream().ok()
                                    .and_then(|u| u.name().ok().flatten().map(|s| s.to_string())),
                            });
                        }
                    }
                }

                // Remote branches
                if let Ok(branch_iter) = repo.branches(Some(git2::BranchType::Remote)) {
                    for branch in branch_iter.flatten() {
                        if let Some(name) = branch.0.name().ok().flatten() {
                            branches.push(GitBranch {
                                name: name.to_string(),
                                is_current: false,
                                is_remote: true,
                                upstream: None,
                            });
                        }
                    }
                }

                Ok(branches)
            }
            Err(e) => Err(format!("Not a git repository: {}", e))
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_checkout(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitCheckoutRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    let branch_name = request.branch;
    let create = request.create.unwrap_or(false);

    tokio::task::spawn_blocking(move || {
        match git2::Repository::open(&path) {
            Ok(repo) => {
                if create {
                    // Create new branch
                    let commit = repo.head()
                        .ok()
                        .and_then(|h| h.target())
                        .and_then(|oid| repo.find_commit(oid).ok())
                        .ok_or("No commit to branch from")?;

                    repo.branch(&branch_name, &commit, false)
                        .map_err(|e| format!("Failed to create branch: {}", e))?;
                }

                // Checkout branch
                let (object, reference) = repo.revparse_ext(&branch_name)
                    .map_err(|e| format!("Failed to find branch: {}", e))?;

                repo.checkout_tree(&object, None)
                    .map_err(|e| format!("Failed to checkout tree: {}", e))?;

                if let Some(reference) = reference {
                    repo.set_head(reference.name().unwrap())
                        .map_err(|e| format!("Failed to set head: {}", e))?;
                }

                Ok(())
            }
            Err(e) => Err(format!("Not a git repository: {}", e))
        }
    }).await.map_err(|e| format!("Task failed: {}", e))? 
}

#[tauri::command]
pub async fn git_clone(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GitCloneRequest,
) -> Result<(), String> {
    let url = request.url;
    let path = PathBuf::from(&request.path);
    let depth = request.depth;

    info!("Cloning repository: {} to {:?}", url, path);

    // Use git command for clone (easier than git2 for this)
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("clone");
    
    if let Some(d) = depth {
        cmd.arg("--depth").arg(d.to_string());
    }
    
    cmd.arg(&url).arg(&path);

    let output = cmd.output().await
        .map_err(|e| format!("Failed to execute git clone: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Git clone failed: {}", stderr))
    }
}
