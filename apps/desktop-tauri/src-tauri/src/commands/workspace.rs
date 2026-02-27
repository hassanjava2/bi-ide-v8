//! Workspace Commands
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;
use tracing::{info, error};

use crate::state::{AppState, WorkspaceState};
use bi_ide_protocol::VectorClock;
use bi_ide_protocol::now_ms;

#[derive(Debug, Serialize)]
pub struct WorkspaceInfo {
    pub id: String,
    pub path: String,
    pub name: String,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub modified_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ActiveWorkspace {
    pub id: Option<String>,
    pub path: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenWorkspaceRequest {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct GetWorkspaceFilesRequest {
    pub workspace_id: String,
    pub path: Option<String>,
}

#[tauri::command]
pub async fn open_workspace(
    state: State<'_, std::sync::Arc<AppState>>,
    request: OpenWorkspaceRequest,
) -> Result<WorkspaceInfo, String> {
    let path = PathBuf::from(&request.path);
    
    info!("Opening workspace: {:?}", path);

    if !path.exists() {
        return Err("Path does not exist".to_string());
    }

    if !path.is_dir() {
        return Err("Path is not a directory".to_string());
    }

    let workspace_id = path.to_string_lossy().to_string();
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace")
        .to_string();

    // Check if already open
    let existing_workspace_id = {
        let workspaces = state.workspaces.read().unwrap();
        workspaces.get(&workspace_id).map(|existing| existing.id.clone())
    };

    if let Some(existing_id) = existing_workspace_id {
        *state.current_workspace.write().unwrap() = Some(path.clone());
        return load_workspace_info(&path, &existing_id).await;
    }

    // Create new workspace state
    let workspace_state = WorkspaceState {
        id: workspace_id.clone(),
        path: path.clone(),
        vector_clock: VectorClock::new(),
        last_sync: 0,
    };

    {
        let mut workspaces = state.workspaces.write().unwrap();
        workspaces.insert(workspace_id.clone(), workspace_state);
    }

    *state.current_workspace.write().unwrap() = Some(path.clone());

    // Save config
    state.save_config().await
        .map_err(|e| format!("Failed to save config: {}", e))?;

    load_workspace_info(&path, &workspace_id).await
}

#[tauri::command]
pub async fn close_workspace(
    state: State<'_, std::sync::Arc<AppState>>,
    workspace_id: String,
) -> Result<(), String> {
    info!("Closing workspace: {}", workspace_id);

    {
        let mut workspaces = state.workspaces.write().unwrap();
        workspaces.remove(&workspace_id);
    }

    // Check if this was the current workspace
    let should_clear_current = {
        let current = state.current_workspace.read().unwrap();
        current
            .as_ref()
            .map(|current_path| current_path.to_string_lossy().to_string() == workspace_id)
            .unwrap_or(false)
    };
    if should_clear_current {
        *state.current_workspace.write().unwrap() = None;
    }

    state.save_config().await
        .map_err(|e| format!("Failed to save config: {}", e))?;

    Ok(())
}

#[tauri::command]
pub async fn get_workspace_files(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GetWorkspaceFilesRequest,
) -> Result<Vec<FileEntry>, String> {
    let workspace_id = request.workspace_id;
    let relative_path = request.path.unwrap_or_default();

    let workspace_path = {
        let workspaces = state.workspaces.read().unwrap();
        workspaces.get(&workspace_id)
            .map(|w| w.path.clone())
            .ok_or("Workspace not found")?
    };

    let target_path = if relative_path.is_empty() {
        workspace_path.clone()
    } else {
        workspace_path.join(&relative_path)
    };

    if !target_path.exists() {
        return Err("Path does not exist".to_string());
    }

    if !target_path.is_dir() {
        return Err("Path is not a directory".to_string());
    }

    let mut entries = Vec::new();

    match tokio::fs::read_dir(&target_path).await {
        Ok(mut dir) => {
            while let Ok(Some(entry)) = dir.next_entry().await {
                let metadata = entry.metadata().await.ok();
                
                let file_entry = FileEntry {
                    path: entry.path()
                        .strip_prefix(&workspace_path)
                        .unwrap_or(&entry.path())
                        .to_string_lossy()
                        .to_string(),
                    name: entry.file_name().to_string_lossy().to_string(),
                    is_dir: metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                    size: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
                    modified_at: metadata.as_ref()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                };
                
                entries.push(file_entry);
            }

            // Sort: directories first, then by name
            entries.sort_by(|a, b| {
                match (a.is_dir, b.is_dir) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                }
            });

            Ok(entries)
        }
        Err(e) => {
            error!("Failed to read directory: {}", e);
            Err(format!("Failed to read directory: {}", e))
        }
    }
}

#[tauri::command]
pub async fn get_active_workspace(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<ActiveWorkspace, String> {
    let current = state.current_workspace.read().unwrap();
    
    if let Some(ref path) = *current {
        let id = path.to_string_lossy().to_string();
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string());

        Ok(ActiveWorkspace {
            id: Some(id),
            path: Some(path.to_string_lossy().to_string()),
            name,
        })
    } else {
        Ok(ActiveWorkspace {
            id: None,
            path: None,
            name: None,
        })
    }
}

async fn load_workspace_info(path: &PathBuf, workspace_id: &str) -> Result<WorkspaceInfo, String> {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace")
        .to_string();

    // Load root level files
    let mut files = Vec::new();

    match tokio::fs::read_dir(path).await {
        Ok(mut dir) => {
            while let Ok(Some(entry)) = dir.next_entry().await {
                let metadata = entry.metadata().await.ok();
                
                files.push(FileEntry {
                    path: entry.path()
                        .strip_prefix(path)
                        .unwrap_or(&entry.path())
                        .to_string_lossy()
                        .to_string(),
                    name: entry.file_name().to_string_lossy().to_string(),
                    is_dir: metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                    size: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
                    modified_at: metadata.as_ref()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                });
            }

            files.sort_by(|a, b| {
                match (a.is_dir, b.is_dir) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                }
            });
        }
        Err(e) => {
            error!("Failed to read workspace directory: {}", e);
        }
    }

    Ok(WorkspaceInfo {
        id: workspace_id.to_string(),
        path: path.to_string_lossy().to_string(),
        name,
        files,
    })
}
