//! File System Commands
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;
use tracing::{info, error};

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct FileInfo {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub modified_at: Option<u64>,
    pub created_at: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ReadFileRequest {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct ReadFileResponse {
    pub content: String,
    pub encoding: String,
}

#[derive(Debug, Deserialize)]
pub struct WriteFileRequest {
    pub path: String,
    pub content: String,
    pub encoding: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ReadDirRequest {
    pub path: String,
    pub recursive: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CreateDirRequest {
    pub path: String,
    pub recursive: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteFileRequest {
    pub path: String,
    pub recursive: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct RenameFileRequest {
    pub from: String,
    pub to: String,
}

#[tauri::command]
pub async fn read_file(
    state: State<'_, std::sync::Arc<AppState>>,
    request: ReadFileRequest,
) -> Result<ReadFileResponse, String> {
    let path = PathBuf::from(&request.path);
    
    info!("Reading file: {:?}", path);

    // Security check: ensure path is within allowed directories
    if !is_path_allowed(&path) {
        return Err("Path not allowed".to_string());
    }

    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Ok(ReadFileResponse {
            content,
            encoding: "utf-8".to_string(),
        }),
        Err(e) => {
            error!("Failed to read file: {}", e);
            Err(format!("Failed to read file: {}", e))
        }
    }
}

#[tauri::command]
pub async fn write_file(
    state: State<'_, std::sync::Arc<AppState>>,
    request: WriteFileRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    
    info!("Writing file: {:?}", path);

    if !is_path_allowed(&path) {
        return Err("Path not allowed".to_string());
    }

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            error!("Failed to create parent directories: {}", e);
            return Err(format!("Failed to create parent directories: {}", e));
        }
    }

    match tokio::fs::write(&path, request.content).await {
        Ok(_) => {
            // Record operation for sync
            // TODO: Add to pending operations
            Ok(())
        }
        Err(e) => {
            error!("Failed to write file: {}", e);
            Err(format!("Failed to write file: {}", e))
        }
    }
}

#[tauri::command]
pub async fn read_dir(
    state: State<'_, std::sync::Arc<AppState>>,
    request: ReadDirRequest,
) -> Result<Vec<FileInfo>, String> {
    let path = PathBuf::from(&request.path);
    
    info!("Reading directory: {:?}", path);

    if !is_path_allowed(&path) {
        return Err("Path not allowed".to_string());
    }

    let mut entries = Vec::new();

    match tokio::fs::read_dir(&path).await {
        Ok(mut dir) => {
            while let Ok(Some(entry)) = dir.next_entry().await {
                let metadata = entry.metadata().await.ok();
                
                let file_info = FileInfo {
                    path: entry.path().to_string_lossy().to_string(),
                    name: entry.file_name().to_string_lossy().to_string(),
                    is_dir: metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                    size: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
                    modified_at: metadata.as_ref()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                    created_at: metadata.as_ref()
                        .and_then(|m| m.created().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                };
                
                entries.push(file_info);
            }
            Ok(entries)
        }
        Err(e) => {
            error!("Failed to read directory: {}", e);
            Err(format!("Failed to read directory: {}", e))
        }
    }
}

#[tauri::command]
pub async fn create_dir(
    state: State<'_, std::sync::Arc<AppState>>,
    request: CreateDirRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    
    info!("Creating directory: {:?}", path);

    if !is_path_allowed(&path) {
        return Err("Path not allowed".to_string());
    }

    let result = if request.recursive.unwrap_or(false) {
        tokio::fs::create_dir_all(&path).await
    } else {
        tokio::fs::create_dir(&path).await
    };

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("Failed to create directory: {}", e);
            Err(format!("Failed to create directory: {}", e))
        }
    }
}

#[tauri::command]
pub async fn delete_file(
    state: State<'_, std::sync::Arc<AppState>>,
    request: DeleteFileRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    
    info!("Deleting file: {:?}", path);

    if !is_path_allowed(&path) {
        return Err("Path not allowed".to_string());
    }

    let metadata = tokio::fs::metadata(&path).await
        .map_err(|e| format!("Failed to get metadata: {}", e))?;

    let result = if metadata.is_dir() {
        if request.recursive.unwrap_or(false) {
            tokio::fs::remove_dir_all(&path).await
        } else {
            tokio::fs::remove_dir(&path).await
        }
    } else {
        tokio::fs::remove_file(&path).await
    };

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("Failed to delete file: {}", e);
            Err(format!("Failed to delete file: {}", e))
        }
    }
}

#[tauri::command]
pub async fn rename_file(
    state: State<'_, std::sync::Arc<AppState>>,
    request: RenameFileRequest,
) -> Result<(), String> {
    let from = PathBuf::from(&request.from);
    let to = PathBuf::from(&request.to);
    
    info!("Renaming file: {:?} -> {:?}", from, to);

    if !is_path_allowed(&from) || !is_path_allowed(&to) {
        return Err("Path not allowed".to_string());
    }

    match tokio::fs::rename(&from, &to).await {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("Failed to rename file: {}", e);
            Err(format!("Failed to rename file: {}", e))
        }
    }
}

#[tauri::command]
pub async fn watch_path(
    state: State<'_, std::sync::Arc<AppState>>,
    path: String,
    workspace_id: String,
) -> Result<(), String> {
    // TODO: Implement file watching using notify crate
    info!("Watching path: {} for workspace: {}", path, workspace_id);
    Ok(())
}

#[tauri::command]
pub async fn unwatch_path(
    state: State<'_, std::sync::Arc<AppState>>,
    path: String,
) -> Result<(), String> {
    info!("Unwatching path: {}", path);
    Ok(())
}

fn is_path_allowed(path: &PathBuf) -> bool {
    // Check if path is within home directory or workspace
    let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
    
    if let Some(home) = dirs::home_dir() {
        if canonical.starts_with(&home) {
            return true;
        }
    }
    
    // Allow temp directory
    if let Some(temp) = std::env::temp_dir().parent() {
        if canonical.starts_with(temp) {
            return true;
        }
    }

    false
}
