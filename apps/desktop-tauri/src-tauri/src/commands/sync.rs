//! Sync Commands
use serde::{Deserialize, Serialize};
use tauri::State;
use tracing::info;

use crate::state::AppState;
use bi_ide_protocol::sync::FileOperation;
use tauri::Emitter;

#[derive(Debug, Serialize)]
pub struct SyncStatus {
    pub enabled: bool,
    pub server_url: String,
    pub is_connected: bool,
    pub last_sync: Option<u64>,
    pub pending_count: usize,
    pub conflicts_count: usize,
}

#[derive(Debug, Serialize)]
pub struct PendingOperations {
    pub operations: Vec<FileOperationInfo>,
}

#[derive(Debug, Serialize)]
pub struct FileOperationInfo {
    pub id: String,
    pub file_path: String,
    pub op_type: String,
    pub timestamp: u64,
}

#[derive(Debug, Deserialize)]
pub struct ForceSyncRequest {
    pub workspace_id: Option<String>,
}

#[tauri::command]
pub async fn get_sync_status(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<SyncStatus, String> {
    let enabled = *state.sync_manager.enabled.read().unwrap();
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let pending = state.sync_manager.pending_ops.read().unwrap();

    // Get last sync from workspaces
    let last_sync = state.workspaces.read().unwrap()
        .values()
        .map(|w| w.last_sync)
        .max();

    Ok(SyncStatus {
        enabled,
        server_url,
        is_connected: enabled, // TODO: Actually check connection
        last_sync,
        pending_count: pending.len(),
        conflicts_count: 0, // TODO: Track conflicts
    })
}

#[tauri::command]
pub async fn force_sync(
    state: State<'_, std::sync::Arc<AppState>>,
    app_handle: tauri::AppHandle,
    request: ForceSyncRequest,
) -> Result<(), String> {
    info!("Force sync requested");

    if !*state.sync_manager.enabled.read().unwrap() {
        return Err("Sync is not enabled".to_string());
    }

    let workspace_id = request.workspace_id.or_else(|| {
        state.current_workspace.read().unwrap()
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
    });

    if let Some(workspace_id) = workspace_id {
        // Perform sync
        perform_sync(&state, &app_handle, &workspace_id).await?;
    }

    Ok(())
}

#[tauri::command]
pub async fn get_pending_operations(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<PendingOperations, String> {
    let pending = state.sync_manager.pending_ops.read().unwrap();
    
    let operations: Vec<FileOperationInfo> = pending
        .iter()
        .map(|op| FileOperationInfo {
            id: op.op_id.to_string(),
            file_path: op.file_path.clone(),
            op_type: format!("{:?}", op.op_type),
            timestamp: op.timestamp,
        })
        .collect();

    Ok(PendingOperations { operations })
}

async fn perform_sync(
    state: &std::sync::Arc<AppState>,
    app_handle: &tauri::AppHandle,
    workspace_id: &str,
) -> Result<(), String> {
    use bi_ide_protocol::sync::SyncRequest;
    use reqwest::Client;

    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let token = {
        let auth = state.auth.read().unwrap();
        auth.access_token.clone().ok_or("Not authenticated")?
    };

    let client = Client::new();

    // Get current vector clock for workspace
    let vector_clock = state.workspaces.read().unwrap()
        .get(workspace_id)
        .map(|w| w.vector_clock.clone())
        .unwrap_or_default();

    // Get pending operations
    let local_operations: Vec<FileOperation> = {
        let pending = state.sync_manager.pending_ops.read().unwrap();
        pending.clone()
    };

    let sync_request = SyncRequest {
        device_id: state.device_id.clone(),
        workspace_id: workspace_id.to_string(),
        since_vector_clock: vector_clock,
        local_operations,
    };

    let response = client
        .post(format!("{}/api/v1/sync", server_url))
        .header("Authorization", format!("Bearer {}", token))
        .json(&sync_request)
        .send()
        .await
        .map_err(|e| format!("Sync request failed: {}", e))?;

    if !response.status().is_success() {
        let error = response.text().await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Sync failed: {}", error));
    }

    let sync_response: bi_ide_protocol::sync::SyncResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse sync response: {}", e))?;

    // Apply remote operations
    for op in &sync_response.operations {
        apply_operation(state, app_handle, op).await?;
    }

    // Clear local pending operations that were accepted
    {
        let mut pending = state.sync_manager.pending_ops.write().unwrap();
        pending.clear();
    }

    // Update workspace vector clock
    {
        let mut workspaces = state.workspaces.write().unwrap();
        if let Some(workspace) = workspaces.get_mut(workspace_id) {
            workspace.vector_clock = sync_response.server_vector_clock;
            workspace.last_sync = bi_ide_protocol::now_ms();
        }
    }

    // Emit sync complete event
    app_handle.emit("sync-complete", serde_json::json!({
        "workspace_id": workspace_id,
        "operations_applied": sync_response.operations.len(),
    })).map_err(|e| e.to_string())?;

    info!("Sync completed: {} operations applied", sync_response.operations.len());

    Ok(())
}

async fn apply_operation(
    _state: &std::sync::Arc<AppState>,
    app_handle: &tauri::AppHandle,
    op: &FileOperation,
) -> Result<(), String> {
    use std::path::PathBuf;

    let path = PathBuf::from(&op.file_path);

    match &op.op_type {
        bi_ide_protocol::sync::FileOpType::Create |
        bi_ide_protocol::sync::FileOpType::Update => {
            if let Some(content) = &op.content {
                tokio::fs::write(&path, content).await
                    .map_err(|e| format!("Failed to write file: {}", e))?;
            }
        }
        bi_ide_protocol::sync::FileOpType::Delete => {
            if path.exists() {
                if path.is_dir() {
                    tokio::fs::remove_dir_all(&path).await
                        .map_err(|e| format!("Failed to remove dir: {}", e))?;
                } else {
                    tokio::fs::remove_file(&path).await
                        .map_err(|e| format!("Failed to remove file: {}", e))?;
                }
            }
        }
        bi_ide_protocol::sync::FileOpType::Rename { old_path } => {
            let old = PathBuf::from(old_path);
            tokio::fs::rename(&old, &path).await
                .map_err(|e| format!("Failed to rename: {}", e))?;
        }
        bi_ide_protocol::sync::FileOpType::Move { old_path } => {
            let old = PathBuf::from(old_path);
            tokio::fs::rename(&old, &path).await
                .map_err(|e| format!("Failed to move: {}", e))?;
        }
    }

    // Emit file changed event
    app_handle.emit("file-changed", serde_json::json!({
        "path": op.file_path,
        "op_type": format!("{:?}", op.op_type),
    })).map_err(|e| e.to_string())?;

    Ok(())
}

