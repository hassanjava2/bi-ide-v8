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

/// Device info for sync
#[derive(Debug, Serialize, Clone)]
pub struct SyncDevice {
    pub device_id: String,
    pub device_name: String,
    pub status: String, // "synced" | "syncing" | "conflict" | "offline"
    pub last_seen: u64,
}

#[derive(Debug, Deserialize)]
pub struct GetSyncDevicesRequest {
    pub workspace_id: String,
}

#[tauri::command]
pub async fn get_sync_status(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<SyncStatus, String> {
    let enabled = *state.sync_manager.enabled.read().unwrap();
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let pending_count = state.sync_manager.pending_ops.read().unwrap().len();
    let token = state.auth.read().unwrap().access_token.clone();

    // Get last sync from workspaces
    let last_sync = state.workspaces.read().unwrap()
        .values()
        .map(|w| w.last_sync)
        .max();

    let is_connected = if enabled {
        let client = reqwest::Client::new();
        let url = format!("{}/api/v1/sync/status", server_url);
        let mut req = client.get(url);
        if let Some(token) = token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        match req.send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    } else {
        false
    };

    Ok(SyncStatus {
        enabled,
        server_url,
        is_connected,
        last_sync,
        pending_count,
        conflicts_count: 0,
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

/// Get sync devices for a workspace
#[tauri::command]
pub async fn get_sync_devices(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GetSyncDevicesRequest,
) -> Result<Vec<SyncDevice>, String> {
    info!("Getting sync devices for workspace: {}", request.workspace_id);

    let enabled = *state.sync_manager.enabled.read().unwrap();
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let token = state.auth.read().unwrap().access_token.clone();
    let device_id = state.device_id.clone();

    // If sync is disabled, return just this device as offline
    if !enabled {
        return Ok(vec![SyncDevice {
            device_id: device_id.clone(),
            device_name: get_device_name(),
            status: "offline".to_string(),
            last_seen: bi_ide_protocol::now_ms(),
        }]);
    }

    // Try to fetch devices from server
    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/sync/devices?workspace_id={}", server_url, request.workspace_id);
    
    let mut req = client.get(&url);
    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    match req.send().await {
        Ok(response) => {
            if response.status().is_success() {
                // Parse server response
                match response.json::<Vec<ServerDevice>>().await {
                    Ok(server_devices) => {
                        let devices: Vec<SyncDevice> = server_devices
                            .into_iter()
                            .map(|d| SyncDevice {
                                device_id: d.device_id,
                                device_name: d.device_name,
                                status: d.status,
                                last_seen: d.last_seen,
                            })
                            .collect();
                        
                        // Ensure this device is in the list
                        let has_this_device = devices.iter().any(|d| d.device_id == device_id);
                        if !has_this_device {
                            let mut devices = devices;
                            devices.push(SyncDevice {
                                device_id: device_id.clone(),
                                device_name: get_device_name(),
                                status: "synced".to_string(),
                                last_seen: bi_ide_protocol::now_ms(),
                            });
                            return Ok(devices);
                        }
                        
                        Ok(devices)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse devices response: {}", e);
                        // Return local device as fallback
                        Ok(vec![SyncDevice {
                            device_id: device_id.clone(),
                            device_name: get_device_name(),
                            status: if enabled { "synced" } else { "offline" }.to_string(),
                            last_seen: bi_ide_protocol::now_ms(),
                        }])
                    }
                }
            } else {
                // Server returned error, return local device
                tracing::warn!("Server returned error for devices: {}", response.status());
                Ok(vec![SyncDevice {
                    device_id: device_id.clone(),
                    device_name: get_device_name(),
                    status: "offline".to_string(),
                    last_seen: bi_ide_protocol::now_ms(),
                }])
            }
        }
        Err(e) => {
            tracing::warn!("Failed to fetch devices from server: {}", e);
            // Return local device as fallback when offline
            Ok(vec![SyncDevice {
                device_id: device_id.clone(),
                device_name: get_device_name(),
                status: "offline".to_string(),
                last_seen: bi_ide_protocol::now_ms(),
            }])
        }
    }
}

/// Server device response structure
#[derive(Debug, Deserialize)]
struct ServerDevice {
    device_id: String,
    device_name: String,
    status: String,
    last_seen: u64,
}

/// Get the device name from system info
fn get_device_name() -> String {
    // Try to get hostname
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "Unknown Device".to_string())
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
