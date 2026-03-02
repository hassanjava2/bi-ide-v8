//! Update Commands
//! Checks update manifest from control plane and reports update status

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tracing::{error, info};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CheckForUpdatesRequest {
    pub current_version: String,
    pub channel: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct UpdateManifest {
    pub version: String,
    pub critical: bool,
    pub size_mb: f32,
    pub download_url: Option<String>,
    pub release_notes: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CheckForUpdatesResponse {
    pub has_update: bool,
    pub manifest: Option<UpdateManifest>,
}

#[derive(Debug, Deserialize)]
pub struct ReportUpdateRequest {
    pub version_from: String,
    pub version_to: String,
    pub status: String,
    pub error_message: Option<String>,
}

#[tauri::command]
pub async fn check_for_updates(
    state: State<'_, Arc<AppState>>,
    request: CheckForUpdatesRequest,
) -> Result<CheckForUpdatesResponse, String> {
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let channel = request.channel.unwrap_or_else(|| "stable".to_string());

    let url = format!(
        "{}/api/v1/updates/manifest?device_id={}&current_version={}&channel={}",
        server_url,
        urlencoding::encode(&state.device_id),
        urlencoding::encode(&request.current_version),
        urlencoding::encode(&channel)
    );

    info!("Checking for updates from {}", url);

    let client = reqwest::Client::new();
    let mut req = client.get(&url);

    if let Some(token) = state.auth.read().unwrap().access_token.clone() {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Update check failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Update check failed with status {}", response.status()));
    }

    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Invalid update manifest payload: {}", e))?;

    let has_update = payload
        .get("has_update")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let manifest = if has_update {
        let version = payload
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .to_string();

        let critical = payload
            .get("critical")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let size_mb = payload
            .get("estimated_download_size_mb")
            .or_else(|| payload.get("size_mb"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let download_url = payload
            .get("download_url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let release_notes = payload
            .get("release_notes")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Some(UpdateManifest {
            version,
            critical,
            size_mb,
            download_url,
            release_notes,
        })
    } else {
        None
    };

    Ok(CheckForUpdatesResponse { has_update, manifest })
}

#[tauri::command]
pub async fn report_update_status(
    state: State<'_, Arc<AppState>>,
    request: ReportUpdateRequest,
) -> Result<(), String> {
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let token = state.auth.read().unwrap().access_token.clone();

    let url = format!("{}/api/v1/updates/report", server_url);
    let body = serde_json::json!({
        "device_id": state.device_id,
        "version_from": request.version_from,
        "version_to": request.version_to,
        "status": request.status,
        "error_message": request.error_message,
        "timestamp": bi_ide_protocol::now_ms(),
    });

    let client = reqwest::Client::new();
    let mut req = client.post(&url).json(&body);

    if let Some(token) = token {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to report update status: {}", e))?;

    if !response.status().is_success() {
        error!("Failed to report update status: {}", response.status());
    }

    Ok(())
}
