//! Authentication Commands
use serde::{Deserialize, Serialize};
use tauri::State;
use tracing::info;

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub is_registered: bool,
}

#[derive(Debug, Deserialize)]
pub struct RegisterDeviceRequest {
    pub server_url: String,
    pub device_name: String,
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct AuthTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
}

#[derive(Debug, Deserialize)]
pub struct SetTokenRequest {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<u64>,
}

#[tauri::command]
pub async fn get_device_id(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<String, String> {
    Ok(state.device_id.clone())
}

#[tauri::command]
pub async fn register_device(
    state: State<'_, std::sync::Arc<AppState>>,
    request: RegisterDeviceRequest,
) -> Result<AuthTokens, String> {
    use bi_ide_protocol::auth::DeviceRegisterRequest;
    use reqwest::Client;
    use sysinfo::System;

    info!("Registering device: {} with server: {}", 
          request.device_name, request.server_url);

    let client = Client::new();
    
    // First, authenticate with username/password
    let auth_response = client
        .post(format!("{}/api/v1/auth/login", request.server_url))
        .json(&serde_json::json!({
            "username": request.username,
            "password": request.password,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to server: {}", e))?;

    if !auth_response.status().is_success() {
        return Err("Invalid credentials".to_string());
    }

    let auth_data: serde_json::Value = auth_response
        .json()
        .await
        .map_err(|e| format!("Failed to parse auth response: {}", e))?;

    let access_token = auth_data["access_token"]
        .as_str()
        .ok_or("No access token in response")?;

    // Generate device keypair (for future mTLS)
    // For now, we'll use a simple registration
    let hostname = System::host_name()
        .unwrap_or_else(|| "unknown".to_string());

    let register_request = DeviceRegisterRequest {
        device_name: request.device_name,
        platform: std::env::consts::OS.to_string(),
        hostname,
        public_key: String::new(), // Would generate keypair here
        labels: vec!["desktop".to_string()],
    };

    let register_response = client
        .post(format!("{}/api/v1/devices/register", request.server_url))
        .header("Authorization", format!("Bearer {}", access_token))
        .json(&register_request)
        .send()
        .await
        .map_err(|e| format!("Failed to register device: {}", e))?;

    if !register_response.status().is_success() {
        let error_text = register_response.text().await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Device registration failed: {}", error_text));
    }

    let device_data: serde_json::Value = register_response
        .json()
        .await
        .map_err(|e| format!("Failed to parse registration response: {}", e))?;

    // Update state
    *state.sync_manager.server_url.write().unwrap() = request.server_url;
    *state.sync_manager.enabled.write().unwrap() = true;

    // Save tokens
    {
        let mut auth = state.auth.write().unwrap();
        auth.access_token = Some(access_token.to_string());
        auth.expires_at = device_data["expires_at"].as_u64();
    }

    // Save config
    state.save_config().await
        .map_err(|e| format!("Failed to save config: {}", e))?;

    Ok(AuthTokens {
        access_token: access_token.to_string(),
        refresh_token: device_data["refresh_token"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        expires_at: device_data["expires_at"]
            .as_u64()
            .unwrap_or(0),
    })
}

#[tauri::command]
pub async fn get_access_token(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<Option<String>, String> {
    let auth = state.auth.read().unwrap();
    Ok(auth.access_token.clone())
}

#[tauri::command]
pub async fn set_access_token(
    state: State<'_, std::sync::Arc<AppState>>,
    request: SetTokenRequest,
) -> Result<(), String> {
    {
        let mut auth = state.auth.write().unwrap();
        auth.access_token = Some(request.access_token);
        auth.refresh_token = request.refresh_token;
        auth.expires_at = request.expires_at;
    }

    state.save_config().await
        .map_err(|e| format!("Failed to save config: {}", e))?;

    Ok(())
}
