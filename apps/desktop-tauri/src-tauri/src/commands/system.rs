//! System Commands
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;
use tracing::info;

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub platform: String,
    pub arch: String,
    pub version: String,
    pub hostname: String,
    pub device_id: String,
    pub app_version: String,
}

#[derive(Debug, Serialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub disk_percent: f32,
}

#[derive(Debug, Deserialize)]
pub struct OpenPathRequest {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct ShowNotificationRequest {
    pub title: String,
    pub body: String,
    pub icon: Option<String>,
}

#[tauri::command]
pub async fn get_system_info(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<SystemInfo, String> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let platform = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    
    let hostname = System::host_name()
        .unwrap_or_else(|| "unknown".to_string());

    Ok(SystemInfo {
        platform,
        arch,
        version: System::os_version()
            .unwrap_or_else(|| "unknown".to_string()),
        hostname,
        device_id: state.device_id.clone(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[tauri::command]
pub async fn get_resource_usage(
    _state: State<'_, std::sync::Arc<AppState>>,
) -> Result<ResourceUsage, String> {
    use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};

    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
    );
    sys.refresh_all();

    let cpu_percent = if !sys.cpus().is_empty() {
        sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32
    } else {
        0.0
    };

    let memory_used = sys.used_memory();
    let memory_total = sys.total_memory();
    let memory_percent = (memory_used as f32 / memory_total as f32) * 100.0;

    // Calculate disk usage
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk_percent = if let Some(disk) = disks.iter().next() {
        let total = disk.total_space();
        let available = disk.available_space();
        let used = total - available;
        (used as f32 / total as f32) * 100.0
    } else {
        0.0
    };

    Ok(ResourceUsage {
        cpu_percent,
        memory_percent,
        memory_used_gb: memory_used as f64 / 1024.0 / 1024.0 / 1024.0,
        memory_total_gb: memory_total as f64 / 1024.0 / 1024.0 / 1024.0,
        disk_percent,
    })
}

#[tauri::command]
pub async fn open_path(
    _state: State<'_, std::sync::Arc<AppState>>,
    request: OpenPathRequest,
) -> Result<(), String> {
    let path = PathBuf::from(&request.path);
    
    info!("Opening path: {:?}", path);

    #[cfg(target_os = "windows")]
    {
        match std::process::Command::new("explorer")
            .arg(&path)
            .spawn() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to open path: {}", e)),
        }
    }

    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("open")
            .arg(&path)
            .spawn() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to open path: {}", e)),
        }
    }

    #[cfg(target_os = "linux")]
    {
        match std::process::Command::new("xdg-open")
            .arg(&path)
            .spawn() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to open path: {}", e)),
        }
    }
}

#[tauri::command]
pub async fn show_notification(
    app_handle: tauri::AppHandle,
    request: ShowNotificationRequest,
) -> Result<(), String> {
    #[cfg(desktop)]
    {
        use tauri_plugin_notification::NotificationExt;
        
        app_handle.notification()
            .builder()
            .title(&request.title)
            .body(&request.body)
            .show()
            .map_err(|e| format!("Failed to show notification: {}", e))?;
    }
    
    Ok(())
}
