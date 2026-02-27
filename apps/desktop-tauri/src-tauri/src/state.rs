//! Application State Management
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use tauri::{AppHandle, Emitter};
use tracing::{info, error};
use uuid::Uuid;

use bi_ide_protocol::auth::{DeviceCapabilities, DeviceStatus};
use bi_ide_protocol::sync::WorkspaceSnapshot;
use bi_ide_protocol::VectorClock;

/// Global application state
pub struct AppState {
    /// Unique device identifier
    pub device_id: String,
    /// Current device status
    pub device_status: RwLock<DeviceStatus>,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
    /// Current workspace path
    pub current_workspace: RwLock<Option<PathBuf>>,
    /// Open workspaces
    pub workspaces: RwLock<HashMap<String, WorkspaceState>>,
    /// Sync manager
    pub sync_manager: SyncManager,
    /// Training manager
    pub training_manager: TrainingManager,
    /// File watcher handles
    pub file_watchers: RwLock<HashMap<String, FileWatcherState>>,
    /// Authentication state
    pub auth: RwLock<AuthState>,
    /// System resource usage
    pub resource_usage: RwLock<ResourceUsage>,
}

pub struct WorkspaceState {
    pub id: String,
    pub path: PathBuf,
    pub vector_clock: VectorClock,
    pub last_sync: u64,
}

pub struct SyncManager {
    pub enabled: RwLock<bool>,
    pub server_url: RwLock<String>,
    pub pending_ops: RwLock<Vec<bi_ide_protocol::sync::FileOperation>>,
}

pub struct TrainingManager {
    pub enabled: RwLock<bool>,
    pub current_job: RwLock<Option<String>>,
    pub metrics: RwLock<TrainingMetrics>,
}

pub struct TrainingMetrics {
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub total_training_time_ms: u64,
    pub last_training_at: Option<u64>,
}

pub struct FileWatcherState {
    pub path: PathBuf,
    pub workspace_id: String,
}

pub struct AuthState {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_at: Option<u64>,
}

pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_percent: f32,
}

impl AppState {
    pub fn new() -> Self {
        let device_id = Self::load_or_create_device_id();
        let capabilities = Self::detect_capabilities();

        Self {
            device_id,
            device_status: RwLock::new(DeviceStatus::Online),
            capabilities,
            current_workspace: RwLock::new(None),
            workspaces: RwLock::new(HashMap::new()),
            sync_manager: SyncManager {
                enabled: RwLock::new(false),
                server_url: RwLock::new("http://localhost:8000".to_string()),
                pending_ops: RwLock::new(Vec::new()),
            },
            training_manager: TrainingManager {
                enabled: RwLock::new(false),
                current_job: RwLock::new(None),
                metrics: RwLock::new(TrainingMetrics {
                    jobs_completed: 0,
                    jobs_failed: 0,
                    total_training_time_ms: 0,
                    last_training_at: None,
                }),
            },
            file_watchers: RwLock::new(HashMap::new()),
            auth: RwLock::new(AuthState {
                access_token: None,
                refresh_token: None,
                expires_at: None,
            }),
            resource_usage: RwLock::new(ResourceUsage {
                cpu_percent: 0.0,
                memory_percent: 0.0,
                disk_percent: 0.0,
            }),
        }
    }

    pub async fn initialize(&self, app_handle: AppHandle) -> anyhow::Result<()> {
        info!("Initializing app state for device: {}", self.device_id);

        // Load configuration
        self.load_config().await?;

        // Start background tasks
        self.spawn_background_tasks(app_handle);

        info!("App state initialized successfully");
        Ok(())
    }

    fn load_or_create_device_id() -> String {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide");
        
        let device_id_file = config_dir.join("device_id");

        if device_id_file.exists() {
            match std::fs::read_to_string(&device_id_file) {
                Ok(id) if !id.trim().is_empty() => {
                    return id.trim().to_string();
                }
                _ => {}
            }
        }

        // Create new device ID
        let new_id = Uuid::new_v4().to_string();
        
        // Ensure directory exists
        if let Err(e) = std::fs::create_dir_all(&config_dir) {
            error!("Failed to create config dir: {}", e);
            return new_id;
        }

        // Save device ID
        if let Err(e) = std::fs::write(&device_id_file, &new_id) {
            error!("Failed to save device ID: {}", e);
        }

        new_id
    }

    fn detect_capabilities() -> DeviceCapabilities {
        use sysinfo::{System, RefreshKind, CpuRefreshKind};

        let mut sys = System::new_with_specifics(
            RefreshKind::new().with_cpu(CpuRefreshKind::everything())
        );
        sys.refresh_all();

        let cpu_cores = sys.cpus().len();
        let memory_gb = sys.total_memory() / 1024 / 1024 / 1024;

        // Simple GPU detection (would need more sophisticated detection for real use)
        let has_gpu = cfg!(target_os = "macos") || std::path::Path::new("/proc/driver/nvidia/gpus").exists();

        DeviceCapabilities {
            cpu_cores,
            memory_gb: memory_gb as usize,
            has_gpu,
            gpu_model: None, // Would detect actual GPU model
            supports_training: has_gpu || memory_gb >= 16,
        }
    }

    async fn load_config(&self) -> anyhow::Result<()> {
        // Load from config file
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide");
        
        let config_file = config_dir.join("config.json");

        if config_file.exists() {
            let content = tokio::fs::read_to_string(&config_file).await?;
            let config: serde_json::Value = serde_json::from_str(&content)?;

            if let Some(sync_url) = config.get("sync_server_url").and_then(|v| v.as_str()) {
                *self.sync_manager.server_url.write().unwrap() = sync_url.to_string();
            }

            if let Some(enabled) = config.get("sync_enabled").and_then(|v| v.as_bool()) {
                *self.sync_manager.enabled.write().unwrap() = enabled;
            }

            if let Some(enabled) = config.get("training_enabled").and_then(|v| v.as_bool()) {
                *self.training_manager.enabled.write().unwrap() = enabled;
            }
        }

        Ok(())
    }

    pub async fn save_config(&self) -> anyhow::Result<()> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide");
        
        let config_file = config_dir.join("config.json");

        let config = serde_json::json!({
            "device_id": self.device_id,
            "sync_server_url": self.sync_manager.server_url.read().unwrap().clone(),
            "sync_enabled": *self.sync_manager.enabled.read().unwrap(),
            "training_enabled": *self.training_manager.enabled.read().unwrap(),
        });

        tokio::fs::create_dir_all(&config_dir).await?;
        tokio::fs::write(&config_file, serde_json::to_string_pretty(&config)?).await?;

        Ok(())
    }

    fn spawn_background_tasks(&self, app_handle: AppHandle) {
        let state_clone = self;

        // Resource monitoring task
        tauri::async_runtime::spawn({
            let app_handle = app_handle.clone();
            async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    if let Err(e) = Self::update_resource_usage(&app_handle).await {
                        error!("Resource monitoring error: {}", e);
                    }
                }
            }
        });

        // Heartbeat task
        tauri::async_runtime::spawn({
            let app_handle = app_handle.clone();
            async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    if let Err(e) = Self::send_heartbeat(&app_handle).await {
                        error!("Heartbeat error: {}", e);
                    }
                }
            }
        });
    }

    async fn update_resource_usage(app_handle: &AppHandle) -> anyhow::Result<()> {
        use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};

        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything())
        );
        sys.refresh_all();

        let cpu_usage = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32;
        let memory_usage = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;

        // Emit to frontend
        app_handle.emit("resource-usage", serde_json::json!({
            "cpu_percent": cpu_usage,
            "memory_percent": memory_usage,
        }))?;

        Ok(())
    }

    async fn send_heartbeat(app_handle: &AppHandle) -> anyhow::Result<()> {
        // Would send heartbeat to control plane
        // For now, just log it
        info!("Sending heartbeat");
        Ok(())
    }
}

