//! Agent configuration
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Server URL
    pub server_url: String,
    /// Authentication token
    pub token: Option<String>,
    /// Device name
    pub device_name: String,
    /// Labels for the device
    pub labels: Vec<String>,
    /// Workspace directories to watch
    pub workspaces: Vec<PathBuf>,
    /// Training configuration
    pub training: TrainingConfig,
    /// Sync configuration
    pub sync: SyncConfig,
    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Enable local training
    pub enabled: bool,
    /// Max CPU usage percent before pausing
    pub max_cpu_percent: f32,
    /// Max memory usage percent before pausing
    pub max_memory_percent: f32,
    /// Only train when on AC power
    pub require_power: bool,
    /// Allowed training hours (24h format)
    pub allowed_hours: Vec<u8>,
    /// Dataset directory
    pub dataset_dir: PathBuf,
    /// Model cache directory
    pub model_cache_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Enable sync
    pub enabled: bool,
    /// Sync server URL
    pub sync_server_url: String,
    /// Auto sync interval in seconds
    pub sync_interval_sec: u64,
    /// Conflict resolution strategy
    pub conflict_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    pub enabled: bool,
    /// Privacy level for collected data
    pub privacy_level: String,
    /// Max events to buffer before upload
    pub max_buffer_size: usize,
    /// Upload interval in seconds
    pub upload_interval_sec: u64,
}

impl AgentConfig {
    pub fn default() -> Self {
        Self {
            server_url: "http://localhost:8000".to_string(),
            token: None,
            device_name: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "desktop-agent".to_string()),
            labels: vec!["desktop".to_string(), "autonomous".to_string()],
            workspaces: vec![],
            training: TrainingConfig {
                enabled: false,
                max_cpu_percent: 50.0,
                max_memory_percent: 70.0,
                require_power: true,
                allowed_hours: (0..24).collect(),
                dataset_dir: dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("bi-ide")
                    .join("datasets"),
                model_cache_dir: dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("bi-ide")
                    .join("models"),
            },
            sync: SyncConfig {
                enabled: false,
                sync_server_url: "ws://localhost:8001".to_string(),
                sync_interval_sec: 30,
                conflict_strategy: "newest".to_string(),
            },
            telemetry: TelemetryConfig {
                enabled: true,
                privacy_level: "anonymized".to_string(),
                max_buffer_size: 1000,
                upload_interval_sec: 300,
            },
        }
    }

    pub async fn load_or_default() -> Result<Self> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide");
        
        let config_file = config_dir.join("agent.toml");

        if config_file.exists() {
            let content = tokio::fs::read_to_string(&config_file).await?;
            let config: AgentConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AgentConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub async fn save(&self) -> Result<()> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide");
        
        tokio::fs::create_dir_all(&config_dir).await?;
        
        let config_file = config_dir.join("agent.toml");
        let content = toml::to_string_pretty(self)?;
        
        tokio::fs::write(&config_file, content).await?;
        
        Ok(())
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self::default()
    }
}
