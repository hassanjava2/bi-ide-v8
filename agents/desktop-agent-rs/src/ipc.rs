//! IPC client for server communication
use anyhow::Result;
use bi_ide_protocol::auth::DeviceRegisterRequest;
use reqwest::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, debug};

use crate::config::AgentConfig;

pub struct IpcClient {
    client: Client,
    server_url: String,
    device_id: Arc<parking_lot::RwLock<Option<String>>>,
    token: Arc<parking_lot::RwLock<Option<String>>>,
    connected: AtomicBool,
}

impl IpcClient {
    pub async fn new(server_url: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            server_url: server_url.to_string(),
            device_id: Arc::new(parking_lot::RwLock::new(None)),
            token: Arc::new(parking_lot::RwLock::new(None)),
            connected: AtomicBool::new(false),
        })
    }

    pub async fn register(&self, config: &AgentConfig) -> Result<()> {
        info!("Registering device with server: {}", self.server_url);

        // Generate or load device ID
        let device_id = self.get_or_create_device_id().await?;

        let request = DeviceRegisterRequest {
            device_name: config.device_name.clone(),
            platform: std::env::consts::OS.to_string(),
            hostname: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            public_key: String::new(), // Would generate keypair
            labels: config.labels.clone(),
        };

        let response = self
            .client
            .post(format!("{}/api/v1/devices/register", self.server_url))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            *self.device_id.write() = Some(device_id);
            *self.token.write() = data["access_token"].as_str().map(|s| s.to_string());
            self.connected.store(true, Ordering::SeqCst);
            
            info!("Device registered successfully");
            Ok(())
        } else {
            let error = response.text().await?;
            Err(anyhow::anyhow!("Registration failed: {}", error))
        }
    }

    pub async fn heartbeat(&self) -> Result<()> {
        let device_id = self.device_id.read().clone();
        let token = self.token.read().clone();

        if let (Some(device_id), Some(token)) = (device_id, token) {
            use bi_ide_protocol::auth::{DeviceHeartbeat, DeviceStatus, ResourceUsage};
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

            let memory_percent = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;

            let heartbeat = DeviceHeartbeat {
                device_id,
                timestamp: bi_ide_protocol::now_ms(),
                status: DeviceStatus::Online,
                current_job_id: None, // Would track current job
                resource_usage: ResourceUsage {
                    cpu_percent,
                    memory_percent,
                    disk_percent: 0.0, // Would calculate
                },
            };

            let response = self
                .client
                .post(format!("{}/api/v1/devices/heartbeat", self.server_url))
                .header("Authorization", format!("Bearer {}", token))
                .json(&heartbeat)
                .send()
                .await?;

            if response.status().is_success() {
                debug!("Heartbeat sent");
                self.connected.store(true, Ordering::SeqCst);
                Ok(())
            } else {
                self.connected.store(false, Ordering::SeqCst);
                Err(anyhow::anyhow!("Heartbeat failed: {}", response.status()))
            }
        } else {
            Err(anyhow::anyhow!("Not registered"))
        }
    }

    pub async fn claim_job(&self) -> Result<Option<serde_json::Value>> {
        let token = self.token.read().clone();
        let device_id = self.device_id.read().clone();

        if let (Some(token), Some(device_id)) = (token, device_id) {
            let response = self
                .client
                .post(format!("{}/api/v1/orchestrator/workers/{}/jobs/next", self.server_url, device_id))
                .header("Authorization", format!("Bearer {}", token))
                .json(&serde_json::json!({}))
                .send()
                .await?;

            if response.status().is_success() {
                let data: serde_json::Value = response.json().await?;
                Ok(data.get("job").cloned())
            } else {
                Err(anyhow::anyhow!("Failed to claim job: {}", response.status()))
            }
        } else {
            Err(anyhow::anyhow!("Not registered"))
        }
    }

    pub async fn update_job_status(&self, job_id: &str, status: &str, logs: Option<&str>) -> Result<()> {
        let token = self.token.read().clone();

        if let Some(token) = token {
            let payload = serde_json::json!({
                "status": status,
                "logs_tail": logs,
            });

            let response = self
                .client
                .post(format!("{}/api/v1/orchestrator/jobs/{}/status", self.server_url, job_id))
                .header("Authorization", format!("Bearer {}", token))
                .json(&payload)
                .send()
                .await?;

            if response.status().is_success() {
                Ok(())
            } else {
                Err(anyhow::anyhow!("Failed to update job status: {}", response.status()))
            }
        } else {
            Err(anyhow::anyhow!("Not registered"))
        }
    }

    pub async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn get_or_create_device_id(&self) -> Result<String> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("bi-ide");
        
        let device_id_file = config_dir.join("device_id");

        if device_id_file.exists() {
            let id = tokio::fs::read_to_string(&device_id_file).await?;
            if !id.trim().is_empty() {
                return Ok(id.trim().to_string());
            }
        }

        // Create new device ID
        let new_id = uuid::Uuid::new_v4().to_string();
        
        tokio::fs::create_dir_all(&config_dir).await?;
        tokio::fs::write(&device_id_file, &new_id).await?;

        Ok(new_id)
    }
}
