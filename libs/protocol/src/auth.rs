//! Authentication contracts

use serde::{Deserialize, Serialize};

/// Device registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRegisterRequest {
    pub device_name: String,
    pub platform: String,  // windows, macos, linux
    pub hostname: String,
    pub public_key: String,
    pub labels: Vec<String>,
}

/// Device registration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRegisterResponse {
    pub device_id: String,
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
}

/// Token refresh request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshRequest {
    pub refresh_token: String,
}

/// Token refresh response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshResponse {
    pub access_token: String,
    pub expires_at: u64,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub has_gpu: bool,
    pub gpu_model: Option<String>,
    pub supports_training: bool,
}

/// Device heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHeartbeat {
    pub device_id: String,
    pub timestamp: u64,
    pub status: DeviceStatus,
    pub current_job_id: Option<String>,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceStatus {
    Online,
    Busy,
    Offline,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_percent: f32,
}
