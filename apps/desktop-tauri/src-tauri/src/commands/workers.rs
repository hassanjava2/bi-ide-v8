//! Worker Resource Governance Commands
//! Manage worker devices and apply resource policies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use tauri::State;
use tracing::info;

use crate::state::AppState;

lazy_static::lazy_static! {
    static ref WORKER_POLICIES: RwLock<HashMap<String, ResourcePolicy>> = RwLock::new(HashMap::new());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDevice {
    pub device_id: String,
    pub device_name: String,
    pub device_type: String,
    pub status: String,
    pub capabilities: DeviceCapabilities,
    pub current_usage: ResourceUsage,
    pub policy: Option<ResourcePolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub has_gpu: bool,
    pub gpu_memory_gb: Option<f32>,
    pub gpu_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub gpu_percent: Option<f32>,
    pub temperature_c: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePolicy {
    pub mode: String, // full, assist, training_only, idle_only, disabled
    pub limits: ResourceLimits,
    pub schedule: Option<SchedulePolicy>,
    pub safety: SafetyPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_max_percent: u32,
    pub ram_max_gb: u32,
    pub gpu_mem_max_percent: Option<u32>,
    pub max_duration_hours: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulePolicy {
    pub timezone: String,
    pub windows: Vec<TimeWindow>,
    pub idle_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: String, // HH:MM format
    pub end: String,
    pub days: Option<Vec<u8>>, // 0=Sunday, 1=Monday, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    pub thermal_cutoff_c: f32,
    pub auto_pause_on_user_activity: bool,
    pub max_consecutive_hours: Option<u32>,
    pub required_break_minutes: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct GetWorkersRequest {
    pub include_offline: Option<bool>,
}

#[tauri::command]
pub async fn get_workers(
    state: State<'_, std::sync::Arc<AppState>>,
    request: GetWorkersRequest,
) -> Result<Vec<WorkerDevice>, String> {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_all();

    let memory_total = sys.total_memory() as f32;
    let memory_used = sys.used_memory() as f32;
    let memory_percent = if memory_total > 0.0 {
        (memory_used / memory_total) * 100.0
    } else {
        0.0
    };

    let default_policy = ResourcePolicy {
        mode: "full".to_string(),
        limits: ResourceLimits {
            cpu_max_percent: 90,
            ram_max_gb: (memory_total / 1024.0 / 1024.0 / 1024.0 * 0.8) as u32,
            gpu_mem_max_percent: None,
            max_duration_hours: Some(8),
        },
        schedule: None,
        safety: SafetyPolicy {
            thermal_cutoff_c: 85.0,
            auto_pause_on_user_activity: true,
            max_consecutive_hours: Some(4),
            required_break_minutes: Some(30),
        },
    };

    let local_device_id = format!("worker-{}", state.device_id);
    let saved_policy = WORKER_POLICIES
        .read()
        .unwrap()
        .get(&local_device_id)
        .cloned();

    let mut workers = vec![WorkerDevice {
        device_id: local_device_id,
        device_name: hostname::get()
            .ok()
            .and_then(|h| h.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "Local Machine".to_string()),
        device_type: "desktop".to_string(),
        status: "online".to_string(),
        capabilities: get_local_capabilities(),
        current_usage: ResourceUsage {
            cpu_percent: sys.global_cpu_info().cpu_usage(),
            memory_percent,
            gpu_percent: None,
            temperature_c: None,
        },
        policy: Some(saved_policy.unwrap_or(default_policy)),
    }];

    if request.include_offline.unwrap_or(false) {
        workers.push(WorkerDevice {
            device_id: "worker-offline".to_string(),
            device_name: "Offline Device".to_string(),
            device_type: "server".to_string(),
            status: "offline".to_string(),
            capabilities: DeviceCapabilities {
                cpu_cores: 0,
                memory_gb: 0.0,
                has_gpu: false,
                gpu_memory_gb: None,
                gpu_model: None,
            },
            current_usage: ResourceUsage {
                cpu_percent: 0.0,
                memory_percent: 0.0,
                gpu_percent: None,
                temperature_c: None,
            },
            policy: None,
        });
    }

    Ok(workers)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPolicyRequest {
    pub device_id: String,
    pub policy: ResourcePolicy,
}

#[derive(Debug, Serialize)]
pub struct ApplyPolicyResponse {
    pub success: bool,
    pub confirmation_code: String,
    pub effective_at: u64,
}

#[tauri::command]
pub async fn apply_worker_policy(
    state: State<'_, std::sync::Arc<AppState>>,
    request: ApplyPolicyRequest,
) -> Result<ApplyPolicyResponse, String> {
    info!("Applying policy to worker: {}", request.device_id);
    
    // Validate policy
    if request.policy.limits.cpu_max_percent > 100 {
        return Err("CPU max percent cannot exceed 100".to_string());
    }
    if request.policy.limits.cpu_max_percent < 10 {
        return Err("CPU max percent must be at least 10".to_string());
    }
    
    // If this is the local device, apply policy immediately
    if request.device_id == format!("worker-{}", state.device_id) || request.device_id == state.device_id {
        WORKER_POLICIES
            .write()
            .unwrap()
            .insert(request.device_id.clone(), request.policy.clone());

        info!("Applying local resource policy: mode={}", request.policy.mode);
    } else {
        // Send policy to remote worker via API
        let server_url = state.sync_manager.server_url.read().unwrap().clone();
        let token = state.auth.read().unwrap().access_token.clone();
        
        if let Some(token) = token {
            let client = reqwest::Client::new();
            let response = client
                .post(format!("{}/api/v1/workers/apply-policy", server_url))
                .header("Authorization", format!("Bearer {}", token))
                .json(&request)
                .send()
                .await
                .map_err(|e| format!("Failed to send policy: {}", e))?;
            
            if !response.status().is_success() {
                return Err(format!("Policy application failed: {}", response.status()));
            }
        } else {
            return Err("Not authenticated".to_string());
        }
    }
    
    Ok(ApplyPolicyResponse {
        success: true,
        confirmation_code: format!("POLICY-{}", bi_ide_protocol::now_ms()),
        effective_at: bi_ide_protocol::now_ms() + 5000, // 5 seconds from now
    })
}

#[derive(Debug, Deserialize)]
pub struct RegisterWorkerRequest {
    pub device_name: String,
    pub device_type: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterWorkerResponse {
    pub device_id: String,
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
}

#[tauri::command]
pub async fn register_worker(
    state: State<'_, std::sync::Arc<AppState>>,
    request: RegisterWorkerRequest,
) -> Result<RegisterWorkerResponse, String> {
    info!("Registering worker: {}", request.device_name);
    
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let capabilities = get_local_capabilities();
    
    let register_request = bi_ide_protocol::contracts::v1::WorkerRegisterRequest {
        device_name: request.device_name,
        device_type: match request.device_type.as_str() {
            "desktop" => bi_ide_protocol::contracts::v1::DeviceType::Desktop,
            "laptop" => bi_ide_protocol::contracts::v1::DeviceType::Laptop,
            "server" => bi_ide_protocol::contracts::v1::DeviceType::Server,
            "workstation" => bi_ide_protocol::contracts::v1::DeviceType::Workstation,
            _ => bi_ide_protocol::contracts::v1::DeviceType::Desktop,
        },
        capabilities: bi_ide_protocol::contracts::v1::DeviceCapabilities {
            cpu_cores: capabilities.cpu_cores,
            memory_gb: capabilities.memory_gb,
            has_gpu: capabilities.has_gpu,
            gpu_memory_gb: capabilities.gpu_memory_gb,
            gpu_model: capabilities.gpu_model,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            supports_training: capabilities.has_gpu || capabilities.cpu_cores >= 8,
            supports_inference: true,
        },
        public_key: format!("bi-ide-{}", uuid::Uuid::new_v4()),
        request_context: bi_ide_protocol::contracts::RequestContext::new(&state.device_id),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/api/v1/workers/register", server_url))
        .json(&register_request)
        .send()
        .await
        .map_err(|e| format!("Registration failed: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Registration failed: {}", response.status()));
    }
    
    let result: bi_ide_protocol::contracts::v1::WorkerRegisterResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    // Store credentials
    {
        let mut auth = state.auth.write().unwrap();
        auth.access_token = Some(result.access_token.clone());
        auth.refresh_token = Some(result.refresh_token.clone());
    }
    
    Ok(RegisterWorkerResponse {
        device_id: result.device_id,
        access_token: result.access_token,
        refresh_token: result.refresh_token,
        expires_at: result.expires_at,
    })
}

fn get_local_capabilities() -> DeviceCapabilities {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};

    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_memory();

    DeviceCapabilities {
        cpu_cores: num_cpus::get() as u32,
        memory_gb: (sys.total_memory() as f32 / 1024.0 / 1024.0 / 1024.0),
        has_gpu: cfg!(target_os = "macos") || std::path::Path::new("/proc/driver/nvidia/gpus").exists(),
        gpu_memory_gb: None,
        gpu_model: None,
    }
}

#[derive(Debug, Deserialize)]
pub struct WorkerHeartbeatRequest {
    pub device_id: String,
    pub resource_usage: ResourceUsage,
}

#[tauri::command]
pub async fn send_worker_heartbeat(
    state: State<'_, std::sync::Arc<AppState>>,
    request: WorkerHeartbeatRequest,
) -> Result<bi_ide_protocol::contracts::v1::WorkerHeartbeatResponse, String> {
    let server_url = state.sync_manager.server_url.read().unwrap().clone();
    let token = state.auth.read().unwrap().access_token.clone()
        .ok_or("Not authenticated")?;
    
    let heartbeat = bi_ide_protocol::contracts::v1::WorkerHeartbeatRequest {
        device_id: request.device_id,
        status: bi_ide_protocol::contracts::v1::WorkerStatus::Online,
        resource_usage: bi_ide_protocol::contracts::v1::ResourceUsage {
            cpu_percent: request.resource_usage.cpu_percent,
            memory_percent: request.resource_usage.memory_percent,
            memory_used_gb: (request.resource_usage.memory_percent * 16.0 / 100.0) as f64,
            memory_total_gb: 16.0,
            gpu_percent: request.resource_usage.gpu_percent,
            gpu_memory_used_gb: None,
            gpu_memory_total_gb: None,
            disk_percent: 0.0,
        },
        active_jobs: vec![],
        request_context: bi_ide_protocol::contracts::RequestContext::new(&state.device_id),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/api/v1/workers/heartbeat", server_url))
        .header("Authorization", format!("Bearer {}", token))
        .json(&heartbeat)
        .send()
        .await
        .map_err(|e| format!("Heartbeat failed: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Heartbeat failed: {}", response.status()));
    }
    
    response.json().await
        .map_err(|e| format!("Failed to parse response: {}", e))
}
