//! BI-IDE API Contracts v1.0.0
//! Canonical endpoint definitions with request/response types
//! 
//! ENDPOINT MATRIX:
//! | Domain    | Method | Path                          | Status     |
//! |-----------|--------|-------------------------------|------------|
//! | Council   | POST   | /api/v1/council/message       | Canonical  |
//! | Council   | GET    | /api/v1/council/status        | Canonical  |
//! | Training  | POST   | /api/v1/training/start        | Canonical  |
//! | Training  | GET    | /api/v1/training/status       | Canonical  |
//! | Training  | POST   | /api/v1/training/stop         | Canonical  |
//! | Sync      | POST   | /api/v1/sync                  | Canonical  |
//! | Sync      | GET    | /api/v1/sync/status           | Canonical  |
//! | Sync      | WS     | /api/v1/sync/ws               | Canonical  |
//! | Workers   | POST   | /api/v1/workers/register      | Canonical  |
//! | Workers   | POST   | /api/v1/workers/heartbeat     | Canonical  |
//! | Workers   | POST   | /api/v1/workers/apply-policy  | Canonical  |
//! | Updates   | GET    | /api/v1/updates/manifest      | Canonical  |
//! | Updates   | POST   | /api/v1/updates/report        | Canonical  |

use serde::{Deserialize, Serialize};

// =============================================================================
// COUNCIL API CONTRACTS
// =============================================================================

/// POST /api/v1/council/message
/// Send a message to the AI Council
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilMessageRequest {
    pub message: String,
    pub context: Option<String>,
    pub wise_man_id: Option<String>,
    pub conversation_id: Option<String>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilMessageResponse {
    pub response: String,
    pub wise_man_id: String,
    pub wise_man_name: String,
    pub confidence: f32,
    pub conversation_id: String,
    pub processing_time_ms: u64,
    pub sources: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub content: Option<String>,
}

/// GET /api/v1/council/status
/// Get council system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilStatusRequest {
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilStatusResponse {
    pub status: String,
    pub connected: bool,
    pub wise_men_count: u32,
    pub active_discussions: u32,
    pub messages_total: u64,
    pub last_message_at: Option<u64>,
    pub latency_ms: u64,
}

/// POST /api/v1/council/discuss
/// Start a multi-wise-man discussion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDiscussRequest {
    pub topic: String,
    pub context: Option<String>,
    pub max_turns: Option<u32>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDiscussResponse {
    pub discussion_id: String,
    pub responses: Vec<WiseManResponse>,
    pub consensus: Option<String>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WiseManResponse {
    pub wise_man_id: String,
    pub wise_man_name: String,
    pub role: String,
    pub response: String,
    pub confidence: f32,
}

// =============================================================================
// TRAINING API CONTRACTS
// =============================================================================

/// POST /api/v1/training/start
/// Start a training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStartRequest {
    pub job_type: TrainingJobType,
    pub priority: Option<u8>,
    pub dataset_query: Option<DatasetQuery>,
    pub resource_limits: Option<ResourceLimits>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingJobType {
    FineTune,
    LoRA,
    Evaluation,
    Embedding,
    Distillation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetQuery {
    pub languages: Vec<String>,
    pub file_patterns: Vec<String>,
    pub since_timestamp: Option<u64>,
    pub min_privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyLevel {
    Public,
    Anonymized,
    Encrypted,
    LocalOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_cpu_percent: Option<f32>,
    pub max_memory_gb: Option<f32>,
    pub max_gpu_memory_percent: Option<f32>,
    pub max_duration_hours: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStartResponse {
    pub job_id: String,
    pub status: TrainingJobStatus,
    pub estimated_start_time: Option<u64>,
    pub estimated_completion_time: Option<u64>,
    pub queue_position: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingJobStatus {
    Queued,
    Starting,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// GET /api/v1/training/status
/// Get training job status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatusRequest {
    pub job_id: Option<String>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatusResponse {
    pub enabled: bool,
    pub current_job: Option<TrainingJobInfo>,
    pub metrics: TrainingMetricsInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobInfo {
    pub job_id: String,
    pub job_type: TrainingJobType,
    pub status: TrainingJobStatus,
    pub progress_percent: f32,
    pub started_at: Option<u64>,
    pub estimated_completion: Option<u64>,
    pub current_epoch: Option<u32>,
    pub total_epochs: Option<u32>,
    pub current_metrics: Option<CurrentMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub samples_processed: u64,
    pub epoch: u32,
    pub total_epochs: u32,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetricsInfo {
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub total_training_time_hours: f64,
    pub last_training_at: Option<u64>,
}

/// POST /api/v1/training/stop
/// Stop/pause a training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStopRequest {
    pub job_id: String,
    pub action: StopAction,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopAction {
    Pause,
    Cancel,
    ForceStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStopResponse {
    pub job_id: String,
    pub previous_status: TrainingJobStatus,
    pub current_status: TrainingJobStatus,
    pub checkpoint_saved: bool,
    pub checkpoint_path: Option<String>,
}

// =============================================================================
// SYNC API CONTRACTS
// =============================================================================

/// POST /api/v1/sync
/// Perform sync operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    pub device_id: String,
    pub workspace_id: String,
    pub since_vector_clock: crate::VectorClock,
    pub local_operations: Vec<FileOperation>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperation {
    pub op_id: crate::OpId,
    pub file_path: String,
    pub op_type: FileOpType,
    pub content: Option<String>,
    pub content_hash: Option<String>,
    pub timestamp: u64,
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileOpType {
    Create,
    Update,
    Delete,
    Rename { old_path: String },
    Move { old_path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    pub server_vector_clock: crate::VectorClock,
    pub operations: Vec<FileOperation>,
    pub conflicts: Vec<SyncConflict>,
    pub server_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub file_path: String,
    pub local_op: FileOperation,
    pub remote_op: FileOperation,
    pub resolution: ConflictResolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    UseLocal,
    UseRemote,
    MergeRequired,
    ManualResolutionRequired,
}

/// GET /api/v1/sync/status
/// Get sync status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatusRequest {
    pub workspace_id: Option<String>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatusResponse {
    pub enabled: bool,
    pub is_connected: bool,
    pub server_url: String,
    pub last_sync: Option<u64>,
    pub pending_count: usize,
    pub conflicts_count: usize,
    pub sync_interval_seconds: u64,
}

// =============================================================================
// WORKERS API CONTRACTS
// =============================================================================

/// POST /api/v1/workers/register
/// Register a new worker device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegisterRequest {
    pub device_name: String,
    pub device_type: DeviceType,
    pub capabilities: DeviceCapabilities,
    pub public_key: String,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Desktop,
    Laptop,
    Server,
    Workstation,
    Vps,
    Embedded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub has_gpu: bool,
    pub gpu_memory_gb: Option<f32>,
    pub gpu_model: Option<String>,
    pub os: String,
    pub arch: String,
    pub supports_training: bool,
    pub supports_inference: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegisterResponse {
    pub device_id: String,
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub assigned_policies: Vec<PolicyAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyAssignment {
    pub policy_id: String,
    pub policy_type: String,
    pub parameters: serde_json::Value,
}

/// POST /api/v1/workers/heartbeat
/// Worker heartbeat with status update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatRequest {
    pub device_id: String,
    pub status: WorkerStatus,
    pub resource_usage: ResourceUsage,
    pub active_jobs: Vec<String>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    Online,
    Idle,
    Busy,
    Training,
    Offline,
    Maintenance,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub gpu_percent: Option<f32>,
    pub gpu_memory_used_gb: Option<f64>,
    pub gpu_memory_total_gb: Option<f64>,
    pub disk_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatResponse {
    pub acknowledged: bool,
    pub policy_updates: Vec<PolicyUpdate>,
    pub command_queue: Vec<WorkerCommand>,
    pub next_heartbeat_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyUpdate {
    pub policy_id: String,
    pub action: PolicyAction,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    Apply,
    Update,
    Remove,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCommand {
    pub command_id: String,
    pub command_type: String,
    pub parameters: serde_json::Value,
    pub priority: u8,
}

/// POST /api/v1/workers/apply-policy
/// Apply or update resource policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPolicyRequest {
    pub device_id: String,
    pub policy: ResourcePolicy,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePolicy {
    pub policy_id: String,
    pub mode: WorkerMode,
    pub limits: ResourceLimits,
    pub schedule: Option<SchedulePolicy>,
    pub safety: SafetyPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    Full,
    Assist,
    TrainingOnly,
    IdleOnly,
    Disabled,
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
    pub end: String,   // HH:MM format
    pub days: Option<Vec<u8>>, // 0=Sunday, 1=Monday, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    pub thermal_cutoff_c: f32,
    pub auto_pause_on_user_activity: bool,
    pub max_consecutive_hours: Option<u32>,
    pub required_break_minutes: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPolicyResponse {
    pub policy_id: String,
    pub applied_at: u64,
    pub effective_at: u64,
    pub confirmation_code: String,
}

// =============================================================================
// UPDATES API CONTRACTS
// =============================================================================

/// GET /api/v1/updates/manifest
/// Get update manifest for device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateManifestRequest {
    pub device_id: String,
    pub current_version: String,
    pub channel: UpdateChannel,
    pub platform: String,
    pub arch: String,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateChannel {
    Stable,
    Beta,
    Canary,
    Dev,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateManifestResponse {
    pub has_update: bool,
    pub version: Option<String>,
    pub download_url: Option<String>,
    pub signature_url: Option<String>,
    pub release_notes: Option<String>,
    pub critical: bool,
    pub mandatory: bool,
    pub estimated_download_size_mb: Option<f32>,
    pub rollout_percentage: Option<f32>,
}

/// POST /api/v1/updates/report
/// Report update installation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReportRequest {
    pub device_id: String,
    pub version_from: String,
    pub version_to: String,
    pub status: UpdateInstallStatus,
    pub error_message: Option<String>,
    pub install_duration_seconds: Option<u64>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateInstallStatus {
    Downloaded,
    Installing,
    Success,
    Failed,
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReportResponse {
    pub acknowledged: bool,
    pub next_check_interval_seconds: u64,
    pub emergency_action: Option<EmergencyAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmergencyAction {
    Rollback,
    DisableUpdates,
    ForceRestart,
}

// =============================================================================
// AUTH API CONTRACTS
// =============================================================================

/// POST /api/v1/auth/token
/// Refresh access token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshRequest {
    pub refresh_token: String,
    pub device_id: String,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub token_type: String,
}

/// POST /api/v1/auth/revoke
/// Revoke device access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRevokeRequest {
    pub device_id: String,
    pub reason: Option<String>,
    pub request_context: super::RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRevokeResponse {
    pub revoked: bool,
    pub effective_immediately: bool,
}
