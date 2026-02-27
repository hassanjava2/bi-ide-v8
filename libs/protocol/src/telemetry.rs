//! Telemetry and training data contracts

use serde::{Deserialize, Serialize};

/// Types of telemetry events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    /// Code edit event
    CodeEdit {
        file_path: String,
        language: String,
        edit_size: usize,
        is_autocomplete: bool,
    },
    /// Build/compilation event
    Build {
        success: bool,
        duration_ms: u64,
        error_count: u32,
        warning_count: u32,
    },
    /// Test execution event
    TestRun {
        passed: u32,
        failed: u32,
        skipped: u32,
        duration_ms: u64,
    },
    /// AI suggestion interaction
    AiSuggestion {
        suggestion_type: String,
        accepted: bool,
        time_to_accept_ms: Option<u64>,
    },
    /// Error/crash event
    Error {
        error_type: String,
        message: String,
        stack_trace: Option<String>,
    },
    /// Performance metric
    Performance {
        metric_name: String,
        value: f64,
        unit: String,
    },
}

/// A single telemetry record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub event_id: String,
    pub timestamp: u64,
    pub device_id: String,
    pub workspace_id: String,
    pub event: TelemetryEvent,
    pub session_id: String,
    pub privacy_level: PrivacyLevel,
}

/// Privacy levels for data redaction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrivacyLevel {
    /// Public - can be shared
    Public,
    /// Anonymized - identifiers removed
    Anonymized,
    /// Aggregated - statistical only
    Aggregated,
    /// Private - local only, never uploaded
    Private,
}

/// Telemetry batch upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBatch {
    pub device_id: String,
    pub batch_id: String,
    pub records: Vec<TelemetryRecord>,
    pub uploaded_at: u64,
}

/// Training artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingArtifact {
    pub artifact_id: String,
    pub artifact_type: ArtifactType,
    pub version: String,
    pub created_at: u64,
    pub device_id: String,
    pub parent_artifact_id: Option<String>,
    pub metrics: TrainingMetrics,
    pub size_bytes: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    ModelCheckpoint,
    LoraAdapter,
    Embedding,
    Dataset,
    Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub perplexity: Option<f64>,
    pub validation_loss: Option<f64>,
    pub epochs_trained: u32,
    pub samples_processed: u64,
}

/// Training job request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub job_id: String,
    pub job_type: TrainingJobType,
    pub priority: u8, // 0-100
    pub resource_requirements: ResourceRequirements,
    pub dataset_query: DatasetQuery,
    pub base_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingJobType {
    FineTune,
    LoRA,
    Evaluation,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: usize,
    pub min_memory_gb: usize,
    pub requires_gpu: bool,
    pub min_gpu_memory_gb: Option<usize>,
    pub max_duration_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetQuery {
    pub languages: Vec<String>,
    pub file_patterns: Vec<String>,
    pub since_timestamp: Option<u64>,
    pub min_privacy_level: PrivacyLevel,
}

/// Training job status update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatusUpdate {
    pub job_id: String,
    pub device_id: String,
    pub status: JobStatus,
    pub progress_percent: f32,
    pub current_metrics: Option<TrainingMetrics>,
    pub estimated_completion: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Preparing,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}
