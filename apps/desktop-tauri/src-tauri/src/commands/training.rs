//! Training Commands
use serde::{Deserialize, Serialize};
use tauri::State;
use tracing::{info, error};

use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct TrainingStatus {
    pub enabled: bool,
    pub current_job: Option<CurrentJob>,
    pub metrics: TrainingMetricsInfo,
}

#[derive(Debug, Serialize)]
pub struct CurrentJob {
    pub job_id: String,
    pub job_type: String,
    pub progress_percent: f32,
    pub status: String,
    pub started_at: u64,
    pub estimated_completion: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct TrainingMetricsInfo {
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub total_training_time_hours: f64,
    pub last_training_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct TrainingJobResponse {
    pub job_id: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct StartTrainingRequest {
    pub job_type: String,
    pub priority: Option<u8>,
    pub dataset_query: Option<DatasetQuery>,
}

#[derive(Debug, Deserialize)]
pub struct DatasetQuery {
    pub languages: Vec<String>,
    pub since_days: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct PauseTrainingRequest {
    pub job_id: String,
}

#[derive(Debug, Serialize)]
pub struct TrainingMetricsResponse {
    pub current: Option<CurrentMetrics>,
    pub history: Vec<MetricPoint>,
}

#[derive(Debug, Serialize)]
pub struct CurrentMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub samples_processed: u64,
    pub epoch: u32,
    pub total_epochs: u32,
}

#[derive(Debug, Serialize)]
pub struct MetricPoint {
    pub timestamp: u64,
    pub loss: f64,
    pub accuracy: f64,
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<TrainingStatus, String> {
    let enabled = *state.training_manager.enabled.read().unwrap();
    let current_job = state.training_manager.current_job.read().unwrap();
    let metrics = state.training_manager.metrics.read().unwrap();

    let current_job_info = current_job.as_ref().map(|job_id| CurrentJob {
        job_id: job_id.clone(),
        job_type: "fine_tune".to_string(), // Would get from actual job
        progress_percent: 45.0, // Mock for now
        status: "running".to_string(),
        started_at: bi_ide_protocol::now_ms() - 3600000, // 1 hour ago
        estimated_completion: Some(bi_ide_protocol::now_ms() + 7200000), // 2 hours from now
    });

    Ok(TrainingStatus {
        enabled,
        current_job: current_job_info,
        metrics: TrainingMetricsInfo {
            jobs_completed: metrics.jobs_completed,
            jobs_failed: metrics.jobs_failed,
            total_training_time_hours: metrics.total_training_time_ms as f64 / 3600000.0,
            last_training_at: metrics.last_training_at,
        },
    })
}

#[tauri::command]
pub async fn start_training_job(
    state: State<'_, std::sync::Arc<AppState>>,
    request: StartTrainingRequest,
) -> Result<TrainingJobResponse, String> {
    use bi_ide_protocol::telemetry::{DatasetQuery, ResourceRequirements, TrainingJob, TrainingJobType};
    use reqwest::Client;

    if !state.capabilities.supports_training {
        return Err("Device does not support training".to_string());
    }

    // Check if already running
    {
        let current = state.training_manager.current_job.read().unwrap();
        if current.is_some() {
            return Err("Training job already running".to_string());
        }
    }

    let job_id = format!("job-{}", bi_ide_protocol::now_ms());

    info!("Starting training job: {} of type {}", job_id, request.job_type);

    // Create training job
    let job_type = match request.job_type.as_str() {
        "lora" => TrainingJobType::LoRA,
        "fine_tune" => TrainingJobType::FineTune,
        "evaluation" => TrainingJobType::Evaluation,
        "embedding" => TrainingJobType::Embedding,
        _ => TrainingJobType::FineTune,
    };

    let job = TrainingJob {
        job_id: job_id.clone(),
        job_type,
        priority: request.priority.unwrap_or(50),
        resource_requirements: ResourceRequirements {
            min_cpu_cores: 4,
            min_memory_gb: 16,
            requires_gpu: true,
            min_gpu_memory_gb: Some(8),
            max_duration_hours: 24,
        },
        dataset_query: DatasetQuery {
            languages: request.dataset_query.as_ref()
                .map(|q| q.languages.clone())
                .unwrap_or_default(),
            file_patterns: vec!["*.py".to_string(), "*.rs".to_string(), "*.ts".to_string()],
            since_timestamp: request.dataset_query
                .as_ref()
                .and_then(|q| q.since_days)
                .map(|days| bi_ide_protocol::now_ms() - (days as u64 * 86400000)),
            min_privacy_level: bi_ide_protocol::telemetry::PrivacyLevel::Anonymized,
        },
        base_model: "code-7b-base".to_string(),
    };

    // If sync is enabled, submit to server
    if *state.sync_manager.enabled.read().unwrap() {
        let server_url = state.sync_manager.server_url.read().unwrap().clone();
        let token = { state.auth.read().unwrap().access_token.clone() };

        if let Some(token) = token {
            let client = Client::new();
            
            let response = client
                .post(format!("{}/api/v1/training/jobs", server_url))
                .header("Authorization", format!("Bearer {}", token))
                .json(&job)
                .send()
                .await
                .map_err(|e| format!("Failed to submit job: {}", e))?;

            if !response.status().is_success() {
                let error = response.text().await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(format!("Failed to submit job: {}", error));
            }
        }
    }

    // Update state
    {
        let mut current = state.training_manager.current_job.write().unwrap();
        *current = Some(job_id.clone());
    }

    // Spawn training task
    let state_clone = state.inner().clone();
    tauri::async_runtime::spawn(async move {
        run_training_job(state_clone, job).await;
    });

    Ok(TrainingJobResponse {
        job_id,
        status: "started".to_string(),
    })
}

#[tauri::command]
pub async fn pause_training_job(
    state: State<'_, std::sync::Arc<AppState>>,
    request: PauseTrainingRequest,
) -> Result<(), String> {
    info!("Pausing training job: {}", request.job_id);

    // Would signal the training process to pause
    // For now, just clear the current job
    {
        let mut current = state.training_manager.current_job.write().unwrap();
        *current = None;
    }

    Ok(())
}

#[tauri::command]
pub async fn get_training_metrics(
    state: State<'_, std::sync::Arc<AppState>>,
) -> Result<TrainingMetricsResponse, String> {
    // Mock metrics for now
    // In reality, this would read from the training process
    
    let current = if state.training_manager.current_job.read().unwrap().is_some() {
        Some(CurrentMetrics {
            loss: 0.0234,
            accuracy: 0.945,
            samples_processed: 150000,
            epoch: 5,
            total_epochs: 10,
        })
    } else {
        None
    };

    let history = vec![
        MetricPoint { timestamp: bi_ide_protocol::now_ms() - 3600000, loss: 0.05, accuracy: 0.85 },
        MetricPoint { timestamp: bi_ide_protocol::now_ms() - 2400000, loss: 0.04, accuracy: 0.88 },
        MetricPoint { timestamp: bi_ide_protocol::now_ms() - 1200000, loss: 0.03, accuracy: 0.91 },
        MetricPoint { timestamp: bi_ide_protocol::now_ms() - 600000, loss: 0.025, accuracy: 0.93 },
    ];

    Ok(TrainingMetricsResponse { current, history })
}

async fn run_training_job(
    state: std::sync::Arc<AppState>,
    job: bi_ide_protocol::telemetry::TrainingJob,
) {
    use bi_ide_protocol::telemetry::TrainingStatusUpdate;

    info!("Training job {} started", job.job_id);

    let start_time = bi_ide_protocol::now_ms();

    // Simulate training process
    // In reality, this would:
    // 1. Load dataset from local telemetry
    // 2. Initialize model
    // 3. Run training loop
    // 4. Save checkpoints periodically
    // 5. Upload artifacts

    for epoch in 0..10 {
        // Check if job was cancelled
        {
            let current = state.training_manager.current_job.read().unwrap();
            if current.as_ref() != Some(&job.job_id) {
                info!("Training job {} cancelled", job.job_id);
                return;
            }
        }

        // Simulate epoch
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;

        // Update progress
        let progress = ((epoch + 1) as f32 / 10.0) * 100.0;
        
        let update = TrainingStatusUpdate {
            job_id: job.job_id.clone(),
            device_id: state.device_id.clone(),
            status: bi_ide_protocol::telemetry::JobStatus::Running,
            progress_percent: progress,
            current_metrics: Some(bi_ide_protocol::telemetry::TrainingMetrics {
                loss: 0.05 - (0.003 * epoch as f64),
                accuracy: Some(0.85 + (0.01 * epoch as f64)),
                perplexity: Some(1.5 - (0.1 * epoch as f64)),
                validation_loss: Some(0.06 - (0.002 * epoch as f64)),
                epochs_trained: epoch as u32 + 1,
                samples_processed: (epoch as u64 + 1) * 10000,
            }),
            estimated_completion: Some(
                bi_ide_protocol::now_ms() + ((9 - epoch) as u64 * 60000)
            ),
        };

        // Send update to server if sync enabled
        if *state.sync_manager.enabled.read().unwrap() {
            // Would send update
        }

        info!("Training job {} epoch {} complete", job.job_id, epoch + 1);
    }

    // Mark as complete
    {
        let mut metrics = state.training_manager.metrics.write().unwrap();
        metrics.jobs_completed += 1;
        metrics.total_training_time_ms += bi_ide_protocol::now_ms() - start_time;
        metrics.last_training_at = Some(bi_ide_protocol::now_ms());
    }

    {
        let mut current = state.training_manager.current_job.write().unwrap();
        *current = None;
    }

    info!("Training job {} completed", job.job_id);
}
