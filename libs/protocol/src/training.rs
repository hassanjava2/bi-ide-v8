//! Training-specific contracts

use serde::{Deserialize, Serialize};

/// Model registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub name: String,
    pub version: String,
    pub base_model: String,
    pub architecture: ModelArchitecture,
    pub size_bytes: u64,
    pub checksum: String,
    pub created_at: u64,
    pub trained_by: String, // device_id
    pub evaluation_score: f64,
    pub status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Transformer,
    LSTM,
    CNN,
    Hybrid,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Draft,
    Evaluating,
    Active,
    Deprecated,
    Archived,
}

/// Model promotion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPromotionRequest {
    pub model_id: String,
    pub from_status: ModelStatus,
    pub to_status: ModelStatus,
    pub justification: String,
    pub evaluation_report: EvaluationReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub perplexity: f64,
    pub accuracy: f64,
    pub latency_ms: u64,
    pub memory_mb: u64,
    pub test_cases_passed: u32,
    pub test_cases_total: u32,
    pub regression_detected: bool,
}

/// Policy for autonomous training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPolicy {
    pub policy_id: String,
    pub version: String,
    pub max_daily_jobs: u32,
    pub max_concurrent_jobs: u32,
    pub cpu_threshold_percent: f32,
    pub memory_threshold_percent: f32,
    pub allowed_hours: Vec<u8>, // 0-23
    pub require_power_connected: bool,
    pub min_dataset_size: u64,
    pub forbidden_paths: Vec<String>,
}

/// Self-improvement proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementProposal {
    pub proposal_id: String,
    pub proposal_type: ProposalType,
    pub detected_by: String, // device_id or rule_id
    pub detection_timestamp: u64,
    pub target_component: String,
    pub current_config: serde_json::Value,
    pub proposed_config: serde_json::Value,
    pub expected_improvement: String,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalType {
    ConfigChange,
    ModelUpdate,
    PromptTemplate,
    ThresholdAdjustment,
    FeatureFlag,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    L0, // No risk - cosmetic only
    L1, // Low risk - easily reversible
    L2, // Medium risk - needs validation
    L3, // High risk - needs human approval
    L4, // Critical - automatic rejection
}

/// Policy gate decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub proposal_id: String,
    pub decision: Decision,
    pub reason: String,
    pub decided_at: u64,
    pub decided_by: String, // policy engine or human
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    AutoApprove,
    Approve,
    Reject,
    Escalate,
    RequireSandboxTest,
}

/// Canary rollout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    pub proposal_id: String,
    pub percentage: f32, // 0-100
    pub duration_minutes: u32,
    pub success_criteria: SuccessCriteria,
    pub rollback_triggers: Vec<RollbackTrigger>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub max_error_rate: f32,
    pub min_performance_improvement: f32,
    pub max_latency_regression: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackTrigger {
    pub metric: String,
    pub threshold: f64,
    pub operator: ComparisonOperator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
}
