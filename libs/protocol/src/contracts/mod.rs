//! BI-IDE API Contracts v1
//! Canonical contract definitions for all API interactions
//! Version: 1.0.0
//! Last Updated: 2026-03-02

pub mod v1;

pub use v1::*;

/// Current API version
pub const API_VERSION: &str = "v1";

/// Contract version for compatibility checking
pub const CONTRACT_VERSION: &str = "1.0.0";

/// Minimum supported client version
pub const MIN_CLIENT_VERSION: &str = "1.0.0";

/// Error codes for contract violations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractErrorCode {
    /// Version mismatch between client and server
    VersionMismatch = 1001,
    /// Endpoint deprecated
    EndpointDeprecated = 1002,
    /// Invalid request format
    InvalidRequestFormat = 1003,
    /// Required field missing
    MissingRequiredField = 1004,
    /// Invalid field type
    InvalidFieldType = 1005,
    /// Contract not found
    ContractNotFound = 1006,
}

impl ContractErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VersionMismatch => "VERSION_MISMATCH",
            Self::EndpointDeprecated => "ENDPOINT_DEPRECATED",
            Self::InvalidRequestFormat => "INVALID_REQUEST_FORMAT",
            Self::MissingRequiredField => "MISSING_REQUIRED_FIELD",
            Self::InvalidFieldType => "INVALID_FIELD_TYPE",
            Self::ContractNotFound => "CONTRACT_NOT_FOUND",
        }
    }
}

/// Standard API response wrapper
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub meta: ResponseMeta,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResponseMeta {
    pub request_id: String,
    pub timestamp: u64,
    pub api_version: String,
    pub contract_version: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T, request_id: String) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            meta: ResponseMeta {
                request_id,
                timestamp: crate::now_ms(),
                api_version: API_VERSION.to_string(),
                contract_version: CONTRACT_VERSION.to_string(),
            },
        }
    }

    pub fn error(code: impl Into<String>, message: impl Into<String>, request_id: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(ApiError {
                code: code.into(),
                message: message.into(),
                details: None,
            }),
            meta: ResponseMeta {
                request_id,
                timestamp: crate::now_ms(),
                api_version: API_VERSION.to_string(),
                contract_version: CONTRACT_VERSION.to_string(),
            },
        }
    }
}

/// Request context for tracing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestContext {
    pub request_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub device_id: String,
    pub user_id: Option<String>,
    pub timestamp: u64,
}

impl RequestContext {
    pub fn new(device_id: impl Into<String>) -> Self {
        let request_id = uuid::Uuid::new_v4().to_string();
        let trace_id = uuid::Uuid::new_v4().to_string().replace("-", "");
        let span_id = format!("{:016x}", rand::random::<u64>());
        
        Self {
            request_id,
            trace_id,
            span_id,
            parent_span_id: None,
            device_id: device_id.into(),
            user_id: None,
            timestamp: crate::now_ms(),
        }
    }
}

/// Pagination parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PaginationParams {
    pub page: u32,
    pub per_page: u32,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: 1,
            per_page: 20,
        }
    }
}

/// Paginated response wrapper
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: u64,
    pub page: u32,
    pub per_page: u32,
    pub total_pages: u32,
}
