//! AI Commands
//! Integration with AI services — real HTTP to VPS/RTX

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;
use tracing::{info, warn};

use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct CouncilMessageRequest {
    pub message: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CouncilMessageResponse {
    pub response: String,
    pub source: String,
    pub confidence: f64,
    pub wise_man: String,
    pub processing_time_ms: u64,
}

/// Send a message to the AI Council via HTTP (bypasses WebKit restrictions)
#[tauri::command]
pub async fn send_council_message(
    _state: State<'_, Arc<AppState>>,
    message: String,
    session_id: Option<String>,
) -> Result<CouncilMessageResponse, String> {
    info!("Sending council message: {} chars", message.len());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    // Try RTX direct first (LAN, fast)
    let rtx_url = "http://192.168.1.164:8090/council/message";
    match try_send(&client, rtx_url, &message, &session_id, 5).await {
        Ok(resp) => {
            info!("RTX direct response received");
            return Ok(resp);
        }
        Err(e) => {
            warn!("RTX direct failed: {}, falling back to VPS", e);
        }
    }

    // Fallback to VPS
    let vps_url = "https://bi-iq.com/api/v1/council/message";
    match try_send(&client, vps_url, &message, &session_id, 55).await {
        Ok(resp) => {
            info!("VPS response received");
            Ok(resp)
        }
        Err(e) => {
            warn!("VPS also failed: {}", e);
            // Offline fallback
            Ok(CouncilMessageResponse {
                response: format!(
                    "⚡ لا يمكن الاتصال بنظام AI حالياً.\n\n🔍 تحقق من:\n• اتصال الإنترنت\n• حالة السيرفر (bi-iq.com)\n• حالة RTX 5090\n\nالخطأ: {}\n\nرسالتك محفوظة: \"{}\"",
                    e, message
                ),
                source: "offline".to_string(),
                confidence: 0.0,
                wise_man: "النظام".to_string(),
                processing_time_ms: 0,
            })
        }
    }
}

async fn try_send(
    client: &reqwest::Client,
    url: &str,
    message: &str,
    session_id: &Option<String>,
    timeout_secs: u64,
) -> Result<CouncilMessageResponse, String> {
    let body = serde_json::json!({
        "message": message,
        "context": {
            "session_id": session_id.as_deref().unwrap_or("desktop-app"),
        }
    });

    let response = client
        .post(url)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {} — {}", status, body));
    }

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    // VPS wraps in ApiResponse, RTX returns directly
    let resp_data = data.get("data").unwrap_or(&data);

    Ok(CouncilMessageResponse {
        response: resp_data
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("لا يوجد رد")
            .to_string(),
        source: resp_data
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        confidence: resp_data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5),
        wise_man: resp_data
            .get("council_member")
            .or_else(|| resp_data.get("wise_man"))
            .and_then(|v| v.as_str())
            .unwrap_or("المجلس")
            .to_string(),
        processing_time_ms: resp_data
            .get("processing_time_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    })
}

// ─── Legacy placeholder commands (kept for compatibility) ───

#[derive(Debug, Deserialize)]
pub struct ExplainCodeRequest {
    pub code: String,
    pub language: Option<String>,
    pub context: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AIResponse {
    pub explanation: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RefactorCodeRequest {
    pub code: String,
    pub language: Option<String>,
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RefactorResponse {
    pub refactored_code: String,
    pub explanation: String,
    pub changes: Vec<String>,
}

#[tauri::command]
pub async fn explain_code(
    _state: State<'_, Arc<AppState>>,
    request: ExplainCodeRequest,
) -> Result<AIResponse, String> {
    info!("Explaining code snippet ({} chars)", request.code.len());
    Ok(AIResponse {
        explanation: "AI analysis placeholder — use council message for real AI".to_string(),
        suggestions: vec![],
    })
}

#[tauri::command]
pub async fn refactor_code(
    _state: State<'_, Arc<AppState>>,
    request: RefactorCodeRequest,
) -> Result<RefactorResponse, String> {
    info!("Refactoring code snippet ({} chars)", request.code.len());
    Ok(RefactorResponse {
        refactored_code: request.code,
        explanation: "AI refactor placeholder".to_string(),
        changes: vec![],
    })
}

#[tauri::command]
pub async fn get_ai_completion(
    _state: State<'_, Arc<AppState>>,
    prompt: String,
    _max_tokens: Option<u32>,
) -> Result<String, String> {
    info!("Getting AI completion for prompt ({} chars)", prompt.len());
    Ok("// AI completion placeholder\n".to_string())
}
