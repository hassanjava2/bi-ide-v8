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

/// Send a message to the AI Council via HTTP
/// Priority: 1. Brain capsules (/brain/ask) → 2. Council (/council/message) → 3. VPS fallback
#[tauri::command]
pub async fn send_council_message(
    _state: State<'_, Arc<AppState>>,
    message: String,
    session_id: Option<String>,
) -> Result<CouncilMessageResponse, String> {
    info!("AI message: {} chars, session: {:?}", message.len(), session_id);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let is_council = session_id.as_deref() == Some("council-panel");
    let rtx_base = "http://100.104.35.44:8090";

    // === Path 1: Brain capsules (for AI Assistant) ===
    if !is_council {
        let brain_url = format!("{}/brain/ask", rtx_base);
        let brain_body = serde_json::json!({
            "question": message,
            "max_tokens": 512
        });
        match try_brain_ask(&client, &brain_url, &brain_body, 30).await {
            Ok(resp) => {
                info!("Brain capsule response received");
                return Ok(resp);
            }
            Err(e) => {
                warn!("Brain capsule failed: {}, falling back to council", e);
            }
        }
    }

    // === Path 2: Council (RTX direct) ===
    let rtx_url = format!("{}/council/message", rtx_base);
    match try_send(&client, &rtx_url, &message, &session_id, 30).await {
        Ok(resp) => {
            info!("RTX council response received");
            return Ok(resp);
        }
        Err(e) => {
            warn!("RTX direct failed: {}, falling back to VPS", e);
        }
    }

    // === Path 3: VPS fallback ===
    let vps_url = "https://bi-iq.com/api/v1/council/message";
    match try_send(&client, vps_url, &message, &session_id, 55).await {
        Ok(resp) => {
            info!("VPS response received");
            Ok(resp)
        }
        Err(e) => {
            warn!("VPS also failed: {}", e);
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

async fn try_brain_ask(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    timeout_secs: u64,
) -> Result<CouncilMessageResponse, String> {
    let response = client
        .post(url)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Brain request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        return Err(format!("Brain HTTP {}", status));
    }

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Brain parse failed: {}", e))?;

    Ok(CouncilMessageResponse {
        response: data.get("answer")
            .or_else(|| data.get("response"))
            .and_then(|v| v.as_str())
            .unwrap_or("لا يوجد رد")
            .to_string(),
        source: data.get("capsule")
            .or_else(|| data.get("source"))
            .and_then(|v| v.as_str())
            .unwrap_or("brain-capsule")
            .to_string(),
        confidence: data.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8),
        wise_man: data.get("capsule")
            .and_then(|v| v.as_str())
            .unwrap_or("الكبسولة")
            .to_string(),
        processing_time_ms: data.get("processing_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    })
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

// ─── Code Intelligence Commands — routed to brain capsules ───

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

/// Route code language to capsule ID
fn language_to_capsule(language: &str) -> &str {
    match language.to_lowercase().as_str() {
        "python" | "py" => "code_python",
        "typescript" | "ts" | "javascript" | "js" => "code_typescript",
        "rust" | "rs" => "code_rust",
        "sql" => "code_sql",
        "css" | "scss" | "less" => "code_css",
        "html" => "code_web",
        _ => "code_python",
    }
}

#[tauri::command]
pub async fn explain_code(
    _state: State<'_, Arc<AppState>>,
    request: ExplainCodeRequest,
) -> Result<AIResponse, String> {
    info!("Explaining code ({} chars, lang: {:?})", request.code.len(), request.language);

    let capsule = language_to_capsule(request.language.as_deref().unwrap_or("python"));
    let prompt = format!("Explain this {} code:\n```\n{}\n```",
        request.language.as_deref().unwrap_or("code"), request.code);

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "question": prompt,
        "capsule_id": capsule,
        "max_tokens": 512
    });

    match try_brain_ask(&client, "http://100.104.35.44:8090/brain/ask", &body, 30).await {
        Ok(resp) => Ok(AIResponse {
            explanation: resp.response,
            suggestions: vec![],
        }),
        Err(e) => Ok(AIResponse {
            explanation: format!("AI غير متاح حالياً: {}", e),
            suggestions: vec![],
        }),
    }
}

#[tauri::command]
pub async fn refactor_code(
    _state: State<'_, Arc<AppState>>,
    request: RefactorCodeRequest,
) -> Result<RefactorResponse, String> {
    info!("Refactoring code ({} chars)", request.code.len());

    let capsule = language_to_capsule(request.language.as_deref().unwrap_or("python"));
    let instruction = request.instruction.as_deref().unwrap_or("Refactor for readability and performance");
    let prompt = format!("{}:\n```\n{}\n```\nReturn only the refactored code.",
        instruction, request.code);

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "question": prompt,
        "capsule_id": capsule,
        "max_tokens": 1024
    });

    match try_brain_ask(&client, "http://100.104.35.44:8090/brain/ask", &body, 30).await {
        Ok(resp) => Ok(RefactorResponse {
            refactored_code: resp.response.clone(),
            explanation: format!("Refactored by capsule: {}", resp.wise_man),
            changes: vec!["Refactored via brain capsule".to_string()],
        }),
        Err(e) => Ok(RefactorResponse {
            refactored_code: request.code,
            explanation: format!("AI غير متاح: {}", e),
            changes: vec![],
        }),
    }
}

#[tauri::command]
pub async fn get_ai_completion(
    _state: State<'_, Arc<AppState>>,
    prompt: String,
    _max_tokens: Option<u32>,
) -> Result<String, String> {
    info!("AI completion ({} chars)", prompt.len());

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "question": prompt,
        "max_tokens": _max_tokens.unwrap_or(256)
    });

    match try_brain_ask(&client, "http://100.104.35.44:8090/brain/ask", &body, 20).await {
        Ok(resp) => Ok(resp.response),
        Err(e) => Ok(format!("// AI غير متاح: {}", e)),
    }
}

/// Direct brain capsule query — for explicit capsule routing from frontend
#[tauri::command]
pub async fn send_brain_ask(
    _state: State<'_, Arc<AppState>>,
    question: String,
    capsule_id: Option<String>,
) -> Result<CouncilMessageResponse, String> {
    info!("Brain ask: {} chars, capsule: {:?}", question.len(), capsule_id);

    let client = reqwest::Client::new();
    let mut body = serde_json::json!({ "question": question, "max_tokens": 512 });
    if let Some(cid) = &capsule_id {
        body["capsule_id"] = serde_json::Value::String(cid.clone());
    }

    try_brain_ask(&client, "http://100.104.35.44:8090/brain/ask", &body, 30).await
}

/// Execute a full project via brain orchestrator
#[tauri::command]
pub async fn send_brain_project(
    _state: State<'_, Arc<AppState>>,
    command: String,
) -> Result<serde_json::Value, String> {
    info!("Brain project: {}", command);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let body = serde_json::json!({ "command": command });

    let response = client
        .post("http://100.104.35.44:8090/brain/project")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Project request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Project HTTP {}", response.status()));
    }

    response.json().await.map_err(|e| format!("Parse error: {}", e))
}
