//! WebSocket handler for real-time sync
use axum::extract::ws::{WebSocket, Message};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error, debug};
use uuid::Uuid;

use crate::AppState;
use bi_ide_protocol::sync::{SyncRequest, FileOperation};

/// Connection handle
pub struct Connection {
    pub id: String,
    pub device_id: String,
    pub workspace_id: String,
    pub sender: mpsc::UnboundedSender<Message>,
}

/// Handle WebSocket connection
pub async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    let conn_id = Uuid::new_v4().to_string();
    let mut device_id = String::new();
    let mut workspace_id = String::new();

    info!("WebSocket connected: {}", conn_id);

    // Spawn task to forward messages from channel to WebSocket
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                debug!("Received message: {}", text);
                
                // Parse message
                match serde_json::from_str::<WsMessage>(&text) {
                    Ok(ws_msg) => {
                        match ws_msg {
                            WsMessage::Auth { device_id: id, workspace_id: ws_id } => {
                                device_id = id;
                                workspace_id = ws_id;
                                
                                // Register connection
                                let conn = Connection {
                                    id: conn_id.clone(),
                                    device_id: device_id.clone(),
                                    workspace_id: workspace_id.clone(),
                                    sender: tx.clone(),
                                };
                                state.add_connection(&workspace_id, conn).await;
                                
                                // Send ack
                                let _ = tx.send(Message::Text(
                                    serde_json::to_string(&WsResponse::AuthOk).unwrap()
                                ));
                                
                                info!("Device {} authenticated for workspace {}", 
                                      device_id, workspace_id);
                            }
                            
                            WsMessage::SyncRequest(request) => {
                                // Handle sync request
                                match handle_sync_request(&state, &request).await {
                                    Ok(response) => {
                                        let _ = tx.send(Message::Text(
                                            serde_json::to_string(&WsResponse::SyncResponse(response)).unwrap()
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Message::Text(
                                            serde_json::to_string(&WsResponse::Error(e.to_string())).unwrap()
                                        ));
                                    }
                                }
                            }
                            
                            WsMessage::Operation(op) => {
                                // Apply operation
                                if state.crdt_engine.write().await.apply_operation(&op) {
                                    // Store operation
                                    let _ = state.store.store_operation(&workspace_id, &op).await;
                                    
                                    // Broadcast to other connections
                                    broadcast_to_others(&state, &conn_id, &workspace_id, &op).await;
                                }
                            }
                            
                            WsMessage::Ping => {
                                let _ = tx.send(Message::Text(
                                    serde_json::to_string(&WsResponse::Pong).unwrap()
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse message: {}", e);
                        let _ = tx.send(Message::Text(
                            serde_json::to_string(&WsResponse::Error("Invalid message".to_string())).unwrap()
                        ));
                    }
                }
            }
            
            Message::Close(_) => {
                info!("WebSocket closed: {}", conn_id);
                break;
            }
            
            _ => {}
        }
    }

    // Clean up
    if !workspace_id.is_empty() {
        state.remove_connection(&workspace_id, &conn_id).await;
    }
    
    send_task.abort();
    info!("WebSocket disconnected: {}", conn_id);
}

async fn handle_sync_request(
    state: &AppState,
    request: &SyncRequest,
) -> anyhow::Result<bi_ide_protocol::sync::SyncResponse> {
    use bi_ide_protocol::sync::SyncResponse;
    
    // Get stored operations
    let stored_ops = state
        .store
        .get_operations_since(&request.workspace_id, &request.since_vector_clock)
        .await?;

    // Apply client operations
    for op in &request.local_operations {
        state.crdt_engine.write().await.apply_operation(op);
        state.store.store_operation(&request.workspace_id, op).await?;
    }

    // Get updated vector clock
    let server_vector_clock = state.store.get_vector_clock(&request.workspace_id).await?;

    Ok(SyncResponse {
        server_vector_clock,
        operations: stored_ops,
        conflicts: vec![],
    })
}

async fn broadcast_to_others(
    state: &AppState,
    sender_id: &str,
    workspace_id: &str,
    op: &FileOperation,
) {
    let connections = state.connections.read().await;
    
    if let Some(conns) = connections.get(workspace_id) {
        for conn in conns {
            if conn.id != sender_id {
                let msg = WsResponse::Operation(op.clone());
                let _ = conn.sender.send(Message::Text(
                    serde_json::to_string(&msg).unwrap()
                ));
            }
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
enum WsMessage {
    #[serde(rename = "auth")]
    Auth { device_id: String, workspace_id: String },
    #[serde(rename = "sync")]
    SyncRequest(SyncRequest),
    #[serde(rename = "op")]
    Operation(FileOperation),
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "type")]
enum WsResponse {
    #[serde(rename = "auth_ok")]
    AuthOk,
    #[serde(rename = "sync_response")]
    SyncResponse(bi_ide_protocol::sync::SyncResponse),
    #[serde(rename = "op")]
    Operation(FileOperation),
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error(String),
}
