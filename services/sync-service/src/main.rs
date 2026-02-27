//! BI-IDE Sync Service
//! CRDT-based synchronization service for multi-device file sync

use axum::{
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use bi_ide_protocol::sync::{SyncRequest, SyncResponse, FileOperation};

mod crdt;
mod store;
mod websocket;

use crdt::CrdtEngine;
use store::SyncStore;

/// Application state
struct AppState {
    /// CRDT engine for conflict resolution
    crdt_engine: RwLock<CrdtEngine>,
    /// Persistent storage
    store: SyncStore,
    /// Active connections per workspace
    connections: RwLock<HashMap<String, Vec<websocket::Connection>>>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    info!("Starting BI-IDE Sync Service");

    // Initialize state
    let state = Arc::new(AppState {
        crdt_engine: RwLock::new(CrdtEngine::new()),
        store: SyncStore::new().await.expect("Failed to initialize store"),
        connections: RwLock::new(HashMap::new()),
    });

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/sync", post(handle_sync))
        .route("/sync/ws", get(websocket_handler))
        .route("/workspace/:id/snapshot", get(get_snapshot))
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8001));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "bi-ide-sync",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Handle sync request
async fn handle_sync(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SyncRequest>,
) -> Result<Json<SyncResponse>, StatusCode> {
    info!(
        "Sync request from device: {} for workspace: {}",
        request.device_id, request.workspace_id
    );

    // Get stored operations since client's vector clock
    let stored_ops = state
        .store
        .get_operations_since(&request.workspace_id, &request.since_vector_clock)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Merge client operations
    let mut new_ops = Vec::new();
    for op in request.local_operations {
        if state.crdt_engine.write().await.apply_operation(&op) {
            // Operation was new and applied
            state
                .store
                .store_operation(&request.workspace_id, &op)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            new_ops.push(op);
        }
    }

    // Get current vector clock
    let server_vector_clock = state
        .store
        .get_vector_clock(&request.workspace_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Broadcast new operations to other connected clients
    broadcast_operations(&state, &request.workspace_id, &new_ops).await;

    Ok(Json(SyncResponse {
        server_vector_clock,
        operations: stored_ops,
        conflicts: vec![], // TODO: Handle conflicts
    }))
}

/// WebSocket handler for real-time sync
async fn websocket_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket::handle_socket(socket, state))
}

/// Get workspace snapshot
async fn get_snapshot(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(workspace_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let snapshot = state
        .store
        .get_snapshot(&workspace_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!(snapshot)))
}

/// Broadcast operations to connected clients
async fn broadcast_operations(state: &AppState, workspace_id: &str, ops: &[FileOperation]) {
    let connections = state.connections.read().await;
    
    if let Some(conns) = connections.get(workspace_id) {
        for conn in conns {
            // Send operation to client
            // TODO: Implement actual send
        }
    }
}

impl AppState {
    async fn add_connection(&self, workspace_id: &str, conn: websocket::Connection) {
        let mut connections = self.connections.write().await;
        connections
            .entry(workspace_id.to_string())
            .or_insert_with(Vec::new)
            .push(conn);
    }

    async fn remove_connection(&self, workspace_id: &str, conn_id: &str) {
        let mut connections = self.connections.write().await;
        if let Some(conns) = connections.get_mut(workspace_id) {
            conns.retain(|c| c.id != conn_id);
        }
    }
}
