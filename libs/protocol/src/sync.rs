//! Synchronization contracts - CRDT Operations

use crate::{OpId, VectorClock};
use serde::{Deserialize, Serialize};

/// Types of file operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileOpType {
    Create,
    Update,
    Delete,
    Rename { old_path: String },
    Move { old_path: String },
}

/// A single file operation in the CRDT log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperation {
    pub op_id: OpId,
    pub vector_clock: VectorClock,
    pub timestamp: u64,
    pub node_id: u64,
    pub workspace_id: String,
    pub file_path: String,
    pub op_type: FileOpType,
    pub content_hash: Option<String>, // For large files, we sync chunks
    pub content: Option<Vec<u8>>,     // For small files (< 1MB)
    pub permissions: Option<u32>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Sync request from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    pub device_id: String,
    pub workspace_id: String,
    pub since_vector_clock: VectorClock,
    pub local_operations: Vec<FileOperation>,
}

/// Sync response from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    pub server_vector_clock: VectorClock,
    pub operations: Vec<FileOperation>,
    pub conflicts: Vec<SyncConflict>,
}

/// A sync conflict that needs resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub file_path: String,
    pub local_op: FileOperation,
    pub remote_op: FileOperation,
    pub resolution: ConflictResolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    UseLocal,
    UseRemote,
    MergeRequired { algorithm: String },
}

/// File chunk for large file transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChunk {
    pub chunk_id: u64,
    pub total_chunks: u64,
    pub file_hash: String,
    pub data: Vec<u8>,
}

/// Workspace snapshot request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub workspace_id: String,
    pub device_id: String,
}

/// Workspace snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    pub workspace_id: String,
    pub vector_clock: VectorClock,
    pub files: Vec<FileSnapshot>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    pub path: String,
    pub content_hash: String,
    pub size: u64,
    pub modified_at: u64,
    pub permissions: u32,
}

/// Presence information for collaborative editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceUpdate {
    pub device_id: String,
    pub user_id: String,
    pub cursor_position: Option<CursorPosition>,
    pub selection: Option<Selection>,
    pub status: UserStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selection {
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserStatus {
    Active,
    Idle,
    Away,
    Offline,
}
