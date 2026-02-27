//! Persistent storage for sync operations
use bi_ide_protocol::sync::{FileOperation, WorkspaceSnapshot};
use bi_ide_protocol::VectorClock;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use std::path::PathBuf;
use tracing::info;

/// Persistent storage for sync data
pub struct SyncStore {
    pool: Pool<Sqlite>,
}

impl SyncStore {
    pub async fn new() -> anyhow::Result<Self> {
        // Create data directory
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("bi-ide")
            .join("sync");
        
        tokio::fs::create_dir_all(&data_dir).await?;

        let db_path = data_dir.join("sync.db");
        info!("Using database: {:?}", db_path);

        // Create connection pool
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&format!("sqlite:{}", db_path.display()))
            .await?;

        // Run migrations
        Self::run_migrations(&pool).await?;

        Ok(Self { pool })
    }

    async fn run_migrations(pool: &Pool<Sqlite>) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                op_id TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                vector_clock TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                op_type TEXT NOT NULL,
                content_hash TEXT,
                content BLOB,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, op_id)
            );

            CREATE INDEX IF NOT EXISTS idx_operations_workspace 
                ON operations(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_operations_timestamp 
                ON operations(timestamp);

            CREATE TABLE IF NOT EXISTS vector_clocks (
                workspace_id TEXT PRIMARY KEY,
                clock_data TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                snapshot_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            "#
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Store a file operation
    pub async fn store_operation(
        &self,
        workspace_id: &str,
        op: &FileOperation,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO operations 
            (workspace_id, op_id, node_id, vector_clock, timestamp, file_path, op_type, content_hash, content, metadata)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#
        )
        .bind(workspace_id)
        .bind(op.op_id.to_string())
        .bind(op.node_id as i64)
        .bind(serde_json::to_string(&op.vector_clock)?)
        .bind(op.timestamp as i64)
        .bind(&op.file_path)
        .bind(serde_json::to_string(&op.op_type)?)
        .bind(&op.content_hash)
        .bind(&op.content)
        .bind(serde_json::to_string(&op.metadata)?)
        .execute(&self.pool)
        .await?;

        // Update vector clock
        self.update_vector_clock(workspace_id, &op.vector_clock).await?;

        Ok(())
    }

    /// Get operations since a vector clock
    pub async fn get_operations_since(
        &self,
        workspace_id: &str,
        since: &VectorClock,
    ) -> anyhow::Result<Vec<FileOperation>> {
        let rows = sqlx::query_as::<_, OperationRow>(
            r#"
            SELECT * FROM operations 
            WHERE workspace_id = ?1
            ORDER BY timestamp ASC
            "#
        )
        .bind(workspace_id)
        .fetch_all(&self.pool)
        .await?;

        let mut operations = Vec::new();

        for row in rows {
            let op = self.row_to_operation(row)?;
            
            // Check if this operation is newer than the since clock
            if since.get(op.node_id) < op.vector_clock.get(op.node_id) {
                operations.push(op);
            }
        }

        Ok(operations)
    }

    /// Get current vector clock for workspace
    pub async fn get_vector_clock(&self, workspace_id: &str) -> anyhow::Result<VectorClock> {
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT clock_data FROM vector_clocks WHERE workspace_id = ?1"
        )
        .bind(workspace_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some((clock_data,)) => {
                Ok(serde_json::from_str(&clock_data)?)
            }
            None => Ok(VectorClock::new()),
        }
    }

    /// Update vector clock for workspace
    async fn update_vector_clock(
        &self,
        workspace_id: &str,
        clock: &VectorClock,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO vector_clocks (workspace_id, clock_data, updated_at)
            VALUES (?1, ?2, CURRENT_TIMESTAMP)
            "#
        )
        .bind(workspace_id)
        .bind(serde_json::to_string(clock)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Store workspace snapshot
    pub async fn store_snapshot(
        &self,
        workspace_id: &str,
        snapshot: &WorkspaceSnapshot,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            INSERT INTO snapshots (workspace_id, snapshot_data)
            VALUES (?1, ?2)
            "#
        )
        .bind(workspace_id)
        .bind(serde_json::to_string(snapshot)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get latest snapshot for workspace
    pub async fn get_snapshot(
        &self,
        workspace_id: &str,
    ) -> anyhow::Result<Option<WorkspaceSnapshot>> {
        let row: Option<(String,)> = sqlx::query_as(
            r#"
            SELECT snapshot_data FROM snapshots 
            WHERE workspace_id = ?1 
            ORDER BY created_at DESC 
            LIMIT 1
            "#
        )
        .bind(workspace_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some((data,)) => Ok(Some(serde_json::from_str(&data)?)),
            None => Ok(None),
        }
    }

    /// Convert database row to FileOperation
    fn row_to_operation(&self, row: OperationRow) -> anyhow::Result<FileOperation> {
        use bi_ide_protocol::OpId;
        
        Ok(FileOperation {
            op_id: OpId::new(row.node_id as u64, row.timestamp as u64),
            vector_clock: serde_json::from_str(&row.vector_clock)?,
            timestamp: row.timestamp as u64,
            node_id: row.node_id as u64,
            workspace_id: row.workspace_id,
            file_path: row.file_path,
            op_type: serde_json::from_str(&row.op_type)?,
            content_hash: row.content_hash,
            content: row.content,
            permissions: None,
            metadata: serde_json::from_str(&row.metadata.unwrap_or_default())?,
        })
    }

    /// Clean up old operations (keep last 30 days)
    pub async fn cleanup_old_operations(&self) -> anyhow::Result<u64> {
        let result = sqlx::query(
            r#"
            DELETE FROM operations 
            WHERE timestamp < strftime('%s', 'now', '-30 days') * 1000
            "#
        )
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }
}

#[derive(sqlx::FromRow)]
struct OperationRow {
    id: i64,
    workspace_id: String,
    op_id: String,
    node_id: i64,
    vector_clock: String,
    timestamp: i64,
    file_path: String,
    op_type: String,
    content_hash: Option<String>,
    content: Option<Vec<u8>>,
    metadata: Option<String>,
    created_at: chrono::NaiveDateTime,
}
