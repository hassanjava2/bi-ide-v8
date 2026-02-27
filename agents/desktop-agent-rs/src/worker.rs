//! Main agent worker
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, error, debug};

use crate::config::AgentConfig;
use crate::fs::FileWatcher;
use crate::ipc::IpcClient;
use crate::telemetry::TelemetryCollector;
use crate::training::TrainingManager;

pub struct AgentWorker {
    config: AgentConfig,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: RwLock<mpsc::Receiver<()>>,
    tasks: RwLock<Vec<JoinHandle<()>>>,
    file_watcher: Arc<FileWatcher>,
    ipc_client: Arc<IpcClient>,
    telemetry: Arc<TelemetryCollector>,
    training: Arc<TrainingManager>,
}

impl AgentWorker {
    pub async fn new(config: AgentConfig) -> Result<Self> {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        
        let file_watcher = Arc::new(FileWatcher::new()?);
        let ipc_client = Arc::new(IpcClient::new(&config.server_url).await?);
        let telemetry = Arc::new(TelemetryCollector::new(&config.telemetry).await?);
        let training = Arc::new(TrainingManager::new(&config.training).await?);

        Ok(Self {
            config,
            shutdown_tx,
            shutdown_rx: RwLock::new(shutdown_rx),
            tasks: RwLock::new(Vec::new()),
            file_watcher,
            ipc_client,
            telemetry,
            training,
        })
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting agent worker");

        // Register with server
        self.ipc_client.register(&self.config).await?;

        // Start file watcher for each workspace
        for workspace in &self.config.workspaces {
            let watcher = self.file_watcher.clone();
            let path = workspace.clone();
            
            let handle = tokio::spawn(async move {
                if let Err(e) = watcher.watch(&path).await {
                    error!("File watcher error for {:?}: {}", path, e);
                }
            });
            
            self.tasks.write().await.push(handle);
        }

        // Start telemetry collector
        let telemetry = self.telemetry.clone();
        let telemetry_handle = tokio::spawn(async move {
            telemetry.run().await;
        });
        self.tasks.write().await.push(telemetry_handle);

        // Start training manager if enabled
        if self.config.training.enabled {
            let training = self.training.clone();
            let training_handle = tokio::spawn(async move {
                training.run().await;
            });
            self.tasks.write().await.push(training_handle);
        }

        // Start heartbeat
        let ipc = self.ipc_client.clone();
        let heartbeat_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = ipc.heartbeat().await {
                    error!("Heartbeat failed: {}", e);
                }
            }
        });
        self.tasks.write().await.push(heartbeat_handle);

        // Wait for shutdown signal
        let mut rx = self.shutdown_rx.write().await;
        let _ = rx.recv().await;

        Ok(())
    }

    pub async fn shutdown(&self) {
        info!("Initiating graceful shutdown...");

        // Signal shutdown
        let _ = self.shutdown_tx.send(()).await;

        // Stop all tasks
        let mut tasks = self.tasks.write().await;
        for handle in tasks.drain(..) {
            handle.abort();
        }

        // Cleanup
        self.file_watcher.stop().await;
        
        info!("Shutdown complete");
    }

    pub async fn get_status(&self) -> AgentStatus {
        AgentStatus {
            config: self.config.clone(),
            file_watcher_active: self.file_watcher.is_active().await,
            ipc_connected: self.ipc_client.is_connected().await,
            telemetry_buffered: self.telemetry.buffered_count().await,
            training_active: self.training.is_active().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentStatus {
    pub config: AgentConfig,
    pub file_watcher_active: bool,
    pub ipc_connected: bool,
    pub telemetry_buffered: usize,
    pub training_active: bool,
}
