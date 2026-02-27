//! File system watching and operations
use anyhow::Result;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, debug, error};

pub struct FileWatcher {
    watcher: RwLock<Option<RecommendedWatcher>>,
    event_tx: mpsc::UnboundedSender<FileEvent>,
    event_rx: RwLock<mpsc::UnboundedReceiver<FileEvent>>,
}

#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: String,
    pub kind: FileEventKind,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum FileEventKind {
    Created,
    Modified,
    Deleted,
    Renamed(String), // old path
}

impl FileWatcher {
    pub fn new() -> Result<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Ok(Self {
            watcher: RwLock::new(None),
            event_tx,
            event_rx: RwLock::new(event_rx),
        })
    }

    pub async fn watch(&self, path: &Path) -> Result<()> {
        info!("Starting file watcher for: {:?}", path);

        let tx = self.event_tx.clone();
        let path = path.to_path_buf();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    debug!("File event: {:?}", event);
                    
                    for path in event.paths {
                        let kind = match event.kind {
                            notify::EventKind::Create(_) => FileEventKind::Created,
                            notify::EventKind::Modify(_) => FileEventKind::Modified,
                            notify::EventKind::Remove(_) => FileEventKind::Deleted,
                            _ => continue,
                        };

                        let file_event = FileEvent {
                            path: path.to_string_lossy().to_string(),
                            kind,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64,
                        };

                        let _ = tx.send(file_event);
                    }
                }
                Err(e) => {
                    error!("Watch error: {}", e);
                }
            }
        })?;

        watcher.watch(path.as_path(), RecursiveMode::Recursive)?;

        *self.watcher.write().await = Some(watcher);

        // Process events
        let mut rx = self.event_rx.write().await;
        while let Some(event) = rx.recv().await {
            self.handle_event(event).await;
        }

        Ok(())
    }

    async fn handle_event(&self, event: FileEvent) {
        debug!("Handling file event: {:?}", event);
        // TODO: Queue for sync
    }

    pub async fn stop(&self) {
        if let Some(mut watcher) = self.watcher.write().await.take() {
            let _ = watcher.unwatch(std::path::Path::new("."));
        }
    }

    pub async fn is_active(&self) -> bool {
        self.watcher.read().await.is_some()
    }
}

/// Read file contents
pub async fn read_file(path: &Path) -> Result<String> {
    Ok(tokio::fs::read_to_string(path).await?)
}

/// Write file contents
pub async fn write_file(path: &Path, content: &str) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    
    tokio::fs::write(path, content).await?;
    Ok(())
}

/// Copy file
pub async fn copy_file(from: &Path, to: &Path) -> Result<()> {
    tokio::fs::copy(from, to).await?;
    Ok(())
}

/// Delete file or directory
pub async fn delete(path: &Path) -> Result<()> {
    let metadata = tokio::fs::metadata(path).await?;
    
    if metadata.is_dir() {
        tokio::fs::remove_dir_all(path).await?;
    } else {
        tokio::fs::remove_file(path).await?;
    }
    
    Ok(())
}

/// Get file metadata
pub async fn get_metadata(path: &Path) -> Result<FileMetadata> {
    let metadata = tokio::fs::metadata(path).await?;
    
    Ok(FileMetadata {
        size: metadata.len(),
        modified: metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        created: metadata.created()?.duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        is_dir: metadata.is_dir(),
        is_file: metadata.is_file(),
    })
}

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub modified: u64,
    pub created: u64,
    pub is_dir: bool,
    pub is_file: bool,
}
