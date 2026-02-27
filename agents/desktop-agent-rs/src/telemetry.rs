//! Telemetry collection and upload
use anyhow::Result;
use bi_ide_protocol::telemetry::{TelemetryEvent, TelemetryRecord, TelemetryBatch, PrivacyLevel};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug};

use crate::config::TelemetryConfig;

pub struct TelemetryCollector {
    config: TelemetryConfig,
    buffer: Arc<RwLock<VecDeque<TelemetryRecord>>>,
    session_id: String,
}

impl TelemetryCollector {
    pub async fn new(config: &TelemetryConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            session_id: uuid::Uuid::new_v4().to_string(),
        })
    }

    pub async fn run(&self) {
        if !self.config.enabled {
            info!("Telemetry disabled");
            return;
        }

        info!("Telemetry collector started");

        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(self.config.upload_interval_sec)
        );

        loop {
            interval.tick().await;
            
            if let Err(e) = self.upload().await {
                tracing::error!("Telemetry upload failed: {}", e);
            }
        }
    }

    pub async fn record(&self, event: TelemetryEvent, workspace_id: String) {
        if !self.config.enabled {
            return;
        }

        let privacy_level = self.determine_privacy_level(&event);
        
        // Filter based on privacy level
        if !self.should_collect(&privacy_level) {
            return;
        }

        let record = TelemetryRecord {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: bi_ide_protocol::now_ms(),
            device_id: String::new(), // Would get from config
            workspace_id,
            event,
            session_id: self.session_id.clone(),
            privacy_level,
        };

        let mut buffer = self.buffer.write().await;
        buffer.push_back(record);

        // Trim buffer if too large
        while buffer.len() > self.config.max_buffer_size {
            buffer.pop_front();
        }
    }

    pub async fn buffered_count(&self) -> usize {
        self.buffer.read().await.len()
    }

    async fn upload(&self) -> Result<()> {
        let records: Vec<TelemetryRecord> = {
            let mut buffer = self.buffer.write().await;
            if buffer.is_empty() {
                return Ok(());
            }
            buffer.drain(..).collect()
        };

        debug!("Uploading {} telemetry records", records.len());

        let batch = TelemetryBatch {
            device_id: String::new(), // Would get from config
            batch_id: uuid::Uuid::new_v4().to_string(),
            records,
            uploaded_at: bi_ide_protocol::now_ms(),
        };

        // Would upload to server
        // For now, just log
        debug!("Telemetry batch: {:?}", batch);

        Ok(())
    }

    fn determine_privacy_level(&self, event: &TelemetryEvent) -> PrivacyLevel {
        match event {
            TelemetryEvent::Performance { .. } => PrivacyLevel::Public,
            TelemetryEvent::Build { .. } => PrivacyLevel::Aggregated,
            TelemetryEvent::TestRun { .. } => PrivacyLevel::Aggregated,
            TelemetryEvent::CodeEdit { .. } => PrivacyLevel::Anonymized,
            TelemetryEvent::AiSuggestion { .. } => PrivacyLevel::Anonymized,
            TelemetryEvent::Error { .. } => PrivacyLevel::Public,
        }
    }

    fn should_collect(&self, level: &PrivacyLevel) -> bool {
        let config_level = match self.config.privacy_level.as_str() {
            "public" => PrivacyLevel::Public,
            "anonymized" => PrivacyLevel::Anonymized,
            "aggregated" => PrivacyLevel::Aggregated,
            "private" => PrivacyLevel::Private,
            _ => PrivacyLevel::Anonymized,
        };

        // Only collect if event level is >= config level
        let level_value = |l: &PrivacyLevel| match l {
            PrivacyLevel::Public => 0,
            PrivacyLevel::Anonymized => 1,
            PrivacyLevel::Aggregated => 2,
            PrivacyLevel::Private => 3,
        };

        level_value(level) >= level_value(&config_level)
    }
}
