//! BI-IDE Desktop Agent v2
//! Enhanced local execution and training agent

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, error};

mod config;
mod fs;
mod git;
mod ipc;
mod telemetry;
mod training;
mod worker;

use config::AgentConfig;
use worker::AgentWorker;

#[derive(Parser, Debug)]
#[command(name = "bi-ide-desktop-agent")]
#[command(about = "BI-IDE Desktop Agent - Local execution and training")]
struct Args {
    /// Server URL
    #[arg(short, long, env = "BI_SERVER", default_value = "http://localhost:8000")]
    server: String,

    /// Device token
    #[arg(short, long, env = "BI_TOKEN")]
    token: Option<String>,

    /// Configuration file path
    #[arg(short, long, env = "BI_CONFIG")]
    config: Option<String>,

    /// Run in background mode
    #[arg(long, env = "BI_DAEMON")]
    daemon: bool,

    /// Log level
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .init();

    info!("BI-IDE Desktop Agent v{} starting", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = if let Some(config_path) = args.config {
        AgentConfig::from_file(&config_path)?
    } else {
        AgentConfig::load_or_default().await?
    };

    // Override with command line args
    let config = AgentConfig {
        server_url: args.server,
        token: args.token.or(config.token),
        ..config
    };

    // Create and start agent
    let agent = Arc::new(AgentWorker::new(config).await?);

    // Start agent
    let agent_clone = agent.clone();
    tokio::spawn(async move {
        if let Err(e) = agent_clone.run().await {
            error!("Agent error: {}", e);
        }
    });

    // Wait for shutdown signal
    info!("Agent running. Press Ctrl+C to stop.");

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received shutdown signal");
            }
            _ = sigterm.recv() => {
                info!("Received SIGTERM");
            }
        }
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c().await?;
        info!("Received shutdown signal");
    }

    // Graceful shutdown
    info!("Shutting down...");
    agent.shutdown().await;
    info!("Agent stopped");

    Ok(())
}
