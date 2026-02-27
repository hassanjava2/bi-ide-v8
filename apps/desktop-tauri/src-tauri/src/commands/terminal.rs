//! Terminal / Process Commands
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tauri::State;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, error};

use crate::state::AppState;

lazy_static::lazy_static! {
    static ref PROCESS_COUNTER: AtomicU64 = AtomicU64::new(1);
    static ref ACTIVE_PROCESSES: Arc<Mutex<HashMap<u64, ProcessHandle>>> = Arc::new(Mutex::new(HashMap::new()));
}

struct ProcessHandle {
    child: Child,
    stdin: Option<tokio::process::ChildStdin>,
    stdout_reader: Option<BufReader<tokio::process::ChildStdout>>,
    stderr_reader: Option<BufReader<tokio::process::ChildStderr>>,
}

#[derive(Debug, Serialize)]
pub struct ProcessOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct SpawnedProcess {
    pub process_id: u64,
    pub command: String,
}

#[derive(Debug, Deserialize)]
pub struct ExecuteCommandRequest {
    pub command: String,
    pub args: Option<Vec<String>>,
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct SpawnProcessRequest {
    pub command: String,
    pub args: Option<Vec<String>>,
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct KillProcessRequest {
    pub process_id: u64,
}

#[derive(Debug, Deserialize)]
pub struct WriteProcessInputRequest {
    pub process_id: u64,
    pub input: String,
}

#[tauri::command]
pub async fn execute_command(
    _state: State<'_, Arc<AppState>>,
    request: ExecuteCommandRequest,
) -> Result<ProcessOutput, String> {
    info!("Executing command: {}", request.command);

    let mut cmd = Command::new(&request.command);
    
    if let Some(args) = request.args {
        cmd.args(args);
    }
    
    if let Some(cwd) = request.cwd {
        cmd.current_dir(cwd);
    }
    
    if let Some(env) = request.env {
        for (key, value) in env {
            cmd.env(key, value);
        }
    }

    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let timeout = request.timeout_ms.unwrap_or(30000);

    match cmd.spawn() {
        Ok(mut child) => {
            let result = tokio::time::timeout(
                tokio::time::Duration::from_millis(timeout),
                child.wait()
            ).await;

            match result {
                Ok(Ok(status)) => {
                    let stdout = if let Some(mut stdout) = child.stdout.take() {
                        let mut buffer = String::new();
                        let _ = tokio::io::AsyncReadExt::read_to_string(&mut stdout, &mut buffer).await;
                        buffer
                    } else {
                        String::new()
                    };

                    let stderr = if let Some(mut stderr) = child.stderr.take() {
                        let mut buffer = String::new();
                        let _ = tokio::io::AsyncReadExt::read_to_string(&mut stderr, &mut buffer).await;
                        buffer
                    } else {
                        String::new()
                    };

                    Ok(ProcessOutput {
                        stdout,
                        stderr,
                        exit_code: status.code(),
                    })
                }
                Ok(Err(e)) => {
                    error!("Process wait error: {}", e);
                    Err(format!("Process error: {}", e))
                }
                Err(_) => {
                    let _ = child.kill().await;
                    Err("Command timed out".to_string())
                }
            }
        }
        Err(e) => {
            error!("Failed to spawn command: {}", e);
            Err(format!("Failed to spawn command: {}", e))
        }
    }
}

#[tauri::command]
pub async fn spawn_process(
    _state: State<'_, Arc<AppState>>,
    request: SpawnProcessRequest,
) -> Result<SpawnedProcess, String> {
    info!("Spawning process: {}", request.command);

    let process_id = PROCESS_COUNTER.fetch_add(1, Ordering::SeqCst);

    let mut cmd = Command::new(&request.command);
    
    if let Some(args) = request.args {
        cmd.args(args);
    }
    
    if let Some(cwd) = request.cwd {
        cmd.current_dir(cwd);
    }

    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped());

    // Shell-specific setup
    #[cfg(target_os = "windows")]
    {
        if request.command == "powershell" || request.command == "pwsh" {
            cmd.arg("-NoLogo").arg("-NoExit");
        }
    }

    match cmd.spawn() {
        Ok(mut child) => {
            let stdin = child.stdin.take();
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let handle = ProcessHandle {
                child,
                stdin,
                stdout_reader: stdout.map(BufReader::new),
                stderr_reader: stderr.map(BufReader::new),
            };

            ACTIVE_PROCESSES.lock().await.insert(process_id, handle);

            Ok(SpawnedProcess {
                process_id,
                command: request.command,
            })
        }
        Err(e) => {
            error!("Failed to spawn process: {}", e);
            Err(format!("Failed to spawn process: {}", e))
        }
    }
}

#[tauri::command]
pub async fn kill_process(
    _state: State<'_, Arc<AppState>>,
    request: KillProcessRequest,
) -> Result<(), String> {
    info!("Killing process: {}", request.process_id);

    let mut processes = ACTIVE_PROCESSES.lock().await;
    
    if let Some(mut handle) = processes.remove(&request.process_id) {
        match handle.child.kill().await {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to kill process: {}", e);
                Err(format!("Failed to kill process: {}", e))
            }
        }
    } else {
        Err("Process not found".to_string())
    }
}

#[tauri::command]
pub async fn read_process_output(
    _state: State<'_, Arc<AppState>>,
    process_id: u64,
) -> Result<ProcessOutput, String> {
    let mut processes = ACTIVE_PROCESSES.lock().await;
    
    if let Some(handle) = processes.get_mut(&process_id) {
        let mut stdout_output = String::new();
        let mut stderr_output = String::new();

        // Try to read available output (non-blocking)
        if let Some(ref mut reader) = handle.stdout_reader {
            let mut line = String::new();
            // This would need proper async handling in a real implementation
            // For now, we'll return empty if not ready
        }

        Ok(ProcessOutput {
            stdout: stdout_output,
            stderr: stderr_output,
            exit_code: handle.child.try_wait().ok().flatten().map(|s| s.code()).flatten(),
        })
    } else {
        Err("Process not found".to_string())
    }
}

#[tauri::command]
pub async fn write_process_input(
    _state: State<'_, Arc<AppState>>,
    request: WriteProcessInputRequest,
) -> Result<(), String> {
    let mut processes = ACTIVE_PROCESSES.lock().await;
    
    if let Some(handle) = processes.get_mut(&request.process_id) {
        if let Some(ref mut stdin) = handle.stdin {
            match stdin.write_all(request.input.as_bytes()).await {
                Ok(_) => {
                    let _ = stdin.flush().await;
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to write to process: {}", e);
                    Err(format!("Failed to write to process: {}", e))
                }
            }
        } else {
            Err("Process has no stdin".to_string())
        }
    } else {
        Err("Process not found".to_string())
    }
}
