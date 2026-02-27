// BI-IDE Desktop - Main Entry Point
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod state;

use std::sync::Arc;
use tauri::{
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent, RunEvent,
};
use tracing::{info, warn, error};

use state::AppState;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    info!("Starting BI-IDE Desktop v0.1.0");

    let state = Arc::new(AppState::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_http::init())
        .manage(state)
        .setup(|app| {
            setup_app(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // File operations
            commands::fs::read_file,
            commands::fs::write_file,
            commands::fs::read_dir,
            commands::fs::create_dir,
            commands::fs::delete_file,
            commands::fs::rename_file,
            commands::fs::watch_path,
            commands::fs::unwatch_path,
            
            // Git operations
            commands::git::git_status,
            commands::git::git_add,
            commands::git::git_commit,
            commands::git::git_push,
            commands::git::git_pull,
            commands::git::git_log,
            commands::git::git_branches,
            commands::git::git_checkout,
            commands::git::git_clone,
            
            // Terminal operations
            commands::terminal::execute_command,
            commands::terminal::spawn_process,
            commands::terminal::kill_process,
            commands::terminal::read_process_output,
            commands::terminal::write_process_input,
            
            // System operations
            commands::system::get_system_info,
            commands::system::get_resource_usage,
            commands::system::open_path,
            commands::system::show_notification,
            
            // Auth operations
            commands::auth::get_device_id,
            commands::auth::register_device,
            commands::auth::get_access_token,
            commands::auth::set_access_token,
            
            // Sync operations
            commands::sync::get_sync_status,
            commands::sync::force_sync,
            commands::sync::get_pending_operations,
            
            // Workspace operations
            commands::workspace::open_workspace,
            commands::workspace::close_workspace,
            commands::workspace::get_workspace_files,
            commands::workspace::get_active_workspace,
            
            // Training operations
            commands::training::get_training_status,
            commands::training::start_training_job,
            commands::training::pause_training_job,
            commands::training::get_training_metrics,
        ])
        .on_window_event(|window, event| {
            match event {
                WindowEvent::CloseRequested { api, .. } => {
                    // Hide instead of close when clicking X
                    window.hide().unwrap();
                    api.prevent_close();
                }
                _ => {}
            }
        })
        .build(tauri::generate_context!())
        .expect("Error while building BI-IDE Desktop")
        .run(|_app_handle, event| {
            match event {
                RunEvent::Exit => {
                    info!("BI-IDE Desktop shutting down");
                }
                _ => {}
            }
        });
}

fn setup_app(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    info!("Setting up BI-IDE Desktop");

    // Get the main window
    let window = app.get_webview_window("main");

    // Setup system tray
    setup_tray(app)?;

    // Initialize state
    let state: Arc<AppState> = app.state::<Arc<AppState>>().inner().clone();
    
    // Spawn background tasks
    let app_handle = app.handle().clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = state.initialize(app_handle).await {
            error!("Failed to initialize app state: {}", e);
        }
    });

    // Show window after a short delay to ensure frontend is ready
    if let Some(window) = window {
        let window_clone = window.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            if let Err(e) = window_clone.show() {
                warn!("Failed to show window: {}", e);
            }
            if let Err(e) = window_clone.set_focus() {
                warn!("Failed to focus window: {}", e);
            }
        });
    }

    info!("BI-IDE Desktop setup complete");
    Ok(())
}

fn setup_tray(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let tray_icon = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("BI-IDE Desktop")
        .on_tray_icon_event(|tray, event| {
            match event {
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } => {
                    let app = tray.app_handle();
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                _ => {}
            }
        })
        .on_menu_event(|app, event| {
            match event.id.as_ref() {
                "show" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                "hide" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.hide();
                    }
                }
                "sync" => {
                    // Trigger force sync
                    let state: tauri::State<Arc<AppState>> = app.state();
                    let state = state.inner().clone();
                    tauri::async_runtime::spawn(async move {
                        if !*state.sync_manager.enabled.read().unwrap() {
                            warn!("Sync is disabled; ignoring tray force sync");
                        }
                    });
                }
                "training" => {
                    // Show training status
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.emit("show-training", ());
                    }
                }
                "quit" => {
                    app.exit(0);
                }
                _ => {}
            }
        })
        .build(app)?;

    info!("System tray setup complete");
    Ok(())
}
