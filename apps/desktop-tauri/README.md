# BI-IDE Desktop

AI-Powered Desktop Development Environment with multi-device sync and autonomous training capabilities.

## Features

- 🖥️ **Native Desktop App** - Built with Tauri v2 for Windows, macOS, and Linux
- 📁 **File Management** - Full file system integration with file watching
- 🌿 **Git Integration** - Native Git operations (status, add, commit, push, pull, etc.)
- 💻 **Integrated Terminal** - Spawn and manage terminal processes
- ☁️ **Multi-Device Sync** - CRDT-based real-time synchronization
- 🧠 **Autonomous Training** - Local ML training when idle
- 🔒 **Security** - mTLS, encrypted storage, secure IPC

## Project Structure

```
apps/desktop-tauri/
├── src/                    # Frontend React code
│   ├── components/         # React components
│   ├── lib/               # Utilities and API wrappers
│   ├── hooks/             # Custom React hooks
│   └── App.tsx            # Main app component
├── src-tauri/             # Rust backend code
│   ├── src/
│   │   ├── commands/      # Tauri command handlers
│   │   ├── state.rs       # App state management
│   │   └── main.rs        # Entry point
│   ├── Cargo.toml         # Rust dependencies
│   └── tauri.conf.json    # Tauri configuration
└── package.json           # Node.js dependencies
```

## Development Setup

### Prerequisites

- Rust 1.75+
- Node.js 20+
- Tauri CLI: `cargo install tauri-cli`

### Quick Start

```bash
# From project root
./scripts/dev-setup.sh      # Setup development environment
./scripts/dev-up.sh         # Start API and Desktop app
```

Or manually:

```bash
# 1. Install dependencies
cd apps/desktop-tauri
npm install

# 2. In one terminal, start the API
cd ../..
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# 3. In another terminal, start the desktop app
cd apps/desktop-tauri
npm run tauri:dev
```

## Building

```bash
# Debug build
./scripts/build-desktop.sh

# Release build
./scripts/build-desktop.sh --release

# Build for specific target
./scripts/build-desktop.sh --release --target aarch64-apple-darwin
```

## Architecture

### Frontend (React + TypeScript)
- **State Management**: Zustand with persistence
- **Styling**: Tailwind CSS
- **IPC**: Tauri API for native commands

### Backend (Rust)
- **Tauri v2**: Desktop app framework
- **Tokio**: Async runtime
- **Git2**: Git operations
- **Notify**: File system watching

### Protocol
Shared protocol definitions in `libs/protocol/` provide:
- Auth contracts
- CRDT sync operations
- Telemetry formats
- Training job definitions

## Commands

### File System
- `read_file` - Read file contents
- `write_file` - Write file contents
- `read_dir` - List directory contents
- `create_dir` - Create directory
- `delete_file` - Delete file/directory
- `rename_file` - Rename/move file
- `watch_path` - Watch path for changes

### Git
- `git_status` - Get repository status
- `git_add` - Stage files
- `git_commit` - Commit changes
- `git_push` - Push to remote
- `git_pull` - Pull from remote
- `git_log` - Get commit history
- `git_branches` - List branches
- `git_checkout` - Switch branches
- `git_clone` - Clone repository

### Terminal
- `execute_command` - Execute shell command
- `spawn_process` - Spawn long-running process
- `kill_process` - Kill running process
- `read_process_output` - Read process output
- `write_process_input` - Write to process stdin

### System
- `get_system_info` - Get device info
- `get_resource_usage` - Get CPU/memory/disk usage
- `open_path` - Open in file explorer
- `show_notification` - Show system notification

### Auth
- `get_device_id` - Get unique device ID
- `register_device` - Register with control plane
- `get_access_token` - Get stored token
- `set_access_token` - Store token

### Sync
- `get_sync_status` - Get sync status
- `force_sync` - Trigger manual sync
- `get_pending_operations` - List pending operations

### Workspace
- `open_workspace` - Open workspace folder
- `close_workspace` - Close workspace
- `get_workspace_files` - Get workspace files
- `get_active_workspace` - Get active workspace

### Training
- `get_training_status` - Get training status
- `start_training_job` - Start training job
- `pause_training_job` - Pause training job
- `get_training_metrics` - Get training metrics

## License

MIT
