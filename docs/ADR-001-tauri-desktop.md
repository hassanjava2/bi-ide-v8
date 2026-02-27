# ADR-001: Tauri v2 for Desktop Application

## Status
Accepted

## Context
We need to build a cross-platform desktop IDE that supports:
- Windows, macOS, and Linux
- Native file system access
- Git operations
- Terminal integration
- Low resource usage
- Fast startup time

## Decision
We will use **Tauri v2** with **React + TypeScript** for the desktop application.

## Rationale

### Why Tauri?
1. **Small Bundle Size**: ~600KB vs ~100MB+ for Electron
2. **Memory Efficient**: Uses system webview instead of bundling Chromium
3. **Native Performance**: Rust backend for performance-critical operations
4. **Security**: Memory-safe Rust with sandboxed frontend
5. **Cross-Platform**: Single codebase for Windows, macOS, Linux

### Why React + TypeScript?
1. **Existing Codebase**: We already have React components in the web UI
2. **Developer Experience**: Type safety and good tooling
3. **Ecosystem**: Rich component libraries and tools

### Architecture
```
┌─────────────────────────────────────┐
│         Frontend (React)            │
│  - File Explorer                    │
│  - Editor                           │
│  - Terminal UI                      │
└──────────────┬──────────────────────┘
               │ Tauri Commands
┌──────────────▼──────────────────────┐
│         Backend (Rust)              │
│  - File System Operations           │
│  - Git Integration                  │
│  - Process Management               │
│  - Sync Engine                      │
└──────────────┬──────────────────────┘
               │ IPC / HTTP
┌──────────────▼──────────────────────┐
│      Control Plane (API)            │
│  - Authentication                   │
│  - Sync Coordination                │
│  - Model Registry                   │
└─────────────────────────────────────┘
```

## Consequences

### Positive
- Fast development iteration
- Shared types between frontend and backend
- Easy migration from existing React UI
- Native performance for file operations

### Negative
- Team needs to learn Rust
- Smaller ecosystem than Electron
- Platform-specific quirks may arise

## Alternatives Considered

### Electron
- **Pros**: Mature ecosystem, familiar to web developers
- **Cons**: Large bundle size, high memory usage, slower startup

### Flutter
- **Pros**: Native performance, single language
- **Cons**: Different paradigm, less suitable for IDE with web tech

### Native (Swift/C#)
- **Pros**: Best performance, native look and feel
- **Cons**: Platform-specific code, harder maintenance

## References
- [Tauri Documentation](https://tauri.app/)
- [Architecture Decision Records](https://adr.github.io/)
