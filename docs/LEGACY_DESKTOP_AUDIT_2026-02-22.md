# Legacy Desktop Audit (Pre-v8)

Date: 2026-02-22
Scope: Review of old BI-IDE desktop code under `D:/bi ide` and imported snapshots inside `D:/bi-ide-v8`.

## Executive Result

Yes — pre-v8 contains real desktop application implementations (not only ideas), especially:
- Electron monolith in `D:/bi ide` (v6 alpha lineage).
- Electron + Vite + React architecture in `D:/bi ide/bi-ide-v7`.

But the legacy stack also has structural issues that explain instability and maintenance pain.

## What was found

### 1) Desktop app existed and was large
- `D:/bi ide/package.json`:
  - `version: 6.0.0-alpha.1`
  - heavy Electron scripts (`build:win`, `build:mac`, `build:linux`, smoke/security/quality scripts)
  - broad dependencies (monaco, xterm, node-pty, onnxruntime-node, transformers, etc.)
- `D:/bi ide/main.js`:
  - full Electron main process with multiple IPC modules
  - terminal integration, training/model handlers, workspace security hooks

### 2) v7 existed as cleaner desktop architecture
- `D:/bi ide/bi-ide-v7/package.json`:
  - `version: 7.0.0`
  - Vite + React renderer + Electron packaging
- `D:/bi ide/bi-ide-v7/src/`:
  - separated `main.ts`, `preload.ts`, `ipc/`, `core/`, `renderer/`

### 3) Evidence of operational fragility
- `D:/bi ide/backend.err` shows runtime dependency failure:
  - `ModuleNotFoundError: No module named 'structlog'`
- Build/process drift signal in v7:
  - `README.md` says `npm run dist`, but `package.json` does not define `dist` script.

### 4) Imported folders inside v8 mostly data/environment remnants
- `import_ok` and `import_from_linux` mainly include envs/snapshots/data, not a complete standalone desktop app ready to run.
- Desktop mentions there are mostly in knowledge templates, not production desktop runtime.

## Why v8 felt weaker than old versions

- Legacy had a bigger desktop surface area (features appeared richer).
- v8 reset reduced complexity intentionally, so current capability breadth is lower.
- However, v8 is cleaner for rebuilding reliability and autonomous operation.

## Migration recommendation (Do this, avoid copy-paste)

1. Reuse concepts, not legacy code bulk.
2. Keep v8 control-plane + worker orchestration as the base.
3. Bring only proven desktop patterns from v7:
   - secure preload bridge
   - modular IPC boundaries
   - terminal/session isolation patterns
4. Rebuild desktop runtime in V6 track (Rust desktop agent + web control), then add UI shell progressively.
5. Define strict acceptance gates before feature parity claims:
   - boot reliability
   - dependency lock integrity
   - e2e smoke
   - crash recovery

## Immediate next parity targets

- P1: command execution + terminal persistence + project scaffolding from desktop nodes.
- P2: model/train job controls from desktop shell.
- P3: full project factory loop (idea → spec → code → test → artifact).
