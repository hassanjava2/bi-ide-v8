# BI-IDE V6 Desktop Agent (Rust)

Native desktop worker for autonomous execution on Windows/Linux/macOS.

## Why this exists
- Browser-only apps cannot safely manage full local build/runtime workflows.
- Desktop agent gives direct local execution, filesystem access, and resilient long-running jobs.
- It connects to the central orchestrator and executes jobs 24/7.

## Run
1. Install Rust toolchain.
2. Start orchestrator API server.
3. Run:

```bash
cargo run --release -- \
  --server http://localhost:8000 \
  --token YOUR_TOKEN \
  --name desktop-node-01 \
  --labels desktop,autonomous,builder \
  --poll-sec 5
```

## Current behavior
- Registers worker with orchestrator.
- Sends heartbeat.
- Claims pending jobs.
- Executes command locally.
- Publishes job status and logs.

## Next upgrades (planned)
- Artifact upload stream (incremental checkpoints/log chunks).
- Local safety policy sandbox (allowlist for commands/paths).
- Resource-aware execution (CPU/RAM/GPU throttling).
- Signed update channel for agent binaries.
