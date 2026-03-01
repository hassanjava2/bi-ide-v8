# ADR-002: Runtime Stack and Data Platform

## Status
Accepted

## Context
BI-IDE v8 is a hybrid product that combines:
- Desktop IDE workflows (interactive, low-latency UX)
- API control plane and orchestration
- AI-assisted coding and council-style decision flows
- Distributed training workers and job scheduling
- Persistent multi-domain operational data

The team needs a stable, production-ready stack with fast delivery speed and low migration risk.

## Decision
We will standardize on the following core stack:

1. **Backend API & orchestration:** Python + FastAPI
2. **Desktop app:** Tauri (Rust runtime) + React + TypeScript UI
3. **Primary database:** PostgreSQL (with Alembic migrations)
4. **Cache/ephemeral coordination (when needed):** Redis

## Rationale

### Why Python + FastAPI for backend?
1. Strong fit for AI/ML ecosystem and orchestration logic
2. High development velocity for iterative product phases
3. Existing codebase already built around FastAPI + Python services

### Why Tauri + React/TS for desktop?
1. Lower memory footprint and package size than Electron-class alternatives
2. Native-safe operations through Rust backend
3. Reuse of existing React/TypeScript UI skills and components

### Why PostgreSQL as primary database?
1. Reliable transactional model for core business data
2. Good support for mixed structured/semi-structured workloads (`JSONB`)
3. Operational maturity, backup tooling, and migration discipline with Alembic

### Why Redis as optional supporting component?
1. Efficient short-lived state and caching
2. Useful for rate limiting, queue hints, and transient orchestration metadata

## Implementation Rules

1. **DB source of truth:** `core/database.py` + `alembic/`
2. No parallel DB access layer that bypasses the core DB module
3. All schema changes must be migration-first through Alembic
4. In-memory fake stores are allowed only behind explicit feature flags and must be temporary

## Consequences

### Positive
- Minimal architecture churn while addressing critical delivery gaps
- Better consistency between app layers and deployment environments
- Reduced operational risk during migration from mock/in-memory paths

### Negative
- Python services may hit CPU-bound limits before compiled alternatives
- Requires disciplined boundaries to prevent “mixed patterns” DB usage

## When to Revisit This ADR
Re-evaluate only if one or more of the following are true:

1. Sustained throughput/latency targets are missed despite optimization
2. Worker/job orchestration scales beyond current backend limits
3. Data access patterns require specialized stores not covered by PostgreSQL + Redis

## Alternatives Considered

### Move backend core to Go/Rust now
- **Pros**: Potentially higher raw throughput
- **Cons**: Large migration cost, slower near-term delivery, high integration risk
- **Decision**: Not now; optimize current stack first

### Keep SQLite/in-memory for major routers
- **Pros**: Simpler short-term setup
- **Cons**: State loss on restart, weak production guarantees
- **Decision**: Rejected for production path

## References
- Existing decision: `docs/ADR-001-tauri-desktop.md`
- Project migration plan: `implementation_plan.md`
