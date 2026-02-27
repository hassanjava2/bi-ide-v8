# BI-IDE v8 — Go/No-Go Report (2026-02-24)

## Decision
**NO-GO (for full production release)**

Current evidence does not support production sign-off because the full backend suite is not stable and the repository state is highly unclean.

## Evidence Snapshot

### 1) Release Risk Snapshot
- Working tree churn: **529 changed items** (`git status --short | Measure-Object`).
- Risk impact: high change surface, difficult traceability, and weak release hygiene.

### 2) Full Backend Test Run
- Command: `pytest -q`
- Duration: ~19m55s
- Result: **293 passed, 28 failed, 70 errors, 1 skipped**
- Main failing domains observed:
  - ERP integration/model conflicts and schema drift
  - Enum/signature mismatches in ERP APIs
  - SQLite/Windows file locking and transaction rollback cascades
  - Auth-dependent E2E failures (`access_token` missing paths)
  - Performance suite failing due to connectivity/throughput assumptions

### 3) Frontend Production Build
- Command: `cd ui && npm run build`
- Result: **PASS**
- Output: Vite build succeeded in **2.82s**.

### 4) Smoke + Diagnostics
- Command: `PYTHONPATH=. python scripts/smoke_test.py` (from repo root)
- Result: **PASS (6/6)**
- VS Code diagnostics (`get_errors`): **No active editor errors found**.

## Conclusion
The project is **operational in smoke scope** and the frontend artifact is deployable, but **not releasable as production** under full-suite quality gates.

## Primary Blockers
1. Full backend suite instability (**98 non-passing tests: 28 fail + 70 error**).
2. Massive unclean working tree (**529 changes**) preventing clean release traceability.

## Minimum Conditions to Flip to GO
1. Reduce backend failures to zero under agreed gate:
   - `pytest -q` => all required tests passing (or approved quarantine list with owners/deadlines).
2. Create clean release snapshot:
   - isolate release branch,
   - remove non-release churn,
   - freeze artifact versions.
3. Re-run sign-off sequence on the release candidate commit:
   - full backend tests,
   - frontend production build,
   - smoke test.

## Practical Release Note
If an urgent beta/demo deployment is required, allow **limited-scope beta** only with explicit caveat:
- "Not production-certified; full backend suite currently failing."
