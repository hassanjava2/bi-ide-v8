# BI-IDE Desktop Supreme Master Plan (2026)
## الخطة الشاملة لتحويل BI-IDE إلى Desktop IDE عملاق متعدد الأنظمة

**Owner:** bi  
**Date:** 2026-02-27  
**Mode:** Private Product (not public SaaS)  
**Primary Sources Reviewed:**
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `PROJECT_STATUS_REAL.md`
- `PROJECT_AUDIT_REPORT_FINAL.md`
- `v6/desktop-agent-rs/README.md`
- `training/v6-scripts/README.md`

---

## 1) Executive Direction (قرار المنتج)

### Product North Star
نبني **Desktop IDE عملاق** يعمل على Windows/macOS/Linux، ويحقق:
1. **Offline-first + Online-sync** بين كل أجهزتك.
2. **Autonomous local training node**: أي جهاز تنزّل عليه التطبيق يبدأ يتعلم محليًا ويبعث النتائج للمركز.
3. **Continuous self-improvement loop** ضمن حدود أمان واضحة.
4. **Enterprise-grade reliability** مع تجربة IDE سريعة جدًا ومستمرة.

### Critical Constraint
هذا المنتج **خاص بك فقط**. لذلك architecture لازم تكون:
- Self-hosted control plane
- Private model registry
- Signed updates
- Zero third-party data leakage by default

---

## 2) Current-State Reality (مراجعة واقعية للمشروع الحالي)

## ما هو قوي الآن
- Backend/API + ERP + UI base موجودة وتشتغل.
- بنية AI hierarchy واسعة موجودة.
- توجد بداية desktop agent في Rust (`v6/desktop-agent-rs`).
- توجد training scripts وبنية learning assets (`training/`, `models/learning/`).

## الفجوات الحقيقية (Root Gaps)
1. **Desktop productization gap**: المشروع الحالي أقرب web platform أكثر من IDE desktop native مكتمل.
2. **Real-time multi-device sync gap**: لا يوجد layer موحد موثوق للتزامن اللحظي بآلية conflict-free قوية.
3. **Autonomous learning governance gap**: التعلم الذاتي موجود كفكرة، لكن يلزم policy engine + verification gates.
4. **Docs/status inconsistency**: بعض الملفات تقول 80% وبعضها 100%؛ لازم مصدر حقيقة واحد.
5. **AI realism gap**: أجزاء من hierarchy موسومة mock (وهذا صحيح وشفاف)، ولازم roadmap تحويلها تدريجيًا لوحدات إنتاجية.

---

## 3) Target Architecture (الهيكلة العملاقة المقترحة)

## 3.1 High-Level Topology

```text
┌────────────────────────────────────────────────────────────────────┐
│                        Private Control Plane                      │
│  (Self-hosted API + Orchestrator + Model Registry + Artifact DB) │
└───────────────▲───────────────────────────────▲────────────────────┘
                │                               │
        encrypted sync                    model/artifact
                │                               │
┌───────────────┴───────────────┐   ┌──────────┴────────────────────┐
│ Desktop Node A (Win/mac/Linux) │   │ Desktop Node B (Win/mac/Linux)│
│ - IDE UI (Tauri + React)       │   │ - IDE UI (Tauri + React)      │
│ - Rust Core Agent              │   │ - Rust Core Agent             │
│ - Local event log + CRDT       │   │ - Local event log + CRDT      │
│ - Local trainer worker          │   │ - Local trainer worker         │
└─────────────────────────────────┘   └───────────────────────────────┘
```

## 3.2 Stack Decisions (أفضل الأدوات المقترحة)

### Desktop Runtime
- **Tauri v2 + Rust** (أفضل مزيج: أداء، أمان، حجم صغير، cross-platform ممتاز).
- السبب: عندك أصلًا Rust agent، فالتكامل يكون طبيعي جدًا.

### IDE Engine
- **Monaco Editor + LSP + DAP + Tree-sitter** كبداية production.
- Option لاحقًا: embedding extension host advanced (Theia/Code-OSS compatible layer) إذا احتجت ecosystem أوسع.

### Backend/Control Plane
- **FastAPI (current)** يبقى مؤقتًا + إدخال خدمات Rust/Go تدريجيًا للـ high-throughput workers.
- Queue decision: **Redis Streams كخيار بداية رسمي** (2-5 أجهزة، أقل تعقيد تشغيلي).
- Upgrade trigger: الانتقال إلى **NATS JetStream** فقط عند تجاوز حدود Redis (throughput/latency/fan-out).

### Sync/Data
- **CRDT-based sync** (Yjs/Automerge concepts) + event log immutable.
- Metadata store: PostgreSQL.
- Local store: SQLite + encrypted op-log.

### Training/MLOps
- Local adapters (LoRA/PEFT style) على الأجهزة، ورفع artifacts incremental.
- Central model registry: versioned + signed manifests.
- Evaluation pipeline إجباري قبل promotion لأي model.
- Base-model policy (v1): Code-focused 7B class model كخط أساس، مع adapters فقط في البداية.
- Hardware split policy:
  - macOS/CPU-only devices: data collection + light eval + optional tiny adapter tests.
  - Ubuntu RTX node: heavy training/fine-tuning + main artifact production.

### Security
- mTLS بين node/control-plane.
- Per-node keys + hardware-bound secrets where possible.
- Signed binaries + signed model artifacts.

---

## 4) Core Subsystems (المكونات الرئيسية)

## 4.1 Desktop IDE Core
- Workspace manager
- Editor + multi-cursor + diagnostics
- Git graph + staging + diff AI assist
- Task runner + terminal orchestration
- Plugin sandbox (phase-4/5 بعد استقرار sync وautonomous core)

## 4.2 Sync Engine (Cross-device live updates)
- File-level and semantic-level operations.
- Presence + device identity.
- Deterministic merge via CRDT op sets.
- Offline queue + eventual consistency.

## 4.3 Autonomous Training Engine
- Local telemetry collector (code/edit/build/test outcomes)
- Privacy filter + redaction
- Feature builder
- Training queue scheduler
- Artifact uploader with retry/resume

## 4.4 Self-Improvement Engine
- Detect bottlenecks/regressions/opportunities
- Generate candidate patches/prompts/model-config deltas
- Sandbox validate
- Policy gate approve/reject
- Rollout canary then full

## 4.5 Governance + Safety Layer
- Policy-as-code (what allowed to modify autonomously)
- Risk tiers (L0..L4)
- Human override & kill switch
- Immutable audit ledger

---

## 5) Algorithms (خوارزميات أساسية)

## 5.1 Multi-device Sync Algorithm (CRDT + Event Sourcing)

### Data Structures
- `op_id = (node_id, logical_clock)`
- `vector_clock[node_id]`
- `document_state` as CRDT structure
- `op_log` append-only

### Flow
1. كل تعديل محلي يتحول إلى operation.
2. العملية تُطبّق محليًا فورًا (optimistic local).
3. تُرسل إلى control-plane + peers.
4. عند الاستلام: إذا op جديدة تُدمج حسب CRDT rule.
5. vector clocks تمنع إعادة التطبيق وتضمن convergence.

### Conflict Rule
- النصوص: CRDT text merge (position IDs not raw offsets).
- metadata: LWW فقط للحقول غير الحرجة.
- الملفات الحرجة (`config`, `schema`) تمر عبر semantic merge check.

## 5.2 Autonomous Training Scheduler (Priority + Budget Aware)

### Objective
تعظيم فائدة التعلم تحت قيود CPU/GPU/bandwidth.

### Score
`score = impact * confidence * freshness / cost`

Where:
- `impact`: تأثير الحالة على جودة المساعد/الإكمال.
- `confidence`: جودة البيانات (pass/fail certainty).
- `freshness`: حداثة السجلات.
- `cost`: تقدير زمن/طاقة التدريب والرفع.

### Policy
- إذا الجهاز مشغول (CPU > 75%) يؤجل training jobs الثقيلة.
- إذا الجهاز idle + power connected يبدأ jobs ذات score الأعلى.
- الرفع incremental chunked مع resume token.

## 5.3 Self-Improvement Loop (Propose → Verify → Promote)

1. **Detect** regression/opportunity.
2. **Propose** patch/config/model delta.
3. **Sandbox Test** (unit + integration + perf budgets).
4. **Policy Gate**:
   - Security checks
   - Quality thresholds
   - No forbidden file zones
5. **Canary Rollout** to one node.
6. **Promote** if KPIs pass; else rollback.

---

## 6) Build Strategy (طريقة البناء الشاملة)

> Timeline note: النسخة الأصلية كانت طموحة؛ التقدير الواقعي الكلي هو **9-14 شهر** حسب عمق التنفيذ.

## Phase 0 — Stabilization Baseline (3-4 weeks)
- توحيد source-of-truth status document.
- تثبيت auth/session/sync contracts.
- منع تضارب localStorage/session keys.
- baseline benchmarks (startup, memory, latency).
- Structured logging + basic metrics من أول أسبوع.

**Exit Criteria:** baseline CI أخضر + smoke tests ثابتة.

## Phase 1 — Desktop Foundation (6-8 weeks)
- Tauri shell + existing React UI migration.
- Rust core process bridge (IPC contract v1).
- Local workspace services (file watcher, git, terminal).
- Error tracking + crash reporting + release channel telemetry.

**Exit Criteria:** desktop app يعمل على Win/mac/Linux + open/edit/build/test.

## Phase 2 — Live Sync Fabric (8-10 weeks)
- CRDT op format + local op-log.
- device identity + encrypted channels.
- offline replay + reconciliation tests.
- Sync operation metrics (queue lag, convergence time, conflict rate).

**Exit Criteria:** تعديل على جهاز A يظهر على B/C بزمن < 2s LAN / < 5s WAN.

## Phase 3 — Autonomous Local Training (5-7 weeks)
- telemetry schema + privacy filter.
- local trainer worker + scheduler.
- artifact chunk uploader + integrity verification.

**Exit Criteria:** كل جهاز جديد يبدأ training تلقائي + رفع artifacts بدون تدخل يدوي.

## Phase 4 — Self-Improvement Automation (6-9 weeks)
- opportunity detector.
- patch proposal engine.
- gated sandbox + canary rollout.

**Exit Criteria:** دورة تحسين ذاتي يومية مع rollback آمن.

## Phase 5 — Production Hardening (4-6 weeks)
- signed updates (desktop + models).
- advanced observability (traces, SLO dashboards).
- chaos testing + disaster drills.

**Exit Criteria:** SLOs محققة + release train أسبوعي مستقر.

---

## 7) Quality Gates & KPIs (معايير النجاح)

## Reliability
- Crash-free sessions ≥ 99.5%
- Background agent uptime ≥ 99%
- Sync convergence errors < 0.1%

## Performance
- IDE startup < 2.5s (warm)
- File open latency p95 < 120ms
- Suggestion latency p95 < 400ms (local-assisted)

## Training
- Daily successful training cycles/node ≥ 1
- Artifact upload success ≥ 99%
- Model promotion only if quality delta > threshold

## Security
- 100% signed binaries/models
- 0 critical secrets in logs
- Full audit trail for autonomous actions

---

## 8) Security-by-Design Blueprint

1. **Private-by-default telemetry**: redact code secrets & PII before storage.
2. **Zero trust node auth**: short-lived certs + rotated tokens.
3. **Policy enforcement**: forbidden ops deny by default.
4. **Secrets management**: no plaintext keys in repo or local logs.
5. **Update trust chain**: verify signature before install/activate.

---

## 9) Deployment Model (كيف تنشره على كل أجهزتك)

## Node Bootstrap (first install)
1. Install desktop app.
2. Device generates keypair.
3. Node registers to control-plane.
4. Pulls policies + base model index.
5. Starts worker, sync, and training services automatically.

## Ongoing Operations
- Background auto-update channels (stable/canary).
- Daily local checkpoint + periodic remote snapshot.
- Auto-heal on service failure (restart with backoff).

---

## 9.1 Backup & Disaster Recovery (خطة النسخ والاستعادة)

### Targets
- **RPO target:** 1 ساعة للعمليات اليومية، 24 ساعة كحد أقصى للحالات الطارئة.
- **RTO target:** 2-4 ساعات لإرجاع control plane الأساسي.

### Backup Policy
- Local incremental snapshots يومية (nightly).
- Remote encrypted snapshot كل 6 ساعات للـ critical state.
- Rotation: يومي 14 نسخة + أسبوعي 8 نسخ + شهري 6 نسخ.

### Restore Discipline
- اختبار استرجاع فعلي كل أسبوعين (tabletop + real restore).
- أي backup لا يمر restore test يعتبر غير صالح.
- DR runbook موحد مع خطوات معتمدة ووقت تنفيذي مسجل.

---

## 9.2 Migration Plan (من الوضع الحالي إلى Desktop-first)

### What migrates as-is
- React UI الحالية (مع refactor تدريجي لفصل طبقة الواجهة عن transport logic).
- API contracts المستقرة (auth/status/ide الأساسية).

### What gets rewritten
- Local execution bridge (Rust IPC layer).
- Sync engine (CRDT + op-log) بدل الاعتماد على تزامن ad-hoc.
- Autonomous policy engine كخدمة مستقلة وقابلة للاختبار.

### ERP Scope Decision
- Phase 1-2: ERP يبقى web-compatible مع desktop embedding.
- Phase 3+: نقرر module-by-module إذا يحتاج native offline capability.

### Migration Gates
1. Contract parity tests pass.
2. No regression in auth/session behavior.
3. Desktop E2E smoke suite green on 3 OS.

---

## 10) One-Repo Structure Proposal (هيكل مجلدات مستهدف)

```text
bi-ide-v8/
  apps/
    desktop-tauri/           # Tauri shell + UI host
    control-center-web/      # optional operations web UI
  services/
    api-control-plane/       # existing FastAPI (transition)
    sync-service/            # CRDT relay + auth
    model-registry/          # artifacts + manifests
  agents/
    desktop-agent-rs/        # rust worker runtime
    trainer-worker-rs/       # local training orchestrator
  libs/
    protocol/                # shared contracts (events, ops, schemas)
    policy-engine/           # rules + risk gates
  data-contracts/
    telemetry/
    sync-ops/
    artifacts/
```

---

## 11) Immediate Backlog (Next 30 Days)

## 10.1 Development Environment Setup (إعداد بيئة التطوير)
- ملف توحيد بيئة (`.env.dev`) + policy واضحة للقيم المحلية.
- Docker Compose profile للتطوير (api + db + cache + observability-lite).
- Scripts موحدة:
  - `scripts/dev-up.*`
  - `scripts/dev-check.*`
  - `scripts/dev-reset.*`
- Onboarding checklist: من clone إلى first green run خلال أقل من 20 دقيقة.

---

### Week 1
- Freeze architecture decision record (ADR-001..ADR-010).
- Build unified auth/session contract.
- Set status source-of-truth doc.

### Week 2
- Bootstrap Tauri app bound to existing UI.
- Add Rust sidecar for file/git/terminal commands.

### Week 3
- Implement op-log format + basic peer sync.
- Add e2e test: edit from node A reflected on node B.

### Week 4
- Add local telemetry collector + privacy filter.
- Start first autonomous training job lifecycle.

---

## 12) Verification Playbook (نتأكد كل الخطة شغالة)

## Test Matrix
- OS: Windows, macOS, Linux
- Network: offline, LAN, WAN, flaky
- Workload: small project, large mono-repo
- Modes: foreground usage, overnight autonomous mode

## Must-pass Scenarios
1. Install on fresh device → auto-register + auto-sync works.
2. Edit code on device A → appears on B without data loss.
3. Device offline 2 hours → comes back and converges cleanly.
4. Local training starts automatically when idle.
5. Artifact upload resumes after network drop.
6. Bad model/policy candidate auto-rolled back.

---

## 13) Risk Register (مختصر)

1. **Sync complexity risk**
   - Mitigation: CRDT + exhaustive simulation tests.
2. **Autonomous unsafe actions**
   - Mitigation: strict policy gates + risk tiers + kill switch.
3. **Model drift / quality decay**
   - Mitigation: continuous eval + promotion guardrails.
4. **Cross-platform packaging issues**
   - Mitigation: CI matrix builds + signing pipeline early.

---

## 14) Final Build Order (الترتيب الذهبي)

1. Stabilize current repo contracts.
2. Desktop shell + Rust core bridge.
3. Sync fabric (CRDT + event log).
4. Autonomous training pipeline.
5. Self-improvement with governance.
6. Security hardening + signed release operations.

**Rule:** لا نفعّل autonomous full mode إلا بعد اجتياز quality + security gates.

---

## 15) Decision Summary for You (قرار تنفيذي)

إذا هدفك “أفضل جودة وأقوى أدوات” مع ملكية خاصة وتعدد أجهزة:
- **نعم**: الاتجاه الصحيح هو **Tauri + Rust Agent Core + CRDT Sync + Controlled Autonomous Training**.
- **نعم**: المشروع الحالي قابل للتحويل، لكن يحتاج تنفيذ مرحلي صارم وليس ترقيعات متفرقة.
- **نعم**: يمكن الوصول لنظام يتطور ذاتيًا، بشرط governance قوية جدًا من البداية.

---

## Appendix A — Definition of Done (DoD) لكل إصدار

- Release signed and reproducible.
- All critical e2e flows green on 3 OS.
- No critical vulns open.
- Sync convergence and rollback drills passed.
- Autonomous actions fully auditable.

## Appendix B — Non-Negotiable Engineering Rules

1. No silent fallback for critical paths.
2. No model promotion without eval report.
3. No cross-device sync without deterministic replay.
4. No secret in source control ever.
5. Every autonomous decision needs evidence + trace id.
6. No autonomous action without rollback capability.
