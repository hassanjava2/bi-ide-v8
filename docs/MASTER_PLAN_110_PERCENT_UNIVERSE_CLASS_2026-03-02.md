# BI-IDE v8 — MASTER PLAN (110% Universe-Class)

> Date: 2026-03-02 (Updated)
> Mission: تحويل BI-IDE إلى منصة عالمية production-grade من كل النواحي (Product + Engineering + Security + Operations + AI + Business).
> Rule: لا نعتبر أي ميزة “مكتملة” إلا إذا نجحت Functional + Reliability + Security + Observability + UX + Docs.
> Verified: كل فجوة أدناه محققة من كود المشروع الفعلي — مو من تقارير سابقة.

---

## 1) Vision (شنو يعني 110%)

"110%" هنا يعني:
1. **مو بس يشتغل** — يشتغل بثبات تحت الضغط.
2. **مو بس واجهة** — كل زر مربوط بقدرة backend حقيقية.
3. **مو بس features** — فيه قياس جودة واضح وSLOs فعلية.
4. **مو بس local success** — قابل للنشر والإدارة على كل البيئات.
5. **مو بس launch** — عنده خطة نمو وصيانة ومخاطر واستجابة حوادث.

---

## 2) Definition of Done (Global)

الميزة/النظام لا يُغلق إلا إذا تحقق:
- Functional tests: ✅
- Integration tests (frontend + backend + desktop bridge): ✅
- Error budgets/SLO: ✅
- Security checks + auth behavior: ✅
- Telemetry/logging/tracing: ✅
- Performance budgets: ✅
- Docs/runbooks/recovery guide: ✅

---

## 3) Reality Gaps to Close First (Critical)

> **ملاحظة:** بعد الفحص الدقيق، بعض الـ Gaps السابقة كانت غير دقيقة. التحديث أدناه يعكس الواقع الفعلي.

## Gap A — Desktop Sync realism 🔴 CRITICAL
- `SyncPanel.tsx:24` يستدعي `invoke("get_sync_devices")` — لكن هذا الأمر **غير موجود** في `main.rs`.
- `main.rs` يسجل فقط: `get_sync_status`, `force_sync`, `get_pending_operations` (سطور 84-86).
- `commands/sync.rs` لا يحتوي على `get_sync_devices` نهائياً — **الـ frontend ينادي API غير موجود!**
- **Action:** إضافة `get_sync_devices` command في `commands/sync.rs` + تسجيله في `main.rs` invoke_handler.

## Gap B — Command Palette partial actions
- الأوامر تستدعي store methods حقيقية (toggleSidebar, toggleTerminal, etc.).
- لكن بعض الأوامر مثل `file.new`, `git.commit` تعتمد على backends قد لا تكون كاملة.
- **Action:** مراجعة كل أمر + ربط بتنفيذ فعلي أو إخفاء الأمر مؤقتاً.

## Gap C — Training Dashboard simulated metrics
- **15+ موضع** `Math.random()` في:
  - `TrainingDashboard.tsx` سطور 315-318 (loss/accuracy)
  - `TrainingDashboard.tsx` سطور 335-341 (GPU stats)
  - `GPUMonitor.tsx` سطور 288-293 (جميع المقاييس)
  - `GPUMonitor.tsx` سطور 405, 415 (read/write speed)
- GPU utilization, VRAM, temperature, fan speed, power — كلها محاكاة وهمية.
- **Action:** استبدال بـ `invoke("get_gpu_metrics")` polling حقيقي مع empty/offline states.

## Gap D — Auth contract across desktop/web/api ⚠️ PARTIAL
> **تحديث:** الـ device ID يعمل، لكن token lifecycle يحتاج مراجعة.

- ✅ `App.tsx:42` يستدعي `setStoreDeviceId(info.device_id)` — يعمل بشكل صحيح
- ⚠️ مسارات API محمية، لكن token refresh mechanism غير مؤكد
- ⚠️ Silent re-auth + error UX يحتاج اختبار end-to-end
- **Action:** مراجعة `commands::auth` module + اختبار token refresh flow

## Gap E — Desktop Component Wiring ✅ DONE (Verified)
> **تحديث:** بعد الفحص الميداني — الـ wiring شغال بشكل صحيح.

- ✅ `Layout.tsx` يستخدم `MonacoEditor` و `RealTerminal` (lazy imports)
- ✅ `App.tsx` يركّب `CommandPalette` و `QuickOpen` مع اختصارات Ctrl+Shift+P و Ctrl+P
- ✅ `Sidebar.tsx` يستخدم lazy imports لـ `GitPanel`, `SearchPanel`, `SyncPanel`, `TrainingDashboard`
- ✅ `App.tsx:42` يستدعي `setStoreDeviceId(info.device_id)` بشكل صحيح
- **الحالة:** لا يوجد action مطلوب — هذا الـ Gap مغلق فعلياً
- **ملاحظة:** التقرير السابق كان يحتوي على معلومات غير دقيقة عن هذا الجزء

## Gap F — GPU Metrics Backend 🔴 NEW
- `TrainingDashboard.tsx` و `GPUMonitor.tsx` يحتاجون `get_gpu_metrics` Tauri command.
- حالياً يستخدمون `Math.random()` (15+ موضع) — بيانات وهمية بالكامل.
- لا يوجد `get_gpu_metrics` في `commands/training.rs` ولا في `main.rs`.
- **Action:** إضافة `get_gpu_metrics` command (حقيقي من nvidia-smi/ROCm أو stub مع empty state) + تسجيله.

---

## 4) Execution Program (10 Massive Tracks)

## Track 1 — Product Integrity
**Goal:** كل تجربة مستخدم من أول فتح لحد daily workflow تكون سلسة.
- توحيد flows: open workspace, edit, search, git, sync, train, council.
- إلغاء أي زر/تبويب ميت.
- توحيد حالات: loading/empty/error/success.

**KPI:**
- Task success rate > 98%
- Dead-end flows = 0

## Track 2 — Architecture & Contracts
**Goal:** عقود API/IPC واضحة versioned ومضبوطة.
- اعتماد schema موحد لطلبات/استجابات desktop-tauri/api.
- Contract tests تمنع كسر التوافق.
- Deprecation policy رسمي.

**KPI:**
- Breaking changes without migration = 0
- Contract test pass = 100%

## Track 3 — Reliability & SRE
**Goal:** المنصة تتحمل الأعطال وتتعافى بسرعة.
- Health/readiness متكاملة لكل service.
- Retry policies مع circuit breakers.
- Incident runbooks + on-call procedure.

**SLO targets:**
- API availability >= 99.9%
- Crash-free desktop sessions >= 99.5%
- P95 critical action latency < 500ms (excluding model inference heavy paths)

## Track 4 — Security & Trust
**Goal:** أمان إنتاجي حقيقي.
- JWT lifecycle محكم + secure storage.
- RBAC verification.
- Secrets management + key rotation.
- Dependency + container scanning + SBOM.

**KPI:**
- Critical vulns open > 0 = release blocked
- Security regression tests pass = 100%

## Track 5 — Performance Engineering
**Goal:** سرعة عالية وثابتة.
- Continue bundle splitting + route/panel lazy loading.
- Editor/terminal memory profiling.
- Startup budget + interaction budget.

**Budgets:**
- Desktop cold start target <= 2.5s (typical machine)
- UI first interaction <= 150ms
- Memory growth leak-free over 2h session

## Track 6 — AI, Training, Hierarchy, Council (Real Mode)
**Goal:** كل منظومة AI مرتبطة telemetry + governance.
- Replace simulated metrics with real training metrics.
- Council responses with source attribution + confidence policy.
- Hierarchy decisions logged + replayable.
- Model lifecycle (train -> evaluate -> promote -> rollback).

**KPI:**
- % real data panels = 100%
- Reproducible training jobs = 100%

## Track 7 — Data & Sync Correctness
**Goal:** لا فقدان بيانات ولا تعارضات صامتة.
- Sync conflict strategy واضحة (detect/resolve/audit).
- Exactly-once semantics where required.
- Backup/restore verification.

**KPI:**
- Data loss incidents = 0
- Unresolved conflict ratio < 1%

## Track 8 — Quality Engineering
**Goal:** CI/CD gatekeeper فعلي.
- Test pyramid: unit + integration + e2e + contract + smoke.
- Golden-path end-to-end for desktop + web + api.
- Mandatory pre-release checklist automation.

**KPI:**
- Flaky tests < 2%
- Main branch red time < 5%

## Track 9 — DevEx & Operations
**Goal:** سرعة تطوير بدون فوضى.
- One-command local bootstrap.
- Consistent environments (dev/stage/prod parity).
- Release automation with rollback.

**KPI:**
- Setup time for new dev < 30 min
- Mean rollback time < 10 min

## Track 10 — Documentation & Governance
**Goal:** المشروع مفهوم وقابل للاستدامة.
- Living architecture docs.
- ADRs لكل قرار كبير.
- Release notes + migration guides.
- Ownership map لكل module.

**KPI:**
- Undocumented critical module = 0
- Outdated runbooks = 0

---

## 5) Phased Delivery (Universe-Class Roadmap)

## Phase 0 — Reality Stabilization (Week 1)
- ~~Gap E~~ ✅ مغلق فعلياً.
- Close gaps A/B/C/D/F بالكامل.
- Freeze interfaces مؤقتًا.
- Add smoke tests for IDE core flows.

**Exit Gate:**
- No known no-op in core actions
- No simulated metrics in production path

## Phase 1 — Production Foundation (Weeks 2-3)
- SLO dashboards + alerting.
- Auth/session hardening end-to-end.
- Sync correctness tests + recovery procedures.

**Exit Gate:**
- Availability/reliability dashboards live
- Incident drill completed

## Phase 2 — AI Systems Hardening (Weeks 4-5)
- Training pipeline observability.
- Council/hierarchy governance + traceability.
- Model promotion and rollback workflow.

**Exit Gate:**
- Full trace from input -> model decision -> output

## Phase 3 — Cross-Platform Excellence (Weeks 6-7)
- Desktop packaging QA (macOS/Windows/Linux).
- Web + API deployment hardening.
- Performance tuning to meet budgets.

**Exit Gate:**
- Platform certification checklist pass

## Phase 4 — Launch Readiness (Week 8)
- Security final audit.
- DR drill + backup restore validation.
- Go-live playbook + war-room readiness.

**Exit Gate:**
- Launch criteria all green

---

## 6) Master KPI Scoreboard

1. Functional Completeness = 100%
2. Real-Data Completeness = 100%
3. Reliability SLO = green for 14 days
4. Security Critical/Open = 0
5. P95 Latency within budget
6. Crash-free sessions >= target
7. Test pass rate >= 98%
8. Rollback tested and successful
9. Docs coverage = 100% critical systems
10. On-call and incident response validated

---

## 7) Non-Negotiable Release Gates

Release ممنوع إذا:
- أي feature critical still mock/no-op
- أي auth bypass غير مقصود
- أي data-loss scenario غير معالج
- أي P0/P1 bug مفتوح
- monitoring ناقص لأجزاء core

---

## 8) Ownership Matrix

| Module | Owner | Backup | Status |
|--------|-------|--------|--------|
| Desktop Core (Layout/App/Sidebar) | Hassan | AI Assistant | Active |
| API Core (FastAPI routes) | Hassan | — | Active |
| AI/Training (Council/Hierarchy/Brain) | Hassan | AI Assistant | Active |
| Sync/Data (CRDT/WebSocket) | Hassan | — | Planned |
| Security (Auth/JWT/Secrets) | Hassan | — | Planned |
| SRE/Deploy (systemd/launchd/scripts) | Hassan | AI Assistant | Active |
| QA Automation (tests/CI) | Hassan | — | Planned |
| Documentation (plans/guides/ADRs) | Hassan | AI Assistant | Active |

> بدون owner واضح، ماكو complete.

---

## 9) Weekly Operating Rhythm

- Daily: build health + incident/bug review (15m)
- Twice weekly: architecture + risk board
- Weekly: KPI review + release readiness score
- Biweekly: disaster recovery rehearsal / game day

---

## 10) Final Command Strategy

الهدف مو “أفضل مشروع بالكون” بالكلام، وإنما:
- قياس صارم
- إغلاق وهم/No-op
- ثبات تشغيلي حقيقي
- جودة أمنية وعملياتية مستمرة

إذا هذه الخطة تنفذت حرفيًا وبالـ gates أعلاه، BI-IDE يدخل مستوى عالمي فعلي، مو تسويقي.

---

## 11) Immediate Next 72 Hours (Action Burst) — UPDATED

> **تحديث:** تم إعادة ترتيب الأولويات بناءً على الفحص الفعلي. Gap E مغلق فعلياً، لذا نركز على المشاكل الحقيقية.

### Day 1 (Gap A — Critical Fix)
1. **Implement `get_sync_devices`** في `commands/sync.rs`:
   ```rust
   #[derive(Debug, Serialize)]
   pub struct SyncDevice {
       pub device_id: String,
       pub device_name: String,
       pub status: String,
       pub last_seen: u64,
   }
   
   #[tauri::command]
   pub async fn get_sync_devices(
       state: State<'_, std::sync::Arc<AppState>>,
       workspace_id: String,
   ) -> Result<Vec<SyncDevice>, String> { ... }
   ```
2. تسجيل الأمر في `main.rs` invoke_handler.
3. اختبار SyncPanel يعرض "This device" بشكل صحيح.

### Day 2 (Gap F + C — GPU Metrics)
4. **Add `get_gpu_metrics` command** — نظام monitoring حقيقي (أو stub مع empty state).
5. **Replace `Math.random()`** في `TrainingDashboard.tsx` و `GPUMonitor.tsx` باستدعاء `get_gpu_metrics`.
6. إضافة empty/offline states للـ GPU monitor.

### Day 3 (Gap B + D — Polish)
7. **Audit CommandPalette** — إخفاء الأوامر الـ no-op أو ربطها:
   - `file.openFolder` → استدعاء `dialog.open`
   - `file.newFile` → استدعاء `fs.create` + `workspace.refresh`
   - `edit.undo/redo` → تمرير للـ Monaco Editor
8. **Auth hardening** — اختبار token refresh + error UX.
9. **Smoke test** — open workspace → edit → search → git → sync → train.

**Success criteria at 72h (UPDATED):**
- [x] ~~Gap E~~: ✅ Already working — Monaco/Terminal/CommandPalette/QuickOpen all functional
- [ ] Gap A: `get_sync_devices` returns real data (or empty array with proper error handling)
- [ ] Gap F: `get_gpu_metrics` command exists (real or stub with empty state)
- [ ] Gap C: 0 instances of `Math.random()` in production metrics paths
- [ ] Gap B: 0 critical no-op commands (or hidden/disabled)
- [ ] Build green + smoke tests green
- [ ] Pushed to `origin/main` with auto-update deployed
