# ✅ قائمة الأعمال المتبقية - BI-IDE v8
## Execution Backlog (Locked to Reality Check v3.1)

> **المرجع التنفيذي المعتمد:**
> `docs/DESKTOP_V8_REALITY_CHECK_2026-03-02.md` (v3.1)
>
> **قاعدة إلزامية:** لا يبدأ أي عمل خارج هذا الـ backlog قبل اجتياز Gate المرحلة السابقة.

---

## 🔒 ثوابت التنفيذ (لا تفاوض)

- [ ] **Canonical Routing Lock** مفعل:
  - [ ] Sync: `/api/v1/sync`, `/api/v1/sync/status`, `/api/v1/sync/ws`
  - [ ] Training: `/api/v1/training/start`, `/api/v1/training/status`, `/api/v1/training/stop`
- [ ] لا استخدام لأي legacy route في كود جديد.
- [ ] أي استثناء يحتاج ADR قصير + موافقة Platform Owner.
- [ ] لا mock في المسارات الحرجة قبل إعلان readiness.

---

## 👥 Ownership (Owner / Backup)

- **Desktop:** UI/UX, Monaco, Git UI, Workspace UX
- **Rust:** Tauri commands, PTY, sync client, performance
- **Backend:** API contracts, routers, gateway, auth
- **Agent:** enrollment, policy enforcement, heartbeat
- **Platform/DevOps:** updates, release channels, rollback, CI gates
- **AI/ML:** council hardening, training/evaluation, self-improvement gates

---

## 🧭 Dependency Order (إلزامي)

1. **W1-W2 (Phase 0)** -> شرط دخول W3+
2. **W3-W4 (Phase 1)** -> شرط دخول W5+
3. **W5-W6 (Phase 2)** -> شرط دخول W7+
4. **W7-W8 (Phase 3)** -> شرط دخول W9+
5. **W9+ (Phase 4/5)** بعد نجاح Gates السابقة

---

## 📅 Weekly Execution Backlog

## Week 1 — Contract Freeze (P0)

### P0 Tasks
- [ ] إنشاء `docs/api_contracts_v1.md`
  - [ ] council request/response/errors
  - [ ] training request/response/errors
  - [ ] sync request/response/errors
  - [ ] workers policy/heartbeat contracts
- [ ] توحيد canonical endpoints داخل:
  - [ ] `apps/desktop-tauri/src/config/api.ts`
  - [ ] `apps/desktop-tauri/src-tauri/src/commands/sync.rs`
  - [ ] `apps/desktop-tauri/src-tauri/src/commands/training.rs`
  - [ ] `api/app.py` / routers mapping
- [ ] إضافة compatibility test للـ routing في CI.

### Owners
- Owner: **Backend**
- Backup: **Rust**

### DoD (Gate A-1)
- [ ] 0 endpoint mismatch في tests.
- [ ] عقود v1 منشورة ومراجعة.

---

## Week 2 — End-to-End Wire-up + Smoke

### P0 Tasks
- [ ] ربط Desktop -> API -> Worker status end-to-end.
- [ ] smoke suite (20 runs) لمسار:
  - [ ] start training
  - [ ] get status
  - [ ] stop training
- [ ] تثبيت `training_service` كـ singleton واعتماد import موحد بالخدمات.

### P1 Tasks
- [ ] تحسين رسائل الخطأ الموحدة (error_code + trace_id).

### Owners
- Owner: **Backend**
- Backup: **Agent**

### DoD (Gate A-2)
- [ ] smoke success >= 95% على 20 تشغيل.
- [ ] trace_id موجود في الاستجابات الفاشلة.

---

## Week 3 — Core IDE: Monaco + Files (P0)

### P0 Tasks
- [ ] إدخال `MonacoEditor.tsx` وربطه بالـ store الحالي.
- [ ] نقل الحفظ/القراءة إلى Rust FS commands فقط.
- [ ] تبويبات حقيقية (dirty state + close + switch).
- [ ] Quick Open (Cmd+P) MVP.

### P1 Tasks
- [ ] Search panel MVP (project search placeholder route).

### Owners
- Owner: **Desktop**
- Backup: **Rust**

### DoD (Gate B-1)
- [ ] فتح/تعديل/حفظ ملفات حقيقية من disk.
- [ ] File open P95 <= 220ms (dataset داخلي).

---

## Week 4 — Core IDE: PTY + Git MVP

### P0 Tasks
- [ ] PTY terminal integration فعلي في:
  - [ ] `src-tauri/src/commands/terminal.rs`
  - [ ] `src/components/Terminal.tsx`
- [ ] Git MVP في UI:
  - [ ] status
  - [ ] stage
  - [ ] commit
- [ ] تنظيف الاستدعاءات في hooks لتوافق Tauri v2 (`@tauri-apps/api/core`).

### P1 Tasks
- [ ] push/pull basic flows مع error UX واضح.

### Owners
- Owner: **Rust**
- Backup: **Desktop**

### DoD (Gate B-2)
- [ ] 3 جلسات terminal متزامنة بدون crash.
- [ ] git status/stage/commit end-to-end شغال.

---

## Week 5 — Worker Fabric + Policy APIs (P0)

### P0 Tasks
- [ ] workers register + heartbeat + apply-policy endpoints.
- [ ] policy schema v1 (cpu/ram/gpu/schedule/safety).
- [ ] Agent policy enforcement (baseline):
  - [ ] cpu cap
  - [ ] memory cap
  - [ ] idle-only behavior

### P1 Tasks
- [ ] واجهة أولية لعرض worker capabilities.

### Owners
- Owner: **Agent**
- Backup: **Backend**

### DoD (Gate C-1)
- [ ] policy apply latency P95 <= 10s.
- [ ] worker يثبت policy revision في heartbeat.

---

## Week 6 — Worker Dashboard + Resource Control UX

### P0 Tasks
- [ ] WorkerPolicy UI داخل desktop:
  - [ ] تعديل limits
  - [ ] تعديل schedule
  - [ ] تفعيل/تعطيل idle-only
- [ ] عرض planned vs actual usage.
- [ ] ربط alerts عند تجاوز thermal cutoff.

### P1 Tasks
- [ ] bulk apply policy لعدة workers.

### Owners
- Owner: **Desktop**
- Backup: **Agent**

### DoD (Gate C-2)
- [ ] تعديل policy من UI ينعكس فعلياً على worker.
- [ ] metrics محدثة كل <= 5 ثواني.

---

## Week 7 — Sync Hardening (P0)

### P0 Tasks
- [ ] إكمال conflict handling في sync service.
- [ ] replay queue عند reconnect.
- [ ] ws broadcast الحقيقي للتغييرات.
- [ ] إغلاق فجوة `is_connected` الوهمية في sync status.

### P1 Tasks
- [ ] sync diagnostics panel داخل desktop.

### Owners
- Owner: **Rust**
- Backup: **Backend**

### DoD (Gate D-1)
- [ ] Sync convergence LAN P95 < 2s.
- [ ] conflict cases مغطاة في tests الأساسية.

---

## Week 8 — Signed Updates + Rollback (P0)

### P0 Tasks
- [ ] ربط updater فعلي (desktop + agent).
- [ ] manifest signing + verification.
- [ ] release channels: canary/stable.
- [ ] rollback automation عند failure threshold.

### P1 Tasks
- [ ] update report endpoint + dashboard metric.

### Owners
- Owner: **Platform/DevOps**
- Backup: **Rust**

### DoD (Gate D-2)
- [ ] canary update ناجح على جهازين على الأقل.
- [ ] rollback اختباري ناجح مرة واحدة على الأقل.

---

## Week 9-10 — AI Code Intelligence (Phase 4a)

### P0 Tasks
- [ ] Inline completion (Monaco inline).
- [ ] Explain/Refactor actions.
- [ ] Error fix suggestions.
- [ ] Model selection settings.

### Owners
- Owner: **AI/ML**
- Backup: **Desktop**

### DoD (Gate E-1)
- [ ] Code completion P95 < 400ms (local/LAN path).
- [ ] Explain/Refactor يعمل على كود محدد من المحرر.

---

## Week 11-12 — Council Hardening + Memory (Phase 4b)

### P0 Tasks
- [ ] provider orchestration (RTX -> provider fallback -> local last).
- [ ] grounding checks + confidence calibration.
- [ ] memory layer (vector context) للمحادثات.

### Owners
- Owner: **AI/ML**
- Backup: **Backend**

### DoD (Gate E-2)
- [ ] Council response P95 < 700ms (local/LAN path).
- [ ] hallucination critical rate <= 2% في benchmark الداخلي.

---

## Week 13+ — Self-Improvement Gated (Phase 5)

### P0 Tasks
- [ ] propose -> sandbox test -> evaluate -> promote loop.
- [ ] promotion gate policy.
- [ ] kill switch + audit trail.

### Owners
- Owner: **AI/ML**
- Backup: **Platform/DevOps**

### DoD (Gate E-3)
- [ ] دورة تحسين يومية داخل sandbox.
- [ ] لا ترقية إنتاجية بدون gate pass.

---

## 📈 KPI Targets (واقعي ثم توسع)

### Foundation (بعد Week 6)
- [ ] Crash-free sessions >= 99.0%
- [ ] Startup P95 < 3.5s
- [ ] File open P95 < 180ms
- [ ] Worker enrollment success >= 95%

### Scale (بعد Week 12)
- [ ] Crash-free sessions >= 99.5%
- [ ] Startup P95 < 2.5s
- [ ] Sync convergence LAN P95 < 2s
- [ ] Update success rate >= 98%

---

## ✅ Gate Checklist (للإدارة اليومية)

- [ ] Gate A مكتمل (Contracts + Wire-up)
- [ ] Gate B مكتمل (Core IDE بلا mock)
- [ ] Gate C مكتمل (Policy enforcement فعلي)
- [ ] Gate D مكتمل (Sync/Update بثبات)
- [ ] Gate E مكتمل (AI gated + آمن)

---

## 🗂️ ملاحظات تشغيلية

- **Database Source of Truth:** `core/database.py` + `alembic/`
- أي artifact داخل `database/` لا يُعتمد إنتاجياً بدون قرار صريح.
- أي مهمة جديدة خارج هذا الملف تُضاف أولاً هنا مع Owner + DoD + Dependency.

---

**آخر تحديث:** 2026-03-02  
**الحالة:** 🔄 Active Execution Backlog
