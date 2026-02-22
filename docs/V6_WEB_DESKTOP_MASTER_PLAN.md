# V6 Web + Desktop Autonomous Master Plan

## قرار معماري نهائي
نعم: المنصة تكون **Web + Desktop** معاً، وليس Web فقط.

- Web: مركز قيادة ومراقبة وتحكم وتخطيط.
- Desktop: التنفيذ الفعلي للبناء/الاختبار/التطوير على الأجهزة.
- Orchestrator: يوزع المهام، يراقب الصحة، ويدير الكلفة.

## لماذا الموقع وحده لا يكفي؟
المتصفح محدود (Sandbox):
- لا يملك تحكم كامل بالنظام/العمليات/الخلفية 24/7.
- لا يناسب تشغيل build chains ثقيلة وعمليات طويلة على أكثر من جهاز.
- لا يضمن الاستمرارية بعد إغلاق التبويب.

لهذا نستخدم Desktop Agent على كل جهاز + Web Control Plane.

## Stack V6 (مختلف عن المسار السابق)
- Desktop Agent: Rust
- Control Plane APIs: Go (target)
- Web Control UI: React/TypeScript
- Data plane/queue: NATS + PostgreSQL + Object Storage (S3-compatible)

> المسار الحالي Python يبقى انتقالي لحين اكتمال نقل الخدمات الحرجة إلى V6.

## ربط طلبك ببنود تنفيذية
1. ذاتي التدريب: scheduler + queue + workers + لحظي artifacts.
2. ذاتي التطوير: auto patch proposals + test gates + staged rollout.
3. ذاتي الاستحداث: idea generator + graph expansion + ranking.
4. ذاتي الترميم: drift detection + rollback + auto-heal.
5. إنشاء/تطوير البرامج: project factory (spec→code→test→deploy) على Desktop nodes.
6. 24/7 بدون انتظار أوامر: policy-driven loops + watchdog + budget guard.
7. فرق كشافين: scouting loops + idea registry + ownership chain.
8. ماكو فكرة تضيع: append-only idea ledger + dedupe + task linkage.

## المتبقي حتى "باكمل وجه"
### P0 (فوري)
- Cost-aware scheduler: يوقف المهام قليلة العائد على السيرفرات الغالية.
- Incremental artifact upload: كل دقيقة/chunk بدل نهاية المهمة فقط.
- Idea registry خدمة مستقلة (append-only + recovery).
- Desktop node health SLO + auto-restart policy.

### P1 (قريب)
- Auto project factory (MVP):
  - ingest idea
  - generate spec
  - generate repo skeleton
  - run tests
  - publish result
- Scout loops live: telemetry scout + market scout + code-quality scout.

### P2 (توسع عملاق)
- Multi-cluster scheduling (region aware + cost arbitrage).
- Model-policy governance (risk/cost/impact).
- Full Rust/Go migration for high-throughput services.

## التنفيذ الذي أُضيف الآن داخل المستودع
- Rust Desktop Agent scaffold:
  - v6/desktop-agent-rs/Cargo.toml
  - v6/desktop-agent-rs/src/main.rs
  - v6/desktop-agent-rs/README.md
- One-command resilient launcher:
  - start_v6_desktop_node.ps1

## تشغيل سريع
1. شغّل API المركزي.
2. على كل جهاز دسكتوب:
   - powershell -ExecutionPolicy Bypass -File .\start_v6_desktop_node.ps1 -ServerUrl http://<SERVER>:8000 -Token <TOKEN> -WorkerName desktop-01 -Labels desktop,autonomous,builder -PollSec 5
3. من Web أنشئ jobs وسيتم تنفيذها على عقد الدسكتوب تلقائياً.

## ضمان عدم الضياع
- كل فكرة تدخل idea registry قبل أي تنفيذ.
- كل job يمتلك owner node + trace id + result artifact.
- عند انقطاع أي worker: auto-heal + replay pending queue.
