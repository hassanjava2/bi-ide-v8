# BI-IDE Desktop v8 — التقرير التنفيذي الشامل (نسخة معتمدة)

التاريخ: 2026-03-02
الإصدار: 4.0 (Final — Execution Ready)
الحالة: معتمد للتنفيذ

---

## 1) الحكم التنفيذي النهائي

الخطة الطموحة قابلة للتنفيذ فعلياً، لكن فقط إذا اشتغلنا وفق هذا الترتيب الصارم:

1. تثبيت العقود وربط المسارات الأساسية بلا أي mismatch.
2. بناء Core IDE Production Surface فعلياً (Editor + Git + Terminal + Files).
3. تمكين الحواسيب كـ Workers مع Resource Governance كاملة من الدسكتوب.
4. توحيد المزامنة والتحديثات الموقعة لكل الأجهزة الحالية والمستقبلية.
5. ثم التوسع في AI Council والتعلم الذاتي عبر بوابات أمان واختبارات.

أي ترتيب آخر سيؤدي إلى ميزات كثيرة فوق أساس غير مستقر.

---

## 2) ما هو مثبت من الواقع الحالي

### مثبت ويُبنى عليه
- الاتجاه التقني الحالي صحيح: Tauri v2 + Rust + React.
- أوامر Rust موجودة ومتصلة على نطاق واسع (fs/git/terminal/sync/training/auth/workspace).
- Desktop Agent Rust موجود كبنية تشغيل على العقد.
- Sync service موجود كنواة CRDT/WS لكنه يحتاج إكمال التشغيل الإنتاجي.
- Council chat مربوط جزئياً بالـ API مع fallback.

### موجود لكنه غير إنتاجي بالكامل
- محرر الكود الحالي Textarea، وليس Monaco.
- لوحات التدريب والعمال تحتوي بيانات تجريبية ومحاكاة.
- auto-update hook في الواجهة ما زال Stub.
- بعض الـ telemetry/training metrics في أوامر Rust ما زالت Mock.

### فجوات ربط حرجة
- sync endpoint mismatch بين الدسكتوب والخدمة.
- training endpoint mismatch بين desktop commands وbackend routers.
- غياب contract versioning إلزامي يمنع الاستقرار عند التوسع.

### الأجهزة الحالية (Hardware Inventory)

| الجهاز | النظام | المعالج | GPU | RAM | الدور | الاتصال |
|--------|--------|---------|-----|-----|-------|--------|
| Mac | macOS | M-series | Integrated | 24GB | تطوير + Desktop App | محلي |
| RTX 5090 | Ubuntu | — | RTX 5090 24GB | — | تدريب AI + Worker | `bi@192.168.1.164` |
| Windows | Win 11 | — | RTX 4050 6GB | — | Worker ثانوي | بحاجة SSH |
| VPS | Ubuntu | 8 cores | — | — | API + Registry | `root@bi-iq.com` |

---

## 3) الرؤية التقنية النهائية (Target Operating Model)

## 3.1 طبقات النظام

1) Desktop Runtime (Tauri)
- UI/UX + أوامر Rust محلية + إدارة الجلسات.

2) Control Plane API
- المصدر الوحيد للعقود، السياسات، التسجيل، الجدولة، التحديثات، الهوية.

3) Worker Agent (Rust)
- يُنصّب على أي جهاز.
- ينفذ المهام حسب policy والموارد المحددة.

4) Sync & Update Services
- مزامنة بيانات وملفات/عمليات.
- تحديثات موقعة تدريجية مع rollback.

5) AI/Training Plane
- inference + training + evaluation gates + registry.

## 3.2 مبدأ حاكم
- لا منطق تشغيلي حرج داخل UI فقط.
- كل قرار حرج يمر عبر Policy في السيرفر ثم يُنفذ على الـ Agent.

---

## 4) المتطلبات غير القابلة للتفاوض (Non-Negotiables)

1. ربط كامل Desktop ↔ Website ↔ Server ↔ Workers.
2. أي جهاز جديد يدعم agent onboarding خلال دقائق.
3. تحكم فعلي من الدسكتوب بحدود الاستهلاك:
- CPU سقف %
- RAM سقف GB
- GPU Memory سقف %
- نافذة زمنية للتشغيل
- Idle-only mode
4. تحديثات تلقائية آمنة وموقعة لكل الأجهزة.
5. توافق عكسي مدروس للعقود عبر versioning.
6. لا بيانات تجريبية في المسارات الإنتاجية.

---

## 5) الفجوات الحالية مقابل المطلوب (Gap Matrix)

| المجال | الحالة الحالية | الفجوة | قرار التنفيذ |
|---|---|---|---|
| API Contracts | غير موحّد بالكامل | mismatch endpoints | Contract Freeze v1 |
| Editor | Textarea | لا Monaco/advanced editing | Monaco Phase |
| Terminal | process spawn موجود | PTY/session isolation ناقص | PTY hardening |
| Git UX | Rust commands جيدة | UI workflows ناقصة | Git MVP ثم توسع |
| Sync | نواة موجودة | conflict/ws/replay ناقص | Sync hardening |
| Auto Update | Stub في UI | لا قناة نشر موحدة | Signed rollout pipeline |
| Resource Control | مراقبة جزئية | لا enforcement policy شامل | Policy + agent enforcement |
| Training Metrics | جزئي/Mock | لا قياسات موثوقة end-to-end | Real telemetry ingestion |
| AI Council | Hybrid مع fallback | governance/reliability ناقص | Provider orchestration |

---

## 6) الخطة التنفيذية المعتمدة (120 يوم)

## المرحلة 0 — Contract Freeze & Wire-Up (الأسبوع 1-2)

الهدف:
- تصفير أي mismatch في الربط.

العمل:
- توحيد مسارات sync/training/council/workers.
- إصدار وثيقة API Contracts v1 versioned.
- بناء smoke e2e إلزامي لمسار:
  Desktop -> API -> Worker -> Status العودة.

DoD:
- لا أي endpoint mismatch.
- smoke e2e يمر في CI.

Fallback Plan:
- إذا Contract Freeze تجاوز 3 أسابيع: نعتمد contract-per-service بدل unified freeze.
- أي service ينجح freeze مستقل يبدأ بيتحرك للمرحلة 1 بدون انتظار الباقي.

## المرحلة 1 — Core IDE Production (الأسبوع 3-6)

الهدف:
- IDE فعلي بمستوى إنتاجي.

العمل:
- Monaco integration كامل (tabs, dirty state, save, language modes).
- File explorer حقيقي + watching.
- Terminal PTY حقيقي مع lifecycle آمن.
- Git MVP: status/stage/commit/push/diff أساسي.
- **Quick Open (Cmd+P)**: fuzzy file search عبر المشروع.
- **Command Palette (Cmd+Shift+P)**: كل الأوامر من مكان واحد.
- **Search & Replace (Cmd+Shift+F)**: بحث في كل الملفات (ripgrep من Rust).
- Minimap + Breadcrumbs (built-in Monaco).

DoD:
- مشروع متوسط/كبير ينفتح ويتعدل ويتبنى من داخل الدسكتوب بدون انقطاع.
- Cmd+P يفتح ملفات بسرعة + Command Palette يشتغل.

## المرحلة 2 — Worker Fabric + Resource Governance (الأسبوع 7-10)

الهدف:
- تحويل كل الأجهزة إلى شبكة تنفيذ قابلة للضبط.

العمل:
- Device enrollment: install/register/heartbeat/capabilities.
- سياسات موارد قابلة للتعديل من UI.
- Enforcement عبر agent لكل نظام تشغيل.
- Worker classes: full/assist/training-only.

DoD:
- تغيير حدود الموارد من UI يطبق فعلياً على الجهاز خلال أقل من 10 ثواني.

## المرحلة 3 — Sync & Signed Auto-Update (الأسبوع 11-14)

الهدف:
- تحديثات ومزامنة موثوقة على كل الأجهزة.

العمل:
- إكمال conflict handling + replay + ws broadcast الحقيقي.
- Device identity rotation + transport encryption.
- signed manifests + rollout channels + staged deployment + rollback.
- auto-update لكل من desktop app وagent.

DoD:
- إصدار جديد يصل لأجهزة canary أولاً ثم stable تلقائياً.
- rollback تلقائي عند failure rate يتجاوز العتبة.

## المرحلة 4a — Code Intelligence (الأسبوع 15-16)

الهدف:
- ذكاء اصطناعي عملي يساعد المطور أثناء الكتابة.

العمل:
- **Inline Code Completion** (Copilot-style) عبر Monaco inline suggestions.
- **Explain Code**: اختيار كود → AI يشرحه.
- **Refactor Code**: AI يقترح إعادة بناء.
- **Error Fix Suggestions**: AI يقترح إصلاح الأخطاء.
- ربط مع RTX 5090 inference + fallback محلي (Ollama/llama.cpp).
- Model Selection UI (اختيار نموذج من الإعدادات).

DoD:
- كتابة كود → اقتراحات inline تظهر < 400ms.
- اختيار كود → Explain/Refactor يشتغل.

## المرحلة 4b — Council Hardening (الأسبوع 17-18)

الهدف:
- مجلس حكماء حقيقي بموثوقية عالية.

العمل:
- Provider orchestration مع fallback policy مرتبة.
- grounding checks + confidence calibration.
- ربط المجلس بمدخلات مشاريع فعلية لا رسائل عامة فقط.
- ربط Shadow Team + Light Team + الكشافة فعلياً.
- Conversation Memory (Vector DB + context awareness).

DoD:
- جودة الاستجابة مستقرة وفق KPIs latency/quality.
- المجلس يحلل كود فعلي مو بس أسئلة عامة.

## المرحلة 5 — Self-Improvement Gated (الأسبوع 19-20)

الهدف:
- تحسين ذاتي مضبوط، لا ترقيع ولا مخاطرة إنتاجية.

العمل:
- loop: propose -> sandbox test -> evaluate -> promote.
- promotion gate + kill switch + audit trail.

DoD:
- دورة يومية ناجحة داخل sandbox مع ترقية آمنة عند اجتياز الشروط.

## 6.1 ربط المراحل مع Gates (تنفيذي)

- Phase 0 -> Gate A
- Phase 1 -> Gate B
- Phase 2 -> Gate C
- Phase 3 -> Gate D
- Phase 4a -> Gate D2
- Phase 4b + Phase 5 -> Gate E

قاعدة تنفيذ:
- لا انتقال للمرحلة التالية بدون نجاح Gate المرحلة الحالية.

---

## 7) عقود الربط الإلزامية (Contract v1)

## 7.1 مبدأ العقود
- كل endpoint يُعرّف input/output/errors/version.
- لا استدعاء مباشر بلا client contract.

## 7.2 مصفوفة العقود القانونية (Canonical Contract Matrix v1)

| المجال | Method | Path | Canonical | Owner |
|---|---|---|---|---|
| Council | POST | `/api/v1/council/message` | ✅ | Backend |
| Council | GET | `/api/v1/council/status` | ✅ | Backend |
| Training | POST | `/api/v1/training/start` | ✅ | Backend |
| Training | GET | `/api/v1/training/status` | ✅ | Backend |
| Training | POST | `/api/v1/training/stop` | ✅ | Backend |
| Sync | POST | `/api/v1/sync` | ✅ | Rust + Backend |
| Sync | GET | `/api/v1/sync/status` | ✅ | Rust + Backend |
| Sync | WS | `/api/v1/sync/ws` | ✅ | Rust + Backend |
| Workers | POST | `/api/v1/workers/register` | ✅ | Agent + Backend |
| Workers | POST | `/api/v1/workers/heartbeat` | ✅ | Agent + Backend |
| Workers | POST | `/api/v1/workers/apply-policy` | ✅ | Agent + Backend |
| Updates | GET | `/api/v1/updates/manifest` | ✅ | Platform |
| Updates | POST | `/api/v1/updates/report` | ✅ | Platform |

## 7.3 قواعد التوافق (Compatibility Rules)

- أي route قديم (legacy) يبقى عبر gateway translation لمدة دورة إصدار واحدة فقط.
- بعد دورة واحدة: إزالة legacy routes إلزامية.
- أي عميل جديد ممنوع يستهلك legacy routes.
- أي تغيير على المسارات canonical يحتاج ADR جديد + توقيع Owner المجال.

---

## 8) إدارة الموارد من الدسكتوب (Core Requirement)

## 8.1 نموذج سياسة الموارد

```json
{
  "device_id": "worker-123",
  "mode": "training-only",
  "limits": {
    "cpu_max_percent": 85,
    "ram_max_gb": 24,
    "gpu_mem_max_percent": 90,
    "io_nice": "normal"
  },
  "schedule": {
    "timezone": "Asia/Baghdad",
    "windows": [
      {"start": "22:00", "end": "07:00"}
    ],
    "idle_only": true
  },
  "safety": {
    "thermal_cutoff_c": 85,
    "auto_pause_on_user_activity": true
  }
}
```

## 8.2 متطلبات التنفيذ
- policy تصدر من control plane.
- agent يستلم policy ويؤكد التفعيل عبر heartbeat.
- أي خرق limits ينتج event + auto-throttle.

## 8.3 معايير القبول
- تعديل policy من UI -> ينعكس على الجهاز بسرعة.
- dashboard تعرض planned vs actual usage لكل جهاز.

---

## 9) التحديثات التلقائية الشاملة لكل الأجهزة

## 9.1 مبادئ النشر
- كل إصدار موقّع.
- قنوات: canary -> beta -> stable.
- phased rollout حسب نسب نجاح وصحة.

## 9.2 خطوات rollout
1. نشر manifest موقّع.
2. canary بنسبة 5%.
3. مراقبة crash/error/rollback signals.
4. التوسع إلى 25% ثم 100%.
5. rollback تلقائي عند تخطي عتبة الفشل.

## 9.3 ما يجب إضافته فوراً
- updater plugin integration حقيقي (desktop + agent).
- update health contract موحد.
- release coordinator service.

---

## 10) الأمن والحوكمة

## 10.1 أمن التشغيل
- mTLS أو TLS mutual trust داخل شبكة العمال.
- device keypair + rotation دوري.
- least privilege للعمليات المنفذة على العمال.

## 10.2 حوكمة الذكاء الذاتي
- no direct production promote.
- policy tiers: safe / guarded / experimental.
- kill switch عالمي من control plane.

## 10.3 السجلات والتدقيق
- audit trail إلزامي لكل:
  - policy change
  - model promotion
  - update rollout
  - remote execution

## 10.4 إدارة مفاتيح التوقيع
- Signing key يُحفظ في hardware-bound store:
  - macOS: Keychain
  - Windows: DPAPI
  - Linux: TPM أو encrypted keyring
- ممنوع تخزين signing key في repo أو `.env` أو أي ملف نصي.
- Key rotation كل 6 أشهر كحد أقصى.
- Owner: **Platform/DevOps**
- Backup Owner: **Security**

---

## 11) مؤشرات النجاح المرحلية (KPIs)

Foundation KPIs (بعد المرحلة 2)
- Crash-free sessions >= 99.0%
- Startup P95 < 3.5s
- File open P95 < 180ms
- Worker enrollment success >= 95%
- Quick Open search P95 < 200ms

Scale KPIs (بعد المرحلة 4)
- Crash-free sessions >= 99.5%
- Startup P95 < 2.5s
- Sync convergence LAN P95 < 2s
- Worker uptime >= 99%
- Update success rate >= 98%
- Policy apply latency P95 < 10s
- Code completion P95 < 400ms

AI KPIs (بعد المرحلة 5)
- Council response P95 < 700ms (local/LAN path)
- Hallucination critical rate <= 2%
- Sandbox promotion pass rate tracked أسبوعياً
- Self-improvement cycle success rate >= 80%

---

## 12) سجل المخاطر (Risk Register)

| الخطر | التأثير | الاحتمال | التخفيف |
|---|---|---|---|
| تضخم نطاق التنفيذ | عالي | عالي | phase gates + strict DoD |
| mismatch عقود API | عالي | عالي | contract freeze + compatibility tests |
| فشل تحديثات جماعي | عالي | متوسط | staged rollout + auto rollback |
| استنزاف موارد أجهزة المستخدم | عالي | متوسط | hard limits + idle mode + thermal cutoff |
| أخطاء sync/conflicts | متوسط | متوسط | CRDT tests + replay validation |
| regressions في AI quality | متوسط | متوسط | evaluation gates + canary models |
| فقدان بيانات تدريب عند crash | عالي | منخفض | checkpoint كل N steps + remote backup + resume token |
| تسريب signing key | حرج | منخفض | hardware-bound store + rotation + audit |

## 12.1 مصفوفة ملكية المخاطر (Risk Ownership Matrix)

| الخطر | Owner | Backup | Gate المراقبة | Trigger | الاستجابة الإلزامية |
|---|---|---|---|---|---|
| تضخم نطاق التنفيذ | Platform | PMO | Gate A/B | تأخر > 20% عن الخطة الأسبوعية | تجميد الميزات الجديدة + إعادة ترتيب backlog |
| mismatch عقود API | Backend | Rust | Gate A | أي فشل contract compatibility test | rollback لآخر عقد ثابت + patch خلال 24 ساعة |
| فشل تحديثات جماعي | Platform/DevOps | Rust | Gate D | update success < 98% أو crash spike | إيقاف rollout + rollback فوري |
| استنزاف موارد أجهزة المستخدم | Agent | Desktop | Gate C | policy drift > 10% أو حرارة > cutoff | auto-throttle + pause jobs + تنبيه فوري |
| أخطاء sync/conflicts | Rust | Backend | Gate D | conflict rate > 2% أو convergence > KPI | تفعيل replay-safe mode + منع نشر sync جديد |
| regressions في AI quality | AI/ML | Backend | Gate D2/E | hallucination critical > 2% | إرجاع model السابق + canary re-eval |
| فقدان بيانات تدريب عند crash | ML | Platform | Gate E | فشل resume أو فقد checkpoint | recovery من remote backup + disable auto-promote |
| تسريب signing key | Security | Platform/DevOps | Gate D | أي دليل compromise | key revoke فوري + rotation + forced update |

قواعد إلزامية:
- أي Risk يدخل Trigger لازم يتسجل Incident خلال 30 دقيقة كحد أقصى.
- مالك الخطر يفتح Corrective Plan خلال 24 ساعة مع موعد إغلاق محدد.
- لا عبور Gate لاحق إذا Risk حرج مفتوح بلا mitigation فعال.

---

## 13) خطة أول 14 يوم (تفصيل تنفيذي)

اليوم 1-2
- اعتماد Contract v1.
- حسم endpoint mapping للـ sync/training.

اليوم 3-4
- تنفيذ clients موحدة وتحديث الاستدعاءات في desktop commands.
- smoke e2e محلي.

اليوم 5-6
- إدخال Monaco بشكل MVP.
- ربط فتح/حفظ ملفات فعلياً.

اليوم 7-8
- PTY terminal integration.
- تحسين lifecycle للعمليات.

اليوم 9-10
- Git panel MVP وربط status/stage/commit.

اليوم 11-12
- worker policy APIs + dashboard controls.

اليوم 13-14
- updater integration أولي + canary release داخلي.

مخرج الأسبوعين:
- نواة تشغيل حقيقية + ربط End-to-End + تحكم موارد أساسي.

### Exit Criteria رقمية (اليومي/الثنائي)

- يوم 1-2:
  - `api_contracts_v1` منشور ومراجع.
  - 0 endpoint mismatch في اختبارات التوافق.
- يوم 3-4:
  - smoke e2e يمر بنسبة نجاح >= 95% على 20 تشغيل متتالي.
- يوم 5-6:
  - Monaco يفتح/يحفظ ملفات فعلية.
  - فتح ملف P95 <= 220ms في dataset داخلي.
- يوم 7-8:
  - PTY يعمل بجلسات متزامنة (>= 3 جلسات) بدون crash.
- يوم 9-10:
  - Git MVP: status/stage/commit شغال end-to-end.
  - فشل عمليات git الحرجة <= 3% في smoke.
- يوم 11-12:
  - تطبيق policy على Worker خلال P95 <= 10s.
  - قياس planned vs actual usage ظاهر في UI.
- يوم 13-14:
  - canary update داخلي لـ 2 أجهزة على الأقل.
  - rollback اختباري ناجح مرة واحدة على الأقل.

---

## 14) بوابات الجودة (Quality Gates)

Gate A (بعد المرحلة 0)
- جميع التعاقدات موثقة ومختبرة.
- ممنوع إضافة feature بدون contract.

Gate B (بعد المرحلة 1)
- Core IDE workflows تعمل بلا mock.

Gate C (بعد المرحلة 2)
- policy enforcement فعلي لكل جهاز مجرب.

Gate D (بعد المرحلة 3)
- sync/update بثبات إنتاجي + rollback مؤكد.

Gate D2 (بعد المرحلة 4a)
- Code completion يشتغل فعلياً + latency ضمن KPI (< 400ms).
- Explain/Refactor يشتغل end-to-end مع model حقيقي.

Gate E (بعد المرحلة 5)
- self-improvement gated وآمن.

---

## 15) سجل الأفكار من النسخ القديمة (Legacy Idea Registry)

> هذه أفكار كانت موجودة بالنسخ القديمة (D:/bi ide/) ولازم **ما تضيع**.
> سياسة: **نقل أفكار فقط — ممنوع نسخ كود قديم** (CODE_FREE_IDEA_MIGRATION_POLICY).

| # | الفكرة | المصدر | المرحلة المقترحة |
|---|--------|--------|----------------|
| 1 | Neural Network + Multi-Head Attention | v6 super-intelligent-learning | 4a |
| 2 | Double DQN + Prioritized Experience Replay | v6 | 5 |
| 3 | Meta-Learning (تعلم كيف يتعلم) | v6 | 5 |
| 4 | Curriculum Learning (صعوبة تدريجية 1→10) | v6 | 2 (training) |
| 5 | TF-IDF + N-gram لفهم الكود | v5/v6 | 4a |
| 6 | Auto-Learning من 95 PDF عربي/إنجليزي | v5 | 2 (training) |
| 7 | Arabic NLP مخصص (تشكيل + جذور + stopwords) | v4/v5 | 4b |
| 8 | 16 حكيم في المجلس الأعلى | v7/v8 hierarchy | 4b |
| 9 | Shadow Team — 4 متشائمين يحللون المخاطر | v7/v8 | 4b |
| 10 | Light Team — 4 متفائلين يقترحون فرص | v7/v8 | 4b |
| 11 | الكشافة (تقني + سوق + منافسين + فرص) | v7/v8 | 5 |
| 12 | Federated Learning (تدريب موزع + خصوصية) | جديد | 2 |
| 13 | Vector DB (HNSW) — ذاكرة ذكية دلالية | جديد | 4b |
| 14 | Predictive Error Detection (توقع الأخطاء) | جديد | 4a |
| 15 | Plugin System + Marketplace + SDK | جديد | بعد المرحلة 5 |
| 16 | Project Factory (idea→spec→code→test→deploy) | v6 | 5 |
| 17 | Self-Repair + Auto-Rollback عند drift | v6 | 5 |
| 18 | Elastic Weight Consolidation (منع النسيان) | جديد | 2 (training) |
| 19 | Sandbox Execution لتشغيل كود آمن | جديد | 4a |
| 20 | Business Intelligence AI (تقارير ERP ذكية) | جديد | بعد المرحلة 5 |
| 21 | Cost-aware H200/GPU scheduler (IDEA-008) | v6 IDEA_PARITY | 2 |
| 22 | Real-time artifact streaming كل دقيقة (IDEA-010) | v6 IDEA_PARITY | 2 |
| 23 | Multi-agent specialist chain: planner→coder→tester→deployer (IDEA-007) | v6 IDEA_PARITY | 5 |
| 24 | 4-level hierarchical memory (IDEA-006) | v6 IDEA_PARITY | 4b |
| 25 | Emergency override governance (IDEA-011) | v6 IDEA_PARITY | 3 |
| 26 | Performance profiling داخل IDE | IDE_IDEAS_MASTER | 1 |
| 27 | Feature flags لتفعيل/تعطيل ميزات | IDE_IDEAS_MASTER | 0 |
| 28 | Live collaboration editing | IDE_IDEAS_MASTER | 3 (sync) |

> **ملاحظة من LEGACY_DESKTOP_AUDIT:** النسخة القديمة v6 (Electron) كانت تحتوي Monaco + xterm + node-pty + onnxruntime-node — كانت أغنى feature-wise من v8 الحالي. لكن كانت غير مستقرة (structlog missing, build drift). v8 أنظف ويتقبل إعادة البناء بشكل أفضل.

---

## 16) خريطة الملفات المطلوبة (Component Map)

```
apps/desktop-tauri/src/components/
├── editor/
│   ├── MonacoEditor.tsx      ← [NEW] المحرر الحقيقي (مرحلة 1)
│   ├── TabBar.tsx            ← [NEW] تبويبات (مرحلة 1)
│   ├── QuickOpen.tsx         ← [NEW] Cmd+P (مرحلة 1)
│   ├── CommandPalette.tsx    ← [NEW] Cmd+Shift+P (مرحلة 1)
│   ├── SearchPanel.tsx       ← [NEW] بحث شامل (مرحلة 1)
│   ├── InlineCompletion.tsx  ← [NEW] AI completion (مرحلة 4a)
│   ├── ProblemsPanel.tsx     ← [NEW] أخطاء (مرحلة 1)
│   └── OutputPanel.tsx       ← [NEW] output (مرحلة 1)
├── git/
│   ├── GitPanel.tsx          ← [NEW] (مرحلة 1)
│   ├── DiffViewer.tsx        ← [NEW] (مرحلة 1)
│   ├── GitGraph.tsx          ← [NEW] (مرحلة 3)
│   └── ConflictResolver.tsx  ← [NEW] (مرحلة 3)
├── workers/
│   └── WorkerPolicy.tsx      ← [NEW] تحكم موارد (مرحلة 2)
├── settings/
│   └── ModelSettings.tsx     ← [NEW] اختيار نموذج AI (مرحلة 4a)
└── themes/                   ← [NEW] dark/light/custom (مرحلة 1)

apps/desktop-tauri/src-tauri/src/commands/
├── ai.rs                     ← [NEW] local inference (مرحلة 4a)
├── search.rs                 ← [NEW] ripgrep search (مرحلة 1)
├── fs.rs                     ← [IMPROVE] more ops
├── git.rs                    ← [IMPROVE] graph/diff
├── terminal.rs               ← [IMPROVE] real PTY
└── sync.rs                   ← [IMPROVE] CRDT

services/
├── training_scheduler.py     ← [NEW] cost-aware (مرحلة 2)
├── model_registry.py         ← [NEW] versions (مرحلة 2)
└── update_coordinator.py     ← [NEW] signed rollout (مرحلة 3)

ai/learning/
├── continuous_learning.py    ← [NEW] EWC (مرحلة 2)
├── federated.py             ← [NEW] FedAvg (مرحلة 2)
└── meta_learner.py          ← [NEW] (مرحلة 5)
```

---

## 17) قرار التنفيذ الملزم

من هذه اللحظة:
1. لا توسع ميزات قبل إكمال Contract Freeze.
2. لا ادعاء readiness مع وجود mock في المسارات الحرجة.
3. الأولوية القصوى لمتطلباتك الأساسية:
   - ربط كل الطبقات فعلياً.
   - تحكم موارد من داخل الدسكتوب.
   - تحديث تلقائي موثوق لكل الأجهزة.
4. كل فكرة من النسخ القديمة تمر عبر Idea Registry (القسم 15) — لا نسخ كود، فقط أفكار.
5. كل مرحلة تمر عبر Quality Gate قبل الانتقال للتالية.

هذا التقرير هو المرجع المعتمد لإعادة هيكلة Desktop v8 بطريقة حقيقية، قابلة للقياس، وقابلة للتوسع العالمي.

---

## 18) Ownership Matrix (منو مسؤول عن شنو)

| المسار | المالك الأساسي | مالك ثانوي | شرط التسليم |
|---|---|---|---|
| Desktop UX/Editor/Git | Desktop Team | Rust Team | اجتياز Gate B |
| Rust Commands/PTY/Sync Client | Rust Team | Desktop Team | اجتياز Gate B/D |
| API Contracts/Gateway | Backend Team | Platform Team | اجتياز Gate A |
| Worker Agent/Policy Enforcement | Agent Team | Platform Team | اجتياز Gate C |
| Updates/Release Coordinator | Platform Team | DevOps | اجتياز Gate D |
| AI Inference/Council | AI Team | Backend Team | اجتياز Gate E |
| Training/Evaluation/Registry | ML Team | Backend Team | اجتياز Gate E |

قاعدة ملزمة:
- كل بند في backlog لازم يحمل `owner` و`backup owner` قبل بدء التنفيذ.

---

## 19) Dependency Graph (ترتيب إلزامي)

1. Phase 0 (Contracts) -> شرط دخول Phase 1.
2. Phase 1 (Core IDE) + جزء API مستقر -> شرط دخول Phase 2.
3. Phase 2 (Workers/Policy) -> شرط دخول Phase 3.
4. Phase 3 (Sync/Updates) -> شرط دخول Phase 4a/4b.
5. Phase 4a/4b -> شرط دخول Phase 5.

Dependency Rules:
- ممنوع بدء Phase أعلى إذا Gate المرحلة السابقة فشل.
- أي استثناء يحتاج ADR قصير مع تحليل أثر وموافقة مالك المنصة.

---

## 20) Ready-to-Execute Checklist (100% Readiness)

### 20.1 جاهزية المنتج
- [ ] Canonical Contract Matrix معتمد وموقع من Owners.
- [ ] Legacy routes محصورة بزمن إيقاف واضح (1 release cycle).
- [ ] جميع Gates معرفة باختبارات pass/fail قابلة للقياس.

### 20.2 جاهزية البنية
- [ ] CI يحتوي: contract tests + smoke e2e + security checks.
- [ ] rollback path مجرب للتطبيق والـ agent.
- [ ] key management مفعل فعلياً (Keychain/DPAPI/TPM).

### 20.3 جاهزية الفريق
- [ ] owner/backup لكل workstream مثبتين كتابياً.
- [ ] on-call rotation أسبوعي محدد (Platform + Backend + Rust).
- [ ] زمن الاستجابة للحوادث متفق عليه (SLA/SLO).

### 20.4 جاهزية القياس
- [ ] dashboard موحد لـ KPI + Risk triggers.
- [ ] trace_id مفعل end-to-end من Desktop إلى API إلى Worker.
- [ ] postmortem template جاهز لأي incident حرج.

---

## 21) Runbook مختصر للحوادث الحرجة

## 21.1 فشل تحديثات جماعي

أول 5 دقائق:
- إيقاف rollout فوراً.
- تثبيت القناة على آخر stable manifest.

أول ساعة:
- تفعيل rollback للأجهزة المتأثرة.
- تحديد النسخة/القناة المتسببة بالفشل.

أول 24 ساعة:
- Patch validation على canary مغلق.
- تقرير RCA + قرار إعادة فتح rollout.

## 21.2 تسريب signing key

أول 5 دقائق:
- revoke للمفتاح المتأثر.
- تعليق التوقيع التلقائي مؤقتاً.

أول ساعة:
- إصدار مفتاح جديد + نشر trust update.
- forced update policy للأجهزة الحرجة.

أول 24 ساعة:
- audit شامل للوصول + root cause.
- تدوير كامل للمفاتيح المرتبطة.

## 21.3 تدهور جودة AI

أول 5 دقائق:
- إيقاف auto-promote.
- إعادة model baseline المعتمد.

أول ساعة:
- تحليل metrics: hallucination/latency/confidence drift.
- عزل canary model.

أول 24 ساعة:
- re-evaluation على benchmark ثابت.
- release جديد فقط بعد Gate D2/E pass.

## 21.4 failure في sync/convergence

أول 5 دقائق:
- تفعيل replay-safe mode.
- تعطيل نشر sync experimental flags.

أول ساعة:
- فحص vector clocks/conflict rates.
- تجميد أي deployment متعلق بالـ sync.

أول 24 ساعة:
- patch + soak test LAN.
- فتح الخدمة تدريجياً بعد تحقق KPI.

---

## 22) اقتراحات تحسين عالية الأثر (أفضل إضافات)

1. **Feature Flags إلزامية** لكل capability جديدة قبل الإطلاق العام.
2. **Dark Launch** لميزات AI الجديدة قبل تمكينها للمستخدمين.
3. **Budget Guardrails** لاستهلاك GPU/CPU لكل worker لتقليل المفاجآت.
4. **Golden Benchmark Set** ثابت لاختبارات جودة AI قبل أي ترقية.
5. **Contract Codegen** لتقليل أخطاء clients بين Desktop/Backend.
6. **Canary-by-Region/Group** بدل canary عشوائي لنتائج أوضح.
7. **Chaos Drills شهرية** لتجربة rollback/sync failure/key revoke.
8. **Warm Start Cache** لتحسين startup latency في الأجهزة الأضعف.
9. **Policy Simulation Mode** لمعاينة أثر limits قبل تطبيقها.
10. **Release Freeze Window** قبل أي milestone production كبير.

---

## 23) قرار Go / No-Go قبل بدء التنفيذ

**Go فقط إذا كل التالي متحقق:**
- [ ] Gate A criteria واضح ومغطى باختبارات.
- [ ] owners + backups + on-call مثبتين.
- [ ] rollback مثبت عملياً للتطبيق والـ agent.
- [ ] dashboard KPI/Risk شغال ومراقب.

**No-Go إذا تحقق أي مما يلي:**
- [ ] endpoint mismatch مستمر.
- [ ] غياب rollback path مجرب.
- [ ] signing key management غير مفعل فعلياً.
- [ ] Risk حرج مفتوح بلا mitigation مقبول.

هذا القسم يجعل الخطة جاهزة للتنفيذ الحقيقي وليس فقط جاهزة للعرض.

---

## 24) سجل إصدارات الوثيقة (Document Version History)

| الإصدار | التاريخ | التغييرات الرئيسية |
|---------|---------|-------------------|
| 2.0 | 2026-03-02 | النسخة الأولى المعتمدة (Execution Grade) |
| 3.0 | 2026-03-02 | Cmd+P/Cmd+Shift+P في مرحلة 1 + تقسيم 4a/4b + 20 فكرة legacy + Component Map |
| 3.1 | 2026-03-02 | Canonical Routing Lock + Gate-Phase mapping + KPI تعديل + Owner matrix |
| 4.0 | 2026-03-02 | Contract Matrix موحد + Risk Ownership Matrix + Runbooks + Go/No-Go + Readiness Checklist + 28 فكرة + Hardware Inventory + Version History |
