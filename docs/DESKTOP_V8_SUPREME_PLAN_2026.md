# BI-IDE Desktop v8 — الخطة العملاقة الشاملة (Supreme Edition)

التاريخ: 2026-03-02 (محدث)
الإصدار: 4.0 (Supreme Execution Grade)
الحالة: جاهز للتنفيذ مع خطط طوارئ
المدة: 180 يوم (6 أشهر) - واقعية وقابلة للتحقق

---

## 1) الحكم التنفيذي النهائي (المحدث)

الخطة الطموحة **قابلة للتنفيذ بنسبة 90%** إذا اتبعنا هذا الترتيب الصارم مع ** buffers واقعية**:

1. **تثبيت العقود** (أسبوعين) → Buffer: +1 أسبوع للمفاجآت
2. **بناء Core IDE** (6 أسابيع) → Buffer: +2 أسبوع لـ Monaco复杂性
3. **Worker Fabric** (5 أسابيع) → Buffer: +2 أسبوع لـ Cross-platform differences
4. **Sync & Updates** (5 أسابيع) → Buffer: +2 أسبوع لـ Security auditing
5. **AI Intelligence** (6 أسابيع) → Buffer: +3 أسابيع لـ Model tuning
6. **Self-Improvement** (4 أسابيع) → Buffer: +2 أسبوع لـ Safety validation

**المجموع: 180 يوم (26 أسبوع)**

**الفرق عن الخطة السابقة:**
- زيادة 60 يوماً للـ Buffers والاختبارات
- إضافة Quality Gates صارمة بين كل مرحلة
- إضافة Contingency Plans لكل مرحلة

---

## 2) ما هو مثبت من الواقع الحالي (Reality Check)

### ✅ مثبت ويُبنى عليه
- الاتجاه التقني الحالي صحيح: Tauri v2 + Rust + React.
- أوامر Rust موجودة ومتصلة على نطاق واسع (~2,245 سطر).
- Desktop Agent Rust موجود كبنية تشغيل على العقد.
- Sync service موجود كنواة CRDT/WS (~887 سطر Rust).
- Council chat مربوط جزئياً بالـ API مع fallback.
- API Routers موجودة (~700 سطر Python).

### ⚠️ موجود لكنه غير إنتاجي بالكامل
- محرر الكود الحالي Textarea، وليس Monaco.
- Terminal: process spawn موجود لكن بدون PTY حقيقي.
- لوحات التدريب والعمال تحتوي بيانات تجريبية (fake_history).
- auto-update hook في الواجهة ما زال Stub.
- بعض الـ telemetry/training metrics في أوامر Rust ما زالت Mock.

### 🔴 فجوات ربط حرجة
- sync endpoint mismatch بين الدسكتوب والخدمة.
- training endpoint mismatch بين desktop commands وbackend routers.
- غياب contract versioning إلزامي يمنع الاستقرار عند التوسع.

---

## 3) الرؤية التقنية النهائية (Target Operating Model)

### 3.1 طبقات النظام (المحدثة)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │   Desktop    │  │    Web       │  │    Mobile    │                   │
│  │   (Tauri)    │  │   (React)    │  │   (React)    │                   │
│  │   Rust+TS    │  │   Next.js    │  │   Native)    │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
├─────────────────────────────────────────────────────────────────────────┤
│                         AI & INTELLIGENCE LAYER                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Supreme   │ │   Local     │ │   Code      │ │   Knowledge         │ │
│  │   Council   │ │   LLM       │ │   Intelligence│ │   Graph             │ │
│  │  (5 Agents) │ │  (7B-70B)   │ │  (AST+ML)   │ │  (Vector+Semantic)  │ │
│  │             │ │             │ │             │ │                     │ │
│  │ • Architect │ │ • Llama 3   │ │ • Completion│ │ • Project Memory    │ │
│  │ • Security  │ │ • Mistral   │ │ • Explain   │ │ • Cross-ref         │ │
│  │ • Performance│ │ • Qwen     │ │ • Refactor  │ │ • Learning          │ │
│  │ • UX        │ │ • Local     │ │ • Debug     │ │ • Context           │ │
│  │ • Ethics    │ │ • Remote    │ │ • Predict   │ │ • Evolution         │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                         SYNC & COLLABORATION LAYER                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   CRDT      │ │  Blockchain │ │   Real-time │ │   Version           │ │
│  │   Engine    │ │  Verification│ │   Presence  │ │   Control           │ │
│  │  (Yjs/Rust) │ │  (Artifacts) │ │  (WebSocket)│ │  (Git+Custom)       │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                         DISTRIBUTED FABRIC LAYER                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Worker    │ │   Federated │ │   Edge      │ │   Swarm             │ │
│  │   Nodes     │ │   Learning  │ │   Inference │ │   Intelligence      │ │
│  │  (Classes)  │ │  (Privacy)  │ │  (TinyML)   │ │  (Emergent)         │ │
│  │             │ │             │ │             │ │                     │ │
│  │ • Full      │ │ • FedAvg    │ │ • Mobile    │ │ • Self-org          │ │
│  │ • Assist    │ │ • Secure    │ │ • IoT       │ │ • Pheromone         │ │
│  │ • Training  │ │ • Aggregation│ │ • Embedded │ │ • Collective        │ │
│  │ • Edge      │ │ • Diff Priv │ │ • Ultra-low │ │ • Behavior          │ │
│  │ • Mobile    │ │             │ │ • Latency   │ │                     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                         SECURITY & GOVERNANCE LAYER                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │  Quantum-   │ │  Homomorphic │ │  Biometric  │ │   Self-Healing      │ │
│  │  Resistant  │ │  Encryption  │ │  & Hardware │ │   & Auto-Remediation│ │
│  │  Crypto     │ │  (Compute)   │ │  Auth       │ │                     │ │
│  │  (Kyber)    │ │  (Future)    │ │  (FIDO2)    │ │ • Auto-fix          │ │
│  │  (Dilithium)│ │              │ │  (WebAuthn) │ │ • Predict           │ │
│  │             │ │              │ │             │ │ • Heal              │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 مبدأ حاكم (Core Principles)

1. **لا منطق تشغيلي حرج داخل UI فقط.**
2. **كل قرار حرج يمر عبر Policy في السيرفر ثم يُنفذ على الـ Agent.**
3. **Privacy by Default**: البيانات تُعالج محلياً أولاً.
4. **Fail-Safe**: أي خطأ يؤدي إلى الإيقاف الآمن لا التلف.
5. **Observability**: كل شيء يُقاس ويُسجل.

---

## 4) المتطلبات غير القابلة للتفاوض (Non-Negotiables)

1. **ربط كامل** Desktop ↔ Website ↔ Server ↔ Workers.
2. **أي جهاز جديد** يدعم agent onboarding خلال **دقيقتين** (مطلوب < 5 دقائق).
3. **تحكم فعلي** من الدسكتوب بحدود الاستهلاك:
   - CPU سقف % (مع throttling تلقائي)
   - RAM سقف GB (مع OOM protection)
   - GPU Memory سقف % (مع thermal throttling)
   - نافذة زمنية للتشغيل (timezone-aware)
   - Idle-only mode (مع activity detection)
   - Thermal cutoff (درجة حرارة قصوى)
4. **تحديثات تلقائية آمنة** وموقعة لكل الأجهزة مع rollback فوري.
5. **توافق عكسي** مدروس للعقود عبر versioning (v1 → v2 migration path).
6. **لا بيانات تجريبية** في المسارات الإنتاجية (Zero mock data in prod).
7. **Code Quality Gates**: كل code يجتاز tests + review + benchmarks.

---

## 5) الفجوات الحالية مقابل المطلوب (Gap Matrix)

| المجال | الحالة الحالية | الفجوة | قرار التنفيذ | الأولوية |
|---|---|---|---|---|
| API Contracts | غير موحّد بالكامل | mismatch endpoints | Contract Freeze v1 | 🔴 P0 |
| Editor | Textarea | لا Monaco/advanced editing | Monaco Integration | 🔴 P0 |
| Terminal | process spawn موجود | PTY/session isolation ناقص | PTY Hardening | 🔴 P0 |
| Git UX | Rust commands جيدة | UI workflows ناقصة | Git MVP + Graph | 🟡 P1 |
| Sync | نواة موجودة | conflict/ws/replay ناقص | Sync Hardening | 🟡 P1 |
| Auto Update | Stub في UI | لا قناة نشر موحدة | Signed Rollout Pipeline | 🟡 P1 |
| Resource Control | مراقبة جزئية | لا enforcement policy شامل | Policy + Agent Enforcement | 🔴 P0 |
| Training Metrics | جزئي/Mock | لا قياسات موثوقة end-to-end | Real Telemetry Ingestion | 🟡 P1 |
| AI Council | Hybrid مع fallback | governance/reliability ناقص | Provider Orchestration | 🟢 P2 |
| Local LLM | غير موجود | dependency على cloud APIs | Local Inference (Ollama) | 🟡 P1 |
| Federated Learning | غير موجود | privacy concerns في التدريب | FedAvg Implementation | 🟢 P2 |
| Edge AI | غير موجود | لا دعم للأجهزة الضعيفة | TinyML Integration | 🟢 P2 |
| Plugin System | غير موجود | closed system | Plugin SDK + Marketplace | 🔵 P3 |
| Swarm Intelligence | غير موجود | centralized scheduling only | Emergent Behavior | 🔵 P3 |

---

## 6) الخطة التنفيذية العملاقة (180 يوم - 26 أسبوع)

### المرحلة 0 — Contract Freeze & Wire-Up (الأسبوع 1-3)

**الهدف:** تصفير أي mismatch في الربط + تثبيت الأساس.

**العمل:**
- توحيد مسارات sync/training/council/workers.
- إصدار وثيقة API Contracts v1 versioned.
- بناء smoke e2e إلزامي لمسار: Desktop -> API -> Worker -> Status العودة.
- **إضافة**: Feature Flags System (يمكن تفعيل/تعطيل ميزات بدون redeploy).
- **إضافة**: Observability Setup (logging, metrics, tracing من اليوم الأول).

**DoD (Definition of Done):**
- [ ] لا أي endpoint mismatch.
- [ ] smoke e2e يمر في CI بنسبة نجاح >= 95%.
- [ ] Feature flags تعمل لـ 3 ميزات على الأقل.
- [ ] Dashboard مراقبة يظهر metrics أساسية.

**Fallback Plan:**
- إذا Contract Freeze تجاوز 3 أسابيع: نعتمد contract-per-service بدل unified freeze.
- أي service ينجح freeze مستقل يبدأ بيتحرك للمرحلة 1 بدون انتظار الباقي.

**Contingency:**
- إذا فشل توحيد الـ contracts: نستخدم GraphQL كـ abstraction layer مؤقت.

---

### المرحلة 1 — Core IDE Supreme (الأسبوع 4-9)

**الهدف:** IDE فعلي بمستوى إنتاجي يتنافس مع VS Code في السرعة.

**العمل:**
- **Monaco integration كامل**:
  - tabs, dirty state, save, language modes
  - minimap, breadcrumbs, folding
  - **Quick Open (Cmd+P)**: fuzzy file search عبر المشروع (ripgrep من Rust)
  - **Command Palette (Cmd+Shift+P)**: كل الأوامر من مكان واحد
  - **Search & Replace (Cmd+Shift+F)**: بحث في كل الملفات
- **File explorer حقيقي + watching** (notify crate).
- **Terminal PTY حقيقي** مع lifecycle آمن:
  - node-pty أو Rust portable-pty
  - دعم multiple sessions
  - دعم tmux/screen
- **Git MVP**: status/stage/commit/push/diff أساسي + **Git Graph** (visualization).
- **إضافة**: Themes (Dark/Light/Custom) + Icon Themes.
- **إضافة**: Settings UI (تعديل الإعدادات بدون JSON).
- **إضافة**: Keyboard Shortcuts customization.

**DoD:**
- [ ] مشروع متوسط/كبير (10k+ files) ينفتح ويتعدل ويتبنى من داخل الدسكتوب بدون انقطاع.
- [ ] Cmd+P يفتح ملفات بسرعة P95 < 200ms.
- [ ] Command Palette يشتغل مع 50+ أمر.
- [ ] Terminal يشغل `npm start` تفاعلياً بدون مشاكل.
- [ ] Git operations نجاح率 >= 97%.

**Contingency:**
- إذا Monaco فشل في التكامل: ننتقل إلى CodeMirror 6 (أسرع لكن أقل مزايا).
- إذا PTY معقد: نستخدم `tmux` كـ backend مؤقت.

---

### المرحلة 2 — Worker Fabric + Resource Governance (الأسبوع 10-14)

**الهدف:** تحويل كل الأجهزة إلى شبكة تنفيذ قابلة للضبط + Federated Learning.

**العمل:**
- **Device enrollment**: install/register/heartbeat/capabilities.
- **سياسات موارد قابلة للتعديل من UI**:
  - CPU/RAM/GPU limits
  - Time windows (timezone-aware)
  - Idle detection
  - Thermal cutoff
- **Enforcement عبر agent** لكل نظام تشغيل:
  - Linux: cgroups v2
  - Windows: Job Objects
  - macOS: resource limits + thermal monitoring
- **Worker classes**: full/assist/training-only/edge/mobile.
- **إضافة**: Federated Learning (FedAvg):
  - تدريب موزع بدون مشاركة بيانات خام
  - Secure Aggregation
  - Differential Privacy
- **إضافة**: Elastic Weight Consolidation (EWC) لمنع "النسيان الكارثي".
- **إضافة**: Curriculum Learning (صعوبة تدريجية).

**DoD:**
- [ ] تغيير حدود الموارد من UI يطبق فعلياً على الجهاز خلال P95 < 10 ثوانٍ.
- [ ] dashboard يعرض planned vs actual usage لكل جهاز بدقة >= 95%.
- [ ] Federated Learning round ناجحة مع 3+ أجهزة.
- [ ] Worker enrollment success rate >= 95%.

**Risk Mitigation:**
- Cross-platform differences: اختبار يومي على Windows/macOS/Linux.
- Resource leaks: monitoring + auto-cleanup.

---

### المرحلة 3 — Sync & Signed Auto-Update (الأسبوع 15-19)

**الهدف:** تحديثات ومزامنة موثوقة على كل الأجهزة + Live Collaboration.

**العمل:**
- **إكمال conflict handling + replay + ws broadcast الحقيقي**.
- **Device identity rotation** + transport encryption (mTLS).
- **Signed manifests** + rollout channels + staged deployment + rollback.
- **auto-update** لكل من desktop app وagent.
- **إضافة**: Live Collaboration Editing (CRDT-based):
  - Multiple users edit same file
  - Cursor presence
  - Comments on code
- **إضافة**: Emergency Override Governance (IDEA-011).
- **إضافة**: Real-time artifact streaming كل دقيقة (IDEA-010).

**DoD:**
- [ ] إصدار جديد يصل لأجهزة canary (5%) أولاً ثم stable تلقائياً.
- [ ] rollback تلقائي عند failure rate > 5%.
- [ ] Sync convergence LAN P95 < 2s.
- [ ] 3 users يحررون نفس الملف بدون conflicts.

**Contingency:**
- إذا CRDT معقد: نستخدم Operational Transformation (OT) المبسط.
- إذا Updates فشلت: نظام manual update كـ fallback.

---

### المرحلة 4a — Code Intelligence & Local AI (الأسبوع 20-25)

**الهدف:** ذكاء اصطناعي عملي يساعد المطور أثناء الكتابة + Local LLM.

**العمل:**
- **Inline Code Completion** (Copilot-style) عبر Monaco inline suggestions.
- **Local LLM Integration**:
  - Ollama/llama.cpp integration
  - Models: Llama 3, Mistral, CodeLlama, Qwen
  - Quantization (4-bit, 8-bit) للسرعة
  - Model selection UI
- **Explain Code**: اختيار كود → AI يشرحه.
- **Refactor Code**: AI يقترح إعادة بناء.
- **Error Fix Suggestions**: AI يقترح إصلاح الأخطاء.
- **Predictive Error Detection**: توقع الأخطاء قبل حدوثها (IDEA جديد).
- **إضافة**: Sandbox Execution لتشغيل كود AI بأمان.
- **إضافة**: Arabic NLP مخصص (تشكيل + جذور + stopwords) - من الأفكار القديمة.

**DoD:**
- [ ] كتابة كود → اقتراحات inline تظهر P95 < 400ms.
- [ ] Local LLM يشتغل بدون اتصال إنترنت.
- [ ] اختيار كود → Explain/Refactor يشتغل end-to-end.
- [ ] Hallucination rate < 5% (قابل للقياس).

**Contingency:**
- إذا Local LLM بطيء: نستخدم hybrid (local للبسيط، cloud للمعقد).
- إذا Models كبيرة: quantization أكثر ت aggressiveness.

---

### المرحلة 4b — Supreme Council & Knowledge Graph (الأسبوع 26-31)

**الهدف:** مجلس حكماء حقيقي بموثوقية عالية + ذاكرة دلالية.

**العمل:**
- **Provider orchestration** مع fallback policy مرتبة:
  - Local LLM → Remote API (OpenAI/Claude) → Council Decision
- **grounding checks + confidence calibration**.
- **ربط المجلس بمدخلات مشاريع فعلية** لا رسائل عامة فقط.
- **ربط Shadow Team + Light Team + الكشافة فعلياً**:
  - Shadow Team: 4 متشائمين يحللون المخاطر
  - Light Team: 4 متفائلين يقترحون فرص
  - الكشافة: تقني + سوق + منافسين + فرص
- **Conversation Memory**: Vector DB (HNSW) + context awareness (IDEA جديد).
- **Knowledge Graph**:
  - Project entities and relationships
  - Code semantics (beyond AST)
  - Cross-project insights (بخصوصية)
  - 4-level hierarchical memory (IDEA-006)
- **إضافة**: 16 حكيم في المجلس الأعلى (من الأفكار القديمة).

**DoD:**
- [ ] جودة الاستجابة مستقرة وفق KPIs latency/quality.
- [ ] المجلس يحلل كود فعلي مو بس أسئلة عامة.
- [ ] Context window >= 10k tokens للمشاريع الكبيرة.
- [ ] Council response P95 < 700ms (local/LAN path).

---

### المرحلة 5 — Self-Improvement & Automation (الأسبوع 32-36)

**الهدف:** تحسين ذاتي مضبوط، لا ترقيع ولا مخاطرة إنتاجية.

**العمل:**
- **loop**: propose -> sandbox test -> evaluate -> promote.
- **promotion gate + kill switch + audit trail**.
- **Project Factory** (idea→spec→code→test→deploy):
  - Multi-agent specialist chain: planner→coder→tester→deployer (IDEA-007)
- **Self-Repair + Auto-Rollback** عند drift.
- **Meta-Learning**: تعلم كيف يتعلم (من الأفكار القديمة).
- **Continuous Learning**:
  - Neural Network + Multi-Head Attention (من v6)
  - Double DQN + Prioritized Experience Replay (من v6)
- **Plugin System + Marketplace + SDK** (للتوسع المستقبلي).

**DoD:**
- [ ] دورة يومية ناجحة داخل sandbox مع ترقية آمنة عند اجتياز الشروط.
- [ ] Self-improvement cycle success rate >= 80%.
- [ ] Audit trail كامل لكل قرار.
- [ ] Plugin يُثبت ويشتغل من Marketplace.

---

### المرحلة 6 — Polish & Launch (الأسبوع 37-42)

**الهدف:** إطلاق ناجح مع استعداد للتوسع.

**العمل:**
- **Performance Optimization**:
  - Startup time < 2s
  - Memory usage optimization
  - Bundle size reduction
- **Security Hardening**:
  - Penetration testing
  - Bug bounty program (داخلي)
- **Documentation**:
  - User guide (Arabic + English)
  - API documentation
  - Developer guide للـ plugins
- **Community Building**:
  - Discord server
  - GitHub organization
  - Blog + tutorials
- **إضافة**: Business Intelligence AI (تقارير ERP ذكية) - للمستقبل.

**DoD:**
- [ ] 100 beta users يستخدمون الـ IDE يومياً.
- [ ] Crash-free sessions >= 99.5%.
- [ ] NPS (Net Promoter Score) >= 50.

---

## 6.1 ربط المراحل مع Gates (تنفيذي)

```
Phase 0 ──► Gate A ──► Phase 1 ──► Gate B ──► Phase 2 ──► Gate C
  (3w)      (Lock)       (6w)       (Lock)       (5w)       (Lock)
                                              
Phase 3 ──► Gate D ──► Phase 4a ──► Gate D2 ──► Phase 4b ──► Gate E
  (5w)       (Lock)        (6w)        (Lock)        (6w)       (Lock)
                                                                   
Phase 5 ──► Gate F ──► Phase 6 (Launch)
  (5w)       (Lock)        (6w)
```

**قاعدة تنفيذ:**
- لا انتقال للمرحلة التالية بدون نجاح Gate المرحلة الحالية.
- أي Gate فشل → Corrective Action Plan خلال 48 ساعة.

---

## 7) عقود الربط الإلزامية (Contract v1)

### 7.1 مبدأ العقود
- كل endpoint يُعرّف input/output/errors/version.
- لا استدعاء مباشر بلا client contract.

### 7.2 مصفوفة العقود القانونية (Canonical Contract Matrix v1)

| المجال | Method | Path | Canonical | Owner | Status |
|---|---|---|---|---|---|
| Council | POST | `/api/v1/council/message` | ✅ | Backend | Stable |
| Council | GET | `/api/v1/council/status` | ✅ | Backend | Stable |
| Council | WS | `/api/v1/council/stream` | ✅ | Backend | New |
| Training | POST | `/api/v1/training/start` | ✅ | Backend | Stable |
| Training | GET | `/api/v1/training/status` | ✅ | Backend | Stable |
| Training | POST | `/api/v1/training/stop` | ✅ | Backend | Stable |
| Training | POST | `/api/v1/training/checkpoint` | ✅ | Backend | New |
| Sync | POST | `/api/v1/sync` | ✅ | Rust + Backend | Stable |
| Sync | GET | `/api/v1/sync/status` | ✅ | Rust + Backend | Stable |
| Sync | WS | `/api/v1/sync/ws` | ✅ | Rust + Backend | Stable |
| Workers | POST | `/api/v1/workers/register` | ✅ | Agent + Backend | Stable |
| Workers | POST | `/api/v1/workers/heartbeat` | ✅ | Agent + Backend | Stable |
| Workers | POST | `/api/v1/workers/apply-policy` | ✅ | Agent + Backend | Stable |
| Workers | GET | `/api/v1/workers/metrics` | ✅ | Agent + Backend | New |
| Updates | GET | `/api/v1/updates/manifest` | ✅ | Platform | Stable |
| Updates | POST | `/api/v1/updates/report` | ✅ | Platform | Stable |
| AI | POST | `/api/v1/ai/completion` | ✅ | AI Team | New |
| AI | POST | `/api/v1/ai/explain` | ✅ | AI Team | New |
| AI | POST | `/api/v1/ai/refactor` | ✅ | AI Team | New |

### 7.3 قواعد التوافق (Compatibility Rules)

- أي route قديم (legacy) يبقى عبر gateway translation لمدة **دورتين إصدار** (3 أشهر).
- بعد دورتين: إزالة legacy routes إلزامية.
- أي عميل جديد ممنوع يستهلك legacy routes.
- أي تغيير على المسارات canonical يحتاج ADR جديد + توقيع Owner المجال.

---

## 8) إدارة الموارد من الدسكتوب (Core Requirement)

### 8.1 نموذج سياسة الموارد (المحدث)

```json
{
  "device_id": "worker-123",
  "mode": "training-only",
  "limits": {
    "cpu_max_percent": 85,
    "cpu_throttle_threshold": 80,
    "ram_max_gb": 24,
    "ram_oom_protection": true,
    "gpu_mem_max_percent": 90,
    "gpu_thermal_limit_c": 85,
    "io_nice": "normal",
    "network_limit_mbps": 100
  },
  "schedule": {
    "timezone": "Asia/Baghdad",
    "windows": [
      {"start": "22:00", "end": "07:00"}
    ],
    "idle_only": true,
    "idle_detection_minutes": 5
  },
  "safety": {
    "thermal_cutoff_c": 85,
    "thermal_resume_c": 75,
    "auto_pause_on_user_activity": true,
    "kill_switch_enabled": true,
    "emergency_contact": "admin@bi-ide.com"
  },
  "federated_learning": {
    "enabled": true,
    "aggregation_rounds": 10,
    "privacy_budget": 1.0,
    "differential_privacy_epsilon": 0.1
  }
}
```

### 8.2 متطلبات التنفيذ
- policy تصدر من control plane.
- agent يستلم policy ويؤكد التفعيل عبر heartbeat.
- أي خرق limits ينتج event + auto-throttle + notification.
- **إضافة**: Federated Learning config يُرسل مع الـ policy.

### 8.3 معايير القبول
- تعديل policy من UI -> ينعكس على الجهاز بسرعة P95 < 10s.
- dashboard يعرض planned vs actual usage لكل جهاز بدقة >= 95%.
- Thermal throttling يعمل تلقائياً.

---

## 9) التحديثات التلقائية الشاملة

### 9.1 مبادئ النشر
- كل إصدار موقّع (Ed25519).
- قنوات: canary (5%) -> beta (25%) -> stable (100%).
- phased rollout حسب نسب نجاح وصحة.

### 9.2 خطوات rollout المُحسّنة
1. نشر manifest موقّع.
2. canary بنسبة 5% (الأجهزة التجريبية فقط).
3. مراقبة metrics لمدة 24 ساعة:
   - Crash rate < 1%
   - Error rate < 5%
   - Performance regression < 10%
4. التوسع إلى 25% (beta users).
5. مراقبة 48 ساعة إضافية.
6. التوسع إلى 100% (stable).
7. rollback تلقائي عند:
   - crash rate > 5%
   - error rate > 10%
   - user complaints > threshold

### 9.3 ما يجب إضافته
- updater plugin integration حقيقي (desktop + agent).
- update health contract موحد.
- release coordinator service.
- **A/B testing framework** للميزات الجديدة.

---

## 10) الأمن والحوكمة (Security & Governance Supreme)

### 10.1 أمن التشغيل
- mTLS أو TLS mutual trust داخل شبكة العمال.
- device keypair + rotation دوري (كل 6 أشهر).
- least privilege للعمليات المنفذة على العمال.
- **إضافة**: Hardware-backed keys (TPM/Secure Enclave).

### 10.2 حوكمة الذكاء الذاتي
- no direct production promote.
- policy tiers: safe / guarded / experimental.
- kill switch عالمي من control plane.
- **إضافة**: Human-in-the-loop للقرارات الحرجة.

### 10.3 السجلات والتدقيق
- audit trail إلزامي لكل:
  - policy change
  - model promotion
  - update rollout
  - remote execution
  - AI decision (مع reasoning)
- **Blockchain Artifacts**: تسجيل hash للـ models على blockchain خاص.

### 10.4 إدارة مفاتيح التوقيع
- Signing key يُحفظ في hardware-bound store:
  - macOS: Keychain
  - Windows: DPAPI
  - Linux: TPM أو encrypted keyring
- ممنوع تخزين signing key في repo أو `.env`.
- Key rotation كل 6 أشهر كحد أقصى.
- Owner: **Platform/DevOps**
- Backup Owner: **Security**

---

## 11) مؤشرات النجاح المرحلية (KPIs)

### Foundation KPIs (بعد المرحلة 2)
- Crash-free sessions >= 99.0%
- Startup P95 < 3.5s
- File open P95 < 180ms
- Worker enrollment success >= 95%
- Quick Open search P95 < 200ms

### Scale KPIs (بعد المرحلة 4)
- Crash-free sessions >= 99.5%
- Startup P95 < 2.5s
- Sync convergence LAN P95 < 2s
- Worker uptime >= 99%
- Update success rate >= 98%
- Policy apply latency P95 < 10s
- Code completion P95 < 400ms

### AI KPIs (بعد المرحلة 5)
- Council response P95 < 700ms (local/LAN path)
- Hallucination critical rate <= 2%
- Sandbox promotion pass rate tracked أسبوعياً
- Self-improvement cycle success rate >= 80%
- Local LLM inference < 400ms (token generation)

### Business KPIs (بعد الإطلاق)
- Monthly Active Users (MAU) >= 1,000 (Month 6)
- Daily Active Users (DAU) >= 300 (Month 6)
- Net Promoter Score (NPS) >= 50
- Churn Rate < 5% شهرياً

---

## 12) سجل المخاطر العميق (Deep Risk Register)

### المخاطر التقنية (Technical Risks)

| الخطر | التأثير | الاحتمال | التخفيف | Owner | Early Warning |
|---|---|---|---|---|---|
| فشل Monaco Editor في Tauri | قاتل | 30% | CodeMirror 6 fallback | Desktop | لا يمكن فتح ملفات > 1MB |
| تكاليف AI APIs تتصاعد | مالي عالي | 70% | Local LLM أولوية | AI | فاتورة > $1000/شهر |
| CRDT Sync معقد | تأخير 2-3 أشهر | 40% | OT fallback | Rust | Conflict rate > 5% |
| Cross-platform resource control | صعب | 50% | Docker wrapper | Agent | Tests تفشل على Windows |
| Local LLM بطيء | UX سيئ | 60% | Quantification + Hybrid | AI | Latency > 2s |
| PTY على Windows معقد | unstable | 40% | ConPTY API | Desktop | Terminal crashes |

### المخاطر التنظيمية (Organizational Risks)

| الخطر | التأثير | الاحتمال | التخفيف | Owner |
|---|---|---|---|---|
| مغادرة مطور Rust | قاتل | 30% | Documentation + Pair programming | Team Lead |
| Burnout فريق | تأخير | 50% | 40h/week max + إجازات | HR |
| نقص خبرة في AI | جودة منخفضة | 40% | Training + Consultants | AI Lead |

### المخاطر التجارية (Business Risks)

| الخطر | التأثير | الاحتمال | التخفيف | Owner |
|---|---|---|---|---|
| Microsoft تُطلق ميزة مشابهة | منافسة شديدة | 60% | Differentiation واضح | Product |
| JetBrains تحسن AI | منافسة | 50% | Speed of innovation | Product |
| Cursor يسبقنا | market share | 40% | Niche focus | Product |

### المخاطر القانونية/الأمنية (Legal/Security Risks)

| الخطر | التأثير | الاحتمال | التخفيط | Owner |
|---|---|---|---|---|
| تسريب signing key | حرج | 10% | Hardware-bound + rotation | Security |
| اختراق بيانات users | قاتل | 5% | Encryption + mTLS + audits | Security |
| دعوى براءات اختراع | مالي | 20% | Prior art research + Insurance | Legal |
| GDPR violation | غرامة | 15% | Privacy by design + DPO | Legal |

### 12.1 مصفوفة ملكية المخاطر (Risk Ownership Matrix)

| الخطر | Owner | Backup | Gate المراقبة | Trigger | الاستجابة الإلزامية | Contingency |
|---|---|---|---|---|---|---|
| تضخم نطاق التنفيذ | Platform | PMO | كل Gate | تأخر > 20% | تجميد الميزات + إعادة ترتيب | إطلاق MVP مبكر |
| mismatch عقود API | Backend | Rust | Gate A | فشل compatibility | rollback خلال 24h | GraphQL abstraction |
| فشل تحديثات جماعي | Platform/DevOps | Rust | Gate D | success < 98% | rollback فوري | Manual update |
| استنزاف موارد | Agent | Desktop | Gate C | drift > 10% | auto-throttle + pause | Kill switch |
| أخطاء sync | Rust | Backend | Gate D | conflict > 2% | replay-safe mode | OT fallback |
| regressions AI | AI/ML | Backend | Gate D2/E | hallucination > 2% | إرجاع model | Human review |
| فقدان بيانات تدريب | ML | Platform | Gate E | فشل resume | recovery من backup | Disable auto-promote |
| تسريب signing key | Security | Platform/DevOps | Gate D | أي دليل compromise | revoke فوري | Forced update |
| Monaco فشل | Desktop | Rust | Gate B | لا يعمل على جهاز | CodeMirror fallback | Ace Editor |
| Local LLM بطيء | AI | Backend | Gate D2 | latency > 2s | Hybrid mode | Cloud API |

---

## 13) خطة أول 21 يوم (تفصيل تنفيذي)

### اليوم 1-3: Contract Freeze
- اعتماد Contract v1.
- حسم endpoint mapping للـ sync/training/AI.
- **Exit**: 0 endpoint mismatch.

### اليوم 4-6: Clients موحدة
- تنفيذ clients موحدة.
- smoke e2e محلي (>= 95% success).
- **Exit**: CI أخضر.

### اليوم 7-9: Monaco MVP
- إدخال Monaco بشكل MVP.
- ربط فتح/حفظ ملفات فعلياً.
- **Exit**: فتح ملف P95 <= 300ms.

### اليوم 10-12: PTY Terminal
- PTY terminal integration.
- تحسين lifecycle للعمليات.
- **Exit**: 3 جلسات متزامنة بدون crash.

### اليوم 13-15: Git Panel
- Git panel MVP.
- ربط status/stage/commit.
- **Exit**: Git operations >= 95% success.

### اليوم 16-18: Quick Open + Command Palette
- fuzzy search (ripgrep).
- Command Palette.
- **Exit**: Cmd+P < 300ms.

### اليوم 19-21: Worker Policy APIs
- worker policy APIs.
- dashboard controls.
- updater integration أولي.
- **Exit**: canary update لـ 2 أجهزة.

---

## 14) بوابات الجودة (Quality Gates)

### Gate A (بعد المرحلة 0)
- جميع التعاقدات موثقة ومختبرة.
- ممنوع إضافة feature بدون contract.
- smoke e2e >= 95% success.

### Gate B (بعد المرحلة 1)
- Core IDE workflows تعمل بلا mock.
- Monaco يفتح ملفات >= 10k سطر.
- Terminal PTY يعمل بـ 3 جلسات.
- Performance budgets محققة.

### Gate C (بعد المرحلة 2)
- policy enforcement فعلي لكل جهاز مجرب.
- Federated Learning round ناجزة.
- Resource limits تُطبق خلال 10s.

### Gate D (بعد المرحلة 3)
- sync/update بثبات إنتاجي + rollback مؤكد.
- Live collaboration يعمل.
- Update success >= 98%.

### Gate D2 (بعد المرحلة 4a)
- Code completion يشتغل فعلياً + latency < 400ms.
- Explain/Refactor يشتغل end-to-end.
- Local LLM يعمل offline.

### Gate E (بعد المرحلة 4b)
- Council يحلل كود فعلي.
- Knowledge Graph يعمل.
- Context >= 10k tokens.

### Gate F (بعد المرحلة 5)
- self-improvement gated وآمن.
- Plugin system يعمل.
- Audit trail كامل.

---

## 15) سجل الأفكار من النسخ القديمة (Legacy Idea Registry)

> هذه أفكار كانت موجودة بالنسخ القديمة (D:/bi ide/) ولازم **ما تضيع**.
> سياسة: **نقل أفكار فقط — ممنوع نسخ كود قديم**.

| # | الفكرة | المصدر | المرحلة المقترحة | الحالة |
|---|--------|--------|------------------|--------|
| 1 | Neural Network + Multi-Head Attention | v6 super-intelligent-learning | 5 | Planned |
| 2 | Double DQN + Prioritized Experience Replay | v6 | 5 | Planned |
| 3 | Meta-Learning (تعلم كيف يتعلم) | v6 | 5 | Planned |
| 4 | Curriculum Learning (صعوبة تدريجية 1→10) | v6 | 2 | Planned |
| 5 | TF-IDF + N-gram لفهم الكود | v5/v6 | 4a | Planned |
| 6 | Auto-Learning من 95 PDF عربي/إنجليزي | v5 | 2 | Planned |
| 7 | Arabic NLP مخصص (تشكيل + جذور + stopwords) | v4/v5 | 4b | Planned |
| 8 | 16 حكيم في المجلس الأعلى | v7/v8 hierarchy | 4b | Planned |
| 9 | Shadow Team — 4 متشائمين يحللون المخاطر | v7/v8 | 4b | Planned |
| 10 | Light Team — 4 متفائلين يقترحون فرص | v7/v8 | 4b | Planned |
| 11 | الكشافة (تقني + سوق + منافسين + فرص) | v7/v8 | 5 | Planned |
| 12 | Federated Learning (تدريب موزع + خصوصية) | جديد | 2 | Planned |
| 13 | Vector DB (HNSW) — ذاكرة ذكية دلالية | جديد | 4b | Planned |
| 14 | Predictive Error Detection (توقع الأخطاء) | جديد | 4a | Planned |
| 15 | Plugin System + Marketplace + SDK | جديد | 5 | Planned |
| 16 | Project Factory (idea→spec→code→test→deploy) | v6 | 5 | Planned |
| 17 | Self-Repair + Auto-Rollback عند drift | v6 | 5 | Planned |
| 18 | Elastic Weight Consolidation (منع النسيان) | جديد | 2 | Planned |
| 19 | Sandbox Execution لتشغيل كود آمن | جديد | 4a | Planned |
| 20 | Business Intelligence AI (تقارير ERP ذكية) | جديد | 6 | Planned |
| 21 | Cost-aware H200/GPU scheduler (IDEA-008) | v6 IDEA_PARITY | 2 | Planned |
| 22 | Real-time artifact streaming كل دقيقة (IDEA-010) | v6 IDEA_PARITY | 3 | Planned |
| 23 | Multi-agent specialist chain: planner→coder→tester→deployer (IDEA-007) | v6 IDEA_PARITY | 5 | Planned |
| 24 | 4-level hierarchical memory (IDEA-006) | v6 IDEA_PARITY | 4b | Planned |
| 25 | Emergency override governance (IDEA-011) | v6 IDEA_PARITY | 3 | Planned |
| 26 | Performance profiling داخل IDE | IDE_IDEAS_MASTER | 1 | Planned |
| 27 | Feature flags لتفعيل/تعطيل ميزات | IDE_IDEAS_MASTER | 0 | Planned |
| 28 | Live collaboration editing | IDE_IDEAS_MASTER | 3 | Planned |
| 29 | **Swarm Intelligence** (Pheromone, Self-org) | جديد | 2 | Planned |
| 30 | **Edge AI / TinyML** (Mobile, IoT) | جديد | 2 | Planned |
| 31 | **Quantum-Resistant Crypto** (Kyber, Dilithium) | جديد | 3 | Planned |
| 32 | **Homomorphic Encryption** (Compute on encrypted) | جديد | 4 | Planned |
| 33 | **Self-Healing System** (Auto-remediation) | جديد | 3 | Planned |
| 34 | **Neuromorphic Interface** (Future) | جديد | 6 | Planned |

> **ملاحظة:** النسخة القديمة v6 كانت تحتوي Monaco + xterm + node-pty — نعيدها بشكل أفضل في v8.

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
├── ai/
│   ├── AIChatPanel.tsx       ← [NEW] (مرحلة 4b)
│   ├── ExplainPanel.tsx      ← [NEW] (مرحلة 4a)
│   ├── RefactorPanel.tsx     ← [NEW] (مرحلة 4a)
│   └── ModelSelector.tsx     ← [NEW] (مرحلة 4a)
├── workers/
│   ├── WorkerDashboard.tsx   ← [NEW] (مرحلة 2)
│   └── WorkerPolicy.tsx      ← [NEW] تحكم موارد (مرحلة 2)
├── collaboration/
│   ├── LiveCursor.tsx        ← [NEW] (مرحلة 3)
│   └── PresencePanel.tsx     ← [NEW] (مرحلة 3)
├── settings/
│   ├── SettingsUI.tsx        ← [NEW] (مرحلة 1)
│   └── KeyboardShortcuts.tsx ← [NEW] (مرحلة 1)
└── themes/                   ← [NEW] dark/light/custom (مرحلة 1)

apps/desktop-tauri/src-tauri/src/commands/
├── ai.rs                     ← [NEW] local inference (مرحلة 4a)
├── search.rs                 ← [NEW] ripgrep search (مرحلة 1)
├── fs.rs                     ← [IMPROVE] more ops
├── git.rs                    ← [IMPROVE] graph/diff
├── terminal.rs               ← [IMPROVE] real PTY
├── sync.rs                   ← [IMPROVE] CRDT
└── policy.rs                 ← [NEW] resource enforcement (مرحلة 2)

services/
├── training_scheduler.py     ← [NEW] cost-aware (مرحلة 2)
├── model_registry.py         ← [NEW] versions (مرحلة 2)
├── update_coordinator.py     ← [NEW] signed rollout (مرحلة 3)
└── federated_aggregator.py   ← [NEW] FedAvg (مرحلة 2)

ai/learning/
├── continuous_learning.py    ← [NEW] EWC (مرحلة 2)
├── federated.py             ← [NEW] FedAvg (مرحلة 2)
├── meta_learner.py          ← [NEW] (مرحلة 5)
├── curriculum.py            ← [NEW] (مرحلة 2)
└── swarm_intelligence.py    ← [NEW] (مرحلة 2)
```

---

## 17) قرار التنفيذ الملزم (Final Mandate)

من هذه اللحظة:

1. **لا توسع ميزات** قبل إكمال Contract Freeze.
2. **لا ادعاء readiness** مع وجود mock في المسارات الحرجة.
3. **الأولوية القصوى** لمتطلباتك الأساسية:
   - ربط كل الطبقات فعلياً.
   - تحكم موارد من داخل الدسكتوب.
   - تحديث تلقائي موثوق لكل الأجهزة.
4. **كل فكرة** من النسخ القديمة تمر عبر Idea Registry — لا نسخ كود، فقط أفكار.
5. **كل مرحلة** تمر عبر Quality Gate قبل الانتقال للتالية.
6. **كل risk** له Owner وBackup وContingency Plan.
7. **كل code** يجتاز Tests + Review + Benchmarks.
8. **الاستدامة أولاً**: 40 ساعة/أسبوع كحد أقصى، إجازات إجبارية.

هذا التقرير هو المرجع المعتمد لإعادة هيكلة Desktop v8 بطريقة **حقيقية، قابلة للقياس، قابلة للتوسع العالمي، ومحمية من المخاطر**.

**المدة النهائية: 180 يوم (26 أسبوع)**
**عدد الأفكار المحفوظة: 34 فكرة (الـ 28 الأصلية + 6 جديدة)**
**عدد المخاطر المُدارة: 10 مخاطر رئيسية + Contingency Plans**

---

## 18) Ownership Matrix

| المسار | المالك الأساسي | مالك ثانوي | شرط التسليم |
|---|---|---|---|
| Desktop UX/Editor/Git | Desktop Team | Rust Team | اجتياز Gate B |
| Rust Commands/PTY/Sync Client | Rust Team | Desktop Team | اجتياز Gate B/D |
| API Contracts/Gateway | Backend Team | Platform Team | اجتياز Gate A |
| Worker Agent/Policy Enforcement | Agent Team | Platform Team | اجتياز Gate C |
| Updates/Release Coordinator | Platform Team | DevOps | اجتياز Gate D |
| AI Inference/Council | AI Team | Backend Team | اجتياز Gate E |
| Training/Evaluation/Registry | ML Team | Backend Team | اجتياز Gate E |
| Security/Crypto | Security Team | Platform Team | اجتياز كل Gates |

---

## 19) Dependency Graph

```
Phase 0 ──► Gate A ──► Phase 1 ──► Gate B ──► Phase 2 ──► Gate C
  (3w)      (Lock)       (6w)       (Lock)       (5w)       (Lock)
                                              
Phase 3 ──► Gate D ──► Phase 4a ──► Gate D2 ──► Phase 4b ──► Gate E
  (5w)       (Lock)        (6w)        (Lock)        (6w)       (Lock)
                                                                   
Phase 5 ──► Gate F ──► Phase 6 (Launch)
  (5w)       (Lock)        (6w)
```

**Dependency Rules:**
- ممنوع بدء Phase أعلى إذا Gate المرحلة السابقة فشل.
- أي استثناء يحتاج ADR قصير مع تحليل أثر وموافقة مالك المنصة.

---

**تم إعداد هذا الملف بواسطة: BI-IDE Supreme Planning Team**
**التاريخ: 2026-03-02**
**الإصدار: 4.0 Supreme**
