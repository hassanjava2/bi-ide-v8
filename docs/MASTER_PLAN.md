# 📋 الخطة الشاملة النهائية — BI-IDE v8

> **المرجع:** [VISION_MASTER.md](file:///Users/bi/Documents/bi-ide-v8/docs/VISION_MASTER.md)
> **آخر تحديث:** 2026-03-03
> **الحالة:** هذا الملف يدمج كل الخطط السابقة بملف واحد

---

# القسم ١ — البيانات والداتا

## 1.1 البيانات المختفية على RTX 5090
| البيانات | الحجم | المسار | مُستخدم؟ |
|----------|-------|--------|----------|
| Training Data | **45GB** | `/home/bi/training_data/` | ❌ |
| Hierarchy Checkpoints | 4.2GB | `training_data/data/checkpoints/` | ❌ |
| Infinite Learning | 159MB | `training_data/data/infinite-learning/` | ❌ |
| LoRA Checkpoint-39 | 82MB | `training_data/models/finetuned/` | ❌ |
| Vocab | - | `training_data/learning_data/vocab.pkl` | ❌ |
| Training Backup | 25GB | `/home/bi/training_backup.tar.gz` | ❌ |
| Old Models | 1.7GB | `/home/bi/Downloads/bi-ide-v8/models/` | ❌ |

## 1.2 البيانات بالمشروع
| البيانات | المسار | مُستخدم؟ |
|----------|--------|----------|
| SQLite DB | `data/bi_ide.db` (422KB) | ⚠️ صغيرة |
| Knowledge Base | `models/knowledge-base.json` | ✅ |
| Learning State | `models/advanced-learning-state.json` | ✅ |
| Council History | `data/council_chat_history.json` | ✅ |
| Vocab | `learning_data/vocab.pkl` | ❌ |
| DB Schema | `database/schema.sql` | ✅ لكن SQLite |
| DB Models ORM | `database/models.py` | ✅ |
| DB Connection | `database/connection.py` | ✅ |

## 1.3 قواعد البيانات — المطلوب
- [ ] PostgreSQL للإنتاج + SQLite للتطوير
- [ ] جداول: طبقة، شجرة، هرم، شخص، فكرة، مصنع، عينة تدريب
- [ ] تاريخ كامل لكل قرار ونقاش وتدريب وفكرة
- [ ] **لا شي يضيع — كل شي يُخزّن**
- [ ] Fake databases بالـ routers (training + monitoring) → استبدال بـ PostgreSQL

---

# القسم ٢ — التدريب التلقائي

## 2.1 أنظمة التدريب الموجودة (4 أنظمة — كلها معطلة!)
| النظام | الملف | `start_all()` | مُستدعى؟ |
|--------|-------|---------------|----------|
| `RealTrainingSystem` | `hierarchy/real_training_system.py` | ✅ | ❌ |
| `InternetTrainingSystem` | `hierarchy/internet_auto_training.py` | ✅ | ❌ |
| `MassiveTrainingSystem` | `hierarchy/massive_training.py` | ✅ | ❌ |
| `AutoLearningSystem` | `hierarchy/auto_learning_system.py` | ✅ | ❌ |

## 2.2 المطلوب — تشغيل التدريب
- [ ] استدعاء `start_all()` عند بدء `rtx_api_server.py`
- [ ] تحميل checkpoints (4.2GB) + vocab عند البدء
- [ ] `InternetDataFetcher` → كل طبقة تجلب عيناتها أوتوماتيكياً كل ساعة
- [ ] استخدام الـ 45GB بدل البيانات الصناعية
- [ ] VPS `.env`: `TRAINING_RELAY_UPSTREAM_URL=http://192.168.1.164:8090`

## 2.3 العتاد الموزّع — 100+ حاسبة
> من `DISTRIBUTED_HIERARCHICAL_TRAINING_PLAN.md` + `VISION_MASTER.md`

**القواعد:**
- أي حاسبة تنصّب BI-IDE ← 100% GPU أوتوماتيكي (default)
- الداتا كلها تنتقل مباشر للـ RTX 5090 (المخزن المركزي)
- **أمر واحد** يبدي التدريب: `start_h200_worker.ps1` أو `bi-ide --train --gpu-all`
- **⚠️ بيانات التدريب لا تتكرر!** — hash/ID + deduplication مركزي

**البنية الموجودة:**
```
RTX 5090 (المركز) ← checkpoints + data + merge
    ↑
    ├── POST /api/v1/network/training/enqueue ← إضافة مهام
    ├── POST /api/v1/network/training/claim   ← worker يأخذ مهمة
    ├── POST /api/v1/network/training/complete ← worker يسلّم
    ├── continuous_training_orchestrator.py    ← يغذي queue دائماً
    └── start_h200_worker.ps1                ← أمر واحد للسيرفر
```

**المطلوب:**
- [ ] Worker auto-enrollment: أي حاسبة → تسجّل نفسها → 100%
- [ ] Batch deduplication مركزي: كل عينة بـ hash/ID
- [ ] Checkpoint merge: gradient updates من كل worker → RTX يدمجهم
- [ ] Scale: 100+ worker بدون تكرار
- [ ] Cost-aware scheduler: لا تحرق ميزانية (IDEA-008)
- [ ] Real-time artifact streaming: لا يضيع تقدم (IDEA-010)
- [ ] Sharded resilient training: تدريب موزّع مقاوم للسقوط (IDEA-009)

---

# القسم ٣ — المجلس والهرم

## 3.1 مجلس الحكماء — المشاكل
- ❌ Mock consensus = 0.75 hardcoded (قرارات مو حقيقية!)
- ❌ لا يتناقش تلقائياً — ينتظر سؤال الرئيس
- ❌ لا حلقة اجتماعات أوتوماتيكية

## 3.2 المطلوب
- [ ] **أوامر الرئيس = تنفيذ فوري** عبر كل الطبقات
- [ ] **مناقشات أوتوماتيكية** كل 30 دقيقة بمواضيع حقيقية
- [ ] استبدال consensus الوهمي بتصويت حقيقي
- [ ] ربط قرارات المجلس → التنفيذ → النتائج ترجع
- [ ] 24/7 autonomous council loop (IDEA-001)
- [ ] Dual shadow-light evaluation (IDEA-002)

## 3.3 شجرة الاختصاصات — المشكلة
**الموجود:** 11 مجال ثابتة بقائمة مسطحة
**المطلوب:**
```
الطب ← شجرة
  ├─ جراحة ← هرم (3-5 أشخاص يفكرون بطرق مختلفة)
  │    ├─ جراحة قلب   │   ├─ جراحة عصبية   │   └─ جراحة عظام
  ├─ طب داخلي ← هرم
  └─ صيدلة ← هرم ... وهكذا لكل اختصاص بالعالم
```
- [ ] تحويل `domain_experts.py` → شجرة ديناميكية
- [ ] اكتشاف تلقائي لتخصصات جديدة من Wikipedia/أبحاث
- [ ] كل تخصص دقيق = عدة أشخاص بأساليب تفكير مختلفة

## 3.4 الكشافة — التطوير
- [ ] كل كشاف = شجرة بأفرع متخصصة قابلة للتوسعة أوتوماتيكياً
- [ ] اكتشاف مستمر + daily findings (IDEA-006)
- [ ] ربط الكشافة ← التدريب مباشر

---

# القسم ٤ — الطبقات الجديدة

## 4.1 طبقة الحياة الواقعية (أسفل طبقة) ⭐
> **غير موجودة — تحتاج إنشاء**

- [ ] إنشاء `hierarchy/real_life_layer.py`
- [ ] كل شخص = agent باختصاص دقيق يتدرب أوتوماتيكياً
- [ ] كل شخص يفكر وينتج أفكار ويطبقها
- [ ] عدة أشخاص لكل اختصاص — كلمن يفكر بطريقة مختلفة
- [ ] معامل ومصانع افتراضية (إعادة بناء الحضارة)
- [ ] طبقة ربط عُليا ← تستخلص الأفكار المُجمّعة

## 4.2 ذاتي التطوير
- [ ] `DynamicLayerGenerator` ← ينتج طبقات أوتوماتيكياً (موجود بـ `meta_architect.py` لكن لا يُستدعى)
- [ ] كل طبقة جديدة تُربط بالشجرة تلقائياً
- [ ] `brain/evaluator.py` ← تقييم أداء كل طبقة (موجود لكن غير متصل)
- [ ] `brain/scheduler.py` ← جدولة التطوير الذاتي (موجود لكن غير متصل)
- [ ] Autonomous self-repair loops (IDEA-015)
- [ ] Self-improvement gated: propose → sandbox → evaluate → promote
- [ ] Kill switch + audit trail

## 4.3 المشروع الموازي
- [ ] مشروع يتابع الأول بطريقة مغايرة
- [ ] Dual-path: current-evolver + zero-reinventor (من خطة التدريب الموزع)
- [ ] مقارنة دورية + Consensus Layer لدمج المخرجات

---

# القسم ٥ — مشاكل حرجة بالكود

## 5.1 🔴 حرجة (فوري)
| # | المشكلة | الملف | الخطر |
|---|---------|-------|-------|
| 1 | SSL معطّل | `hierarchy/auto_learning_system.py` | أمني |
| 2 | Duplicate IDs (S002) | `hierarchy/high_council.py` | تكرار |
| 3 | Mock consensus 0.75 | `hierarchy/__init__.py` | قرارات وهمية |
| 4 | Mock AI services | `services/ai_service.py` | generate/complete/explain/review كلها fake |
| 5 | No PostgreSQL | كل الـ routers | بيانات بالذاكرة تضيع بالريستارت |
| 6 | Password reset وهمي | `api/routes/users.py` | لا يرسل إيميل |
| 7 | Fake databases | `api/routers/training.py` + `monitoring.py` | بيانات وهمية |

## 5.2 🟡 مهمة (قبل الإنتاج)
| # | المشكلة | الملف |
|---|---------|-------|
| 8 | RTX config غير متسق | عدة ملفات (`192.168.68.125` vs `192.168.1.164`) |
| 9 | No data pipeline | لا تنظيف/تحقق بيانات |
| 10 | Missing metrics | fallback_rate + median_latency غير كاملة |
| 11 | No idle training | workers لا يتدربون أثناء الخمول |

---

# القسم ٦ — كود موجود يحتاج تفعيل (لا تبدي من الصفر!)

> ⚠️ **15,448 سطر كود موجود عبر 33+ ملف — يحتاج تفعيل وتحسين لا إنشاء من جديد**

## 6.1 🧠 Brain — الدماغ (موجود 840 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `brain/bi_brain.py` | 241 | الدماغ الرئيسي | **تحسين** — ربطه بالطبقات |
| `brain/evaluator.py` | 301 | تقييم النماذج | **تحسين** — تفعيل التقييم الدوري |
| `brain/scheduler.py` | 298 | جدولة مهام | **تحسين** — تفعيل الجدولة التلقائية |

## 6.2 🎓 AI Training — تدريب (موجود 2,910 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ai/training/continuous_trainer.py` | 734 | تدريب مستمر | **تفعيل** على RTX |
| `ai/training/multi_gpu_trainer.py` | 607 | تدريب متعدد GPUs | **تفعيل** للـ H200 |
| `ai/training/rtx4090_trainer.py` | 460 | تدريب RTX مخصص | **تفعيل** وربط |
| `ai/training/auto_evaluation.py` | 547 | تقييم أوتوماتيكي | **تفعيل** |
| `ai/training/data_collection.py` | 562 | جمع بيانات | **تفعيل** ← الإنترنت |

## 6.3 🧪 AI Memory + Tokenizer + Optimization (موجود 1,882 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ai/memory/vector_db.py` | 608 | ذاكرة vector | **تحسين** — 4 مستويات |
| `ai/memory/conversation_history.py` | 590 | تاريخ المحادثات | **تفعيل** |
| `ai/tokenizer/arabic_processor.py` | 175 | **معالج عربي!** | **تحسين** — ربط بالتدريب |
| `ai/optimization/quantization.py` | 509 | تقليل حجم النموذج | **تفعيل** |

## 6.4 🔧 Data Pipeline (موجود 718 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `data/pipeline/data_cleaner.py` | 330 | تنظيف بيانات | **تفعيل** |
| `data/pipeline/data_validator.py` | 388 | تحقق بيانات | **تفعيل** |

## 6.5 👥 Community (موجود 1,417 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `community/forums.py` | 464 | منتديات | **تفعيل** وربط بالـ UI |
| `community/code_sharing.py` | 489 | مشاركة كود | **تفعيل** |
| `community/knowledge_base.py` | 464 | قاعدة معرفة | **تفعيل** |

## 6.6 🔒 Security (موجود 1,108 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `security/ddos_protection.py` | 636 | حماية DDoS | **تفعيل** |
| `security/encryption.py` | 472 | تشفير | **تفعيل** |

## 6.7 📊 Monitoring (موجود 1,170+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `monitoring/alert_manager.py` | 599 | تنبيهات | **تفعيل** |
| `monitoring/metrics_exporter.py` | 571 | مقاييس | **تفعيل** + Prometheus |
| `monitoring/grafana/` | — | لوحات عرض | **تفعيل** |
| `monitoring/elk/` | — | سجلات | **تفعيل** |

## 6.8 🌐 Network (موجود 752+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `network/auto_reconnect.py` | 752 | إعادة اتصال تلقائي | **تفعيل** |
| `network/firewall_manager.py` | — | جدار ناري | **تفعيل** |

## 6.9 ⚙️ Services (موجود 2,406 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `services/training_service.py` | 285 | خدمة تدريب | **تحسين** — ربط بـ RTX |
| `services/council_service.py` | 501 | خدمة المجلس | **تحسين** — إزالة mocks |
| `services/ai_service.py` | 525 | خدمة AI | **تحسين** — إزالة mocks |
| `services/sync_service.py` | 556 | خدمة مزامنة | **تحسين** |
| `services/backup_service.py` | 539 | خدمة نسخ احتياطي | **تفعيل** |

## 6.10 🏗️ Hierarchy extras (موجود 1,018 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `hierarchy/checkpoint_loader.py` | 233 | تحميل checkpoints | **تفعيل** — ربط بالـ 4.2GB |
| `hierarchy/specialized_ai_network.py` | 414 | شبكة AI متخصصة | **تفعيل** |
| `hierarchy/autonomous_learning.py` | 250 | تعلم ذاتي | **تفعيل** |
| `hierarchy/connect_services.py` | 121 | ربط خدمات | **تفعيل** |

## 6.11 🔄 Worker + Orchestrator (موجود 1,187 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `worker/bi_worker.py` | 711 | عامل تدريب | **تحسين** — idle training |
| `orchestrator_api.py` | 476 | توزيع مهام | **تحسين** — deduplication |

---

# القسم ٧ — أفكار الأولوية (من IDEA_PARITY_TOP15)

| # | الفكرة | الهدف | الحالة |
|---|--------|-------|--------|
| 1 | Cost-aware H200 scheduler | تقليل حرق ميزانية السيرفرات | ❌ |
| 2 | Real-time artifact streaming | لا يضيع تقدم تدريب عند انقطاع | ❌ |
| 3 | No-idea-loss registry | كل فكرة لها owner + task + trace | ❌ |
| 4 | Autonomous self-repair | إصلاح ذاتي سريع | ❌ |
| 5 | Project factory pipeline | فكرة → مشروع قابل للتشغيل | ❌ |
| 6 | Scout persistent discovery | اكتشاف مستمر + daily findings | ⚠️ جزئي |
| 7 | Multi-agent specialist chain | planner→researcher→coder→tester→reviewer→deployer | ❌ |
| 8 | 24/7 autonomous council loop | المجلس يشتغل بدون أوامر | ❌ |
| 9 | Desktop + web dual interface | واجهتين | ❌ |
| 10 | Sharded resilient training | تدريب مقاوم للسقوط | ❌ |
| 11 | Hierarchical memory 4 levels | ذاكرة 4 مستويات | ❌ |
| 12 | Emergency override governance | حوكمة طوارئ | ❌ |
| 13 | Security-first zero-trust | أمان بلا ثقة | ❌ |
| 14 | Language-agnostic mesh | شبكة خدمات متعددة اللغات | ❌ |

---

# القسم ٨ — IDE والديسكتوب

## 8.1 ✅ مكتمل
- بناء Tauri v2 (0 أخطاء TypeScript + Rust)
- AI Chat عبر invoke (إصلاح TypeError)
- Command Palette (25+ أمر)
- Training Dashboard (بيانات GPU حقيقية)
- Sync Panel + Auth + GPU Metrics

## 8.2 ❌ مطلوب
- [ ] Monaco editor حقيقي (فتح/تعديل/حفظ ملفات)
- [ ] PTY terminal integration
- [ ] Git integration حقيقي (status/diff/commit/push/pull)
- [ ] Code completion (Monaco inline) — P95 < 400ms
- [ ] Explain/Refactor/Fix actions
- [ ] Multi-language depth
- [ ] Live collaboration
- [ ] Quick Open (Cmd+P)

---

# القسم ٩ — الأمان والعمليات

- [ ] SSL إصلاح (تفعيل verification)
- [ ] Signed updates + rollback
- [ ] Zero-trust security gates
- [ ] DDoS protection (موجود غير مفعّل)
- [ ] Encryption (موجود غير مفعّل)
- [ ] Incident response plan
- [ ] Disaster recovery rehearsal
- [ ] Deploy systemd services (brain, training, etc.)
- [ ] Monitoring: Prometheus + Grafana + ELK (موجود غير مفعّل)

---

# القسم ١٠ — ترتيب التنفيذ

## المرحلة A — التدريب التلقائي (الأهم)
1. ربط أنظمة التدريب الـ 4 (`start_all()`)
2. تحميل checkpoints + vocab عند البدء
3. `InternetDataFetcher` ← تدريب لا نهائي
4. VPS→RTX relay
5. Batch deduplication (hash/ID لكل عينة)
6. Worker auto-enrollment (أي جهاز → 100% فوراً)
7. أمر واحد يشغّل كل الموارد

## المرحلة B — المجلس والأوامر
8. أوامر الرئيس → تنفيذ فوري
9. حلقة اجتماعات أوتوماتيكية كل 30 دقيقة
10. استبدال mock consensus بتصويت حقيقي

## المرحلة C — الشجرة والهرم
11. تحويل domain_experts → شجرة ديناميكية
12. اكتشاف تلقائي لتخصصات جديدة
13. كشافة بشجرات وأهرام

## المرحلة D — طبقة الحياة الواقعية
14. إنشاء real_life_layer.py
15. agents باختصاصات دقيقة
16. معامل ومصانع افتراضية

## المرحلة E — ذاتي التطوير + DB + أمان
17. DynamicLayerGenerator أوتوماتيكي
18. PostgreSQL + حفظ كل شيء
19. brain/evaluator + scheduler
20. إصلاح المشاكل الحرجة (SSL, mocks, fakes)
21. تفعيل Monitoring + Security

## المرحلة F — IDE والديسكتوب
22. Monaco editor + PTY + Git حقيقي
23. Code completion + AI actions
24. Signed updates + rollback

## المرحلة G — المشروع الموازي + المجتمع
25. إنشاء مشروع موازي (dual-path)
26. تفعيل community/ (forums, code sharing)
27. Project factory pipeline
28. No-idea-loss registry

---

# القسم ١١ — مقاييس النجاح (KPIs)

| المقياس | الهدف | الحالي |
|---------|-------|--------|
| Training systems active | 4/4 | 0/4 |
| تدريب ذاتي بلا تدخل | 24/7 | ❌ |
| Workers متصلين | 100+ | 2 |
| Batch deduplication | 0% تكرار | ❌ غير موجود |
| Council autonomous | 24/7 | ❌ ينتظر أوامر |
| Mock endpoints | 0 | 5+ |
| Data persistence | PostgreSQL | SQLite/Memory |
| Crash-free sessions | ≥99% | ⚠️ |
| Code completion P95 | <400ms | ❌ غير موجود |
| GPU utilization | 100% | 28% (idle) |
