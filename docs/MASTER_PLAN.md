# 📋 الخطة الشاملة — BI-IDE v8 Master Execution Plan

> **المرجع:** [VISION_MASTER.md](file:///Users/bi/Documents/bi-ide-v8/docs/VISION_MASTER.md)
> **تاريخ:** 2026-03-03

---

## 🗄️ البيانات المكتشفة (المختفية)

### على RTX 5090 (192.168.1.164)
| البيانات | الحجم | المسار | الحالة |
|----------|-------|--------|--------|
| Training Data الكلي | **45GB** | `/home/bi/training_data/` | ❌ غير مُستخدم |
| Hierarchy Checkpoints | 4.2GB | `/home/bi/training_data/data/checkpoints/` | ❌ لا يُحمّل |
| Infinite Learning | 159MB | `/home/bi/training_data/data/infinite-learning/` | ❌ غير متصل |
| LoRA Finetuned (checkpoint-39) | 82MB | `/home/bi/training_data/models/finetuned/` | ❌ لا يُحمّل |
| Vocab | - | `/home/bi/training_data/learning_data/vocab.pkl` | ❌ لا يُستخدم |
| Training Backup | 25GB | `/home/bi/training_backup.tar.gz` | ❌ غير مفتوح |
| Old Project Models | 1.7GB | `/home/bi/Downloads/bi-ide-v8/models/` | ❌ مُهمل |

### في المشروع (محلي + VPS)
| البيانات | المسار | الحالة |
|----------|--------|--------|
| SQLite DB | `data/bi_ide.db` (422KB) | ⚠️ صغيرة |
| Knowledge Base | `models/knowledge-base.json` | ✅ موجود |
| Advanced Learning State | `models/advanced-learning-state.json` | ✅ موجود |
| Council Chat History | `data/council_chat_history.json` | ✅ موجود |
| Learning Data + Vocab | `learning_data/checkpoints/`, `vocab.pkl` | ✅ موجود |
| Database Schema SQL | `database/schema.sql` | ✅ موجود |
| Database Models ORM | `database/models.py` | ✅ موجود |
| Database Connection | `database/connection.py` | ✅ موجود |

---

## 🔴 النقص الحرج (مرتب حسب الأولوية)

### 1. التدريب التلقائي معطّل
**المشكلة:** 4 أنظمة تدريب موجودة لكن **لا تُستدعى أبداً**
| النظام | الملف | `start_all()` | مُستدعى؟ |
|--------|-------|---------------|----------|
| `RealTrainingSystem` | `hierarchy/real_training_system.py` | ✅ | ❌ |
| `InternetTrainingSystem` | `hierarchy/internet_auto_training.py` | ✅ | ❌ |
| `MassiveTrainingSystem` | `hierarchy/massive_training.py` | ✅ | ❌ |
| `AutoLearningSystem` | `hierarchy/auto_learning_system.py` | ✅ | ❌ |

**المطلوب:**
- [ ] `rtx_api_server.py`: استدعاء `start_all()` عند بدء السيرفر
- [ ] تحميل الـ checkpoints (4.2GB) عند البدء
- [ ] ربط `InternetDataFetcher` ← كل طبقة تجلب عيناتها أوتوماتيكياً
- [ ] توجيه أنظمة التدريب للـ 45GB بدل البيانات الصناعية

### 2. طبقة الإضافة الأوتوماتيكية (Internet Auto-Training) معطلة
**المشكلة:** `InternetDataFetcher` يجلب من 5 مصادر لكن لا يُشغّل
- أخبار (HackerNews) ← ✅ API يعمل
- تقنيات (GitHub Trending) ← ✅ API يعمل
- سوق (Yahoo Finance) ← ⚠️ يحتاج مفتاح
- أمن (CVE/NVD) ← ⚠️ قد يحتاج مفتاح
- أبحاث (arXiv) ← ✅ مفتوح

**المطلوب:**
- [ ] تشغيل `InternetTrainingSystem.start_all()` ← تدريب مستمر لا نهائي
- [ ] كل طبقة تجلب بياناتها من الإنترنت كل ساعة
- [ ] العينات تتراكم أوتوماتيكياً → تدريب أوتوماتيكي → تطوير ذاتي

### 3. مجلس الحكماء — كواجهة أوامر الرئيس
**الموجود:** `high_council.py` (16 حكيم) + `AIHierarchy.ask()` + `execute_command()`
**المشكلة:**
- المجلس **لا يتناقش تلقائياً** بينهم — ينتظر سؤال الرئيس
- لا يوجد **حلقة اجتماعات أوتوماتيكية** حقيقية
- الأوامر لا تنفّذ مباشرة عبر كل الطبقات

**المطلوب:**
- [ ] **أوامر الرئيس → تنفيذ فوري**: لما الرئيس يقول "سووا هيج" → كل الطبقات تنفذ
- [ ] **مناقشات أوتوماتيكية**: المجلس يجتمع دورياً ويتناقش بدون أوامر
- [ ] **ربط قرارات المجلس → فريق التنفيذ → النتائج ترجع للمجلس**
- [ ] `_council_meeting_loop` → يعمل كل 30 دقيقة بمواضيع حقيقية

### 4. بنية الشجرة + الهرم (Domain Experts)
**الموجود:** 11 مجال ثابتة في قائمة مسطحة
**المطلوب:**
```
الطب ← شجرة
  ├─ جراحة ← هرم
  │    ├─ جراحة قلب ← هرم فرعي (3-5 أشخاص يفكرون بطرق مختلفة)
  │    ├─ جراحة عصبية
  │    └─ جراحة عظام
  ├─ طب داخلي ← هرم
  │    ├─ أمراض قلب
  │    └─ أمراض صدرية
  └─ صيدلة ← هرم

الهندسة ← شجرة
  ├─ مدنية ← هرم
  │    ├─ إنشائية (3 أشخاص × أساليب تفكير مختلفة)
  │    ├─ طرق ← هرم فرعي
  ... وهكذا لكل اختصاص بالعالم
```

**المطلوب:**
- [ ] تحويل `domain_experts.py` من قائمة → شجرة ديناميكية
- [ ] اكتشاف تلقائي لتخصصات جديدة من Wikipedia / الأبحاث
- [ ] كل تخصص دقيق = عدة "أشخاص" بأساليب تفكير مختلفة
- [ ] إنشاء معامل ومصانع افتراضية لكل اختصاص

### 5. ذاتي التطوير
**الموجود:** `meta_architect.py` (25KB) — `DynamicLayerGenerator`
**المشكلة:** الـ `DynamicLayerGenerator` لا يُستدعى تلقائياً

**المطلوب:**
- [ ] `DynamicLayerGenerator` → ينتج طبقات جديدة أوتوماتيكياً
- [ ] كل طبقة جديدة تُربط بالشجرة تلقائياً
- [ ] الطبقات تقيّم نفسها وتحسّن نفسها
- [ ] `brain/evaluator.py` → يقيّم أداء كل طبقة
- [ ] `brain/scheduler.py` → يجدول مهام التطوير الذاتي

### 6. طبقة الحياة الواقعية (أسفل طبقة)
**الموجود:** غير موجودة
**المطلوب:**
- [ ] **إنشاء `hierarchy/real_life_layer.py`** — الطبقة التحتية
- [ ] كل شخص = agent باختصاص دقيق
- [ ] كل شخص يتدرب أوتوماتيكياً على اختصاصه
- [ ] كل شخص يفكر وينتج أفكار ويطبقها
- [ ] كل اختصاص فيه عدة أشخاص بأساليب تفكير مختلفة
- [ ] معامل ومصانع افتراضية (إعادة بناء الحضارة)
- [ ] طبقة ربط عُليا → تستخلص الأفكار المُجمّعة

### 7. الكشافة — شجرات وأهرام قابلة للتوسعة
**الموجود:** 4 كشافين ثابتين (Tech, Market, Competitor, Opportunity)
**المطلوب:**
- [ ] كل كشاف = شجرة بأفرع متخصصة
- [ ] أفرع جديدة تُضاف أوتوماتيكياً حسب المجالات المكتشفة
- [ ] كل فرع = هرم بكشافين دقيقين
- [ ] ربط الكشافة ← طبقة الإضافة ← التدريب

### 8. VPS ↔ RTX Relay
**المشكلة:** `TRAINING_RELAY_UPSTREAM_URL` فارغ
- [ ] ضبط VPS `.env`: `TRAINING_RELAY_UPSTREAM_URL=http://192.168.1.164:8090`
- [ ] كل بيانات كشافة VPS → ترحّل للـ RTX للتدريب
- [ ] sync ثنائي الاتجاه

### 9. قاعدة بيانات عملاقة
**الموجود:** SQLite `data/bi_ide.db` (422KB) + `database/schema.sql`
**المطلوب:**
- [ ] PostgreSQL للإنتاج + SQLite للتطوير
- [ ] جداول لكل: طبقة، شجرة، هرم، شخص، فكرة، مصنع
- [ ] تاريخ كامل لكل قرار ونقاش وتدريب
- [ ] لا شي يضيع — كل شي يُخزّن

### 10. المشروع الموازي
**الموجود:** غير موجود
**المطلوب:**
- [ ] مشروع يتابع الأول لكن يخطط بطريقة مغايرة
- [ ] كل اختصاص يُعاد بناؤه من الصفر بطرق أفضل
- [ ] مقارنة دورية بين المشروعين

---

### 11. العتاد الموزّع — تدريب على 100+ حاسبة
**الموجود:** `orchestrator_api.py` (15KB) — يوزع مهام، workers يسجلون أنفسهم
**المشكلة:**
- Workers لا يتدربون — يأخذون مهام فقط
- لا يوجد **deduplication** للبيانات (أخطر نقطة!)
- لا يوجد **batch assignment** (كل worker يأخذ batch جديد)
- لا يوجد **أمر واحد** يشغّل 100% GPU فوراً

**المطلوب:**
- [ ] **Worker auto-enrollment:** أي حاسبة تنصّب BI-IDE → تسجل نفسها → تبدي 100%
- [ ] **Batch deduplication مركزي:** RTX يتتبع كل عينة تم التدريب عليها (hash/ID)
- [ ] **Central data store:** كل الداتا على RTX 5090، workers يسحبون batches جديدة فقط
- [ ] **Checkpoint merge:** كل worker يرسل gradient updates → RTX يدمجهم
- [ ] **أمر واحد:** `bi-ide --train --gpu-all` → 100% استهلاك فوري
- [ ] **Scale:** دعم 100+ worker متزامن بدون تكرار

---

## 🔗 أنظمة موجودة تحتاج ربط

| النظام | المسار | الحالة |
|--------|--------|--------|
| Brain (Scheduler + Evaluator) | `brain/` | ❌ غير متصل بالطبقات |
| AI Training (15 ملف) | `ai/training/` | ❌ غير متصل بـ RTX |
| Services (Training + Council + AI + Sync) | `services/` | ⚠️ جزئي |
| ERP (Accounting + CRM + HR + Inventory) | `erp/` | ⚠️ جزئي |
| Monitoring (Prometheus + Grafana + ELK) | `monitoring/` | ❌ غير مفعّل |
| Network (Auto-Reconnect + Firewall) | `network/` | ❌ غير مفعّل |
| Agents (Desktop Agent Rust) | `agents/desktop-agent-rs/` | ❌ غير مكتمل |
| Orchestrator API | `orchestrator_api.py` | ⚠️ يعمل جزئياً |

---

## 📅 ترتيب التنفيذ المقترح

### المرحلة A — تشغيل التدريب (الأهم)
1. ربط أنظمة التدريب الـ 4 بالسيرفر (`start_all()`)
2. تحميل الـ checkpoints والـ vocab عند البدء
3. ربط `InternetDataFetcher` ← تدريب أوتوماتيكي لا نهائي
4. VPS→RTX relay
5. **Batch deduplication** — كل عينة تُعلّم ولا تتكرر
6. **Worker auto-enrollment** — أي جهاز يبدي 100% فوراً
7. **أمر واحد يشغّل كل الموارد** (`bi-ide --train --gpu-all`)

### المرحلة B — المجلس والأوامر
5. أوامر الرئيس → تنفيذ فوري عبر كل الطبقات
6. حلقة اجتماعات أوتوماتيكية كل 30 دقيقة
7. ربط المجلس ← التنفيذ ← النتائج

### المرحلة C — الشجرة والهرم
8. تحويل `domain_experts.py` ← شجرة ديناميكية
9. كل تخصص = عدة أشخاص بأساليب مختلفة
10. اكتشاف تلقائي لتخصصات جديدة
11. كشافة بشجرات وأهرام

### المرحلة D — طبقة الحياة الواقعية
12. إنشاء `real_life_layer.py`
13. agents باختصاصات دقيقة
14. معامل ومصانع افتراضية
15. طبقة ربط عُليا

### المرحلة E — ذاتي التطوير + DB
16. `DynamicLayerGenerator` → إنتاج طبقات أوتوماتيكياً
17. PostgreSQL + حفظ كل شيء
18. brain/evaluator + scheduler → تقييم وتحسين ذاتي

### المرحلة F — المشروع الموازي
19. إنشاء مشروع موازي يعيد بناء كل اختصاص من الصفر
20. مقارنة دورية بين المشروعين
