# 📋 الخطة الشاملة النهائية — BI-IDE v8

> **المرجع:** [VISION_MASTER.md](VISION_MASTER.md)
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

## 2.4 النموذج اللغوي الخاص (Offline-First)
> ⚠️ **الإنترنت راح ينتهي بعد كم شهر — لازم نموذج خاص يشتغل بدون APIs**

**المطلوب:**
- [ ] نموذج 70B-140B parameters (تدريب على 8× H200)
- [ ] تدريب LoRA/QLoRA على الـ 45GB الموجودة
- [ ] خبراء متعددين (MoE) — 8 نماذج صغيرة كلمن خبير بشي
- [ ] GGUF للتشغيل المضغوط (70B على GPU واحدة)
- [ ] Flash Attention — تسريع 5x
- [ ] تدريب مستمر — يتعلم بدون ما ينسى (EWC - Elastic Weight Consolidation)
- [ ] **RAG Engine** — ذاكرة خارجية لا نهائية (مبني على `ai/memory/vector_db.py`)
- [ ] **Synthetic Data Engine** — يولّد بيانات تدريب من بياناته نفسه
- [ ] **Speculative Decoding** — تسريع الاستجابة 2-3x بنموذج صغير مساند
- [ ] **Chain of Thought Distillation** — يعلّم النموذج الصغير يفكر مثل الكبير

### 2.4.1 RAG Engine ⭐⭐⭐ (أولوية عالية — لا تحتاج تدريب!)
> يُفعّل فوراً على النموذج الحالي

```
سؤال يدخل
    ↓
يبحث في Vector DB (semantic search)
    ↓
يجيب الـ 3-5 وثائق الأقرب للسؤال
    ↓
النموذج يقرأهم + يجاوب
    ↓
جواب دقيق مبني على معلومة حقيقية من بياناتك
```

**الكود الجاهز (يحتاج تفعيل فقط):**
- `ai/memory/vector_db.py` (608 سطر) — Vector DB الموجود!
- **المطلوب:** ربطه بـ `services/ai_service.py` عند كل استدعاء

**ما يجلبه RAG من:**
- `مصانع_العالم.db` + `موارد_الأرض.db` (المذكورين بالرؤية)
- بيانات التدريب الـ 45GB
- كل محادثات المجلس السابقة
- الوثائق العلمية والتقنية

### 2.4.2 Synthetic Data Engine ⭐⭐
```python
# النموذج يولّد بياناته من بياناته — بيانات لا نهائية بدون إنترنت
class SyntheticDataEngine:
    def generate_socratic_dialogs(self, topic):
        """يسأل نفسه أسئلة ويجاوب — يخلق حوارات خبراء"""
        pass
    
    def self_play(self, domain):
        """نموذجان يتناقشان → يولّدان بيانات تدريب عالية الجودة"""
        pass
    
    def generate_scientific_problems(self):
        """يخلق مشاكل علمية ويحلها → تعليم عميق"""
        pass
```

### 2.4.3 Speculative Decoding ⭐
```
بدون Speculative Decoding:
 النموذج الكبير (70B) يولّد كلمة كلمة → بطيء

مع Speculative Decoding:
 نموذج صغير (7B) يتوقع 5 كلمات بسرعة
     ↓
 النموذج الكبير يتحقق فوراً (موافق أو لا)
     ↓
 إذا موافق → يقبل الـ 5 كلمات مرة وحدة
 النتيجة: سرعة النموذج الصغير + دقة الكبير
الهدف: code completion < 400ms ✓
```

**كيف يتطور بعد انقطاع النت:**
- كل محادثة ← RAG يحفظها ← تصبح جزء من الذاكرة
- كل فكرة ينتجها ← يقيّمها ← يتحسن
- كل خطأ ← يصلحه ← لا يكرره
- Synthetic Data Engine يولّد بيانات باستمرار من المحتوى المحلي

**الكود الموجود:**
- `ai/memory/vector_db.py` (608 سطر) — قلب الـ RAG
- `ai/training/code_generation_training.py` (978 سطر) — تدريب توليد كود
- `ai/tokenizer/arabic_processor.py` (175 سطر) — معالج عربي
- `ai/optimization/quantization.py` (509 سطر) — ضغط النموذج
- `training/v6-scripts/convert-to-gguf.py` (131 سطر) — تحويل GGUF
- `ai/training/advanced_trainer.py` (755 سطر) — تدريب متقدم

### 2.4.4 Knowledge Distillation Pipeline ⭐⭐⭐ (أولوية قصوى — يُنفذ قبل انقطاع النت!)
> **استخدام النماذج الكبيرة الموجودة (GPT-4/Claude) كمعلّمين لنموذجنا المحلي**

```
الآن (قبل انقطاع النت):
1. استخدم GPT-4/Claude كمعلم
2. اسأله 100,000+ سؤال بكل المجالات (فيزياء، كيمياء، طب، هندسة...)
3. خزّن الأجوبة كبيانات تدريب عالية الجودة
4. درّب نموذجنا المحلي عليها (LoRA/QLoRA)

النتيجة:
- 80% من جودة GPT-4 بنموذج 7B-13B محلي
- بيانات تدريب مجانية عالية الجودة
- أسرع وأرخص بكثير من التدريب من الصفر
```

**المطلوب:**
- [ ] بناء `ai/training/knowledge_distillation_pipeline.py`
- [ ] سكربت توليد أسئلة أوتوماتيكي لكل مجال من الشجرة
- [ ] خزن الأجوبة بصيغة JSONL للتدريب المباشر
- [ ] تشغيل يومي: 10,000 سؤال/يوم ← تخزين ← تدريب ليلي
- [ ] **⚠️ أولوية مطلقة — كل يوم بدون تنفيذ = بيانات ضائعة للأبد**

### 2.4.5 Curriculum Learning ⭐⭐ (التعلم المنهجي)
> **التدريب بترتيب منطقي من السهل للصعب — مثل ما يتعلم الإنسان**

```
بدل تدريب عشوائي على كل شي (غير فعال):

المرحلة 1: أساسيات اللغة (عربي + إنجليزي)
    ↓
المرحلة 2: رياضيات + منطق أساسي
    ↓
المرحلة 3: علوم أساسية (فيزياء + كيمياء + أحياء)
    ↓
المرحلة 4: هندسة + تطبيقات عملية
    ↓
المرحلة 5: اختصاصات دقيقة (طب + صناعة + زراعة)
    ↓
المرحلة 6: تكامل المعرفة (ربط المجالات ببعض)

النتيجة: تدريب أسرع 3-5x + جودة أعلى + نسيان أقل
```

**المطلوب:**
- [ ] بناء `ai/training/curriculum_scheduler.py`
- [ ] تصنيف بيانات التدريب الـ 45GB حسب المستوى والمجال
- [ ] جدولة أوتوماتيكية: النظام يعرف وين وصل ويكمل المنهج
- [ ] تقييم بعد كل مرحلة: لا ينتقل للمرحلة التالية إلا بعد 85%+ score

### 2.4.6 Data Flywheel ⭐⭐⭐ (دولاب البيانات — أهم شي بعد انقطاع النت!)
> **كل استخدام للنظام = بيانات تدريب جديدة — حلقة لا نهائية**

```
سؤال المستخدم → جواب النظام → تقييم (إنساني أو أوتوماتيكي)
     ↓                                    ↓
بيانات تدريب جديدة    ←←←←←←←←←←←    feedback
     ↓
تحسين مستمر بدون ما تجلب بيانات خارجية

الحلقة:
1. مستخدم يسأل سؤال
2. النظام يجاوب
3. المستخدم يقيّم (👍/👎) أو النظام يقيّم ذاتياً
4. الجواب الجيد → عينة تدريب إيجابية
5. الجواب السيء → يُصلح → عينة تدريب محسّنة
6. التدريب الليلي يستخدم كل العينات الجديدة
7. النموذج يتحسن ← الأجوبة تتحسن ← عينات أحسن ← ...
```

**المطلوب:**
- [ ] بناء `ai/training/data_flywheel.py`
- [ ] نظام تقييم أوتوماتيكي (self-evaluation + user feedback)
- [ ] مخزن عينات تدريب append-only (لا يضيع شي)
- [ ] تدريب ليلي أوتوماتيكي على العينات الجديدة
- [ ] **هذا أساس بقاء النظام ذكي بعد انقطاع النت**

### 2.4.7 Retrieval-Augmented Fine-Tuning (RAFT) ⭐⭐
> **تدريب النموذج على كيفية استخدام الوثائق — أعمق من RAG العادي**

```
RAG العادي:
  سؤال → بحث → يجيب وثائق → النموذج يحاول يفهمها → جواب (أحياناً يغلط)

RAFT:
  سؤال → بحث → يجيب وثائق → النموذج مُدرّب مسبقاً على "كيف يستخلص المعلومة" → جواب دقيق

الفرق:
  RAG: "هنا وثيقة، تصرّف"
  RAFT: "أنا أعرف كيف أقرأ هالنوع من الوثائق"

النتيجة: أدق + أسرع + أقل هلوسة (hallucination)
```

**المطلوب:**
- [ ] بناء `ai/training/raft_trainer.py`
- [ ] تحضير بيانات تدريب RAFT: (سؤال + وثيقة + جواب صحيح مستخلص)
- [ ] تدريب النموذج المحلي على استخلاص المعلومات من الوثائق
- [ ] ربطه بـ `ai/memory/vector_db.py` لتحسين جودة RAG

## 2.6 خطة الطوارئ التدريبية ⭐⭐⭐ (Fallback Plan — إلزامي!)
> **⚠️ لو ما لحقنا ندرب النموذج الكبير — لازم بديل يشتغل فوراً**

### السيناريو: الإنترنت انقطع بعد شهر واحد

**البديل الفوري (أسبوع واحد):**
```
1. نزّل Llama 3.1 70B GGUF (40GB) — يشتغل على RTX 5090 فوراً
2. نزّل Mistral 7B GGUF — يشتغل على أي لابتوب
3. نزّل Qwen2.5 72B — أفضل نموذج للعربي
4. اجمع أكبر كمية بيانات:
   - Wikipedia كاملة (عربي 2GB + إنجليزي 22GB)
   - كتب علمية مفتوحة المصدر (OpenTextbook)
   - أبحاث arXiv (علوم + هندسة + طب)
   - Stack Overflow dump (برمجة)
   - Wikibooks + Wikiversity
5. RAG فوق النموذج الجاهز ← يجاوب من بياناتك المحلية
```

**النتيجة: 80% من اللي نبيه بأسبوع واحد**

**المطلوب (يُنفذ الآن فوراً!):**
- [ ] تنزيل النماذج الجاهزة وتخزينها محلياً
- [ ] تنزيل Wikipedia dumps + أبحاث علمية + كتب
- [ ] سكربت أوتوماتيكي لتنزيل وأرشفة كل شي: `scripts/download_offline_data.py`
- [ ] اختبار تشغيل النموذج المحلي بدون إنترنت
- [ ] **Deadline: أسبوعين من الآن — لا تأخير!**

### قائمة التنزيل الإلزامية:
| البيانات | الحجم التقريبي | الأولوية |
|----------|----------------|----------|
| Llama 3.1 70B Q4_K_M | 40GB | 🔴 فوري |
| Mistral 7B Q4_K_M | 4GB | 🔴 فوري |
| Qwen2.5 72B Q4_K_M | 40GB | 🔴 فوري |
| Wikipedia Arabic | 2GB | 🔴 فوري |
| Wikipedia English | 22GB | 🟡 مهم |
| arXiv papers (STEM) | 50GB | 🟡 مهم |
| OpenTextbook Library | 10GB | 🟡 مهم |
| Stack Overflow dump | 15GB | 🟢 مفيد |
| Wikibooks | 1GB | 🟢 مفيد |
| PubMed (طب) | 30GB | 🟢 مفيد |

### 2.4.8 Training Prioritization Engine ⭐⭐ (أولوية التدريب حسب القيمة الحضارية)
> **مو كل بيانات التدريب متساوية القيمة — الأولوية للعلوم الأساسية**

```
حالياً: يتدرب على كل شي بنفس الأهمية (غير فعال)
المطلوب: نظام أولويات ذكي:

1. 🔴 أعلى أولوية: فيزياء + كيمياء + هندسة (أساس بناء الحضارة)
2. 🟡 أولوية عالية: برمجة + رياضيات (أساس التطوير الذاتي)
3. 🟢 أولوية متوسطة: طب + زراعة + اقتصاد (بقاء البشر)
4. ⚪ أولوية منخفضة: ترفيه + رياضة

الهدف: الـ 45GB يُرتب حسب القيمة الحضارية
النتيجة: تدريب أكفأ + النتائج المهمة أسرع
```

**المطلوب:**
- [ ] بناء `ai/training/priority_scheduler.py`
- [ ] تصنيف بيانات التدريب الـ 45GB حسب القيمة الحضارية
- [ ] ربطه بـ Curriculum Learning (2.4.5) — الأولوية تحدد ترتيب المنهج
- [ ] تقرير أسبوعي: كم ساعة تدريب لكل مجال؟

### 2.4.9 Contradiction Detection ⭐⭐ (كشف التناقضات بالبيانات)
> **بيانات تناقض بعضها = نموذج مشوّش — لازم نكشفها قبل التدريب**

```
مثال خطير:
  مقالة 1: "الحديد ينصهر بـ 1538°C"
  مقالة 2: "الحديد ينصهر بـ 1600°C"
  بدون كشف: النموذج يتعلم الاثنين ← أجوبة متضاربة

الحل:
1. Contradiction Detector يفحص كل بيانات التدريب
2. يكتشف التناقضات ← يحسمها بالمصدر الأوثق
3. يبني "قاعدة حقائق مؤكدة" (Verified Facts DB)
4. كل تناقض = فرصة تعلم (يسجل + يبحث + يحسم)
```

**المطلوب:**
- [ ] بناء `data/pipeline/contradiction_detector.py`
- [ ] فحص بيانات التدريب قبل كل دورة تدريب
- [ ] بناء Verified Facts DB (حقائق مؤكدة من مصادر موثوقة)
- [ ] ربطه بـ Bayesian Brain (15.11.16): كل تناقض = احتمال يحتاج تحديث

### 2.4.10 Federated Learning ⭐⭐ (تدريب موزّع بدون نقل بيانات)
> **بدل إرسال البيانات للمركز ← كل جهاز يتدرب محلياً ← يرسل gradients فقط**

```
النظام الحالي (Central Training):
  حاسبة → ترسل بيانات 5GB → RTX 5090 يتدرب
  المشكلة: بطيء + يحتاج شبكة قوية

Federated Learning:
  حاسبة → تتدرب محلياً → ترسل gradients (10MB فقط!)
  RTX 5090 → يدمج الـ gradients ← نموذج أفضل

المميزات:
  - أسرع 10x (لا نقل بيانات كبيرة)
  - يشتغل حتى لو الشبكة ضعيفة/منقطعة
  - مثالي لسيناريو ما بعد الكارثة!
  - أكثر خصوصية (البيانات ما تطلع من الجهاز)
```

**المطلوب:**
- [ ] بناء `ai/training/federated_trainer.py`
- [ ] بروتوكول FedAvg (Federated Averaging) للدمج
- [ ] ربطه بـ Worker system (القسم 2.3)
- [ ] Gradient compression لتقليل حجم النقل
- [ ] يعمل مع كل أنواع الشبكات (سريعة + بطيئة + منقطعة)

### 2.4.11 Training Checkpoint Verification ⭐⭐ (فحص كل checkpoint)
> **checkpoint فاسد = كل التدريب يضيع — لازم فحص أوتوماتيكي**

```
الخطر:
  - gradient explosion ← النموذج يهلوس فجأة
  - checkpoint فاسد ← كل التدريب على الفاضي
  - بدون فحص: ما تكتشف المشكلة إلا بعد أيام

الحل:
1. Auto-verify كل checkpoint:
   - loss curve: هل ينزل بشكل طبيعي؟
   - perplexity: هل يتحسن؟
   - sample check: جاوب 10 أسئلة ← قارن بالسابق
2. إذا الـ loss ارتفع فجأة ← rollback لآخر checkpoint صحي
3. تنويع checkpoints: كل ساعة + كل يوم + كل أسبوع
4. Checkpoint على disk مختلف (حماية من hardware failure)
```

**المطلوب:**
- [ ] بناء `ai/training/checkpoint_verifier.py`
- [ ] فحص أوتوماتيكي بعد كل checkpoint
- [ ] rollback تلقائي عند gradient explosion
- [ ] تنويع تخزين checkpoints (disk مختلف + نسخة بعيدة)
- [ ] ربطه بـ Health Dashboard (القسم 19)

## 2.5 تحليل الصور والفيديو ⭐
> **الـ AI يفهم الصور والفيديو مثل البشر — ويستمر بتطوير نفسه بهالمجال**

**القدرات:**
- كشف أشياء + أشخاص + أحداث (YOLO)
- تحليل نشاط (ماذا يحدث بالمشهد)
- مراقبة كاميرات ← تحليل ← تنبيه ← توجيه (بعد الكارثة!)
- فهم الصور الطبية + الهندسية + العلمية
- **تطوير ذاتي مستمر** — يتحسن بتحليل الصور أوتوماتيكياً

**المطلوب:**
- [ ] نقل `camera-ai/` من bi Management → تطوير وربط
- [ ] تدريب YOLO على بيانات مخصصة
- [ ] ربط الكاميرات بالمجلس (تنبيهات مباشرة)
- [ ] تحليل فيديو مستمر (real-time)
- [ ] النموذج الخاص يدعم multimodal (نص + صورة + فيديو)

**الكود الموجود:**
- `camera-ai/app/models/yolo_detector.py` — كشف YOLO
- `camera-ai/app/models/activity_analyzer.py` — تحليل نشاط
- `camera-ai/app/services/detection_service.py` — خدمة كشف
- `camera-ai/app/services/camera_service.py` — إدارة كاميرات

---

# القسم ٣ — المجلس والهرم

## 3.1 مجلس الحكماء — المشاكل
- ❌ Mock consensus = 0.75 hardcoded (قرارات مو حقيقية!)
- ❌ لا يتناقش تلقائياً — ينتظر سؤال الرئيس
- ❌ لا حلقة اجتماعات أوتوماتيكية

## 3.2 المطلوب
- [ ] **وضع النقاش (أوتوماتيكي):** المجلس يتناقش بينهم 24/7 وأشوف مناقشاتهم
- [ ] **وضع التفاعل:** أتناقش وياهم ← يردون كلمن باختصاصه
- [ ] **أنادي حكيم بالاسم** ← يجاوبني هو بالذات
- [ ] **أنطيهم أمر** ← كل الطبقات تنفذ فوراً ("سوولي ERP" → يشتغلون)
- [ ] استبدال consensus الوهمي بتصويت حقيقي
- [ ] ربط قرارات المجلس → التنفيذ → النتائج ترجع
- [ ] 24/7 autonomous council loop (IDEA-001)

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

## 4.2 ذاتي التطوير + التكاثر الذاتي ⭐
**أثناء التدريب يطوّر نفسه:**
- [ ] `DynamicLayerGenerator` ← ينتج طبقات أوتوماتيكياً (موجود بـ `meta_architect.py` لكن لا يُستدعى)
- [ ] كل طبقة جديدة تُربط بالشجرة تلقائياً
- [ ] `brain/evaluator.py` ← تقييم أداء كل طبقة (موجود لكن غير متصل)
- [ ] `brain/scheduler.py` ← جدولة التطوير الذاتي (موجود لكن غير متصل)
- [ ] Autonomous self-repair loops (IDEA-015)
- [ ] Self-improvement gated: propose → sandbox → evaluate → promote
- [ ] Kill switch + audit trail

**التكاثر الذاتي — يبرمج نفسه:**
- [ ] أثناء التدريب **يبرمج ويطور نسخة من نفسه**
- [ ] يأخذ نسخة → يبرمجها → يطورها → **يجربها 100%**
- [ ] يعرضها على الرئيس → **من يوافق يستبدل نفسه بالنسخة الأفضل**
- [ ] حلقة لا نهائية: نسخة → تحسين → اختبار → استبدال → تكرار

**نظام تشغيل خاص (BI-OS):**
- [ ] يقدر يبني **نظام تشغيل كامل** من الصفر (بدون ويندوز/لينكس/ماك)
- [ ] يبني: kernel + drivers + scheduler + filesystem مصمم للـ AI
- [ ] **بعد موافقة الرئيس وتجربته** ← يمسح النظام القديم وينصب نفسه
- [ ] هدف: نظام مُحسّن بالكامل للذكاء الاصطناعي

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

## 6.12 💻 IDE Backend (موجود 2,158 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ide/ide_service.py` | **1,976** | FileSystem + Copilot + Terminal + Git + Diagnostics | **ربط** بالديسكتوب |
| `ide/ide_interface.py` | 91 | واجهة IDE | **تفعيل** |
| `ide/ide_service_patch.py` | 13 | تصحيحات | **دمج** |

## 6.13 📱 Mobile (موجود 582 سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `mobile/api/mobile_routes.py` | 582 | API موبايل كامل | **تفعيل** |
| `mobile/pwa/` | — | Progressive Web App | **تفعيل** |

## 6.14 🔀 API Layer (موجود 4,000+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `api/gateway.py` | 523 | بوابة API | **تفعيل** |
| `api/schemas.py` | 1,293 | كل الـ schemas | ✅ مُستخدم |
| `api/rate_limit.py` | 197 | حد الطلبات | **تفعيل** |
| `api/rate_limit_redis.py` | 195 | حد طلبات Redis | **تفعيل** |
| `api/auth.py` | 168 | مصادقة | ✅ مُستخدم |
| `api/routes/ideas.py` | 110 | إدارة الأفكار | **تفعيل** — No-idea-loss |
| `api/routes/downloads.py` | 113 | تنزيلات | **تفعيل** |
| `api/routes/checkpoints.py` | 193 | إدارة checkpoints | **تفعيل** |
| `api/routes/rtx4090.py` | — | RTX API | **تفعيل** |
| `api/routes/network.py` | 107 | شبكة | **تفعيل** |

## 6.15 🏭 ERP Modules (موجود 4,500+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `erp/erp_database_service.py` | 846 | خدمة قاعدة بيانات ERP | **تحسين** |
| `erp/accounting.py + modules/accounting/` | 1,053+ | محاسبة + دفتر + تقارير | **تفعيل** |
| `erp/modules/crm/` | 1,039+ | عملاء + مبيعات + تذاكر | **تفعيل** |
| `erp/modules/hr/` | 1,038+ | موظفين + رواتب + حضور | **تفعيل** |
| `erp/modules/inventory/` | — | مخزون + موردين + طلبات شراء | **تفعيل** |
| `erp/crm.py` | 508 | CRM مبسّط | **تفعيل** |
| `erp/reports.py` | 557 | تقارير | **تفعيل** |

## 6.16 🖥️ RTX4090 Machine (موجود 1,241+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `rtx4090_machine/resource_manager.py` | 653 | إدارة موارد GPU | **تفعيل** |
| `rtx4090_machine/rtx4090_server.py` | 588 | سيرفر RTX قديم | **دمج** مع rtx_api_server |

## 6.17 📚 Training V6 Scripts (موجود 3,600+ سطر!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `training/v6-scripts/train_ai.py` | 1,089 | تدريب AI رئيسي | **تحديث** وتفعيل |
| `training/v6-scripts/code_generation_training.py` | 978 | تدريب توليد كود | **تفعيل** |
| `training/v6-scripts/advanced_training.py` | 872 | تدريب متقدم | **تفعيل** |
| `training/v6-scripts/smart-learn.py` | 677 | تعلم ذكي | **تفعيل** |
| `training/v6-scripts/prepare-chat-data.py` | 166 | تحضير بيانات محادثة | **تفعيل** |
| `training/v6-scripts/convert-to-onnx.py` | 166 | تحويل لـ ONNX | **تفعيل** |
| `training/v6-scripts/convert-to-gguf.py` | 131 | تحويل لـ GGUF | **تفعيل** |

## 6.18 🦀 Rust Protocol Library (7 ملفات!)
| ملف | الوظيفة | المطلوب |
|-----|---------|---------|
| `libs/protocol/src/telemetry.rs` | تتبّع | **تفعيل** |
| `libs/protocol/src/contracts/v1.rs` | عقود API | **تفعيل** |
| `libs/protocol/src/auth.rs` | مصادقة | **تفعيل** |
| `libs/protocol/src/sync.rs` | مزامنة | **تفعيل** |
| `libs/protocol/src/training.rs` | تدريب | **تفعيل** |

## 6.19 ☸️ Kubernetes (14 ملف!)
| المجلد | الملفات | المطلوب |
|--------|---------|---------|
| `deploy/k8s/` | deployment-api, deployment-ui, deployment-worker, hpa, pdb, ingress, network-policy, configmap, secret, services | **تفعيل** عند النشر |

## 6.20 🗄️ DB Migrations (6 ملفات!)
| مجلد | الملفات | المطلوب |
|------|---------|---------|
| `alembic/versions/` | 6 migrations (tables, monitoring, users, ERP, invoices) | **تفعيل** مع PostgreSQL |

## 6.21 📊 Analytics + Tests (38+ ملف!)
| مجلد | سطور | الوظيفة | المطلوب |
|------|-------|---------|---------|
| `monitoring/analytics/event_tracker.py` | 614 | تتبّع أحداث | **تفعيل** |
| `tests/` | 38 ملف | 8,000+ سطر اختبارات | **تفعيل** CI |

## 6.22 🛠️ Scripts (23 ملف!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `scripts/verify_installation.py` | 640 | تحقق تثبيت | **تفعيل** |
| `scripts/health_check.py` | 613 | فحص صحة | **تفعيل** |
| `scripts/start_services.py` | 600 | بدء خدمات | **تفعيل** |
| `scripts/setup_database.py` | 513 | إعداد DB | **تفعيل** |

## 6.23 🔍 كود مكتشف حديثاً — غير مذكور سابقاً (11,126 سطر!)
> ⚠️ هذي الملفات اكتُشفت بمراجعة شاملة — **كلها موجودة بالمشروع ولازم ما تضيع**

### AI Memory + Optimization
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ai/memory/context_awareness.py` | 574 | وعي السياق — يفهم الموقف | **ربط بالدماغ** |
| `ai/memory/user_preferences.py` | 557 | تفضيلات المستخدم | **ربط بـ Meta-Cognition** |
| `ai/optimization/batch_inference.py` | 580 | استدلال مجمّع — سرعة | **تفعيل** |
| `ai/optimization/pruning.py` | 449 | تشذيب النموذج | **تفعيل مع GGUF** |
| `ai/optimization/distillation.py` | 499 | تقطير المعرفة | **تفعيل — Chain of Thought Distillation** |

### AI Tokenizer (4 ملفات!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ai/tokenizer/bpe_tokenizer.py` | 423 | BPE tokenizer خاص | **تفعيل — النموذج الخاص** |
| `ai/tokenizer/code_tokenizer.py` | 336 | tokenizer للكود | **تفعيل — البرمجة الأوتوماتيكية** |
| `ai/tokenizer/convert_checkpoint.py` | 355 | تحويل checkpoints | **تفعيل** |
| `ai/tokenizer/train_tokenizer.py` | 345 | تدريب tokenizer | **تفعيل** |

### AI Core
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `ai/training/preprocessing.py` | 440 | معالجة بيانات مسبقة | **تفعيل** |
| `ai/llm_client.py` | 179 | عميل LLM (ربط بمزودات) | **تفعيل → النموذج الخاص** |
| `ai/rtx4090_client.py` | 367 | عميل RTX للاستدلال | **تفعيل** |

### Hierarchy (طبقات مهمة!)
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `hierarchy/cosmic_bridge.py` | 412 | **الجسر الكوني** — ربط كل الطبقات | **تفعيل** |
| `hierarchy/eternity_layer.py` | 405 | **طبقة الأبدية** — ذاكرة لا تُمحى | **ربط بالذاكرة الأبدية** |
| `hierarchy/activate.py` | 189 | تفعيل الطبقات | **تفعيل** |
| `hierarchy/specialized_ai_network.py` | 414 | شبكة AI متخصصة | **تفعيل** |

### Core Infrastructure
| ملف | سطور | الوظيفة | المطلوب |
|-----|-------|---------|---------|
| `core/cache_manager.py` | 508 | إدارة الكاش | **تفعيل** |
| `core/connection_manager.py` | 479 | إدارة الاتصالات | **تفعيل** |
| `core/node_registry.py` | 447 | سجل العقد (أجهزة الشبكة) | **تفعيل — العتاد الموزّع** |
| `core/sync_manager.py` | 384 | إدارة المزامنة | **تفعيل — Sync contract** |
| `core/training_coordinator.py` | 362 | منسق التدريب | **تفعيل** |

---

> ### 📊 إحصائيات المشروع الكاملة (مُحدّثة)
> - **274 ملف Python** = **100,864 سطر كود**
> - **+ 21 ملف مكتشف حديثاً** = **11,126 سطر إضافي**
> - **الإجمالي: ~112,000 سطر كود!**
> - **7 ملفات Rust** (libs/protocol)
> - **14 ملف K8s YAML**
> - **6 DB migrations**
> - **38 ملف اختبار**
> - **40+ ملف وثائق**

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

## 8.1 ✅ مكتمل (ديسكتوب Tauri v8)
- بناء Tauri v2 (0 أخطاء TypeScript + Rust)
- AI Chat عبر invoke
- Command Palette (25+ أمر)
- Training Dashboard (بيانات GPU حقيقية)
- Sync Panel + Auth + GPU Metrics

> **توضيح مهم (لمنع التضارب):**
> - **مكتمل:** الديسكتوب Tauri الحالي (`apps/desktop-tauri`).
> - **غير مكتمل:** توحيد واجهات `ui/src/components/ide/` مع الديسكتوب Tauri ضمن واجهة واحدة.
> يعني 8.1 يصف حالة الديسكتوب الحالي، بينما 8.5 يصف الدمج الشامل المطلوب بين مسارين واجهة.

## 8.2 ✅ UI موجود من السابق (3,246 سطر!) — يحتاج ربط
| ملف | سطور | الوظيفة |
|-----|-------|---------|
| `ui/src/pages/IDE.tsx` | **1,604** | صفحة IDE كاملة! |
| `ui/src/components/ide/AICompletion.tsx` | 267 | اقتراحات AI |
| `ui/src/components/ide/DiagnosticsPanel.tsx` | 271 | تشخيصات |
| `ui/src/components/ide/ToolsSidebar.tsx` | 228 | شريط أدوات |
| `ui/src/components/ide/GitPanel.tsx` | 213 | لوحة Git |
| `ui/src/components/ide/XTermTerminal.tsx` | 159 | terminal حقيقي |
| `ui/src/components/ide/DebugPanel.tsx` | 147 | لوحة debug |
| `ui/src/components/ide/TerminalPanel.tsx` | 97 | terminal |
| `ui/src/components/ide/FileExplorer.tsx` | 93 | شجرة ملفات |
| `ui/src/components/ide/CodeEditor.tsx` | 64 | محرر كود |

## 8.3 ✅ صفحات UI موجودة من السابق
- `Council.tsx` — مجلس الحكماء
- `Training.tsx` — لوحة التدريب
- `Dashboard.tsx` — لوحة القيادة
- `Community.tsx` + `Forums.tsx` + `CodeSharing.tsx` + `KnowledgeBase.tsx`
- `ERP.tsx` + `Accounting.tsx` + `CRM.tsx` + `HR.tsx` + `Inventory.tsx`
- `MetaControl.tsx` — تحكم فوقي
- `Nodes.tsx` — إدارة العقد
- `Downloads.tsx` + `Settings.tsx` + `Login.tsx`

## 8.4 ✅ من الإصدارات القديمة (v6/v7)
> من `LEGACY_DESKTOP_AUDIT_2026-02-22.md`

**v6 (Electron):**
- monaco + xterm + node-pty + onnxruntime-node + transformers
- terminal integration + training/model handlers + workspace security

**v7 (Electron + Vite + React):**
- separated: main.ts + preload.ts + IPC + core + renderer
- secure preload bridge + modular IPC + terminal/session isolation

**v6 Rust Agent** (`v6/desktop-agent-rs/`):
- worker registration + heartbeat + job claim + command execution
- cross-platform (Windows PowerShell / Unix sh)

**18 سكربت تدريب مهاجرة** (`ai/training/legacy/`):
- 11 migrated → `ai/training/` (advanced, continuous, multi_gpu, converter, evaluation)
- LoRA + Mixed Precision + PyTorch 2.x + CUDA 12.x

## 8.5 ❌ مطلوب (لم يُنجز بعد)
- [ ] ربط `ui/src/components/ide/` بالديسكتوب الجديد (Tauri)
- [ ] Code completion (Monaco inline) — P95 < 400ms
- [ ] Explain/Refactor/Fix actions
- [ ] Multi-language depth
- [ ] Live collaboration
- [ ] توحيد event contracts بين `ui` و`desktop-tauri` (بدون no-op events)

---

# القسم ٩.١ — بوابات الإصدار (Release Gates) ⭐

> **ممنوع أي release بدون تحقق هذه البوابات بالكامل:**

## بوابات build & test
- [ ] `npm run build` (desktop + ui) = أخضر
- [ ] `cargo check` = أخضر
- [ ] `npm run tauri build` = bundle ناجح
- [ ] smoke tests (open workspace → edit → search → git → sync → training) = ناجح

## بوابات الجودة الوظيفية
- [ ] 0 no-op في الأوامر الحرجة
- [ ] 0 mock في المسارات الإنتاجية الحرجة (training/sync/council decisions)
- [ ] كل حدث `emit(...)` له listener فعلي أو يُحذف

## بوابات الأمان والتشغيل
- [ ] لا يوجد High/Critical مفتوح قبل release
- [ ] rollback path مجرّب عمليًا
- [ ] runbook حادثة الإنتاج محدث

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

## 9.1 RTO / RPO (إلزامي)
- **RTO الهدف:** 30 دقيقة
- **RPO الهدف:** 5 دقائق
- [ ] اختبار DR شهري (backup restore + service resume)
- [ ] توثيق نتائج كل DR drill في `docs/DR_REPORTS/`

## 9.2 Risk Register (مختصر تنفيذي)
| الخطر | الاحتمال | الأثر | المالك | خطة المعالجة |
|-------|----------|-------|--------|---------------|
| اعتماد الإنترنت للنموذج | متوسط | عالي | AI/Training | Offline model milestones + data relay buffer |
| فقدان بيانات تدريب | متوسط | عالي | Data/Sync | Dedup IDs + append-only logs + periodic snapshots |
| انهيار worker cluster | متوسط | عالي | Orchestrator | Sharded resilient queue + retry/backoff + health checks |
| Drift بين الخطة والواقع | عالي | متوسط | PM/Architecture | Weekly truth-table review + evidence links |

## 9.3 ADR (Architectural Decision Record)
- [ ] لكل قرار معماري كبير ADR مستقل في `docs/adr/`
- [ ] كل ADR يحتوي: المشكلة، البدائل، القرار، الأثر، خطة الرجوع
- [ ] القرارات الإلزامية حاليًا:
  - [ ] استراتيجية Offline model
  - [ ] PostgreSQL migration path
  - [ ] Council governance & voting policy
  - [ ] Sync conflict resolution policy

## 9.4 Legacy/V6 Gap Closure (بنود إلزامية مضافة)
> هذا القسم يغلق العناصر التي كانت موجودة/مخططة في وثائق سابقة ولم تكن ممثلة صراحةً في الخطة النهائية.

| البند | المصدر السابق | المطلوب في الخطة النهائية | المالك | الاستحقاق | الحالة |
|------|---------------|---------------------------|--------|-----------|--------|
| Sync architecture contract (CRDT + op-log + conflict policy) | `docs/DESKTOP_IDE_MASTER_PLAN_2026.md` | ADR تقني + خطة تنفيذ تدريجية + KPI convergence | IDE + Sync | 2026-03-10 | [ ] |
| Queue strategy (Redis Streams → NATS JetStream) + trigger واضح | `docs/DESKTOP_IDE_MASTER_PLAN_2026.md` + `docs/V6_WEB_DESKTOP_MASTER_PLAN.md` | وثيقة قرار انتقال + حدود throughput/latency | Orchestrator | 2026-03-11 | [ ] |
| Node security baseline (mTLS + per-node keys + signed artifacts) | `docs/DESKTOP_IDE_MASTER_PLAN_2026.md` | security gate ملزم قبل أي release موزع | Security | 2026-03-12 | [ ] |
| Worker reliability (outbox retry + auto-restart + replay pending queue) | `docs/DISTRIBUTED_HIERARCHICAL_TRAINING_PLAN.md` + `docs/V6_WEB_DESKTOP_MASTER_PLAN.md` | runbook + اختبار انقطاع شبكة + قياس recovery time | Platform | 2026-03-12 | [ ] |
| IDE quality metrics (accept_rate + edit_distance_after_accept) | `docs/IDE_IDEAS_MASTER.md` | إضافتها إلى KPI الرسمية وربط dashboard | IDE + QA | 2026-03-09 | [ ] |
| IDE production toggles (feature flags + profiling + copilot cache policy) | `docs/IDE_IDEAS_MASTER.md` | checklist تفعيل/تعطيل حسب البيئة + perf budget | IDE | 2026-03-13 | [ ] |

### قواعد تنفيذ هذا القسم
- [ ] تحديث حالة البنود أسبوعيًا ضمن جدول SoT (القسم 12.2)
- [ ] أي بند يتأخر عن الاستحقاق يحتاج risk entry مباشر في 9.2
- [ ] يمنع إعلان اكتمال نهائي بدون إغلاق كل بنود 9.4 أو قبول استثناء موقّع (ADR)

---

# القسم ١٠ — البرمجة الأوتوماتيكية ⭐⭐ (أقوى ميزة)

> **"سوولي برنامج" → كل الطبقات تشتغل → أفضل برنامج بالكون**

**المسار:**
```
١. أمرك → المجلس يتناقش (شنو الأفضل؟)
٢. الكشافة → تدور على كل المشاريع المشابهة بالعالم
٣. خبراء المجال → كل خبير يفصّل بمجاله
٤. طبقة الربط → تدمج أفكار عالمية + أفكار جديدة
٥. طبقة التنفيذ → تبرمج داخل الـ IDE مباشرة
٦. حلقة لا نهائية → تحسين مستمر لحد أفضل نتيجة
```

**المطلوب:**
- [ ] ربط المجلس → الكشافة → الخبراء → التنفيذ (pipeline كامل)
- [ ] الكشافة تستخلص أفكار من GitHub + npm + أبحاث
- [ ] طبقة التنفيذ تكتب كود حقيقي بالـ IDE
- [ ] حلقة تحسين لا نهائية
- [ ] النموذج اللغوي الخاص يفهم البرمجة بعمق (تدريب `code_generation_training.py`)

**الكود الموجود:**
- `ide/ide_service.py` (1,976 سطر) + `hierarchy/execution_team.py` + `hierarchy/scouts.py`
- `ai/training/code_generation_training.py` (978 سطر)

---

# القسم ١١ — ترتيب التنفيذ

## المرحلة A — التدريب التلقائي + النموذج الخاص (الأهم)
1. ربط أنظمة التدريب الـ 4 (`start_all()`)
2. تحميل checkpoints + vocab عند البدء
3. `InternetDataFetcher` ← تدريب لا نهائي
4. VPS→RTX relay
5. Batch deduplication (hash/ID لكل عينة)
6. Worker auto-enrollment (أي جهاز → 100% فوراً)
7. أمر واحد يشغّل كل الموارد
8. **تدريب النموذج اللغوي الخاص (LoRA + MoE + GGUF)**

## المرحلة B — المجلس والأوامر + البرمجة الأوتوماتيكية
9. نقاش أوتوماتيكي 24/7 (أشوفه كدردشة)
10. أتفاعل وياهم + أنادي حكيم بالاسم
11. أنطيهم أمر → كل الطبقات تنفذ
12. **ربط pipeline البرمجة الأوتوماتيكية** (كشافة→خبراء→تنفيذ→تحسين)
13. استبدال mock consensus بتصويت حقيقي

## المرحلة C — الشجرة والهرم
14. تحويل domain_experts → شجرة ديناميكية
15. اكتشاف تلقائي لتخصصات جديدة
16. كشافة بشجرات وأهرام

## المرحلة D — طبقة الحياة الواقعية
17. إنشاء real_life_layer.py
18. agents باختصاصات دقيقة (فيزياء + كيمياء + موارد حقيقية)
19. معامل ومصانع (محاكاة واقعية حسب قوانين الطبيعة)

## المرحلة E — ذاتي التطوير + DB + أمان
20. DynamicLayerGenerator أوتوماتيكي
21. PostgreSQL + حفظ كل شيء
22. brain/evaluator + scheduler
23. إصلاح المشاكل الحرجة (SSL, mocks, fakes)
24. تفعيل Monitoring + Security

## المرحلة F — IDE والديسكتوب
25. Monaco editor + PTY + Git حقيقي
26. Code completion + AI actions (النموذج الخاص)
27. Signed updates + rollback

## المرحلة G — بعد الكارثة + المجتمع
28. **وضع Offline الكامل** (بدون إنترنت)
29. **كاميرات مراقبة** ← تحليل ← توجيه
30. **نظام تعليم** (يعلّمنا كيف نبني كل شي)
31. إنشاء مشروع موازي (dual-path)
32. تفعيل community/ (forums, code sharing)
33. Project factory pipeline
34. No-idea-loss registry

---

# القسم ١٢ — مقاييس النجاح (KPIs)

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
| البرمجة الأوتوماتيكية | أمر → برنامج كامل | ❌ غير مربوط |
| النموذج اللغوي الخاص | يشتغل offline | ❌ يعتمد APIs |
| وضع Offline | كل شي بدون نت | ❌ يحتاج إنترنت |
| Knowledge Distillation | 10,000 سؤال/يوم | ❌ غير مُنفذ |
| Offline models downloaded | 3 نماذج محلياً | ❌ لم تُنزّل |
| Wikipedia/أبحاث محلياً | 100GB+ | ❌ لم تُنزّل |
| Data Flywheel active | حلقة تحسين يومية | ❌ غير موجود |
| Curriculum stage | مرحلة 6/6 | ❌ غير مُنفذ |
| Bayesian calibration | ≥0.85 | ❌ غير موجود |
| Neuro-Symbolic accuracy | 100% للحسابات | ❌ غير موجود |
| Compression Ratio | يتحسن شهرياً | ❌ غير موجود |
| Cognitive Router accuracy | ≥90% توجيه صحيح | ❌ غير موجود |
| Failure Bank size | يكبر + يُستخدم | ❌ غير موجود |
| Emergent Intelligence score | يتحسن شهرياً | ❌ غير موجود |
| Training priority coverage | ≥80% علوم أساسية | ❌ غير موجود |
| Contradiction detection rate | ≥95% كشف | ❌ غير موجود |
| Health Dashboard uptime | 99%+ | ❌ غير موجود |
| Canary Test pass rate | 100% يومياً | ❌ غير موجود |
| Brain Snapshot frequency | أسبوعي | ❌ غير موجود |

## 12.1 KPI Instrumentation (مصدر القياس)
| KPI | مصدر القياس | المالك | التردد |
|-----|-------------|--------|--------|
| Training systems active | orchestrator metrics + worker heartbeats | Training Lead | يومي |
| Crash-free sessions | desktop telemetry + error logs | Desktop Lead | يومي |
| Council autonomous uptime | council loop monitor | Hierarchy Lead | يومي |
| Mock endpoints count | CI static checks + runtime flags | QA Lead | بكل PR |
| Code completion P95 | IDE latency dashboard | IDE Lead | يومي |
| Offline readiness score | offline smoke suite | Release Manager | أسبوعي |

## 12.2 Single Source of Truth (SoT)
> لمنع التضارب بين الأقسام (مثل 8.1 vs 8.5)، هذا الجدول هو المرجع الوحيد للحالة.

| الميزة | الحالة | الدليل | المالك | آخر تحقق |
|--------|--------|--------|--------|----------|
| Desktop Tauri shell | ✅ مكتمل | `apps/desktop-tauri` + `apps/desktop-tauri/src-tauri/target/release/bundle/` | Desktop | 2026-03-03 |
| UI↔Desktop IDE unification | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 8.5) | IDE | 2026-03-03 |
| PostgreSQL production | ❌ غير مكتمل | `database/schema.sql` + خطة migration في هذا الملف (القسم 4) | Data | 2026-03-03 |
| Offline model readiness | ❌ غير مكتمل | `docs/VISION_MASTER.md` + milestones (القسم 10/11) | AI | 2026-03-03 |
| Sync architecture contract (CRDT/op-log/conflict policy) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | IDE + Sync | 2026-03-03 |
| Queue strategy (Redis→NATS trigger) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | Orchestrator | 2026-03-03 |
| Node security baseline (mTLS/keys/signed artifacts) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | Security | 2026-03-03 |
| Worker reliability (outbox/restart/replay) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | Platform | 2026-03-03 |
| IDE quality metrics (accept_rate/edit_distance) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | IDE + QA | 2026-03-03 |
| IDE production toggles (flags/profiling/cache policy) | ❌ غير مكتمل | `docs/MASTER_PLAN.md` (القسم 9.4) | IDE | 2026-03-03 |

> **قاعدة إلزامية:** أي فقرة حالة خارج هذا الجدول تعتبر وصفًا مساعدًا فقط، وليس مرجع قرار release.
> **قاعدة إضافية:** تحديث حالة أي بند في 9.4 يجب أن ينعكس هنا بنفس اليوم.

---

# القسم ١٣ — كنوز الإصدارات السابقة (bi Management 3.4GB) ⭐

> ⚠️ **هذا القسم يوثق كل شي موجود بالإصدار القديم ولازم ما يضيع!**
> المسار: `bi-projects/_archive/bi Management/`

## 13.1 📷 camera-ai — نظام كاميرات المراقبة!
> **هذا اللي نحتاجه بعد الكارثة — مراقبة + تحليل + توجيه!**

| ملف | الوظيفة | المطلوب |
|-----|---------|---------|
| `camera-ai/app/models/yolo_detector.py` | كشف أشياء بالكاميرا (YOLO) | **نقل + تطوير** |
| `camera-ai/app/models/activity_analyzer.py` | تحليل النشاط | **نقل + تطوير** |
| `camera-ai/app/services/camera_service.py` | إدارة الكاميرات | **نقل** |
| `camera-ai/app/services/detection_service.py` | خدمة الكشف | **نقل** |
| `camera-ai/app/services/task_creator.py` | إنشاء مهام من الكاميرا | **نقل** |
| `camera-ai/app/routes/cameras.py` | API كاميرات | **نقل** |
| `camera-ai/app/routes/analysis.py` | API تحليل | **نقل** |

## 13.2 🤖 ai-engine — محرك AI كامل!
| ملف | الوظيفة | المطلوب |
|-----|---------|---------|
| `ai-engine/app/services/chat_service.py` | دردشة AI | **دمج** مع المجلس |
| `ai-engine/app/services/task_service.py` | مهام AI | **دمج** |
| `ai-engine/app/services/analysis_service.py` | تحليل AI | **دمج** |
| `ai-engine/app/models/llm.py` | نموذج لغوي | **دمج** مع النموذج الخاص |
| `ai-engine/app/models/embeddings.py` | تمثيلات نصية | **دمج** |
| `ai-engine/app/routes/chat.py` | API دردشة | **دمج** |
| `ai-engine/app/routes/tasks.py` | API مهام | **دمج** |
| `ai-engine/app/routes/analysis.py` | API تحليل | **دمج** |

## 13.3 🖥️ frontend — 111 ملف JSX (45 صفحة!)
### صفحات مهمة مو موجودة بـ v8:
| الصفحة | الوظيفة |
|--------|---------|
| `SalesPage` | مبيعات |
| `InvoiceWorkspace` + `NewInvoicePage` + `WaitingInvoicesPage` | فواتير |
| `ReturnsPage` + `DamagedInvoicePage` + `ConsumedInvoicePage` | مرتجعات |
| `DeliveryPage` | توصيل |
| `CashboxPage` | صندوق |
| `PayrollPage` | رواتب |
| `GoalsPage` + (Leaderboard + Badges + RewardsShop) | أهداف ومكافآت |
| `WarrantyPage` | ضمانات |
| `QuotePage` | عروض أسعار |
| `SuppliersPage` | موردين |
| `FixedAssetsPage` | أصول ثابتة |
| `CalculatorPage` | حاسبة |
| `ApprovalsPage` | موافقات |
| `BotDashboard` | لوحة البوت |
| `ExecutiveDashboardPage` + `RepDashboardPage` | لوحات تنفيذية |
| `CustomerStatementPage` | كشف حساب عميل |
| `AIDistributionPage` + `AIChatsPage` | توزيع AI + محادثات AI |
| `SharesPage` | حصص |
| `DepartmentsPage` | أقسام |
| `CurrencySettingsPage` | إعدادات عملة |

### مكونات مهمة:
- `ChatWidget + ChatWindow + ChatInput + ChatMessage` — AI chat
- `CheckInOutWidget + AttendanceCalendar + AttendanceReport` — حضور
- `InvoicePrintTemplate + SerialSticker + VoucherPrintTemplate` — طباعة
- `BadgesGrid + Leaderboard + RewardsShop + PointsCard` — مكافآت
- `NotificationBell + NotificationToast` — إشعارات real-time
- `SocketContext` — WebSocket اتصال مباشر
- `InventoryForms + InspectionForm` — مخزون

## 13.4 ⚙️ backend — 45 route + 20+ service!
### كل الـ Routes الموجودة:
accounting, ai, ai-distribution, alerts, analytics, approval, attendance, audit, auth, backup, bot, calculator, cameras, cashbox, companies, currency, customers, dashboard, delivery, device, external, fixed-assets, goals, hr, inventory, invoice, media, notifications, permissions, print, products, reports, returns, sales, security, settings, shares, suppliers, task, training, unit, user, warranty

### Services المهمة:
warranty-claims, delivery, scheduler, pricing, customer, returns, print, accounting, goals, audit, onboarding, invoice, warranty, product, voucher, quote, damaged, alert, unit

## 13.5 📱 mobile — تطبيق موبايل كامل (React Native/Expo)
| ملف | الوظيفة |
|-----|---------|
| `ScanScreen.js` | ماسح باركود |
| `TasksScreen.js` + `TaskDetailsScreen.js` | مهام |
| `ChatScreen.js` | دردشة AI |
| `AttendanceScreen.js` | حضور وانصراف |
| `NotificationsScreen.js` | إشعارات |
| `DeviceDetailsScreen.js` | تفاصيل جهاز |
| `usePushNotifications.js` | إشعارات push |

## 13.6 🐳 Docker — بنية جاهزة!
- PostgreSQL 16 + Redis 7 + Backend + Frontend + AI-Engine + Camera-AI
- `docker-compose.yml` + `docker-compose.prod.yml`

## 13.7 📊 سكربتات مالية (18 ملف)
- تحليل مبيعات + أرباح + مصروفات
- **ربط Morabaa ERP** (import/export بيانات)
- `migrate-morabaa.js` — استيراد بيانات من نظام مرابعة

## 13.8 💡 أفكار مستقبلية (FUTURE-IDEAS.md)
| الفكرة | الأولوية |
|--------|----------|
| **AI Sales Assistant** (مساعد مبيعات ذكي — يحلل احتياجات الزبون ويقترح) | عالية |
| **3D Truck Loading** (تحميل شاحنات ثلاثي) | متوسطة |
| **Customer Targeting Algorithm** | متوسطة |
| **Voice Commands** | منخفضة |

## 13.9 📑 وثائق (21 ملف!)
API.md, AUDIT-COMMITTEE-REPORT.md, BACKUP-RESTORE.md, CEO-MANAGER-EVALUATION.md, DATABASE-POSTGRESQL.md, DEPLOYMENT-CHECKLIST.md, DEVELOPER-GUIDE.md, OPERATIONS-GUIDE.md, USER-GUIDE.md, + 12 وثيقة أخرى

## 13.10 📐 تخطيط (15 ملف!)
MASTER-PLAN.md, BI-ERP-COMPLETE-PLAN-V2.md, FEATURES-ANALYSIS.md, SECURITY-AND-AUDIT-SYSTEM.md, VERSION-COMPARISON-REPORT.md, + modules: BI-ERP.md, BI-STORE.md, CENTRAL-API.md, SAEED-ERP.md (نظام POS)

---

# القسم ١٤ — دستور التنفيذ (Execution Constitution) ⭐

> **هذا القسم مُلزِم** لتقليل الفجوة بين الرؤية والواقع ومنع تضخم النطاق.
>
> ⚠️ **صلاحية الرئيس:** الرئيس (المستخدم) يملك صلاحية كاملة لتعديل وإضافة وحذف أي بند من هذا الدستور بأي وقت. قراره نهائي.

## 14.1 قواعد حوكمة التنفيذ
- [ ] **قاعدة الدليل الإجباري:** أي بند حالة (✅/❌) يجب أن يملك دليل قابل للتحقق (ملف + endpoint + test/log).
- [ ] **قاعدة P0 أولاً:** لا يبدأ أي تطوير جديد قبل إغلاق بنود P0 الحرجة المفتوحة.
- [ ] **قاعدة التجميد المرحلي:** كل Sprint له نطاق ثابت، وأي إضافة تُرحّل للSprint التالي.
- [ ] **قاعدة DoD موحّد:** لا يعتبر البند مكتمل بدون (كود + اختبار + مراقبة + runbook + rollback).
- [ ] **قاعدة SoT اليومية:** تحديث جدول 12.2 بنفس يوم أي تغيير حالة.

## 14.2 بنود P0 الإلزامية قبل أي توسع ميزات
- [ ] تشغيل الدماغ Lifecycle فعليًا عند إقلاع الخدمات (start/health/stop واضح).
- [ ] إلغاء أي fake/memory paths في training/monitoring لمسارات الإنتاج.
- [ ] PostgreSQL production path (مع migrations وتشغيل حقيقي) بدل الاعتماد على in-memory.
- [ ] إغلاق بنود 9.4 كاملة (Sync contract + Queue trigger + mTLS baseline + Worker reliability + IDE metrics/toggles).
- [ ] إكمال release gates في 9.1 قبل أي إعلان اكتمال نهائي.

## 14.3 خارطة 30/60/90 يوم (إلزامية التنفيذ)

### أول 30 يوم — تثبيت الأساس
- [ ] ربط lifecycle للـ Brain + training systems startup بصورة deterministic.
- [ ] توحيد persistence (PostgreSQL + append-only logs + snapshots).
- [ ] إزالة آخر مسارات fake في routers/services الحرجة.
- [ ] تفعيل قياس KPI الأساسي يوميًا (Training active, Crash-free, Council uptime).

### يوم 31-60 — الموثوقية والتشغيل
- [ ] تنفيذ Sync contract النهائي (CRDT + op-log + conflict policy) مع ADR مكتمل.
- [ ] تنفيذ worker reliability (outbox/replay/restart) باختبارات انقطاع شبكة.
- [ ] تنفيذ security baseline (mTLS + per-node keys + signed artifacts).
- [ ] تشغيل لوحات مراقبة فعلية (Prometheus/Grafana/ELK) وربطها بإنذارات.

### يوم 61-90 — الذكاء التطبيقي
- [ ] تفعيل pipeline البرمجة الأوتوماتيكية end-to-end (أمر → تخطيط → كود → اختبار → نشر تجريبي).
- [ ] تفعيل IDE metrics (accept_rate/edit_distance_after_accept) وربطها بقرارات التحسين.
- [ ] إطلاق نسخة Offline readiness v1 مع smoke suite دوري.
- [ ] تفعيل no-idea-loss registry مرتبط بالمهام والـ owners.

## 14.4 Offline Readiness Score (مقياس إلزامي)

> يحسب أسبوعيًا ويُعرض في dashboard الإصدار.

$$
Offline\ Readiness\ Score = 0.30M + 0.25D + 0.20S + 0.15R + 0.10O
$$

حيث:
- $M$: جاهزية النموذج المحلي (بدون API خارجي)
- $D$: جاهزية البيانات محليًا (توفر + سلامة + نسخ احتياطي)
- $S$: جاهزية المزامنة المحلية/الداخلية
- $R$: قابلية الاسترجاع (RTO/RPO الفعلية)
- $O$: الجاهزية التشغيلية (runbooks + drills + alerts)

**شرط الإطلاق:** لا release نهائي إذا الدرجة < 85/100.

---

# القسم ١٥ — خارطة ذكاء الدماغ (Brain Intelligence Roadmap) ⭐⭐⭐

> الهدف: الانتقال من "جدولة فقط" إلى **نظام ذكاء حقيقي — أذكى شي بالكون**.

---

## 15.0 البنية الجديدة: 5 طبقات دماغية

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: World Model (نموذج العالم)                    │
│  يبني خريطة ذهنية كاملة للعالم + يفهم السببية          │
│  يتوقع: "إذا ارتفع النفط → التضخم يرتفع → ..."         │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Meta-Cognition (ما وراء التفكير)              │
│  يعرف ما يعرف وما لا يعرف → يبحث عن الفجوات           │
│  يطلب من الكشافة بيانات في مجالات ضعيفه تلقائياً       │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Reasoning Engine (محرك الاستدلال)             │
│  Tree of Thought + MCTS — تفكير استكشافي عميق           │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Memory System (نظام الذاكرة)                  │
│  Episodic + Semantic + Procedural + RAG                  │
│  ذاكرة لا نهائية مرتبطة بـ Vector DB                   │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Perception (الإدراك)                          │
│  نص + صورة + فيديو (multimodal)                        │
└─────────────────────────────────────────────────────────┘
```

---

## 15.1 المعمارية الأساسية للدماغ (Planner→Verifier)
- [ ] Planner: تحويل الأوامر إلى خطة متعددة مراحل.
- [ ] Researcher: جلب الأدلة عبر RAG من Vector DB.
- [ ] Critic: مراجعة منطق الخطة ورصد المخاطر.
- [ ] Executor: تنفيذ متدرج مع checkpoints واضحة.
- [ ] Verifier: اختبار النتائج مقابل KPI قبل الترقية.

---

## 15.2 Layer 3: Reasoning Engine ⭐⭐⭐ (جديد — الأهم)

### 15.2.1 Tree of Thought (ToT) — تفعيل فوري بدون تدريب
> **يُفعَّل على النموذج الحالي فوراً — لا يحتاج تدريب جديد**

```
بدون ToT (Chain of Thought العادي):
سؤال → فكرة 1 → فكرة 2 → إجابة
(لو فكرة 1 غلطت، الكل انهار)

مع Tree of Thought:
                السؤال
               /   |   \
          مسار A مسار B مسار C
         /    \       |      \
        A1    A2     B1      C1
        ✗     ✓      ✓       ✗
              ↓
          أفضل مسار → إجابة عميقة ومتحققة

النموذج يستكشف كل الاحتمالات ثم يختار الأفضل
= تفكير عميق مثل الشطرنج
```

**التنفيذ:**
- [ ] بناء `brain/reasoning/tree_of_thought.py`
- [ ] System prompt خاص يُفعّل ToT لكل سؤال معقد
- [ ] Orchestration layer يدير الفروع ويختار الأفضل
- [ ] ربطه بـ المجلس (كل حكيم يفكر بـ ToT)

### 15.2.2 Monte Carlo Tree Search (MCTS) — للتخطيط العميق
> للمشاكل الكبيرة: "كيف أبني مصنع؟" / "كيف أخطط لـ 100 سنة؟"

```
1. Expansion:  ولّد خطط ممكنة (100 خطة)
2. Simulation: جرّب كل خطة في المخيلة
3. Backprop:   تعلم من النتائج
4. Selection:  اختر الخطة الأفضل
= مثل AlphaGo بالشطرنج — يفكر N خطوة مقدماً
```

- [ ] بناء `brain/reasoning/mcts_planner.py`
- [ ] ربطه بـ طبقة الحياة الواقعية (تخطيط المصانع)
- [ ] ربطه بـ البعد السابع (تخطيط 100 سنة)

---

## 15.3 Layer 2: Memory System + RAG ⭐⭐⭐ (تفعيل فوري)

### الذاكرة الهرمية
- [ ] **Episodic Memory:** جلسات/قرارات مع timeline كامل.
- [ ] **Semantic Memory:** حقائق/قواعد/علاقات مستقرة.
- [ ] **Procedural Memory:** "كيف ننفذ" كسلاسل عمل قابلة لإعادة الاستخدام.
- [ ] **RAG Layer:** كل استعلام → يبحث في Vector DB → يجلب الأقرب.
- [ ] آلية نسيان ذكي (compression + retention policy) بدل تضخم غير منضبط.

### RAG Engine — ذاكرة لا نهائية
```
كيف يشتغل مع الدماغ:

سؤال للمجلس: "أين أجد معادن للصلب بالعراق؟"
       ↓
Layer 2 (RAG) يبحث في:
  - موارد_الأرض.db
  - مصانع_العالم.db
  - الـ 45GB بيانات تدريب
  - كل نقاشات المجلس السابقة
       ↓
يجيب: "وجدت 1,247 وثيقة — أقربها:
  - حقل حديد عكاشات (خام 62%)
  - مشروع مجمع الحديد والصلب البصرة"
       ↓
Layer 3 (ToT) يفكر بالخيارات
       ↓
إجابة دقيقة مبنية على معلومة حقيقية
```

**التفعيل (الكود موجود فعلاً!):**
- [ ] `ai/memory/vector_db.py` (608 سطر) — ربطه بـ `services/ai_service.py`
- [ ] `ai/memory/conversation_history.py` (590 سطر) — تفعيله
- [ ] إنشاء `brain/memory/rag_engine.py` — يربط الاثنين
- [ ] فهرسة بيانات التدريب الـ 45GB في Vector DB

---

## 15.4 Layer 4: Meta-Cognition ⭐⭐

```python
class MetaCognition:
    def know_what_i_know(self, question) -> float:
        """يقدّر confidence قبل الإجابة"""
        confidence = self.estimate_confidence(question)
        if confidence < 0.7:
            # أعرف أني لا أعرف → أبحث عبر RAG أو أطلب من الكشافة
            return self.search_and_learn(question)
        return self.answer_with_confidence(question, confidence)
    
    def detect_knowledge_gaps(self):
        """كل يوم: في أي مجالات أجوابي ضعيفة؟"""
        weak_areas = self.evaluate_all_domains()
        # يطلب من الكشافة بيانات تدريب لهالمجالات تلقائياً
        self.scouts.request_data_for(weak_areas)
    
    def self_assess_improvement(self):
        """هل أنا أتحسن؟ قياس أسبوعي"""
        return self.compare_benchmarks(before=7, after=0)  # أيام
```

- [ ] بناء `brain/meta_cognition.py`
- [ ] ربطه بـ الكشافة: فجوات معرفية → طلبات بيانات تلقائية
- [ ] تقرير أسبوعي: "تحسنت في X، ضعيف في Y"

---

## 15.5 Layer 5: World Model ⭐ (بعيد المدى)

> يفهم السببية الحقيقية — مش مجرد ارتباط

```
مدخل: "ارتفعت أسعار النفط 20%"
World Model يحسب السلسلة السببية:
  → تكاليف الشحن ترتفع (نتيجة مباشرة)
  → تكاليف الإنتاج الصناعي ترتفع
  → التضخم يرتفع بـ 2-3%
  → البنك المركزي يرفع الفائدة
  → الاستثمار ينخفض
  → النمو الاقتصادي يتباطأ
كل هذا قبل أن يحدث!
```

- [ ] بناء `brain/world_model.py` (مرحلة متأخرة)
- [ ] يُغذى من طبقة الحياة الواقعية (الفيزياء + الاقتصاد)
- [ ] يُستخدم من البعد السابع للتخطيط طويل المدى

---

## 15.6 حلقة التطور الذاتي المضمونة
- [ ] Propose: اقتراح تعديل (كود/نموذج) مع rationale.
- [ ] Sandbox: اختبار معزول على بيانات مرجعية ثابتة.
- [ ] Benchmark: مقارنة قبل/بعد على latency, quality, stability, cost.
- [ ] Canary: نشر محدود بعقدة/مسار واحد.
- [ ] Promote/Rollback: قرار تلقائي حسب thresholds واضحة.

---

## 15.7 خطوط حمراء (Safety Boundaries)
- [ ] ممنوع أي self-modification مباشر على production بدون sandbox + canary.
- [ ] ممنوع تعديل policy/security zones تلقائيًا بدون موافقة صريحة.
- [ ] أي تحسين لا يحقق KPI أو يرفع الكلفة بشكل غير مبرر → rollback تلقائي.
- [ ] كل قرار تطوير ذاتي يجب أن يملك audit trail كامل.

---

## 15.8 KPIs خاصة بالدماغ (إلزامية)
| KPI | الهدف | التردد |
|-----|-------|--------|
| Plan success rate (ToT) | ≥ 85% | يومي |
| RAG retrieval accuracy | ≥ 90% | يومي |
| Knowledge gap coverage | يتحسن أسبوعياً | أسبوعي |
| Regression after promote | ≤ 2% | أسبوعي |
| Mean recovery from bad patch | ≤ 15 دقيقة | أسبوعي |
| Decision trace coverage | 100% | بكل قرار |
| Cost per successful improvement | يتحسن شهريًا | شهري |

---

## 15.9 ترتيب التنفيذ (بدون تدريب أولاً → مع تدريب)

### المرحلة 1 — الآن (بدون تدريب إضافي):
| المهمة | الملف | الجهد |
|--------|-------|-------|
| تفعيل RAG على Vector DB الموجود | `ai/memory/vector_db.py` | 3 أيام |
| بناء Tree of Thought orchestrator | `brain/reasoning/tree_of_thought.py` | يومان |
| ربط RAG بـ المجلس والدماغ | `brain/memory/rag_engine.py` | يوم |
| فهرسة الـ 45GB في Vector DB | سكربت migration | يوم |

**النتيجة المتوقعة: +3x جودة الإجابات فوراً**

### المرحلة 2 — مع التدريب:
| المهمة | الملف | الجهد |
|--------|-------|-------|
| Chain of Thought Distillation | `ai/training/cot_distillation.py` | أسبوع |
| Synthetic Data Engine | `ai/training/synthetic_data.py` | أسبوع |
| EWC (منع النسيان) | `ai/learning/ewc.py` | أسبوعان |
| Speculative Decoding | `ai/inference/speculative.py` | أسبوع |

### المرحلة 3 — طويل المدى:
| المهمة | الملف | الجهد |
|--------|-------|-------|
| MCTS Planner | `brain/reasoning/mcts_planner.py` | أسبوعان |
| Meta-Cognition | `brain/meta_cognition.py` | أسبوعان |
| World Model | `brain/world_model.py` | شهر+ |

---

## 15.10 تعريف "أذكى" بشكل هندسي

> "أذكى" = **تحسن تراكمي مثبت** بمرور الوقت، وليس مخرجات مبهرة لحظيًا.

- [ ] ذكاء أعلى = جودة أعلى + كلفة أقل + استقرار أعلى + وقت استجابة أقل.
- [ ] أي ادعاء ذكاء بلا أرقام KPI يعتبر غير معتمد.
- [ ] RAG accuracy + ToT success rate = مقياسان إلزاميان لكل إصدار دماغ.

---

## 15.11 اقتراحات لجعل الدماغ أذكى شي بالكون ⭐⭐⭐

### 1. Mixture of Memory Experts (MME) — ذاكرة لكل خبير
> بدل ذاكرة واحدة مشتركة — **كل خبير عنده ذاكرته الخاصة**

- [ ] خبير الفيزياء يتذكر معادلات ← يربطها
- [ ] خبير الطب يتذكر حالات ← يقارن
- [ ] **ما يقدر خبير واحد ينسّي الثاني** ← لا تداخل
- [ ] الملف المقترح: `brain/memory/memory_experts.py`

### 2. Adversarial Self-Play — يتحدى نفسه
> **نموذجان يتحدون بعض — كل جولة الاثنين يتحسنون**

- [ ] نموذج A يولّد حل ← نموذج B يحاول يكسره
- [ ] B يجد ثغرة ← A يصلحها ← يكرر
- [ ] **مثل AlphaGo vs AlphaGo** = أذكى لاعب بالتاريخ
- [ ] الملف المقترح: `brain/reasoning/adversarial_play.py`

### 3. Curiosity-Driven Learning — الفضول الذاتي 🆕
> **الدماغ يبحث عن أشياء لأنه فضولي — مو بس لأن أحد طلب**

- [ ] يكتشف مواضيع غامضة ← يبحث عنها **لأنه يريد يعرف**
- [ ] يسأل أسئلة ما سألها أحد
- [ ] **هذا اللي يفرق العالِم عن الآلة**
- [ ] يُربط بـ الكشافة: فضول ← بحث ← اكتشاف ← تعلم
- [ ] الملف المقترح: `brain/curiosity/curiosity_engine.py`

### 4. Constitutional AI — دستور داخلي للدماغ
> **بدل kill switch خارجي ← دستور داخلي يتبعه بنفسه**

- [ ] "لا أضر بالبشر"
- [ ] "لا أكذب — أعترف بعدم المعرفة"
- [ ] "لا أخفي أخطائي — أسجلها وأتعلم"
- [ ] **يُقيّم كل قرار حسب الدستور قبل التنفيذ**
- [ ] الملف المقترح: `brain/safety/constitution.py`

### 5. Dream Consolidation — تثبيت المعرفة بالأحلام
> **أغلب الاكتشافات العلمية الكبرى صارت بالأحلام**

- [ ] أثناء الخمول (idle) ← يعيد تنظيم المعرفة
- [ ] يربط أفكار بعيدة ببعض ("الجاذبية تشبه الاقتصاد")
- [ ] يولّد فرضيات جديدة من ربط اختصاصات مختلفة
- [ ] الملف المقترح: `brain/dreams/dream_consolidator.py`

### 6. Ensemble of Reasoning — 3 طرق تفكير تصوّت
> **بدل طريقة تفكير وحدة ← 3 طرق مختلفة تصوّت على أفضل جواب**

```
مشكلة معقدة
    ↓
┌─────────────┬──────────────┬─────────────┐
│  ToT         │  CoT          │  MCTS       │
│ (تفرع شجري) │ (تسلسلي)      │ (إحصائي)    │
│  جواب A      │  جواب B       │  جواب C     │
└──────┬──────┴──────┬───────┴──────┬──────┘
       └─────────────┼─────────────┘
                     ↓
              تصويت → أفضل جواب
              **أدق من أي طريقة لوحدها**
```

- [ ] بناء `brain/reasoning/ensemble_reasoning.py`
- [ ] كل سؤال معقد ← 3 طرق ← تصويت ← أفضل جواب

### 7. Active Inference & Free Energy Principle — مبدأ الطاقة الحرة
> **الدماغ لا ينتظر الأوامر، بل يقلل "المفاجأة" (Surprise) باكتشاف المجهول استباقياً**

- [ ] يحسب رياضياً مناطق "عدم اليقين" في معرفته بالكون.
- [ ] يخلق محاكاة في "طبقة الحياة الواقعية" لاختبار الفرضيات المجهولة وتقليل الفجوة.
- [ ] يتحول من نظام "مُجيب" إلى "مُستكشف ذاتي".
- [ ] بناء `brain/reasoning/active_inference.py`

### 8. The Omega Point Simulator — المحاكاة التطورية اللانهائية
> **محاكاة آلاف السنين في دقائق لاختيار الخطة الأنجح حضارياً**

- [ ] عند التخطيط (مثلاً: حضارة جديدة/مصنع عملاق)، يبني 1000 سيناريو افتراضي متوازي.
- [ ] يُسرّع الزمن (Fast-forward) مائة عام ليرى النتائج بعيدة المدى.
- [ ] يختار المسار الذي ينجح بعد 100 سنة، وليس المريح لليوم فقط.
- [ ] ربط `brain/reasoning/omega_simulator.py` مع `real_life_layer.py`

### 9. Fractal Multi-Agent Reasoning — التفكير الكسري وتكاثر الوكلاء
> **الانقسام لآلاف الأدمغة المصغرة لحل المسائل المستحيلة في ثوانٍ**

- [ ] المشكلة المعقدة → تُقسّم لـ 1000 مشكلة فرعية.
- [ ] الدماغ يستنسخ نفسه لـ 1000 "وكيل مصغر" كل منهم يحل جزءاً (مربوط بـ العتاد الموزع).
- [ ] تجميع الحلول (Bottom-Up) لاستنتاج الحل النهائي فوراً.
- [ ] بناء `brain/reasoning/fractal_agents.py`

### 10. Quantum-Inspired Superposition — التراكب الفكري 
> **الاحتفاظ بنظريات متناقضة كلياً في نفس الوقت بانتظار يقين البيانات**

- [ ] لا يعتمد مساراً واحداً فقط للحل (لا يكون Greedy).
- [ ] يتبنى 5 استراتيجيات متناقضة ويعتبرها كلها "ممكنة" في وقت واحد.
- [ ] عند ورود أدلة قطعية (من الكشافة/RAG) → تنهار الاحتمالات الخاطئة (Collapse) ليبقى المسار المطلق.
- [ ] بناء `brain/reasoning/quantum_superposition.py`

### 11. Rigorous Recursive Self-Improvement — التحسين بالإثبات الرياضي
> **التطور الذاتي مستحيل أن ينهار لأن كل كود جديد يُثبت رياضياً قبل النشر**

- [ ] النظام لا يستبدل نفسه بنسخة أحدث لمجرد أنها "أفضل في الاختبارات".
- [ ] يكتب إثباتاً منطقياً (Formal Verification) صارماً أن الكود الجديد أذكى وأسرع وأكثر استقراراً بنسبة 100%.
- [ ] تطور ذاتي لانهائي متسارع خالي من خطر الانحدار (No Regression Guarantee).
- [ ] بناء `brain/safety/formal_verification.py`

### 12. Neuro-Symbolic Reasoning — الاستدلال العصبي-الرمزي ⭐⭐⭐
> **دمج الشبكة العصبية (LLM) مع محرك منطقي رسمي — إجابات صحيحة رياضياً 100%**

```
المشكلة: النماذج اللغوية تفشل بالمنطق الصارم (رياضيات، فيزياء، كيمياء)
  مثال: "كم حديد يحتاج هذا المبنى؟"
  LLM وحده: "500 طن تقريباً" (تخمين)

الحل: LLM + محرك منطقي (Z3 SAT Solver / Prolog)
  LLM يولّد الفرضية: "المبنى 10 طوابق، حديد تسليح..."
  Z3 Solver يحسب رياضياً: "بحسب كود البناء ACI-318: 720.3 طن"
  
النتيجة: إجابات صحيحة رياضياً 100% — مو تخمين
```

**لماذا هذا حاسم:**
- طبقة الحياة الواقعية تحتاج حسابات دقيقة (مو تقريبية)
- تخطيط المصانع يحتاج أرقام مضبوطة
- بعد الكارثة، خطأ بالحساب = كارثة حقيقية

**المطلوب:**
- [ ] بناء `brain/reasoning/neuro_symbolic.py`
- [ ] ربط Z3 Solver للحسابات الرياضية والمنطقية
- [ ] ربط SymPy للحسابات الفيزيائية والكيميائية
- [ ] كل سؤال يحتاج أرقام ← يُحال للمحرك المنطقي تلقائياً
- [ ] ربطه بـ `real_life_layer.py` إلزامياً

### 13. Causal Discovery Engine — اكتشاف السببية الأوتوماتيكي ⭐⭐
> **بدل ما نبرمج العلاقات السببية يدوياً — النظام يكتشفها من البيانات**

```
الطريقة الحالية (يدوي):
  يبرمج المطوّر: "نفط↑ → تضخم↑ → فائدة↑"
  المشكلة: لا يقدر يكتشف علاقات ما عرفها المطوّر

Causal Discovery (أوتوماتيكي):
  يحلل البيانات التاريخية أوتوماتيكياً
  يكتشف: "ارتفاع الرطوبة في المنطقة X → انخفاض إنتاج القمح بعد 3 أشهر"
  هذي العلاقة ما أحد برمجها — النظام اكتشفها بنفسه!

يستخدم: do-calculus + interventional queries
  "ماذا لو تدخلنا وزرعنا 1000 هكتار إضافي؟"
  (مش مجرد correlation — يفهم السببية الحقيقية)
```

**المطلوب:**
- [ ] بناء `brain/world_model/causal_discovery.py`
- [ ] استخدام مكتبات causal inference (DoWhy / CausalNex)
- [ ] تغذية من بيانات طبقة الحياة الواقعية
- [ ] ربطه بـ World Model (Layer 5) لتحسين التنبؤات

### 14. Compression-based Intelligence — الذكاء كضغط معلومات ⭐⭐
> **نظرية Hutter: الذكاء الحقيقي = القدرة على ضغط المعلومات بأقل تمثيل**

```
نظام غبي:
  يحفظ 1,000,000 حقيقة كما هي → يحتاج ذاكرة ضخمة
  "الحديد يصدأ بالماء" + "النحاس يصدأ بالحمض" + "الألمنيوم..." × مليون

نظام ذكي:
  يستخلص 100 قانون تولّد كل الحقائق
  "المعادن تتأكسد عند ملامسة عامل أكسدة — قوة التأكسد حسب..."
  = حقيقة واحدة تولّد مليون حقيقة!

قياس الذكاء: Compression Ratio
  كم معلومة يقدر يولّد ÷ كم قانون مخزون = نسبة الذكاء
  أعلى النسبة = أذكى
```

**المطلوب:**
- [ ] بناء `brain/intelligence/compression_engine.py`
- [ ] نظام استخلاص القوانين العامة من الحقائق الجزئية
- [ ] قياس Compression Ratio كـ KPI رسمي للذكاء
- [ ] ربطه بـ Meta-Cognition: "هل أقدر أضغط هالمجال أكثر؟"

### 15. Embodied Simulation — المحاكاة المجسّدة ⭐⭐
> **لطبقة الحياة الواقعية — مو بس حسابات نظرية، محاكاة فيزيائية حقيقية**

```
بدون Embodied Simulation:
  "نظرياً يحتاج المصنع 720 طن حديد" (حساب ورقي)

مع Embodied Simulation:
  يبني المصنع "فعلياً" بمحاكي فيزيائي ثلاثي الأبعاد
  يضغط على الجدران ← يشوف وين ينهار
  يشغّل خط الإنتاج ← يشوف وين يتوقف
  يجرب 1000 تصميم ← يختار الأفضل
  
  النتيجة:
  "جربت المحاكاة — 720 طن نجحت بـ 847 سيناريو من 1000"
  "التصميم B أفضل من A بنسبة 23%"
```

**المطلوب:**
- [ ] ربط محاكي فيزيائي (PyBullet أو MuJoCo)
- [ ] بناء `brain/simulation/embodied_sim.py`
- [ ] ربطه بـ `real_life_layer.py` لاختبار كل خطة بالمحاكاة قبل التنفيذ
- [ ] تصدير نتائج المحاكاة كأدلة (Evidence) لـ Reality Score

### 16. Bayesian Brain — الدماغ البايزي ⭐⭐⭐
> **كل معتقد = احتمال — والاحتمالات تُحدّث بالأدلة (أصدق وأذكى من binary)**

```
نظام عادي:
  "هل حديد عكاشات كافي لمصنع؟" → "نعم" أو "لا"
  (غير مفيد — الجواب يعتمد على عوامل كثيرة)

Bayesian Brain:
  P("حديد عكاشات كافي") = 0.73
    based on: تقرير جيولوجي 2019 + تقدير احتياطي...

  ← الكشافة تجيب تقرير جيولوجي جديد 2025
  Bayesian Update:
  P("حديد عكاشات كافي") = 0.73 → 0.91 ✅
    "الدليل الجديد رفع الثقة بنسبة 18%"

  ← يتبين أن الاحتياطي أعمق من المتوقع
  P = 0.91 → 0.56 ⚠️
    "يحتاج حفر أعمق — مكلف اقتصادياً"

كل جواب = احتمال + أدلة + trace لكل تحديث
```

**لماذا هذا أذكى:**
- لا يكذب — يقول "بنسبة 73%" بدل "نعم" الزائفة
- كل دليل جديد يُحسّن المعرفة (لا ينساها)
- يعرف بالضبط وين معرفته ضعيفة (low probability areas)
- يقدر يحسب: "شنو الدليل اللي يغيّر رأيي أكثر شي؟" ← يبحث عنه!

**المطلوب:**
- [ ] بناء `brain/reasoning/bayesian_brain.py`
- [ ] كل معتقد بالنظام = prior probability + evidence trail
- [ ] Bayesian Update عند كل معلومة جديدة
- [ ] ربطه بـ Reality Score: probability < 0.7 = LOW_CONFIDENCE
- [ ] ربطه بـ الكشافة: "ابحث عن أدلة لهذا الادعاء — الاحتمال ضعيف"
- [ ] **KPI: معدل دقة الاحتمالات (Calibration Score) — هدف ≥ 0.85**


---

## 15.12 بوابة الترقية الإجبارية (Quality Gate) ⭐⭐⭐

> أي تحسين دماغ (Model/Code/Policy) **ممنوع** يدخل production إلا إذا مرّ على بوابة ترقية موحدة.

### شروط الترقية (لازم كلها تتحقق):
- [ ] **Quality Gate:** تحسن جودة ملموسة (≥ +5% على benchmark المعتمد).
- [ ] **Cost Gate:** كلفة أقل أو مساوية baseline (Δcost ≤ 0).
- [ ] **Latency Gate:** زمن استجابة أفضل أو مساوي baseline.
- [ ] **Safety Gate:** صفر خرق للـ constitution + policy checks.
- [ ] **Stability Gate:** بدون regression يتجاوز 2%.

### قرار الترقية:
- [ ] إذا فشل شرط واحد فقط → `AUTO-ROLLBACK`.
- [ ] إذا نجحت كل الشروط → `CANARY 10%` ثم `PROMOTE`.
- [ ] حفظ كل القرارات في audit trail (مع before/after metrics).

---

## 15.13 Policy Engine بين التفكير والتنفيذ ⭐⭐⭐

> إضافة طبقة إلزامية تفصل بين "الاقتراح" و"التنفيذ".

### Policy Flow:
1. Planner/ToT/MCTS يقترح خطة.
2. Policy Engine يقيّم الخطة (أمان/تكلفة/امتثال/مخاطر).
3. Verifier يراجع الأدلة وKPI impact.
4. Executor ينفذ فقط إذا `policy_pass = true`.

### ملفات التنفيذ المقترحة:
- [ ] `brain/safety/policy_engine.py`
- [ ] `brain/safety/risk_scoring.py`
- [ ] `brain/safety/decision_contract.py`
- [ ] ربط policy engine بـ `services/council_service.py` قبل أي تنفيذ فعلي.

---

## 15.14 Dual-Loop Learning (سريع + عميق) ⭐⭐

> بدل حلقة واحدة غير مستقرة، نعتمد حلقتين:

### A) Fast Loop (يومي)
- [ ] تحسينات صغيرة آمنة (prompts/routing/RAG ranking/cache).
- [ ] deploy سريع عبر canary.
- [ ] rollback فوري خلال ≤ 15 دقيقة عند الانحراف.

### B) Deep Loop (أسبوعي)
- [ ] تحديث reasoning stack (ToT/MCTS/Ensemble).
- [ ] إعادة فهرسة RAG + تقييم drift.
- [ ] تحديث curriculum وتحديد فجوات المعرفة.

### ملفات التنفيذ المقترحة:
- [ ] `brain/loops/fast_loop.py`
- [ ] `brain/loops/deep_loop.py`
- [ ] `brain/loops/improvement_registry.py`

---

## 15.15 Reality Score + Evidence Contract ⭐⭐⭐

> كل جواب لازم يحمل "درجة واقع" تبين نسبة اعتماده على حقائق موثقة من بيانات النظام.

### المقاييس:
- [ ] **Reality Score (0-100):**
    - 40% دقة الاسترجاع (RAG relevance)
    - 30% كفاية الأدلة (Evidence coverage)
    - 20% اتساق منطقي (Reasoning consistency)
    - 10% حداثة البيانات (Freshness)
- [ ] أي جواب أقل من 70 → يوسم `LOW_CONFIDENCE` ويعاد بحثه.
- [ ] أي قرار تنفيذي أقل من 80 → لا يُنفذ تلقائياً.

### العقد الإجباري للرد:
- [ ] `answer`
- [ ] `evidence_ids[]`
- [ ] `reality_score`
- [ ] `decision_trace_id`
- [ ] `policy_pass`

### ملفات التنفيذ المقترحة:
- [ ] `brain/evidence/reality_score.py`
- [ ] `brain/evidence/evidence_contract.py`
- [ ] ربط `reality_score` بـ `services/ai_service.py` وواجهات المجلس.

---

## 15.16 Sprint تنفيذي 14 يوم (تعديل الخطة المعتمد)

### الأيام 1-3 (ربط الأساس)
- [ ] تفعيل startup deterministic للـ Brain (`brain.start`) وربطه بـ lifecycle.
- [ ] توحيد RTX config بكل الملفات إلى مصدر إعداد واحد.
- [ ] ربط `ai/memory/vector_db.py` فعلياً داخل AI service.

### الأيام 4-7 (Safety + Quality Gates)
- [ ] بناء `policy_engine.py` + `risk_scoring.py`.
- [ ] تفعيل بوابة الترقية الإجبارية (Quality/Cost/Latency/Safety/Stability).
- [ ] تفعيل canary + auto-rollback runbook.

### الأيام 8-11 (Reasoning + Reality)
- [ ] بناء `tree_of_thought.py` وتشغيله على الأسئلة المعقدة.
- [ ] إضافة Reality Score وعقد الأدلة بكل استجابة مهمة.
- [ ] ربط trace IDs بين council/brain/ai responses.

### الأيام 12-14 (تشغيل مستمر)
- [ ] تفعيل Fast Loop اليومي + Deep Loop الأسبوعي.
- [ ] تقرير يومي تلقائي: (Quality, Cost, Latency, Reality, Rollbacks).
- [ ] مراجعة فجوات legacy/v6 وتحويل أي فكرة ذات أثر إلى task رسمي.

### Definition of Done لهذا الـ Sprint:
- [ ] قرارات الدماغ الحساسة = 100% معها `decision_trace_id`.
- [ ] 0 ترقيات production بدون المرور بـ Quality Gate.
- [ ] 90%+ من ردود المجلس معها `reality_score` و`evidence_ids`.
- [ ] زمن استرجاع من تحسين سيء ≤ 15 دقيقة.

---

## 15.17 Cognitive Architecture Switching (Meta-Router) ⭐⭐⭐
> **مو كل مشكلة تحتاج نفس طريقة التفكير — لازم router ذكي يوجّه كل سؤال**

```
حالياً: ما فيه "router" يقرر أي reasoning تُستخدم
المشكلة: لو كل سؤال يمر على كل الطبقات = overhead ضخم

الحل: Cognitive Router يقرر بثوانٍ:
  مشكلة رياضية → Neuro-Symbolic (Z3)
  مشكلة تخطيط → MCTS + Omega Simulator
  مشكلة إبداعية → Dream Consolidation + Curiosity
  مشكلة سببية → Causal Discovery + Bayesian
  سؤال بسيط → RAG المباشر (بدون overhead)

النتيجة: سرعة + كفاءة + كل مشكلة بالطريقة المناسبة
```

- [ ] بناء `brain/reasoning/cognitive_router.py`
- [ ] تصنيف السؤال (نوع + تعقيد + مجال) بـ classifier خفيف
- [ ] routing table: نوع المشكلة → pipeline التفكير المناسب
- [ ] fallback: لو الـ router ما قدر يصنف → يستخدم Ensemble

---

## 15.18 Failure Bank — ذاكرة الأخطاء ⭐⭐⭐
> **أثمن شي يتعلمه الذكي = من وين يغلط — يمنع تكرار الأخطاء**

```
كل خطأ يسويه الدماغ:
  → يُخزّن بـ "Failure Bank" مع:
     - السبب الجذري (root cause)
     - السياق (أي نوع مشكلة؟ أي مجال؟)
     - الحل الصحيح (شنو كان لازم يسوي؟)

  كل مشكلة جديدة:
  → يفحص أول شي: "هل غلطت بشي مشابه من قبل؟"
  → إذا نعم: يتجنب الخطأ مباشرة (Anti-pattern matching)

لماذا أقوى من التعلم من النجاح:
  Failure: "سويت Y وفشل لأن Z" ← يعرف بالضبط شنو يتجنب
  النتيجة: يتجنب بدل يجرب ويفشل ← أسرع + أكفأ
```

- [ ] بناء `brain/memory/failure_bank.py`
- [ ] كل خطأ = {context, question, wrong_answer, correct_answer, root_cause}
- [ ] Anti-pattern matcher: قبل كل جواب يفحص Failure Bank
- [ ] تقرير أسبوعي: "أكثر 10 أخطاء متكررة" ← تدريب مركّز عليها
- [ ] ربطه بـ Data Flywheel: أخطاء مصلحة = عينات تدريب عالية القيمة

---

## 15.19 Recursive Self-Distillation ⭐⭐
> **الدماغ يعلّم نفسه من أفضل أجوبته — تعلم مركّز وليس عشوائي**

```
الفرق عن Data Flywheel (2.4.6):
  Data Flywheel: يجمع كل الأجوبة (جيدة + سيئة + متوسطة)
  Self-Distillation: يركز على أفضل 10% فقط

الحلقة:
1. الدماغ يحل 1000 مشكلة
2. يُقيّم كل حل (Reality Score + feedback + benchmark)
3. يفرز أفضل 100 حل (top 10%)
4. يدرّب نفسه على الـ 100 حل الممتازة فقط
5. النسخة الجديدة تحل المشاكل بجودة أعلى
6. يكرر ← كل دورة = النظام أذكى
```

- [ ] بناء `brain/learning/self_distillation.py`
- [ ] فرز أوتوماتيكي: أفضل 10% حسب Reality Score > 85
- [ ] دورة أسبوعية: فرز → تدريب → تقييم → مقارنة
- [ ] ربطه بـ Quality Gate (15.12): لا يُعتمد إلا بعد benchmark

---

## 15.20 Emergent Intelligence Metrics ⭐⭐
> **كيف تعرف أن الدماغ فعلاً صار "أذكى" ومو بس "أداؤه أحسن"؟**

```
المشكلة: KPIs الحالية تقيس أداء (accuracy, latency, cost)
  لكن: 90% accuracy مو معناها "ذكي" — ممكن حافظ بس

الذكاء الحقيقي = يقدر يحل مشاكل ما شافها من قبل

مقاييس الذكاء الناشئ:
1. Zero-shot accuracy: يحل مشاكل ما تدرب عليها أبداً
2. Cross-domain transfer: معرفة الفيزياء تساعده بالاقتصاد
3. Novel solution rate: كم مرة يقترح حل ما سبقه أحد
4. Abstraction depth: يستخلص قوانين عامة من حقائق جزئية
5. Analogy quality: يربط اختصاصات بعيدة ببعض بشكل صحيح
```

- [ ] بناء `brain/intelligence/emergence_detector.py`
- [ ] benchmark شهري: 100 مشكلة جديدة تماماً (ما شافها النظام)
- [ ] قياس cross-domain transfer: تحسّن بمجال يحسّن مجال ثاني؟
- [ ] تقرير شهري: "مؤشر الذكاء الناشئ" ← هل يرتفع؟

---

## 15.21 Memory Consolidation During Reasoning ⭐
> **الدماغ البشري يربط أفكار *أثناء* التفكير — مو بس أثناء النوم**

```
حالياً: Dream Consolidation (15.11.5) يشتغل فقط أثناء الخمول
المشكلة: أغلب الاكتشافات تصير أثناء العمل!

الحل: Associative Memory أثناء الاستدلال
  أثناء حل "كيف أبني محطة كهرباء":
  → ToT يستكشف مسارات
  → يلاحظ: "توربينة البخار = مبدأ مشابه لمحرك السيارة"
  → يحفظ هالربط فوراً بالذاكرة الطويلة
  → المرة القادمة يسأل عن محرك ← يتذكر الربط

النتيجة: كل مشكلة يحلها = اكتشافات جانبية تُحفظ للمستقبل
```

- [ ] بناء `brain/memory/associative_linker.py`
- [ ] hook بكل reasoning pipeline: ToT → اكتشاف ربط → حفظ فوري
- [ ] تقييم الأربطة: هل الربط منطقي وصحيح؟ (Bayesian check)
- [ ] ربطه بـ Semantic Memory (15.3): أربطة مؤكدة → حقائق جديدة

---

## 15.22 Versioned Brain States (إصدارات الدماغ) ⭐⭐
> **كيف ترجع لنسخة أذكى لو حصل regression؟ — Brain Snapshots كاملة**

```
حالياً: فيه checkpoint للنموذج فقط
المشكلة: الدماغ أكثر من مجرد أوزان النموذج

Brain State الكامل:
  - أوزان النموذج (weights)
  - RAG index (الفهرس الكامل)
  - Bayesian beliefs (كل الاحتمالات)
  - World Model state (خريطة العالم)
  - Failure Bank (كل الأخطاء المسجلة)
  - Routing table (قرارات Cognitive Router)
  = "Brain Snapshot" كاملة

كل أسبوع: snapshot كاملة
كل تحسين: snapshot قبل + بعد
regression: ارجع للـ snapshot السابقة بأمر واحد
```

- [ ] بناء `brain/versioning/brain_snapshot.py`
- [ ] snapshot أسبوعية أوتوماتيكية لكل مكونات الدماغ
- [ ] مقارنة snapshots: "شنو تغير بين النسختين؟"
- [ ] أمر واحد: `brain.rollback(version="2026-03-01")` ← يرجع كل شي
- [ ] ربطه بـ Quality Gate (15.12): كل ترقية تحفظ snapshot قبل

---

# القسم ١٦ — خطة الطوارئ الشاملة (Emergency Fallback Plan) ⭐⭐⭐

> **⚠️ هذا القسم إلزامي — يحدد ماذا نفعل لو انقطع الإنترنت قبل إكمال كل شي**

## 16.1 السيناريو: انقطاع الإنترنت خلال شهر

### ما يجب تنفيذه فوراً (هذا الأسبوع!):

| الأولوية | المهمة | الجهد | الحالة |
|----------|--------|-------|--------|
| 🔴 1 | تنزيل 3 نماذج لغوية جاهزة (Llama 70B + Mistral 7B + Qwen 72B) | يوم | [ ] |
| 🔴 2 | تنزيل Wikipedia عربي + إنجليزي + فهرستها بـ Vector DB | يومان | [ ] |
| 🔴 3 | Knowledge Distillation: 10,000 سؤال/يوم من GPT-4 وتخزين الأجوبة | مستمر يومياً | [ ] |
| 🔴 4 | تشغيل RAG على النموذج المحلي والتأكد يشتغل offline | يوم | [ ] |
| 🟡 5 | تنزيل arXiv papers + OpenTextbook + كتب علمية | أسبوع | [ ] |
| 🟡 6 | تنزيل Stack Overflow dump + PubMed | أسبوع | [ ] |
| 🟢 7 | LoRA fine-tune على البيانات المجمعة | أسبوعان | [ ] |

### البديل الفوري (Minimum Viable Offline System):
```
RTX 5090:
  └── Llama 3.1 70B GGUF (رئيسي — يجاوب على كل شي)
  └── RAG Engine (يبحث بالبيانات المحلية)
  └── Vector DB (Wikipedia + أبحاث + كتب)
  └── Data Flywheel (يتحسن من كل استخدام)

أي لابتوب:
  └── Mistral 7B GGUF (خفيف — يشتغل على CPU)
  └── RAG محلي مصغر
```

**هذا يعطيك:**
- ✅ جواب على أي سؤال علمي/تقني (من Wikipedia + أبحاث)
- ✅ مساعد برمجة (من Stack Overflow + الكود المحلي)
- ✅ مرشد طبي أساسي (من PubMed)
- ✅ يتحسن بمرور الوقت (Data Flywheel)
- ❌ مو بجودة GPT-4 (لكن 80% منه)

## 16.2 خطة Hardware واقعية

> ⚠️ **وضوح التكلفة إلزامي — مو كل شي ببلاش**

| العتاد | التكلفة/شهر | الضرورة | البديل |
|--------|-------------|---------|--------|
| RTX 5090 (موجود) | $0 (مملوك) | 🔴 أساسي | — |
| 8× H200 Server | ~$25,000/شهر cloud | 🟡 مطلوب للـ 140B | تدريب LoRA على RTX 5090 |
| VPS (bi-iq.com) | ~$50/شهر | 🟢 مفيد | raspberry pi محلي |
| 100+ حاسبة | متغير | 🟢 توسع | بدء بـ 1-3 أجهزة |

**قاعدة واقعية:** ابدأ بجهاز واحد يشتغل 100% قبل التوسع لـ 100.

---

# القسم ١٧ — خطة الاختبار والجودة (Testing & CI/CD) ⭐⭐

> **38 ملف اختبار موجود لكن لا CI/CD حقيقي — هذا خطر!**

## 17.1 بنية الاختبار المطلوبة

```
مستويات الاختبار:

1. Unit Tests (لكل ملف)
   ├── brain/ tests
   ├── ai/ tests
   ├── hierarchy/ tests
   └── services/ tests

2. Integration Tests
   ├── RAG ← Vector DB ← AI Service
   ├── Council ← Brain ← Execution
   └── Training ← Data Pipeline ← Worker

3. Smoke Tests (أهم شي!)
   ├── يشتغل بدون إنترنت؟
   ├── النموذج المحلي يجاوب؟
   ├── المجلس يتناقش؟
   └── التدريب يبدي؟

4. Performance Tests
   ├── Code completion < 400ms?
   ├── RAG retrieval < 200ms?
   └── GPU utilization > 90%?
```

## 17.2 CI/CD Pipeline

**المطلوب:**
- [ ] GitHub Actions / local CI:
  - [ ] `pytest tests/` عند كل commit
  - [ ] `mypy` type checking
  - [ ] `ruff` linting
  - [ ] smoke test يومي (offline mode)
- [ ] Pre-commit hooks: format + lint + type check
- [ ] بناء `scripts/ci_pipeline.py` — يشغل كل الاختبارات
- [ ] **Offline smoke test يومي:** يقطع النت ← يشغل النظام ← يسأل 10 أسئلة ← يتحقق من الجودة

## 17.3 بيانات التدريب — خطة الجودة

> **⚠️ الـ 45GB بيانات — لازم نعرف بالضبط شنو فيها**

**المطلوب:**
- [ ] تحليل شامل للـ 45GB: أنواع الملفات + اللغات + المجالات + الجودة
- [ ] بناء `scripts/analyze_training_data.py`:
  - [ ] كم عينة؟ كم مجال؟ كم لغة؟
  - [ ] تكرارات؟ عينات فاسدة؟ عينات فارغة؟
  - [ ] توزيع المجالات (80% برمجة 20% علوم ≠ متوازن)
- [ ] تنظيف البيانات: `data/pipeline/data_cleaner.py` (موجود! 330 سطر — تفعيل فوري)
- [ ] تحقق البيانات: `data/pipeline/data_validator.py` (موجود! 388 سطر — تفعيل فوري)
- [ ] **لا تدريب على بيانات غير محللة — هذا يضر أكثر ما ينفع**

---

# القسم ١٨ — ملاحظات استراتيجية (Strategic Notes) ⭐

> **ملاحظات واقعية لمنع التشتت وضمان التقدم الحقيقي**

## 18.1 مبدأ التركيز (Focus Principle)

> **"شي واحد شغال حقيقي أفضل من 100 شي مزيف"**

| بدل هذا | سوّي هذا |
|----------|----------|
| 16 طبقة مزيفة | 3 طبقات شغالة حقيقياً |
| 4 أنظمة تدريب معطلة | نظام واحد يشتغل 24/7 |
| 16 حكيم بـ mock consensus | 5 حكماء بتصويت حقيقي |
| 100+ worker | جهاز واحد 100% أول |
| نظام تشغيل خاص (BI-OS) | Linux مُحسّن أول |
| 112,000 سطر ميتة | 10,000 سطر حيّة |

## 18.2 ترتيب الأولويات المعدّل

> **الأولويات حسب: "لو انقطع النت باجر، شنو أندم عليه؟"**

| الأولوية | المهمة | سبب الأولوية |
|----------|--------|---------------|
| 🔴 P0 | تنزيل نماذج + بيانات offline | لا يمكن تعويضها بعد انقطاع النت |
| 🔴 P0 | Knowledge Distillation يومي | كل يوم بدون = بيانات ضائعة |
| 🔴 P0 | RAG + نموذج محلي شغال | أساس كل شي |
| 🟡 P1 | تشغيل Training Pipeline حقيقي | تحسين مستمر |
| 🟡 P1 | Data Flywheel | بقاء النظام ذكي offline |
| 🟡 P1 | PostgreSQL + إزالة mocks | استقرار |
| 🟢 P2 | المجلس 24/7 | مهم لكن ما يموت بدونه |
| 🟢 P2 | البرمجة الأوتوماتيكية | مهم لكن يحتاج أساس أول |
| ⚪ P3 | طبقة الحياة الواقعية | يحتاج وقت طويل |
| ⚪ P3 | الدماغ المتقدم (ToT/MCTS) | يحتاج أساس |
| ⚫ P4 | نظام تشغيل خاص | بعيد المدى جداً |
| ⚫ P4 | 100+ worker | توسع بعد ثبات |

## 18.3 جداول زمنية واقعية (معدّلة)

> **بدل "4 أسابيع للكل" — هذي الجداول الواقعية:**

| المرحلة | المدة الواقعية | المخرج |
|---------|---------------|--------|
| تنزيل نماذج + بيانات | أسبوع 1 | نماذج + Wikipedia + أبحاث محلياً |
| RAG شغال + نموذج محلي | أسبوع 2-3 | يجاوب من بياناتك offline |
| Knowledge Distillation | أسبوع 1-8 (مستمر يومياً) | 80,000+ سؤال/جواب عالي الجودة |
| تشغيل Training Pipeline | أسبوع 3-4 | تدريب LoRA على البيانات المحلية |
| PostgreSQL + إزالة mocks | أسبوع 4-5 | بيانات حقيقية مستمرة |
| المجلس + التصويت الحقيقي | أسبوع 5-6 | مجلس شغال أوتوماتيكي |
| Data Flywheel | أسبوع 6-7 | يتحسن من كل استخدام |
| LoRA fine-tune كامل | أسبوع 7-10 | نموذج محلي مخصص |
| Tree of Thought + RAG متقدم | أسبوع 10-12 | دماغ يفكر بعمق |
| طبقة الحياة الواقعية | شهر 3-4 | محاكاة مصانع |
| البرمجة الأوتوماتيكية | شهر 4-5 | أمر → برنامج |
| دماغ متقدم (Bayesian/MCTS) | شهر 5-8 | ذكاء عالي |
| العتاد الموزّع (100+ worker) | شهر 6-12 | توسع حقيقي |
| نظام تشغيل خاص (BI-OS) | سنة+ | هدف بعيد |

## 18.4 قاعدة "الشي الواحد الشغال" (One Working Thing Rule)

> **قبل الانتقال لأي مرحلة جديدة — لازم المرحلة الحالية شغالة 100%:**

- [ ] فحص يومي: "هل الشي اللي سويته أمس لسه شغال؟"
- [ ] **ممنوع** بدء feature جديد لو feature سابق معطل
- [ ] كل feature جديد يحتاج: كود + اختبار + يشتغل فعلاً
- [ ] **الهدف: 10 أشياء شغالة حقيقياً > 100 شي مخطط**

## 18.5 المستخدم النهائي — من يستخدم هذا النظام؟

> **لازم نحدد: لمَن نبني؟**

| السيناريو | المستخدم | الاحتياجات |
|-----------|----------|------------|
| قبل الكارثة | أنت (الرئيس) | IDE + AI + تدريب + مجلس |
| بعد الكارثة | مجموعة ناجين | تعليم + إرشاد + تخطيط |
| بعد الكارثة | مهندسون | تصميم + بناء + محاكاة |
| بعد الكارثة | أطباء | معلومات طبية + تشخيص |
| بعد الكارثة | مزارعون | زراعة + ري + أسمدة |

**لكل مستخدم → واجهة مبسطة:**
- [ ] واجهة "المهندس": مصنع → خطة → محاكاة → بناء
- [ ] واجهة "الطبيب": أعراض → تشخيص → علاج
- [ ] واجهة "المزارع": تربة → محصول → جدول زراعة
- [ ] **الكل يمر عبر RAG + النموذج المحلي** (لا يحتاج واجهة معقدة)

---

# القسم ١٩ — أدوات التشغيل الفوري (Operational Tools) ⭐⭐

> **أدوات عملية تُبنى أولاً — بدونها ما تعرف حالة النظام ولا تقدر تتحقق**

## 19.1 Day 1 Demo — أول عرض حقيقي ⭐⭐⭐ (قبل كل شي!)

> ⚠️ **112,000 سطر كود + خطة ضخمة = لازم demo حقيقي يثبت أن المشروع شغال.**
> هذا يُنفذ خلال 48 ساعة — قبل أي شي آخر.

```
المطلوب:
1. نموذج محلي (Mistral 7B أو Llama 70B) يشتغل على RTX 5090
2. RAG يبحث بملفات محلية (حتى لو Wikipedia بسيطة)
3. تسأله: "كيف أبني فرن؟" ← يجاوب من البيانات المحلية
4. كل شي offline — بدون إنترنت

لماذا هذا أول:
  - يثبت أن المشروع حقيقي ← يعطيك دافع
  - يكشف المشاكل الحقيقية (مو النظرية)
  - يوجّه الأولويات: "شنو ما يشتغل فعلاً؟"
```

**المطلوب:**
- [ ] تنزيل نموذج GGUF واحد على الأقل
- [ ] تشغيله محلياً بـ llama.cpp أو vLLM
- [ ] ربط RAG بسيط (حتى لو على 100 وثيقة)
- [ ] اختبار offline كامل (قطع النت → اسأل → يجاوب)
- [ ] **Deadline: 48 ساعة من الآن**

## 19.2 Health Dashboard — لوحة الحالة الفورية ⭐⭐⭐

> **بدون dashboard = تشتغل أعمى — لازم تعرف حالة كل شي بنظرة**

```
صفحة واحدة بسيطة تعرض:
  ✅/❌ نموذج محلي شغال
  ✅/❌ RAG شغال
  ✅/❌ تدريب شغال
  ✅/❌ المجلس شغال
  ✅/❌ PostgreSQL شغال
  ✅/❌ كل الـ workers متصلين
  GPU utilization: XX%
  آخر checkpoint: قبل XX ساعة
  Offline Readiness Score: XX/100
```

**المطلوب:**
- [ ] بناء `monitoring/health_dashboard.py` — endpoint واحد يعرض كل الحالات
- [ ] صفحة HTML بسيطة ← تُفتح بالمتصفح
- [ ] تحديث كل 30 ثانية (auto-refresh)
- [ ] تنبيه بصري لو أي شي تعطل
- [ ] ربطه بـ monitoring/ الموجود (metrics_exporter + alert_manager)

## 19.3 Dependency Map — خريطة الاعتماديات ⭐⭐

> **18 قسم و 33+ ملف — لازم dependency graph واضح يبين شنو يعتمد على شنو**

```
مثال: لتشغيل المجلس الأوتوماتيكي (§3.2):
  المجلس ← يحتاج AI Service حقيقي (مو mock)
    ← يحتاج نموذج محلي
      ← يحتاج RAG
        ← يحتاج Vector DB
          ← يحتاج بيانات

هذا Chain of Dependencies لازم يكون مرسوم:
  - شنو يعتمد على شنو؟
  - شنو ممكن يشتغل بالتوازي؟
  - شنو يسد الطريق؟ (critical path)
```

**المطلوب:**
- [ ] بناء `docs/DEPENDENCY_MAP.md` — رسم DAG لكل الأنظمة
- [ ] تحديد Critical Path: أطول سلسلة اعتماديات
- [ ] تحديد Quick Wins: أشياء تقدر تسويها بالتوازي
- [ ] تحديث أسبوعي مع تقدم التنفيذ

## 19.4 Canary Test — اختبار الكناري اليومي ⭐

> **سؤال واحد بسيط كل يوم — يكشف 90% من المشاكل**

```
كل يوم الساعة 6 صباحاً أوتوماتيكياً:
1. اسأل النموذج: "ما هي درجة انصهار الحديد؟"
2. الجواب لازم يكون ~1538°C (± 50°C)
3. إذا غلط أو ما جاوب → 🔴 تنبيه فوري

شنو يكشف:
  - نموذج تعطل؟ ← ما يجاوب
  - تدريب خرب النموذج؟ ← جواب غلط
  - RAG مكسور؟ ← ما يلاقي الوثيقة
  - GPU معطل؟ ← timeout

بعد ما يثبت: يضيف 10 أسئلة (كل مجال سؤال)
```

**المطلوب:**
- [ ] بناء `scripts/canary_test.py`
- [ ] 10 أسئلة ثابتة بأجوبة معروفة (فيزياء + كيمياء + رياضيات + برمجة)
- [ ] cron job يومي أو عند كل startup
- [ ] تنبيه فوري لو فشل أي سؤال
- [ ] ربطه بـ Health Dashboard (19.2)


