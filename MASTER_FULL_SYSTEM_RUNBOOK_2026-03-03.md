# BI-IDE v8 — MASTER FULL SYSTEM RUNBOOK

**Date:** 2026-03-03  
**Goal:** تشغيل المشروع بالكامل (AI + ERP + UI + IDE + Training) بدون ضياع بيانات، وبربط فعلي بين كل الطبقات والأجهزة.

---

## 1) الهدف التشغيلي النهائي (Definition of Done)

النظام يعتبر **Ready** فقط إذا تحقق التالي معًا:

1. API + ERP + UI + IDE شغالين بإنتاجية.
2. طبقات AI (الكشافة + الحكماء + مجلس الحكماء + التنفيذ) شغالة عبر endpoints الفعلية.
3. كل المحادثات/القرارات محفوظة دائمًا في PostgreSQL.
4. التدريب الفعلي يتم على جهاز RTX 5090 فقط.
5. نقل البيانات من Windows + Hostinger إلى 5090 مؤتمت ومراقب.
6. لا يوجد جزء “مقطوع” بين الموقع وIDE وAI.

---

## 2) طوبولوجيا التشغيل المعتمدة (Current Target Topology)

### A) Hostinger VPS (واجهة + API + ERP + DB/Queue)
- Nginx + FastAPI + Workers + PostgreSQL + Redis
- Domain production: `app.bi-iq.com`
- File references:
  - `deploy_hostinger.sh`
  - `docker-compose.prod.yml`
  - `DEPLOY.md`

### B) RTX 5090 Machine (Training + AI heavy compute)
- Training server على `0.0.0.0:8080`
- File references:
  - `start_training_ubuntu.sh`
  - `rtx4090_server.py`
  - `RTX4090_SETUP.md`

### C) Windows Nodes (IDE/dev + feeders)
- UI/IDE usage + preprocessing + push data/checkpoints إلى 5090
- File references:
  - `sync_updates_to_ubuntu.ps1`
  - `start_with_rtx4090.bat`

---

## 3) قاعدة عدم ضياع البيانات (Zero-Loss Policy)

1. PostgreSQL هو المصدر الوحيد للحقيقة (Single Source of Truth) للمحادثات والقرارات والـ ERP.
2. أي دردشة مجلس/AI يجب أن تُكتب transactionally في DB قبل التأكيد للمستخدم.
3. ممنوع تشغيل production بدون:
   - volume دائم لقاعدة البيانات
   - نسخ احتياطي يومي
   - اختبار restore ناجح
4. أي تدريب/checkpoints يجب أن يكتب بمسار versioned timestamped.

---

## 4) خطة تشغيل موحّدة (Execution Order)

## PHASE 0 — Preflight (إجباري)

نفّذ على بيئة المشروع:

```bash
python3 -V
docker --version
docker compose version
python3 -m pytest tests/e2e/test_full_workflow.py::TestCompleteBusinessWorkflow::test_business_workflow_e2e -vv --tb=short
```

معيار النجاح:
- أدوات التشغيل موجودة.
- E2E critical workflow ينجح.

---

## PHASE 1 — Database Foundation (PostgreSQL + Migrations)

1. تشغيل PostgreSQL وRedis (prod stack).
2. تشغيل migrations.
3. إنشاء admin افتراضي.

```bash
docker compose -f docker-compose.prod.yml up -d postgres redis
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
docker compose -f docker-compose.prod.yml exec api python scripts/create_default_admin.py
```

معيار النجاح:
- `/health` = 200
- `/ready` = 200
- login admin ناجح

---

## PHASE 2 — API + ERP + UI + IDE Integration

1. تشغيل API + worker + nginx + ui dist.
2. التحقق من endpoints الأساسية:
   - `/api/v1/erp/dashboard`
   - `/api/v1/council/status`
   - `/api/v1/wisdom`
   - `/api/v1/guardian/status`

```bash
docker compose -f docker-compose.prod.yml up -d api worker nginx
```

معيار النجاح:
- ERP endpoints ترجع بنية صحيحة.
- Council/Hierarchy endpoints تعمل بدون 404/401 غير متوقعة.
- UI يفتح من الدومين ويربط API.

---

## PHASE 3 — RTX 5090 Training Node (Compute Layer)

على جهاز 5090:

```bash
cd ~/bi-ide-v8
chmod +x start_training_ubuntu.sh
./start_training_ubuntu.sh
```

تحقق من health:

```bash
curl http://127.0.0.1:8080/status
```

معيار النجاح:
- training server up
- استجابة status صحيحة

---

## PHASE 4 — Auto Data Flow (Windows + Hostinger → 5090)

### 4.1 نقل من Windows إلى 5090 (existing)

```powershell
./sync_updates_to_ubuntu.ps1 -UbuntuUser bi -UbuntuHost 192.168.68.125 -RemoteProjectPath ~/bi-ide-v8
```

### 4.2 نقل بيانات التدريب/الـ checkpoints (mandatory schedule)

اعتمد rsync/scp مجدول لكل من:
- `learning_data/`
- `models/`
- `training/`

صيغة تشغيل مقترحة (من كل feeder node):

```bash
rsync -az --delete /path/to/bi-ide-v8/learning_data/ bi@192.168.68.125:~/bi-ide-v8/learning_data/
rsync -az --delete /path/to/bi-ide-v8/models/ bi@192.168.68.125:~/bi-ide-v8/models/
rsync -az --delete /path/to/bi-ide-v8/training/ bi@192.168.68.125:~/bi-ide-v8/training/
```

مطلوب جدولة (Task Scheduler/Cron) كل 5-15 دقيقة حسب الحمل.

---

## PHASE 5 — ربط AI Layers بالكامل

يجب التحقق من كل طبقة:

1. Scouts (gathering)
2. High Council + Wise Men
3. Guardian
4. Execution layer
5. Persistent chat/decision logging

Checks:

```bash
python3 -m pytest tests/e2e/test_full_workflow.py::TestAIIntegrationWorkflow::test_council_workflow -vv --tb=short
python3 -m pytest tests/e2e/test_full_workflow.py::TestAIIntegrationWorkflow::test_hierarchy_workflow -vv --tb=short
```

معيار النجاح:
- workflow AI ينجح end-to-end
- metrics/history/status endpoints كلها تعمل

---

## PHASE 6 — Full Regression Gate

```bash
python3 _run_tests.py
```

لا ننتقل للإنتاج النهائي إلا عند:
- لا يوجد Failures blocking
- أي warnings معروفة موثقة بخطة إزالة

---

## 5) خطة مراجعة الكود “حرفيًا” (File/Class/Module Audit)

هذه خطة مراجعة كاملة بدون فقدان أي جزء:

### Batch A (Core Runtime)
- `api/`
- `core/`
- `database/`

### Batch B (AI Brain)
- `ai/`
- `hierarchy/`
- `brain/`

### Batch C (Business + ERP)
- `erp/`
- `services/`

### Batch D (UX + Desktop)
- `ui/`
- `apps/desktop-tauri/`
- `ide/`

### Batch E (Ops + Infra)
- `deploy/`
- `docker-compose*.yml`
- `scripts/`

لكل Batch لازم ينفذ:
1. Static diagnostics
2. Targeted tests
3. Integration tests
4. Risk log
5. Fix + re-verify

---

## 6) ربط الموقع مع IDE (No Missing Link)

1. Frontend routes لازم تتصل بنفس API gateway المعتمد (v1/v2 mapping).
2. IDE APIs لازم تكون تحت نفس auth/session strategy.
3. كل operation داخل IDE ينتج:
   - event log
   - DB record (عند الحاجة)
   - trace id للمراقبة

اختبار إلزامي:
- فتح UI
- تنفيذ action من IDE
- تحقق نتيجة في API + DB + logs

---

## 7) تخزين دائم لمحادثة المجلس والحكماء

متطلبات إلزامية:

1. كل message/decision يتخزن في PostgreSQL.
2. وجود endpoint history يعتمد DB وليس ذاكرة مؤقتة.
3. backup يومي + retention policy (7/30/90).

فحص:
- بعد restart كامل للخدمات، history تبقى متاحة بنفس البيانات.

---

## 8) مصفوفة التشغيل على الأجهزة المتصلة (Windows + 5090 + Server)

### Hostinger VPS
- API/ERP/UI/DB/Redis/Nginx
- لا تدريب ثقيل

### RTX 5090
- كل التدريب والاستدلال الثقيل
- checkpoints الرئيسية

### Windows Nodes
- IDE/عمليات dev
- preprocessing/feeders
- مزامنة دورية إلى 5090

---

## 9) أوامر التشغيل القياسية النهائية (Operational Core)

### VPS
```bash
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### 5090
```bash
./start_training_ubuntu.sh
```

### Windows feeder
```powershell
./sync_updates_to_ubuntu.ps1
```

### Validation
```bash
python3 -m pytest tests/e2e/test_full_workflow.py::TestCompleteBusinessWorkflow::test_business_workflow_e2e -vv --tb=short
python3 _run_tests.py
```

---

## 10) خطة الطوارئ (Rollback + Recovery)

1. Snapshot DB قبل كل release.
2. Snapshot learning_data/models قبل migration تدريب.
3. لو فشل deployment:
   - rollback containers للصورة السابقة
   - restore DB snapshot
   - re-run smoke tests

---

## 11) ملاحظات واقعية مهمة

1. عبارة “بدون أي خطأ حرفيًا” تتحقق عمليًا عبر Gates أعلاه، مو بالوصف فقط.
2. أي جزء non-deterministic (network/GPU/external APIs) لازم يبقى تحت monitoring + retries + fallback.
3. هذا الملف هو المرجع التنفيذي الموحد، وأي خطوة تشغيل خارج هذا الملف تعتبر خارج الضمان التشغيلي.

---

## 12) القرار التنفيذي

ابدأ التشغيل من **PHASE 0** إلى **PHASE 6** بالتسلسل، ولا تنتقل لمرحلة قبل تحقيق معيار النجاح للمرحلة الحالية.

إذا تريد، الخطوة التالية أقدر أحول هذا الـ runbook إلى **سكريبت orchestrator واحد** (تنفيذي) باسم:

`scripts/run_full_system_orchestrator.sh`

حتى يصير عندك تنفيذ شبه-أوتوماتيكي بدل التنفيذ اليدوي.
---

## 13) MASTER ORCHESTRATOR - التنفيذ الكامل (2026-03-03)

### 🎯 الهدف
تشغيل **كل** طبقات المشروع بنقرة واحدة:
- ✅ Training System (RTX 5090)
- ✅ AI Layers (Council + Scouts + Execution)
- ✅ Unified UI (Dashboard + IDE + Training)
- ✅ PostgreSQL (Chat storage)
- ✅ Data Sync (Windows → RTX 5090)
- ✅ Resource Monitoring

### 📋 المتطلبات المسبقة
```bash
# From your Mac/Control machine
# 1. SSH access to RTX 5090
ssh bi@192.168.1.164

# 2. Project exists at /home/bi/bi-ide-v8
ls /home/bi/bi-ide-v8

# 3. GPU drivers installed
nvidia-smi
```

### 🚀 التشغيل (خطوة واحدة)

```bash
# على RTX 5090 (192.168.1.164)
ssh bi@192.168.1.164
cd ~

# تشغيل الأوركستر
./MASTER_ORCHESTRATOR.sh
```

### 📁 ما يسويه الأوركستر

| المرحلة | الوصف | المدة |
|---------|-------|-------|
| **PHASE 0** | فحص RTX 5090 + GPU + Disk | 10 ثواني |
| **PHASE 1** | مراجعة ملفات الكود الرئيسية | 5 ثواني |
| **PHASE 2** | إعداد PostgreSQL + المجلدات | 10 ثواني |
| **PHASE 3** | تشغيل نظام التدريب 24/7 | 15 ثانية |
| **PHASE 4** | تفعيل طبقات AI | 5 ثواني |
| **PHASE 5** | تشغيل الواجهة الموحدة port 8080 | 10 ثواني |
| **PHASE 6** | إعداد مزامنة البيانات | 5 ثواني |
| **PHASE 7** | تفعيل المراقبة | 5 ثواني |
| **PHASE 8** | التحقق النهائي من كل شي | 10 ثواني |

### 🌐 روابط الوصول بعد التشغيل

| الخدمة | الرابط | الوصف |
|--------|--------|-------|
| **Dashboard** | http://192.168.1.164:8080/ | GPU + Training + Disk stats |
| **Training** | http://192.168.1.164:8080/training | سجلات التدريب المباشرة |
| **IDE** | http://192.168.1.164:8080/ide | محرر Python يتنفذ على RTX 5090 |
| **Logs** | http://192.168.1.164:8080/logs | سجلات النظام |
| **Files Index** | http://192.168.1.164:8080/static/docs/files-index.html | فهرس الملفات |

### 🖥️ لوحة التحكم الموحدة

بعد التشغيل، الـ Dashboard يعرض:
- **GPU Usage**: الاستخدام الفعلي لـ RTX 5090
- **Temperature**: درجة الحرارة مع تنبيه إذا عالية
- **Training Status**: حالة التدريب (يعمل/متوقف)
- **Disk Space**: المساحة المتبقية مع تحذير إذا امتلأت
- **Active Batches**: عدد الباتشات في queue

### 📊 مراقبة الموارد لكل جهاز

يمكن التحكم باستهلاك الموارد:

```bash
# تعديل استهلاك RTX 5090 (بداية الملف)
RTX5090_GPU_PERCENT=100
RTX5090_CPU_PERCENT=100

# تعديل استهلاك Windows
WINDOWS_GPU_PERCENT=90
WINDOWS_CPU_PERCENT=80

# تعديل استهلاك VPS
VPS_CPU_PERCENT=60
```

### 🔁 مزامنة البيانات التلقائية

**من Windows إلى RTX 5090:**
```powershell
# على Windows (192.168.1.130)
.\sync_updates_to_ubuntu.ps1
```

**من Hostinger VPS إلى RTX 5090:**
```bash
# على VPS
/tmp/push_to_rtx.sh
```

**استلام البيانات على RTX 5090:**
- المسار: `/home/bi/incoming_data/`
- يتم نقلها تلقائياً لـ `/home/bi/data_pipeline/downloading/`
- التدريب يبدأ فوراً

### 🧠 طبقات AI الشغالة

1. **President (الرئيس)**: أعلى مستوى قرار
2. **High Council (المجلس الأعلى)**: 16 حكيم
3. **Execution Team (فريق التنفيذ)**: تنفيذ المهام
4. **Scouts (الكشافة)**: جمع المعلومات
5. **Guardian Layer (طبقة الحماية)**: مراجعة السلامة
6. **Real Training System**: التدريب الحقيقي

**تخزين المحادثات:**
- PostgreSQL: `council_discussions` table
- File backup: `/home/bi/chat_history/`

### 💾 المواقع المهمة على RTX 5090

| المسار | المحتوى |
|--------|---------|
| `/home/bi/data_pipeline/` | بيانات التدريب |
| `/home/bi/checkpoints/` | نقاط الحفظ |
| `/home/bi/chat_history/` | محادثات AI |
| `/tmp/infinite_training_master.log` | سجلات التدريب |
| `/tmp/master_orchestrator_*.log` | سجلات الأوركستر |

### 🆘 استكشاف الأخطاء

**المشكلة: Training not starting**
```bash
# التحقق من GPU
nvidia-smi

# إعادة تشغيل التدريب
pkill -f master_training
~/MASTER_ORCHESTRATOR.sh
```

**المشكلة: UI not accessible**
```bash
# التحقق من port
netstat -tlnp | grep 8080

# إعادة تشغيل UI
cd ~/unified-ui
python3 app.py
```

**المشكلة: Disk full**
```bash
# تنظيف البيانات القديمة
rm -rf /home/bi/data_pipeline/completed/*
rm -rf /home/bi/checkpoints/ol_*
```

### 📈 مراحل التحقق (Validation Gates)

بعد تشغيل الأوركستر، تحقق:

```bash
# 1. GPU شغال
ssh bi@192.168.1.164 "nvidia-smi"

# 2. Training شغال
ssh bi@192.168.1.164 "pgrep -f master_training"

# 3. UI شغال
curl http://192.168.1.164:8080/api/status

# 4. سجلات التدريب
ssh bi@192.168.1.164 "tail -20 /tmp/infinite_training_master.log"
```

### ✅ معايير النجاح

المشروع يعتبر **شغال بالكامل** إذا:
- ✅ Training system يشتغل 24/7 بدون errors
- ✅ UI يعرض GPU stats في real-time
- ✅ IDE يشغل Python code على RTX 5090
- ✅ AI layers تستجيب للـ API calls
- ✅ PostgreSQL يخزن المحادثات
- ✅ لا يوجد disk space warnings
- ✅ GPU temp تحت 85°C

### 🔄 التحديث المستقبلي

لإضافة أي نظام جديد:
1. أضف PHASE جديدة للـ `MASTER_ORCHESTRATOR.sh`
2. حدث هذا الـ runbook
3. اختبر على RTX 5090
4. نفذ `./MASTER_ORCHESTRATOR.sh`

---

**الملفات المنشأة:**
- `~/MASTER_ORCHESTRATOR.sh` (على RTX 5090)
- `~/unified-ui/app.py` (الواجهة الموحدة)
- `/tmp/master_training.sh` (التدريب 24/7)

**تاريخ التحديث:** 2026-03-03  
**الحالة:** ✅ جاهز للتنفيذ
