# ✅ قائمة الأعمال المتبقية - BI-IDE v8
## بناءً على Implementation Plan V2

> **Database Source of Truth:** التنفيذ الرسمي للقاعدة يعتمد `core/database.py` + `alembic/`.
> أي ملفات داخل `database/` تعتبر artifacts مساعدة/مرجعية ما لم يتم اعتماد دمجها رسمياً.

---

## 🔴 المرحلة 0: Contract & Consistency Freeze (يوم 0.5)

### الملفات الجديدة المطلوبة:

- [ ] `docs/api_contracts_v2.md`
  - [ ] توثيق `/api/v1/council/message`
  - [ ] توثيق `/api/v1/council/status`
  - [ ] توثيق orchestrator job APIs
  - [ ] توحيد response schema

- [ ] `database/schema.sql`
  - [ ] جدول `users`
  - [ ] جدول `council_members`
  - [ ] جدول `council_decisions`
  - [ ] جدول `training_jobs`
  - [ ] جدول `workers`

- [ ] `database/models.py`
  - [ ] SQLAlchemy ORM models
  - [ ] Relationships
  - [ ] Indexes

- [ ] `database/connection.py`
  - [ ] Connection pool
  - [ ] Session management
  - [ ] Async support

- [ ] `apps/desktop-tauri/src/config/api.ts`
  - [ ] API_URL central config
  - [ ] Environment handling

### الملفات المطلوب تعديلها:

- [ ] `api/routes/council.py` (Line 202-204)
  ```python
  # من: 192.168.68.125:8080
  # إلى: 192.168.1.164:8090
  ```

- [ ] `hierarchy/__init__.py` (Line 412-413)
  ```python
  # توحيد مع القيمة الصحيحة
  rtx_host = os.getenv("RTX4090_HOST", "192.168.1.164")
  rtx_port = os.getenv("RTX4090_PORT", "8090")
  ```

---

## 🔴 المرحلة 1: Chat Surface Unification (يوم 1)

### الملفات المطلوب تعديلها:

- [ ] `apps/desktop-tauri/src/components/chat/AIChat.tsx`
  - [ ] إزالة Line 306: `const response = responses[Math.floor(Math.random() * responses.length)];`
  - [ ] إزالة Line 310: `Math.random() * 100`
  - [ ] إضافة استدعاء حقيقي لـ `/api/v1/council/message`
  - [ ] إضافة error handling
  - [ ] إضافة retry logic
  - [ ] إضافة offline state

- [ ] `apps/desktop-tauri/src/lib/tauri.ts` (تعديل)
  - [ ] توحيد API base handler
  - [ ] دالة موحدة للاستدعاءات
  - [ ] Interceptors للتوكن

---

## 🔴 المرحلة 2: Council Intelligence Hardening (يوم 2-3)

### الملفات المطلوب تعديلها:

- [ ] `hierarchy/__init__.py`
  - [ ] Line 191-200: إزالة mock consensus
    ```python
    # حذف كامل:
    consensus = {
        '_warning': 'MOCK DATA...',
        'consensus': 0.75,  # حذف
        ...
    }
    ```
  - [ ] إضافة استدعاء حقيقي لـ HighCouncil deliberation
  - [ ] إضافة provider fallback order:
    1. RTX endpoint
    2. Provider fallback
    3. Local heuristic (آخر مرحلة فقط)

- [ ] `hierarchy/high_council.py`
  - [ ] Line 90-91: إصلاح ID المكرر `S002`
    ```python
    # تغيير:
    SageRole.ETHICS: Sage("S003", "حكيم الأخلاق", SageRole.ETHICS),
    ```
  - [ ] Line 210-218: إصلاح `get_status()`
    ```python
    # استخدام getattr بشكل آمن
    def get_status(self) -> dict:
        return {
            'is_meeting': getattr(self, 'meeting_active', True),
            'wise_men_count': len(getattr(self, 'sages', {})),
            'meeting_status': 'continuous',
            'president_present': getattr(self, 'president_present', False),
            'topics_discussed': len(getattr(self, 'discussion_history', []))
        }
    ```

- [ ] `services/council_service.py`
  - [ ] Line 325-359: استبدال `_simulate_council_discussion`
    - [ ] محرك آراء weighted حقيقي
    - [ ] Confidence calculation حقيقي
    - [ ] لا `asyncio.sleep(0.1)` وهمي

---

## 🔴 المرحلة 3: Security & Data Integrity (يوم 4)

### الملفات الجديدة المطلوبة:

- [ ] `data/pipeline/__init__.py`
- [ ] `data/pipeline/data_cleaner.py`
  - [ ] دالة `clean_dataset()`
  - [ ] إزالة duplicates
  - [ ] معالجة missing values
  
- [ ] `data/pipeline/data_validator.py`
  - [ ] دالة `validate_dataset()`
  - [ ] Schema validation
  - [ ] Quality checks

### الملفات المطلوب تعديلها:

- [ ] `hierarchy/auto_learning_system.py`
  - [ ] Line 35-37: إعادة SSL verification
    ```python
    # من:
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # إلى:
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_default_certs()
    ```

---

## 🔴 المرحلة 4: PostgreSQL Migration (يوم 5)

### الملفات المطلوب تعديلها:

- [ ] `api/routers/auth.py`
  - [ ] إزالة `fake_users_db` (Line 81-82)
  - [ ] إضافة PostgreSQL integration
  - [ ] استخدام SQLAlchemy models
  - [ ] Feature flag: `USE_FAKE_AUTH = False`

- [ ] `api/routers/council.py`
  - [ ] إزالة fake data
  - [ ] PostgreSQL للـ members/decisions
  - [ ] Feature flag: `USE_FAKE_COUNCIL = False`

- [ ] `api/routers/training.py`
  - [ ] إزالة `fake_devices` (Line 128+)
  - [ ] إزالة `fake_models`
  - [ ] إزالة `fake_jobs`
  - [ ] PostgreSQL للـ training data
  - [ ] Feature flag: `USE_FAKE_TRAINING = False`

- [ ] `api/routers/monitoring.py`
  - [ ] إزالة `fake_workers` (Line 126+)
  - [ ] إزالة `fake_alerts`
  - [ ] PostgreSQL للـ monitoring data
  - [ ] Feature flag: `USE_FAKE_MONITORING = False`

---

## 🔴 المرحلة 5: Brain MVP + Auto-Scheduling (يوم 6-7)

### الملفات الجديدة المطلوبة:

- [ ] `brain/__init__.py`
- [ ] `brain/config.py`
  - [ ] إعدادات الجدولة
  - [ ] Thresholds
  
- [ ] `brain/scheduler.py`
  - [ ] `class JobScheduler`
  - [ ] دالة `schedule_job()`
  - [ ] دالة `get_priorities()`
  - [ ] Priority queue

- [ ] `brain/evaluator.py`
  - [ ] `class ModelEvaluator`
  - [ ] دالة `evaluate_model()`
  - [ ] دالة `should_deploy()` (minimum delta gate)
  - [ ] Benchmark set

- [ ] `brain/bi_brain.py`
  - [ ] `class BIBrain`
  - [ ] تكامل Scheduler + Evaluator
  - [ ] Dry-run cycle

### الملفات المطلوب تعديلها:

- [ ] `orchestrator_api.py`
  - [ ] إضافة endpoint: `/api/v1/orchestrator/auto-schedule`
  - [ ] إضافة endpoint: `/api/v1/orchestrator/brain/status`

- [ ] `worker/bi_worker.py`
  - [ ] إضافة idle training logic
    ```python
    async def idle_training_loop(self):
        """Training when no jobs available"""
        while self.running:
            if not self.current_job and self.is_idle():
                await self.run_idle_training()
            await asyncio.sleep(60)
    ```

- [ ] `services/training_service.py`
  - [ ] إضافة evaluation gate قبل auto-deploy
  - [ ] دالة `can_deploy_model()` (minimum 2% improvement)

---

## 🔴 المرحلة 6: AI Services De-Mock (يوم 8)

### الملفات المطلوب تعديلها:

- [ ] `services/ai_service.py`
  - [ ] استبدال `_mock_generate_code` (Line 408-434)
    - [ ] استدعاء RTX 4090 API
    - [ ] Provider fallback
    - [ ] Caching
    
  - [ ] استبدال `_mock_complete_code` (Line 436-443)
    - [ ] استدعاء AI endpoint حقيقي
    
  - [ ] استبدال `_mock_explain_code` (Line 445-458)
    - [ ] استدعاء AI endpoint حقيقي
    
  - [ ] استبدال `_mock_review_code` (Line 460-480)
    - [ ] استدعاء AI endpoint حقيقي
    - [ ] Static analysis integration

  - [ ] إضافة provider interface:
    ```python
    class AIProvider(ABC):
        @abstractmethod
        async def generate(self, prompt: str) -> str:
            pass
    
    class RTXProvider(AIProvider): ...
    class LocalProvider(AIProvider): ...
    ```

---

## 🔴 المرحلة 7: Deploy & Observability (يوم 9-10)

### الملفات الجديدة المطلوبة:

- [ ] `deploy/systemd/bi-brain.service`
  ```ini
  [Unit]
  Description=BI-IDE Brain Service
  After=network.target postgresql.service
  
  [Service]
  Type=simple
  User=bi
  WorkingDirectory=/opt/bi-ide
  ExecStart=/opt/bi-ide/venv/bin/python -m brain.bi_brain
  Restart=always
  RestartSec=5
  
  [Install]
  WantedBy=multi-user.target
  ```

### الملفات المطلوب تعديلها:

- [ ] `deploy/` scripts
  - [ ] تثبيت bi-brain.service
  - [ ] إعداد PostgreSQL
  - [ ] إعداد backups

- [ ] Monitoring dashboard
  - [ ] fallback_rate_pct
  - [ ] p95_council_latency
  - [ ] training_success_ratio
  - [ ] council_decision_confidence

---

## 📊 ملخص الأعمال المتبقية

| المرحلة | الملفات الجديدة | الملفات المعدلة | الوقت المقدر |
|---------|-----------------|-----------------|--------------|
| Phase 0 | 6 | 2 | 0.5 يوم |
| Phase 1 | 1 | 1 | 1 يوم |
| Phase 2 | 0 | 3 | 2 أيام |
| Phase 3 | 3 | 1 | 1 يوم |
| Phase 4 | 0 | 4 | 1 يوم |
| Phase 5 | 4 | 3 | 2 أيام |
| Phase 6 | 0 | 1 | 1 يوم |
| Phase 7 | 1 | 2 | 2 أيام |
| **المجموع** | **15** | **17** | **10.5 أيام** |

---

## 🎯 أولويات التنفيذ (ما يجب البدء به فوراً)

### الأسبوع الأول:
1. **اليوم 1:** Phase 0 (Contracts + PostgreSQL setup)
2. **اليوم 2:** Phase 1 (Chat Unification)
3. **اليوم 3-4:** Phase 2 (Council Intelligence)
4. **اليوم 5:** Phase 3 (Security) + Phase 4 (PostgreSQL Migration)

### الأسبوع الثاني:
5. **اليوم 6-7:** Phase 5 (Brain MVP)
6. **اليوم 8:** Phase 6 (AI Services)
7. **اليوم 9-10:** Phase 7 (Deploy + Testing)

---

**آخر تحديث:** 2026-03-01  
**الحالة:** 🔄 جاري المراقبة
