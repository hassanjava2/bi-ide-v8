# 🚀 BI-IDE — خطة التنفيذ والخوارزميات الشاملة
### من v8.1 إلى v10+ — الخارطة الكاملة

---

## 📋 الوحدات الأساسية الموجودة (v8)

### التسلسل الهرمي (`hierarchy/`)
| الملف | الوظيفة | الحالة |
|-------|---------|--------|
| `president.py` | الرئيس — المستخدم 24/7 | ✅ |
| `high_council.py` | المجلس الأعلى — 16 حكيم | ✅ |
| `shadow_light.py` | الظل والنور — 4 متشائمين + 4 متفائلين | ✅ |
| `scouts.py` | الكشافة — تقني، سوق، منافسين، فرص | ✅ |
| `meta_team.py` | الفريق الفوقي — أداء، جودة، تعلم، تطور | ✅ |
| `domain_experts.py` | خبراء التخصص — 11 مجال | ✅ |
| `execution_team.py` | فريق التنفيذ | ✅ |
| `meta_architect.py` | المعماري الفوقي | ✅ |
| `seventh_dimension.py` | البعد السابع | ✅ |
| `cosmic_bridge.py` | الجسر الكوني | ✅ |
| `gpu_trainer.py` | مدرب GPU | ✅ |
| `massive_training.py` | التدريب الضخم | ✅ |
| `internet_auto_training.py` | التدريب الآلي من الإنترنت | ✅ |
| `auto_learning_system.py` | نظام التعلم التلقائي | ✅ |
| `real_training_system.py` | نظام التدريب الحقيقي | ✅ |
| `guardian_layer.py` | طبقة الحماية | ✅ |
| `security_layer.py` | طبقة الأمان | ✅ |
| `compliance_layer.py` | طبقة الامتثال | ✅ |
| `eternity_layer.py` | طبقة الخلود | ✅ |

### الذكاء الاصطناعي (`ai/`)
| الوحدة | الملفات | الحالة |
|--------|---------|--------|
| Tokenizer | `arabic_processor`, `bpe_tokenizer`, `code_tokenizer` | ✅ |
| Training | `data_collection`, `preprocessing`, `rtx4090_trainer`, `deployment`, `auto_evaluation` | ✅ |
| Optimization | `quantization`, `pruning`, `distillation`, `batch_inference`, `benchmark` | ✅ |
| Memory | `vector_db`, `context_awareness`, `conversation_history`, `user_preferences` | ✅ |
| LLM | `llm_client`, `rtx4090_client` | ✅ |

### البنية التحتية
| المكون | الحالة |
|--------|--------|
| Desktop App (Tauri) | ✅ `apps/desktop-tauri/` |
| Web UI | ✅ `ui/` |
| API Backend (FastAPI) | ✅ `api/` |
| ERP | ✅ `erp/` |
| Monitoring | ✅ `monitoring/` |
| Docker | ✅ `Dockerfile`, `docker-compose.yml` |
| Worker System | ✅ `worker/bi_worker.py` |
| RTX Server | ✅ `rtx4090_machine/` |
| Orchestrator | ✅ `orchestrator_api.py` |

---

## 🔧 خطة الإصلاح الشاملة لـ v8 الحالي

> **نتائج الفحص الكامل**: 9,852 سطر في hierarchy، 28 ملف AI، 14 endpoint في الأوركستريتر،
> 9 مكونات Desktop، 11 ملف اختبار، 7 مكونات مجتمع، 5 شبكة، 5 أمان.
> **أخطاء بناء**: 2 syntax errors. **مجلدات فارغة**: `services/`, `monitoring/` (ملف واحد فقط).

---

### 🔴 إصلاحات حرجة (يجب فوراً)

#### 1. أخطاء بناء (Syntax Errors)
| الملف | السطر | الخطأ | الحل |
|-------|-------|-------|------|
| `hierarchy/connect_services.py` | 19 | `SyntaxError: invalid syntax` | مراجعة وإصلاح التركيب |
| `security/security_audit.py` | 374 | `unexpected character after line continuation` | إصلاح رمز `\` الخاطئ |

```python
# الإجراء:
# 1. فتح كل ملف وتصحيح الخطأ
# 2. تشغيل python -m py_compile <file> للتأكد
# 3. تشغيل pytest tests/ للتحقق من عدم وجود تأثيرات جانبية
```

#### 2. GPU RTX 5090 — الدرايفر معطل
```
المشكلة: nvidia-smi يُظهر ERR! | torch.cuda.is_available() = False
Driver: 590.48.01 (غير مستقر مع RTX 5090)
CUDA: 13.1 (PyTorch لا يدعمها)

الحل:
  sudo apt purge nvidia-driver-590
  sudo apt install nvidia-driver-565   # مستقر + CUDA 12.8
  pip3 install torch --index-url https://download.pytorch.org/whl/cu128
  sudo reboot
```

#### 3. Worker يتوقف على Windows
```
المشكلة: العمليات تموت لأن SSH session يغلق
المشكلة الثانية: مسارات Linux (/home/bi/) لا تعمل على Windows

الحل 1 — Windows Service:
  nssm install bi-server "python.exe" "-X utf8 C:\Users\BI\rtx4090_server.py"
  nssm install bi-worker "python.exe" "-X utf8 C:\Users\BI\bi_worker.py --server..."
  nssm start bi-server
  nssm start bi-worker

الحل 2 — إصلاح المسارات في bi_worker.py:
  if platform.system() == "Windows":
      job_path = job_path.replace("/home/bi/", "C:\\Users\\BI\\")
```

---

### 🟡 إصلاحات مهمة (خلال أسبوع)

#### 4. API Backend — شبه فارغ
```
الوضع الحالي:
  api/app.py      → route "/" + catch-all "/{path:path}" فقط!
  api/auth.py     → موجود لكن غير مربوط بالتطبيق
  api/gateway.py  → موجود لكن غير مفعل
  api/rbac.py     → route واحد فقط (invoices + users)
  api/middleware.py → موجود لكن غير مفعل
  api/rate_limit.py → موجود لكن غير مفعل

المطلوب:
  ✅ ربط auth.py بـ app.py (تم) (login, register, refresh, logout)
  ✅ تفعيل middleware.py (تم) (CORS, logging, error handling)
  ✅ تفعيل rate_limit.py (تم) (حماية من DDoS)
  ✅ ربط gateway.py (تم) كـ API Gateway
  ✅ إضافة routes للمجلس (تم) (/council/*)
  ✅ إضافة routes للتدريب (تم) (/training/*)
  ✅ إضافة routes للذكاء (تم) (/ai/*)
  ✅ إضافة routes للمراقبة (تم) (/monitoring/*)
  ✅ ربط ERP routes (تم) (/erp/*)
```

#### 5. Desktop App (Tauri) — واجهات ناقصة
```
المكونات الموجودة (9):
  Sidebar.tsx          586 سطر ✅ الأكبر
  HierarchyPanel.tsx   210 سطر ✅
  Terminal.tsx          202 سطر ✅
  CouncilPanel.tsx      190 سطر ✅
  StatusBar.tsx         180 سطر ✅
  Editor.tsx           175 سطر ✅
  WelcomeScreen.tsx    155 سطر ✅
  Header.tsx           144 سطر ✅
  Layout.tsx           130 سطر ✅

المكونات الناقصة:
  ✅ TrainingDashboard.tsx (تم)   — لوحة مراقبة التدريب
  ✅ WorkerStatus.tsx (تم)        — حالة العمال (4 أجهزة)
  ✅ GPUMonitor.tsx (تم)          — استهلاك GPU مباشر
  ✅ AIChat.tsx (تم)              — محادثة مع الذكاء
  ✅ SettingsPanel.tsx (تم)       — إعدادات البرنامج
  ✅ FileExplorer.tsx (تم)        — مستكشف الملفات
  ✅ ProjectManager.tsx (تم)      — إدارة المشاريع
  ✅ ERPDashboard.tsx (تم)        — لوحة ERP
  ✅ UpdateNotification.tsx (تم)  — إشعارات التحديث
  ✅ NetworkStatus.tsx (تم)       — حالة الشبكة
```

#### 6. الأوركستريتر — endpoints ناقصة
```
الموجود (14 endpoint):
  ✅ GET  /health
  ✅ POST /workers/register
  ✅ POST /workers/heartbeat
  ✅ GET  /workers
  ✅ POST /workers/{id}/command
  ✅ DELETE /workers/{id}

الناقص:
  ✅ GET  /training/status       — حالة التدريب على كل الأجهزة
  ✅ POST /training/start-all    — بدء تدريب على كل الأجهزة
  ✅ POST /training/stop-all     — إيقاف التدريب
  ✅ GET  /training/metrics      — مقاييس الأداء
  ✅ POST /training/distribute   — توزيع البيانات
  ✅ GET  /models/list           — قائمة النماذج
  ✅ POST /models/deploy         — نشر نموذج
  ✅ GET  /system/resources      — موارد كل الأجهزة
  ✅ WS   /ws/realtime           — WebSocket للمراقبة الحية
  ✅ POST /council/query         — استعلام المجلس
  ✅ GET  /council/status        — حالة المجلس
```

---

### 🟠 إصلاحات متوسطة (خلال 2-3 أسابيع)

#### 7. Monitoring — شبه فارغ
```
الوضع:
  monitoring/analytics/event_tracker.py — ملف واحد فقط!

المطلوب:
  ✅ system_monitor.py (تم)    — CPU/GPU/RAM لكل الأجهزة
  ✅ training_monitor.py (تم)  — loss/accuracy/throughput
  ✅ alert_manager.py (تم)     — تنبيهات (GPU حرارة عالية، worker سقط)
  ✅ log_aggregator.py (تم)    — تجميع logs من كل الأجهزة
  ✅ health_dashboard.py  — صفحة صحة النظام
  ✅ metrics_exporter.py (تم)  — تصدير لـ Prometheus/Grafana
```

#### 8. Services — مجلد فارغ
```
المطلوب:
  ✅ training_service.py (تم)    — خدمة التدريب الموحدة
  ✅ council_service.py (تم)     — خدمة المجلس
  ✅ ai_service.py (تم)          — خدمة الذكاء (inference)
  ✅ notification_service.py (تم) — خدمة الإشعارات
  ✅ sync_service.py (تم)        — خدمة المزامنة بين الأجهزة
  ✅ backup_service.py (تم)      — خدمة النسخ الاحتياطي
```

#### 9. الاختبارات — ✅ مكتمل (20 ملف)
```
الموجود (20 ملف):
  ✅ test_api.py               — API الرئيسي
  ✅ test_auth_db_integration.py — المصادقة + قاعدة البيانات
  ✅ test_auth_e2e.py          — مصادقة end-to-end
  ✅ test_coverage.py          — تغطية الاختبارات
  ✅ test_erp_integration.py   — تكامل ERP
  ✅ test_hierarchy.py         — التسلسل الهرمي
  ✅ test_rate_limit.py        — حدود الطلبات
  ✅ test_rbac.py              — صلاحيات الأدوار
  ✅ test_services.py          — الخدمات
  ✅ test_training.py          — اختبار التدريب
  ✅ test_gpu_training.py      — اختبار GPU
  ✅ test_orchestrator.py      — اختبار الأوركستريتر
  ✅ test_worker.py            — اختبار العمال
  ✅ test_council.py           — اختبار المجلس
  ✅ test_ai_memory.py         — اختبار الذاكرة
  ✅ test_tokenizer.py         — اختبار Tokenizers
  ✅ test_network.py           — اختبار الشبكة
  ✅ test_security.py          — اختبار الأمان
  ✅ test_desktop_api.py       — اختبار Tauri API
  ✅ test_monitoring.py        — اختبار المراقبة
```

#### 10. قاعدة البيانات و Redis
```
الوضع:
  .env → DATABASE_URL, REDIS_URL موجودة
  core/database.py → موجود
  core/cache.py → موجود
  alembic/ → migrations موجودة
  init.sql → موجود

المطلوب:
  ✅ التأكد من اتصال قاعدة البيانات الفعلي
  ✅ تشغيل migrations (alembic upgrade head)
  ✅ التأكد من Redis يعمل
  ✅ ربط cache.py بالتطبيق
  ✅ إنشاء جداول التدريب (training_runs, model_checkpoints)
  ✅ إنشاء جداول المجلس (council_decisions, council_votes)
  ✅ إنشاء جداول المراقبة (worker_metrics, training_metrics)
```

#### 11. Community — غير مفعل
```
الموجود:
  community/code_sharing.py    — مشاركة الكود
  community/db_service.py      — قاعدة بيانات المجتمع
  community/forums.py          — المنتديات
  community/knowledge_base.py  — قاعدة المعرفة
  community/models.py          — النماذج
  community/profiles.py        — الملفات الشخصية

المطلوب:
  ✅ ربط community routes بـ API
  ✅ إنشاء واجهة المجتمع بالـ Desktop
  ✅ ربط مع نظام المصادقة (تم)
```

#### 12. Network — غير مفعل
```
الموجود:
  network/connection_tester.py  — اختبار الاتصال
  network/firewall_config.py   — إعدادات الجدار الناري
  network/health_check.py      — فحص الصحة
  network/monitor.py           — مراقبة الشبكة

المطلوب:
  ✅ تفعيل health_check بين كل الأجهزة
  ✅ ربط monitor.py بالـ Dashboard
  ✅ تفعيل الجدار الناري
  ✅ إضافة auto-reconnect عند سقوط الاتصال
```

---

### 🔵 تحسينات (خلال شهر)

#### 13. Docker — تحديث
```
المطلوب:
  ✅ تحديث Dockerfile لدعم GPU (تم) (nvidia-docker)
  ✅ تحديث docker-compose.yml بكل الخدمات (تم)
  ✅ إضافة docker-compose.gpu.yml للتدريب (تم)
  ✅ إضافة health checks لكل container (تم)
  ✅ Docker volumes للبيانات المستمرة (تم)
```

#### 14. Deploy — تحسين
```
الموجود: deploy_hostinger.sh, deploy_remaining.sh
المطلوب:
  ✅ deploy_all.sh (تم)      — نشر واحد لكل الأجهزة
  ✅ deploy_windows.ps1 (تم) — نشر Windows خاص
  ✅ deploy_rtx.sh (تم)      — نشر RTX 5090 خاص
  ✅ CI/CD pipeline (تم)      — GitHub Actions
  ✅ Zero-downtime deployment (تم) — تحديث بدون توقف
```

#### 15. ERP — تكامل
```
الموجود: erp/ (31 ملف)
المطلوب:
  ✅ ربط ERP بالـ API الرئيسي (تم)
  ✅ ربط ERP بلوحة المراقبة (تم)
  ✅ إضافة تقارير ذكية بالـ AI (تم)
  ✅ ربط ERP بالمجلس (تم) (تحليل القرارات المالية)
```

#### 16. Tauri Desktop App — تحسينات
```
المطلوب:
  ✅ Auto-update checker (فحص تحديثات تلقائي)
  ✅ Offline mode (العمل بدون إنترنت)
  ✅ Local AI inference (تشغيل AI محلياً)
  ✅ File watcher (مراقبة تغييرات الملفات)
  ✅ Git integration (تكامل مع Git)
  ✅ Multi-language support (عربي/إنجليزي)
  ✅ Themes (dark/light/custom)
  ✅ Keyboard shortcuts
  ✅ Drag & drop file support
  ✅ Split editor view
```

#### 17. v6 Training Scripts — ترحيل
```
الموجود في training/v6-scripts/ (18 ملف):
  advanced_training.py, auto-finetune.py, code_generation_training.py,
  continuous-train.py, convert-to-gguf.py, convert-to-onnx.py,
  evaluate-model.py, finetune-chat.py, finetune-extended.py,
  finetune.py, monitor.py, prepare-chat-data.py, run-16h.py,
  run_full_training.py, smart-learn.py, train_ai.py, validate-data.py

المطلوب:
  ✅ نقل السكربتات المفيدة إلى ai/training/
  ✅ دمج finetune scripts في rtx4090_trainer.py
  ✅ دمج evaluation scripts في auto_evaluation.py
  ✅ تحديث للعمل مع PyTorch الحديث
  ✅ إضافة دعم multi-GPU
```

---

### 📊 ملخص الإصلاحات حسب الأولوية

| الأولوية | العدد | الوصف |
|---------|-------|-------|
| 🔴 حرج | 3 | Syntax errors + GPU driver + Windows worker |
| 🟡 مهم | 3 | API routes + Desktop components + Orchestrator |
| 🟠 متوسط | 6 | Monitoring + Services + Tests + DB + Community + Network |
| 🔵 تحسين | 5 | Docker + Deploy + ERP + Desktop + v6 scripts |
| **المجموع** | **17** | **مجال إصلاح** |

---

## 🎯 المرحلة 1: v8.1 — إصلاحات حرجة (1-2 أسبوع)

### 1.1 إصلاح GPU RTX 5090
```
الخوارزمية:
1. تنصيب nvidia-driver-565 (مستقر مع CUDA 12.8)
2. تنصيب PyTorch cu128 المتوافق
3. اختبار torch.cuda.is_available() == True
4. إعادة تشغيل التدريب على GPU بالكامل
```

### 1.2 إصلاح Worker Path على Windows
```python
# المشكلة: Worker يحاول تنفيذ linux paths على Windows
# الحل: تحويل المسارات تلقائياً

def resolve_job_path(path: str) -> str:
    if platform.system() == "Windows":
        path = path.replace("/home/bi/", "C:\\Users\\BI\\")
        path = path.replace("/", "\\")
    return path
```

### 1.3 ربط بيانات التدريب القديمة
```python
# نقل 9 ملفات AI من النسخ القديمة إلى v8
MIGRATION_MAP = {
    "super-intelligent-learning.json": "ai/training/legacy_data/",
    "auto-learning.json": "ai/training/legacy_data/",
    "custom-training.json": "ai/training/legacy_data/",
    "intelligent-learning.json": "ai/training/legacy_data/",
    "ai-knowledge-db.json": "ai/memory/legacy_data/",
    "intensive-learning.json": "ai/training/legacy_data/",
    "professional-learning.json": "ai/training/legacy_data/",
}
```

### 1.4 Worker Auto-Restart على Windows
```python
# خوارزمية: Windows Service Wrapper
# nssm install bi-worker python.exe -X utf8 bi_worker.py --server...
# nssm install bi-server python.exe -X utf8 rtx4090_server.py
```

---

## 🧠 المرحلة 2: v8.5 — تكامل الذكاء (2-4 أسابيع)

### 2.1 نقل نظام Super Intelligent Learning
```python
class SuperIntelligentLearning:
    """نقل من النسخ القديمة + تحسين"""
    
    def __init__(self):
        # 1. Deep Network مع Attention
        self.deep_network = DeepAttentionNetwork(
            input_dim=768, hidden_dims=[512, 256, 128],
            attention_heads=8, dropout=0.1
        )
        
        # 2. Q-Learning مزدوج (Double DQN)
        self.q_primary = QNetwork(state_dim=128, action_dim=64)
        self.q_target = QNetwork(state_dim=128, action_dim=64)
        
        # 3. Experience Replay مع أولويات
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=100000, alpha=0.6, beta=0.4
        )
        
        # 4. Meta-Learning
        self.meta_learner = MetaLearner(
            task_memory_size=1000,
            strategy_pool=['gradient', 'evolution', 'bayesian']
        )
        
        # 5. Curriculum Learning
        self.curriculum = CurriculumManager(
            min_difficulty=0.1, max_difficulty=1.0,
            increment=0.05, progress_threshold=0.8
        )
```

### 2.2 خوارزمية التعلم الذاتي المستمر
```python
class ContinuousLearningEngine:
    """
    كل تفاعل مع المستخدم = بيانات تدريب
    Elastic Weight Consolidation لمنع نسيان المعرفة القديمة
    """
    
    def learn_from_interaction(self, interaction):
        fisher_matrix = self.compute_fisher(self.model)
        loss = self.train_step(interaction)
        
        # عقوبة على تغيير الأوزان المهمة
        ewc_penalty = sum(
            (fisher * (param - old_param) ** 2).sum()
            for fisher, param, old_param 
            in zip(fisher_matrix, self.model.parameters(), self.old_params)
        )
        
        total_loss = loss + self.lambda_ewc * ewc_penalty
        total_loss.backward()
        
    def evaluate_improvement(self) -> float:
        old_score = self.benchmark(self.old_model)
        new_score = self.benchmark(self.model)
        return (new_score - old_score) / old_score
```

### 2.3 تكامل المجلس مع التدريب
```python
class CouncilTrainingIntegration:
    async def council_guided_training(self):
        # 1. المجلس يحلل ويحدد الأولويات
        priorities = await self.high_council.analyze_training_needs()
        
        # 2. Shadow Team يحلل نقاط الضعف
        weaknesses = await self.shadow_team.find_weaknesses(self.model)
        
        # 3. Light Team يقترح تحسينات
        improvements = await self.light_team.suggest_improvements(weaknesses)
        
        # 4. الكشافة يبحثون عن بيانات جديدة
        new_data = await self.scouts.find_training_data(priorities)
        
        # 5. التدريب الموجه
        return await self.trainer.train(TrainingConfig(
            focus_areas=priorities, data_sources=new_data,
            improvements=improvements, validation_criteria=weaknesses
        ))
```

### 2.4 Dashboard مركزي للمراقبة
```
المكونات:
- WebSocket حي لكل الأجهزة
- GPU/CPU/RAM graphs بالوقت الحقيقي
- Training loss curves + accuracy
- Worker health (أخضر/أصفر/أحمر)
- تحكم بالموارد (sliders)
- إشعارات عند مشاكل
```

---

## ⚡ المرحلة 3: v9.0 — الذكاء الموزع (4-8 أسابيع)

### 3.1 التعلم الموزع (Distributed Training)
```python
class DistributedTrainer:
    """
    RTX 5090 (24 cores, 24GB VRAM) → Primary Trainer
    Windows RTX 4050 (20 cores, 6GB) → Secondary Trainer
    Mac M5 (10 cores, 24GB) → Evaluation + Inference
    Hostinger (8 cores) → Data Pipeline + Orchestration
    """
    
    async def distributed_step(self):
        # 1. توزيع البيانات
        data_shards = self.split_data(self.dataset, num_workers=2)
        
        # 2. حساب gradients محلياً (بالتوازي)
        grads_rtx = await self.rtx5090.compute_gradients(data_shards[0])
        grads_win = await self.windows.compute_gradients(data_shards[1])
        
        # 3. AllReduce
        avg_grads = self.average_gradients([grads_rtx, grads_win])
        
        # 4. تحديث + مزامنة
        self.model.apply_gradients(avg_grads)
        await self.sync_weights_to_all()
        
        # 5. Mac يقيّم
        return await self.mac.evaluate(self.model)
```

### 3.2 Federated Learning
```python
class FederatedLearning:
    """كل جهاز يدرب محلياً ويشارك الأوزان فقط"""
    
    async def federated_round(self):
        # 1. إرسال النموذج العالمي
        for worker in self.workers:
            await worker.receive_model(self.global_model)
        
        # 2. تدريب محلي
        local_models = await asyncio.gather(*[
            worker.train_local(epochs=5) for worker in self.workers
        ])
        
        # 3. FedAvg
        self.global_model = self.federated_average(
            local_models, weights=[w.data_size for w in self.workers]
        )
```

### 3.3 نظام التحديث الذكي
```python
class SmartUpdateSystem:
    """
    1. فحص Git كل 5 دقائق
    2. تحديث diff فقط
    3. Hot-reload بدون إيقاف
    4. Rollback تلقائي إذا فشل
    """
    
    async def check_and_update(self):
        remote = await self.get_remote_version()
        if remote == self.current_version: return
        
        diff = await self.download_diff(self.current_version, remote)
        backup = self.create_backup()
        
        try:
            self.apply_diff(diff)
            self.hot_reload()
            if not await self.health_check():
                raise Exception("Health check failed")
            self.current_version = remote
        except:
            self.restore_backup(backup)
            self.hot_reload()
```

### 3.4 TF-IDF + N-gram لفهم الكود
```python
class CodeUnderstanding:
    """نقل وتحسين من النسخ القديمة"""
    
    def build_code_index(self, codebase):
        tokens = [self.code_tokenizer.tokenize(f) for f in codebase]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3))
        self.tfidf_matrix = self.tfidf.fit_transform(tokens)
        self.ngram_model = NGramModel(n=3)
        for t in tokens: self.ngram_model.train(t)
    
    def suggest_completion(self, context, top_k=5):
        similar = self.find_similar(context, top_k=10)
        predictions = self.ngram_model.predict_next(context[-3:])
        return self.merge_suggestions(similar, predictions, top_k)
```

---

## 🌟 المرحلة 4: v9.5 — الوعي السياقي (4-6 أسابيع)

### 4.1 Vector Database للذاكرة الذكية
```python
class IntelligentMemory:
    def __init__(self):
        self.vector_db = VectorDB(
            embedding_dim=768, distance_metric='cosine',
            index_type='HNSW', ef_construction=200, M=16
        )
        self.context_window = ContextWindow(
            short_term=50, medium_term=500, long_term=10000
        )
    
    async def remember(self, interaction):
        embedding = self.encoder.encode(interaction['content'])
        self.vector_db.insert(vector=embedding, metadata={
            'type': interaction['type'],
            'timestamp': time.time(),
            'importance': self.calculate_importance(interaction)
        })
    
    async def recall(self, query, top_k=10):
        return self.vector_db.search(self.encoder.encode(query), top_k)
```

### 4.2 توقع الأخطاء قبل حدوثها
```python
class PredictiveErrorDetection:
    def predict_errors(self, code, context):
        static_issues = self.static_analyzer.check(code)
        ml_probability = self.error_model.predict(
            self.encode_code(code), self.encode_context(context)
        )
        similar_bugs = self.recall_similar_bugs(code)
        flow_issues = self.data_flow_analyzer.check(code)
        
        return [PredictedError(
            location=issue.location, message=issue.message,
            severity=self.calculate_severity(issue, ml_probability),
            suggested_fix=self.generate_fix(issue)
        ) for issue in static_issues + flow_issues
          if self.calculate_severity(issue, ml_probability) > 0.6]
```

### 4.3 فهم بنية المشروع
```python
class ProjectUnderstanding:
    async def analyze_project(self, root_path):
        graph = await self.build_dependency_graph(root_path)
        complexity = self.analyze_complexity(root_path)
        patterns = self.detect_patterns(graph)
        bottlenecks = self.find_bottlenecks(graph, complexity)
        
        return ProjectAnalysis(
            graph=graph, complexity=complexity,
            patterns=patterns, bottlenecks=bottlenecks,
            suggestions=self.generate_suggestions(patterns, bottlenecks)
        )
```

---

## 🏗️ المرحلة 5: v10.0 — النظام المتكامل (8-12 أسبوع)

### 5.1 BI-IDE Runtime
```python
class BIRuntime:
    # Tier 1 (v10.0 MVP)
    SUPPORTED = ['python', 'javascript', 'typescript', 'rust', 'go']
    # Tier 2 (بعد استقرار v10): java, cpp, csharp
    # Tier 3 (لاحقاً): php, ruby, swift
    
    async def execute(self, code, lang, sandbox=True):
        if sandbox:
            container = await self.create_sandbox(lang)
            return await container.exec(code, timeout=30)
        return await self.direct_exec(code, lang)
```

### 5.2 ERP + AI
```python
class ERPIntelligence:
    async def analyze_business(self):
        return BusinessReport(
            trends=self.ai.predict_sales(self.erp.get_sales(months=12)),
            optimizations=self.ai.optimize_costs(self.erp.get_expenses()),
            forecast=self.ai.forecast_demand(self.erp.get_inventory()),
            recommendations=self.ai.generate_recommendations()
        )
```

### 5.3 Plugin System + Marketplace
### 5.4 Mobile App (React Native)
### 5.5 Advanced Security (Zero Trust + E2E + AI Intrusion Detection)

---

## 🔑 ملخص الأفكار

### من النسخ القديمة (يجب نقلها):
1. ✅ Neural Network + Attention
2. ✅ Q-Learning مزدوج + Experience Replay
3. ✅ Meta-Learning (تعلم التعلم)
4. ✅ Curriculum Learning (تدريج 1→10)
5. ✅ TF-IDF + N-gram
6. ✅ Auto-Learning من PDFs (95 PDF)
7. ✅ Arabic NLP مخصص

### أفكار جديدة (v9-v10):
8. 🆕 Federated Learning (تدريب موزع + خصوصية)
9. 🆕 Predictive Error Detection (توقع أخطاء)
10. 🆕 Vector Memory HNSW (بحث دلالي)
11. 🆕 A/B Model Testing
12. 🆕 Elastic Weight Consolidation (منع النسيان)
13. 🆕 Hot Code Reload
14. 🆕 Plugin Marketplace
15. 🆕 Project Understanding Graph
16. 🆕 Sandbox Execution
17. 🆕 Business Intelligence AI

---

## 📁 هيكل الملفات المقترح (الإضافات)

```
bi-ide-v8/
├── ai/
│   ├── learning/                    # 🆕
│   │   ├── super_intelligent.py     # نقل + تحسين
│   │   ├── continuous_learning.py   # EWC
│   │   ├── curriculum_manager.py
│   │   ├── meta_learner.py
│   │   └── federated.py
│   ├── understanding/               # 🆕
│   │   ├── code_index.py           # TF-IDF + N-gram
│   │   ├── project_graph.py
│   │   ├── error_predictor.py
│   │   └── completion_engine.py
│   └── training/
│       ├── legacy_data/            # 🆕 بيانات قديمة
│       └── distributed_trainer.py  # 🆕
├── dashboard/                       # 🆕
│   ├── realtime_monitor.py
│   ├── gpu_charts.py
│   └── worker_health.py
└── plugins/                         # 🆕
    ├── plugin_manager.py
    ├── marketplace.py
    └── sdk/
```

---

## 🧭 إضافات تنفيذية مقترحة (لتحويل الخطة إلى تسليمات قابلة للقياس)

### 1) Definition of Done لكل مرحلة

#### ✅ DoD — v8.1
- لا يوجد `SyntaxError` في المستودع (فحص شامل بالـ `py_compile`).
- اختبارات `auth + rbac + rate_limit` تمر بنسبة 100%.
- `torch.cuda.is_available() == True` على جهاز التدريب الرئيسي.
- `worker` يعمل 24 ساعة بدون انقطاع (heartbeat مستقر).

#### ✅ DoD — v8.5
- تفعيل مسارات `auth/gateway/middleware/rate_limit` في الـ API الرئيسي.
- توفر لوحة مراقبة حيّة (WebSocket) مع تحديث ≤ 2 ثانية.
- تغطية اختبارية لا تقل عن 70% لوحدات API + orchestrator.

#### ✅ DoD — v9.0
- تنفيذ تدريب موزع ناجح لجولتين على الأقل بدون فشل مزامنة.
- تحسن throughput الفعلي ≥ 30% مقابل التدريب الأحادي.
- وجود rollback تلقائي مجرّب عند فشل تحديث حي.

#### ✅ DoD — v10.0
- Runtime يدعم تنفيذ آمن لـ 4 لغات على الأقل كبداية (`python`, `typescript`, `rust`, `go`).
- ERP+AI يصدر تقريرًا تنبؤيًا أسبوعيًا تلقائيًا.
- Plugin SDK يعمل مع مثال plugin فعلي قابل للتثبيت.

---

### 2) مؤشرات أداء (KPIs) موحدة

| المجال | KPI | الهدف |
|-------|-----|-------|
| الاستقرار | Crash-free runtime | ≥ 99.5% |
| البنية الخلفية | P95 API latency | ≤ 300ms |
| التدريب | GPU utilization | ≥ 70% أثناء التدريب |
| العمال | Worker uptime | ≥ 99% |
| الجودة | Test coverage (core paths) | ≥ 75% |
| الأمان | High/Critical vulnerabilities | = 0 قبل أي إصدار |

---

### 3) بوابات الإصدار (Release Gates)

لا يتم الانتقال لمرحلة أعلى إلا بعد تحقق الشروط التالية:
1. ✅ نجاح الاختبارات الحرجة + smoke tests.
2. ✅ عدم وجود أخطاء بناء أو lint حرجة.
3. ✅ توثيق API/changes في `docs/`.
4. ✅ خطة rollback مجربة على بيئة staging.
5. ✅ موافقة أمنية أساسية (secrets, auth, rate limit).

---

### 4) سجل مخاطر مختصر

| الخطر | التأثير | الاحتمال | التخفيف |
|------|---------|----------|---------|
| عدم توافق GPU/Driver | توقف التدريب | عالٍ | تثبيت نسخة driver معتمدة + runbook استرجاع |
| تعطل worker على Windows | فقدان مهام | متوسط | تشغيل كـ service + heartbeat + auto-restart |
| تضخم scope في v9-v10 | تأخير الإطلاق | عالٍ | تجميد نطاق كل sprint + MVP صارم لكل مرحلة |
| ضعف الاختبارات | أعطال إنتاجية | متوسط | فرض حد أدنى تغطية + smoke E2E قبل الدمج |
| اختناقات API تحت الحمل | بطء النظام | متوسط | rate limit + profiling + caching تدريجي |

---

### 5) إيقاع تنفيذ أسبوعي (مقترح)

- **السبت:** تخطيط sprint وتجميد النطاق.
- **الأحد-الثلاثاء:** تنفيذ features/إصلاحات حرجة.
- **الأربعاء:** تكامل + اختبارات + إصلاح regressions.
- **الخميس:** قياسات KPI + hardening + توثيق.
- **الجمعة:** release candidate أو patch release.

---

### 6) أول Sprint تنفيذي (7 أيام)

1. إصلاح أخطاء البناء (`connect_services.py`, `security_audit.py`).
2. تفعيل auth + middleware + rate limit داخل `api/app.py`.
3. إضافة endpointين: `/training/status` و`/system/resources` في orchestrator.
4. إنشاء `monitoring/system_monitor.py` و`services/training_service.py` بنسخة MVP.
5. تشغيل اختبارات مستهدفة وتسجيل baseline لـ KPIs.

---

## 🧪 نتائج مراجعة فعلية للمشروع (2026-02-28)

> هذه البنود مبنية على تشغيل فعلي داخل البيئة الحالية، وليست مراجعة نظرية فقط.

### 🔴 خلل مؤكد يجب إصلاحه فوراً

1. **`_run_tests.py` غير قابل للتشغيل على macOS/Linux**
    - **المشكلة:** يعتمد مسار Windows ثابت `d:\\bi-ide-v8`.
    - **الأثر:** تعطّل runner الأساسي للاختبارات على البيئات غير Windows.
    - **الإجراء:** تحويله إلى مسار ديناميكي (`Path(__file__).resolve().parent`) وإزالة أي `cwd` ثابت.

2. **`hierarchy/connect_services.py` يحتوي SyntaxError**
    - **المشكلة:** string/docstring غير مغلق بشكل صحيح (`unterminated triple-quoted string literal`).
    - **الأثر:** فشل `py_compile` وتوقف أي فحص بناء يعتمد على هذا الملف.
    - **الإجراء:** إصلاح الـ docstring/quotes + إضافة فحص `py_compile` ضمن CI قبل الدمج.

### 🟡 خلل استقراري عالي التأثير

3. **الاختبارات تتأثر بخدمة RTX خارجية أثناء التشغيل**
    - **الدليل:** ظهور retries/timeout على `192.168.68.125:8080` أثناء pytest.
    - **المصدر:** `api/routes/council.py` + إعدادات افتراضية في `core/config.py` و`core/tasks.py`.
    - **الأثر:** بطء، flakiness، واحتمال تعليق teardown.
    - **الإجراء:**
      - إضافة `TEST_MODE=true` لتعطيل أي network call خارجي داخل الاختبارات.
      - حقن RTX client عبر dependency injection بدل الاستدعاء المباشر.
      - Mock افتراضي لخدمات RTX في `tests/conftest.py`.

4. **تعليق/إبطاء في إغلاق TestClient عند نهاية الجلسة**
    - **الدليل:** `KeyboardInterrupt` أثناء `pytest_sessionfinish` في `tests/conftest.py`.
    - **الأثر:** عدم موثوقية نتائج CI وزمن اختبار غير ثابت.
    - **الإجراء:**
      - تقليل retries/timeouts أثناء الاختبار (`RTX4090_MAX_RETRIES=0` في test env).
      - مراجعة startup/shutdown hooks لمنع أي polling thread طويل في test mode.

### 🟠 دين تقني يجب جدولته

5. **مهام أساسية في `core/tasks.py` ما زالت TODO (learning/cleanup/embeddings)**
    - **الأثر:** أجزاء من مسار “التعلم الفعلي” غير مكتملة رغم وجود واجهات.
    - **الإجراء:** تحويل كل TODO إلى تذاكر تنفيذية مع DoD واختبار لكل مهمة.

### ✅ إضافات مباشرة على خطة التنفيذ (تحديث Sprint 1)

- إضافة مهمة: **Cross-platform Test Runner Hardening** (`_run_tests.py`).
- إضافة مهمة: **Syntax Gate in CI** (py_compile على `hierarchy/`, `api/`, `core/`, `security/`).
- إضافة مهمة: **Test Isolation Layer** (تعطيل network الخارجي + mocks).
- إضافة KPI جديد: **CI Determinism** (نسبة نجاح الاختبارات المتكررة ≥ 95% بدون flaky).

---
---

# 📐 المواصفات التقنية التفصيلية — BI-IDE

---

## I. خريطة اللغات والتقنيات لكل طبقة

```
┌─────────────────────────────────────────────────────────┐
│                    Desktop (Tauri)                        │
│  Rust (core) + TypeScript/React (UI) + CSS               │
├─────────────────────────────────────────────────────────┤
│                    Web Frontend                          │
│  TypeScript + React 18 + Vite + CSS Variables            │
├─────────────────────────────────────────────────────────┤
│                    API Gateway                           │
│  Python 3.11+ + FastAPI + Pydantic v2 + Uvicorn          │
├─────────────────────────────────────────────────────────┤
│                    Business Logic                        │
│  Python 3.11+ (hierarchy, council, AI, training)         │
├─────────────────────────────────────────────────────────┤
│             AI / ML / Training Pipeline                  │
│  Python + PyTorch 2.x + CUDA + HuggingFace Transformers  │
├─────────────────────────────────────────────────────────┤
│             Distributed Workers                         │
│  Python + asyncio + aiohttp + WebSocket                  │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│  PostgreSQL 16 + Redis 7 + SQLAlchemy 2 + Alembic        │
├─────────────────────────────────────────────────────────┤
│                    Infrastructure                        │
│  Docker + nginx + GitHub Actions CI/CD                    │
├─────────────────────────────────────────────────────────┤
│                    Mobile (مستقبلي)                       │
│  React Native + TypeScript                               │
└─────────────────────────────────────────────────────────┘
```

### ليش هالاختيارات؟

| التقنية | السبب |
|---------|-------|
| **Python** للـ Backend | النظام الأساسي مبني عليه، PyTorch يشتغل بس على Python، أسرع تطوير |
| **FastAPI** | أسرع framework على Python، async native، auto-docs (Swagger)، typing |
| **TypeScript + React** للواجهة | Type safety، أكبر ecosystem، Tauri يدعمها |
| **Tauri (Rust)** للـ Desktop | أخف 10x من Electron، أمان عالي، أداء native |
| **PostgreSQL** | أقوى قاعدة بيانات مفتوحة، JSON support، full-text search |
| **Redis** | Cache + pub/sub + message queue — أسرع in-memory store |
| **PyTorch** | الأفضل للبحث + الإنتاج، dynamic computation graph، CUDA native |
| **Docker** | بيئة موحدة، نشر سريع، عزل الخدمات |

---

## I.1 سياسة دعم اللغات (Language Support Policy)

### 🎯 الهدف
منع تضخم النطاق (Scope Creep) وضمان جودة وأمان الـ Runtime قبل توسيع عدد اللغات.

### مستويات الدعم

| المستوى | الحالة | اللغات | معيار التنفيذ |
|--------|--------|--------|---------------|
| **Tier 1** | دعم كامل (MVP) | Python, JavaScript/TypeScript, Rust, Go | Sandbox + اختبارات + مراقبة + توثيق كامل |
| **Tier 2** | دعم تجريبي (Beta) | Java, C++, C# | تفعيل خلف Feature Flag + قياس أداء فعلي |
| **Tier 3** | مخطط مستقبلي | PHP, Ruby, Swift | لا يبدأ قبل استقرار Tier 1 و Tier 2 |

### بوابة قبول أي لغة جديدة

لا تُضاف أي لغة إلى Tier أعلى إلا بعد تحقق جميع الشروط:
1. تنفيذ آمن داخل Sandbox مع حدود CPU/RAM/Timeout واضحة.
2. Passing tests: Unit + Integration + Security + Smoke.
3. P95 زمن تنفيذ ضمن الهدف المحدد في KPI.
4. دعم أخطاء واضح (stderr mapping + error codes).
5. توثيق رسمي: أمثلة تشغيل + قيود + حالات فشل معروفة.

### قرار معماري مقترح

- في v10.0: الالتزام فقط بـ Tier 1.
- بعد 2 إصدارات مستقرة: تقييم Tier 2 لغة-بلغة (وليس دفعة واحدة).
- Tier 3 يبقى Backlog ولا يدخل Sprint إلا بقرار صريح.

---

## II. هيكل الكلاسات الكامل — Backend

### A. طبقة الـ API (`api/`)

```python
# ═══════════════════════════════════════════
# api/app.py — نقطة الدخول الرئيسية
# ═══════════════════════════════════════════

from fastapi import FastAPI
from api.routers import auth, council, training, ai, erp, monitoring, community

class BIIDEApp:
    """التطبيق الرئيسي — يجمع كل الـ routers"""
    
    def __init__(self):
        self.app = FastAPI(
            title="BI-IDE API",
            version="8.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self._setup_middleware()
        self._register_routers()
        self._register_events()
    
    def _setup_middleware(self):
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"])
        self.app.add_middleware(RateLimitMiddleware, max_requests=100, window=60)
        self.app.add_middleware(LoggingMiddleware)
        self.app.add_middleware(AuthMiddleware)
        self.app.add_middleware(ErrorHandlerMiddleware)
    
    def _register_routers(self):
        self.app.include_router(auth.router,       prefix="/api/v1/auth",       tags=["auth"])
        self.app.include_router(council.router,     prefix="/api/v1/council",    tags=["council"])
        self.app.include_router(training.router,    prefix="/api/v1/training",   tags=["training"])
        self.app.include_router(ai.router,          prefix="/api/v1/ai",         tags=["ai"])
        self.app.include_router(erp.router,         prefix="/api/v1/erp",        tags=["erp"])
        self.app.include_router(monitoring.router,   prefix="/api/v1/monitoring", tags=["monitoring"])
        self.app.include_router(community.router,    prefix="/api/v1/community",  tags=["community"])
```

```python
# ═══════════════════════════════════════════
# api/routers/auth.py — المصادقة
# ═══════════════════════════════════════════

router = APIRouter()

@router.post("/login")
async def login(credentials: LoginRequest, db: AsyncSession = Depends(get_db)):
    """تسجيل الدخول — JWT tokens"""

@router.post("/register")
async def register(user: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """تسجيل مستخدم جديد"""

@router.post("/refresh")
async def refresh_token(token: RefreshRequest):
    """تجديد token"""

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """تسجيل الخروج — إبطال tokens"""

# ═══════════════════════════════════════════
# api/routers/council.py — المجلس
# ═══════════════════════════════════════════

router = APIRouter()

@router.post("/query")
async def query_council(query: CouncilQuery):
    """إرسال استعلام للمجلس"""

@router.get("/status")
async def council_status():
    """حالة المجلس — أعضاء، جلسات، قرارات"""

@router.get("/decisions")
async def list_decisions(skip: int = 0, limit: int = 20):
    """قائمة القرارات"""

@router.get("/members")
async def list_members():
    """أعضاء المجلس — 16 حكيم + فرق"""

@router.post("/vote")
async def submit_vote(vote: CouncilVote):
    """تصويت على قرار"""

# ═══════════════════════════════════════════
# api/routers/training.py — التدريب
# ═══════════════════════════════════════════

router = APIRouter()

@router.get("/status")
async def training_status():
    """حالة التدريب على كل الأجهزة"""

@router.post("/start")
async def start_training(config: TrainingConfig):
    """بدء تدريب — يوزع على الأجهزة"""

@router.post("/stop")
async def stop_training():
    """إيقاف التدريب"""

@router.get("/metrics")
async def get_metrics(period: str = "1h"):
    """مقاييس: loss, accuracy, throughput"""

@router.get("/models")
async def list_models():
    """قائمة النماذج المدربة"""

@router.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """نشر نموذج للإنتاج"""

@router.get("/history")
async def training_history(days: int = 30):
    """تاريخ التدريب"""

# ═══════════════════════════════════════════
# api/routers/monitoring.py — المراقبة
# ═══════════════════════════════════════════

router = APIRouter()

@router.get("/system/resources")
async def system_resources():
    """CPU/GPU/RAM لكل الأجهزة"""

@router.get("/workers")
async def workers_status():
    """حالة كل العمال"""

@router.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    """WebSocket للمراقبة الحية — تحديث كل 2 ثانية"""

@router.get("/alerts")
async def get_alerts(severity: str = "all"):
    """التنبيهات النشطة"""

@router.get("/logs")
async def get_logs(service: str = "all", lines: int = 100):
    """سجلات كل الخدمات"""
```

### B. Pydantic Models (عقود البيانات)

```python
# ═══════════════════════════════════════════
# api/schemas.py — كل النماذج
# ═══════════════════════════════════════════

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

# ─── Auth ───
class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

class UserProfile(BaseModel):
    id: int
    username: str
    role: str  # admin, developer, viewer
    created_at: datetime

# ─── Council ───
class CouncilQuery(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    context: Optional[Dict] = None
    urgency: str = "normal"  # low, normal, high, critical
    require_full_council: bool = False

class CouncilDecision(BaseModel):
    id: str
    question: str
    decision: str
    confidence: float = Field(ge=0, le=1)
    votes: Dict[str, str]  # member_id → vote
    reasoning: str
    shadow_analysis: str   # تحليل Shadow Team
    light_suggestion: str  # اقتراح Light Team
    timestamp: datetime

class CouncilMember(BaseModel):
    id: str
    name: str
    role: str  # president, high_council, shadow, light, scout, meta, domain_expert
    team: str
    specialization: str
    is_active: bool

# ─── Training ───
class TrainingConfig(BaseModel):
    model_preset: str = "xlarge"  # small, medium, large, xlarge
    epochs: int = Field(ge=1, le=1000, default=200)
    batch_size: int = Field(ge=1, le=256, default=32)
    learning_rate: float = Field(ge=1e-6, le=1e-1, default=3e-4)
    cpu_percent: int = Field(ge=10, le=100, default=80)
    gpu_percent: int = Field(ge=0, le=100, default=100)
    devices: List[str] = ["all"]  # ["rtx5090", "windows", "mac", "hostinger"]
    distributed: bool = False
    resume_from: Optional[str] = None  # checkpoint path

class TrainingStatus(BaseModel):
    is_active: bool
    device: str           # cuda, cpu, mps
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    throughput_sps: float  # samples per second
    gpu_utilization: float
    gpu_vram_gb: float
    gpu_temp_c: float
    cpu_utilization: float
    elapsed_seconds: float
    eta_seconds: float
    model_params: int
    samples_processed: int

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    params: int
    size_mb: float
    accuracy: float
    trained_at: datetime
    epochs_trained: int
    status: str  # training, ready, deployed, archived

# ─── Worker ───
class WorkerInfo(BaseModel):
    worker_id: str
    hostname: str
    status: str  # online, training, offline, error
    labels: List[str]
    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    uptime_seconds: float
    last_heartbeat: datetime
    training: Optional[TrainingStatus]
    usage: Dict[str, float]  # cpu_percent, gpu_percent, ram_percent

# ─── Monitoring ───
class SystemResources(BaseModel):
    workers: List[WorkerInfo]
    total_cpu_cores: int
    total_ram_gb: float
    total_gpu_vram_gb: float
    active_trainings: int
    total_throughput_sps: float

class Alert(BaseModel):
    id: str
    severity: str  # info, warning, error, critical
    source: str    # worker_id or service name
    message: str
    timestamp: datetime
    resolved: bool
```

### C. التسلسل الهرمي — هيكل الوراثة الكامل

```python
# ═══════════════════════════════════════════
# hierarchy/ — التصميم الكائني (OOP)
# ═══════════════════════════════════════════

from abc import ABC, abstractmethod

# ─── الكلاس الأساسي لكل عضو مجلس ───
class CouncilEntity(ABC):
    """كل كيان في التسلسل يرث من هنا"""
    
    def __init__(self, entity_id: str, name: str, role: str):
        self.entity_id = entity_id
        self.name = name
        self.role = role
        self.is_active = True
        self.created_at = datetime.now()
        self.decisions_count = 0
        self.accuracy_score = 0.0
    
    @abstractmethod
    async def analyze(self, query: str, context: Dict) -> Dict:
        """تحليل استعلام — كل كيان يحلل بطريقته"""
        pass
    
    @abstractmethod
    async def vote(self, proposal: str) -> str:
        """تصويت على اقتراح — approve/reject/abstain"""
        pass
    
    def update_accuracy(self, was_correct: bool):
        """تحديث دقة التنبؤات"""
        self.decisions_count += 1
        if was_correct:
            self.accuracy_score = (
                self.accuracy_score * (self.decisions_count - 1) + 1
            ) / self.decisions_count

# ─── الرئيس ───
class President(CouncilEntity):
    """المستخدم — الرئيس الأعلى 24/7"""
    
    def __init__(self):
        super().__init__("president-001", "الرئيس", "president")
        self.veto_power = True    # حق النقض
        self.override_power = True # تجاوز أي قرار
    
    async def analyze(self, query, context):
        return {"source": "president", "directive": query}
    
    async def vote(self, proposal):
        return "approve"  # الرئيس يوافق دائماً (يدخل بصفته المراقب)

# ─── عضو المجلس الأعلى ───
class HighCouncilMember(CouncilEntity):
    """حكيم من الـ 16 — يصوت على القرارات الكبرى"""
    
    def __init__(self, member_id: str, name: str, expertise: str):
        super().__init__(member_id, name, "high_council")
        self.expertise = expertise
        self.wisdom_score = 0.5
    
    async def analyze(self, query, context):
        relevance = self._calculate_relevance(query, self.expertise)
        analysis = await self._deep_analyze(query, context)
        return {
            "member": self.name,
            "expertise": self.expertise,
            "relevance": relevance,
            "analysis": analysis,
            "confidence": relevance * self.wisdom_score
        }
    
    async def vote(self, proposal):
        score = await self._evaluate_proposal(proposal)
        return "approve" if score > 0.6 else "reject" if score < 0.3 else "abstain"

# ─── Shadow Team (المتشائمون) ───
class ShadowMember(CouncilEntity):
    """يبحث عن المخاطر والمشاكل المحتملة"""
    
    ANALYSIS_FOCUS = ["security_risks", "scalability_issues", 
                      "performance_bottlenecks", "edge_cases"]
    
    async def analyze(self, query, context):
        risks = []
        for focus in self.ANALYSIS_FOCUS:
            risk = await self._evaluate_risk(query, focus)
            if risk["severity"] > 0.3:
                risks.append(risk)
        return {"risks": risks, "overall_risk": sum(r["severity"] for r in risks) / max(len(risks), 1)}

# ─── Light Team (المتفائلون) ───
class LightMember(CouncilEntity):
    """يبحث عن الفرص والتحسينات"""
    
    ANALYSIS_FOCUS = ["innovation_opportunities", "efficiency_gains",
                      "user_experience_improvements", "growth_potential"]
    
    async def analyze(self, query, context):
        opportunities = []
        for focus in self.ANALYSIS_FOCUS:
            opp = await self._find_opportunity(query, focus)
            if opp["potential"] > 0.3:
                opportunities.append(opp)
        return {"opportunities": opportunities, "overall_potential": max(o["potential"] for o in opportunities) if opportunities else 0}

# ─── Scout (الكشاف) ───
class Scout(CouncilEntity):
    """يبحث في الإنترنت والمصادر الخارجية"""
    
    SCOUT_TYPES = {
        "tech": "تقنيات جديدة، أدوات، frameworks",
        "market": "اتجاهات السوق، طلب المستخدمين",
        "competitor": "ماذا يفعل المنافسون",
        "opportunity": "فرص لم تُستغل"
    }
    
    def __init__(self, scout_type: str):
        super().__init__(f"scout-{scout_type}", f"كشاف {scout_type}", "scout")
        self.scout_type = scout_type
        self.sources = []
    
    async def analyze(self, query, context):
        findings = await self._search_external(query, self.scout_type)
        return {"type": self.scout_type, "findings": findings}

# ─── Meta Team ───
class MetaTeamMember(CouncilEntity):
    """فريق المراقبة الفوقية"""
    
    META_ROLES = {
        "performance": "يراقب أداء النظام ويقترح تحسينات",
        "quality": "يراقب جودة الكود والقرارات",
        "learning": "يراقب تقدم التعلم ويوجه التدريب",
        "evolution": "يراقب تطور النظام ويقترح تغييرات هيكلية"
    }

# ─── Domain Expert ───
class DomainExpert(CouncilEntity):
    """خبير في مجال محدد"""
    
    DOMAINS = [
        "web_development", "mobile_development", "database",
        "security", "devops", "machine_learning", "nlp",
        "arabic_processing", "erp", "ui_ux", "networking"
    ]

# ═══════════════════════════════════════════
# المجلس الكامل — يجمع الكل
# ═══════════════════════════════════════════

class FullCouncil:
    """التشكيلة الكاملة"""
    
    def __init__(self):
        self.president = President()
        
        self.high_council = [
            HighCouncilMember(f"hc-{i}", f"حكيم-{i}", exp)
            for i, exp in enumerate([
                "architecture", "security", "ai", "data",
                "performance", "ux", "business", "innovation",
                "quality", "scalability", "arabic_nlp", "devops",
                "mobile", "web", "testing", "strategy"
            ])
        ]  # 16 حكيم
        
        self.shadow_team = [ShadowMember(f"shadow-{i}", f"ظل-{i}", "shadow") for i in range(4)]
        self.light_team = [LightMember(f"light-{i}", f"نور-{i}", "light") for i in range(4)]
        self.scouts = [Scout(t) for t in Scout.SCOUT_TYPES]
        self.meta_team = [MetaTeamMember(f"meta-{r}", f"فوقي-{r}", "meta") for r in MetaTeamMember.META_ROLES]
        self.domain_experts = [DomainExpert(f"expert-{d}", f"خبير-{d}", "domain_expert") for d in DomainExpert.DOMAINS]
    
    async def full_deliberation(self, query: str, context: Dict = None) -> CouncilDecision:
        """مداولة كاملة — كل الأعضاء يشاركون"""
        
        # 1. تحليل متوازي من كل الفرق
        analyses = await asyncio.gather(
            *[m.analyze(query, context) for m in self.high_council],
            *[s.analyze(query, context) for s in self.shadow_team],
            *[l.analyze(query, context) for l in self.light_team],
            *[sc.analyze(query, context) for sc in self.scouts],
        )
        
        # 2. تصويت المجلس الأعلى
        votes = {}
        for member in self.high_council:
            votes[member.entity_id] = await member.vote(query)
        
        # 3. تجميع القرار
        approve_count = sum(1 for v in votes.values() if v == "approve")
        decision = "approved" if approve_count > len(self.high_council) / 2 else "rejected"
        
        # 4. حساب الثقة
        confidence = approve_count / len(self.high_council)
        
        return CouncilDecision(
            id=str(uuid.uuid4()),
            question=query,
            decision=decision,
            confidence=confidence,
            votes=votes,
            reasoning=self._synthesize_reasoning(analyses),
            shadow_analysis=str([a for a in analyses if "risks" in a]),
            light_suggestion=str([a for a in analyses if "opportunities" in a]),
            timestamp=datetime.now()
        )
```

### D. طبقة التدريب — التصميم الكامل

```python
# ═══════════════════════════════════════════
# ai/training/ — خط أنابيب التدريب
# ═══════════════════════════════════════════

class TrainingPipeline:
    """خط أنابيب التدريب الكامل"""
    
    STAGES = [
        "data_collection",    # جمع البيانات
        "preprocessing",      # معالجة مسبقة
        "tokenization",       # تحويل لـ tokens
        "model_init",         # تهيئة النموذج
        "training_loop",      # حلقة التدريب
        "evaluation",         # تقييم
        "optimization",       # تحسين (quantization, pruning)
        "deployment"          # نشر
    ]
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._select_device()
        self.model = self._init_model()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed Precision
        self.metrics = MetricsTracker()
    
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available() and self.config.gpu_percent > 0:
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        return torch.device("cpu")


# ═══════════════════════════════════════════
# خوارزمية التدريب المفصلة
# ═══════════════════════════════════════════

class AdvancedTrainer:
    """مدرب متقدم مع كل التقنيات الحديثة"""
    
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        
        # Optimizer: AdamW مع weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler: Cosine Annealing مع warmup
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            warmup_steps=100,
            total_steps=config.epochs * config.steps_per_epoch,
            min_lr=1e-6
        )
        
        # Mixed Precision للسرعة
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient Accumulation للـ batch sizes كبيرة
        self.accumulation_steps = 4
        
        # Early Stopping
        self.patience = 10
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpoint
        self.checkpoint_dir = Path("checkpoints/")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    async def train_loop(self):
        """حلقة التدريب الرئيسية"""
        
        for epoch in range(self.config.epochs):
            # ─── Training Phase ───
            self.model.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_loader):
                # نقل البيانات للـ GPU
                input_ids = batch["input_ids"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                
                # Mixed Precision Forward
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.accumulation_steps
                
                # Backward مع scaling
                self.scaler.scale(loss).backward()
                
                # Gradient Accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    # Gradient Clipping (منع الانفجار)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                epoch_loss += loss.item() * self.accumulation_steps
                
                # إرسال metrics كل 10 خطوات
                if step % 10 == 0:
                    await self.report_metrics(epoch, step, loss.item())
            
            # ─── Validation Phase ───
            val_loss, val_accuracy = await self.validate()
            
            # ─── Early Stopping ───
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # ─── Checkpoint كل 10 epochs ───
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch}")
    
    def save_checkpoint(self, epoch: int, name: str):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.dict()
        }, self.checkpoint_dir / f"{name}.pt")
```

### E. النموذج — بنية Transformer كاملة

```python
# ═══════════════════════════════════════════
# ai/models/bi_transformer.py — النموذج المخصص
# ═══════════════════════════════════════════

class BITransformerConfig:
    """إعدادات النموذج"""
    PRESETS = {
        "small":  {"d_model": 256,  "n_heads": 4,  "n_layers": 4,  "d_ff": 1024,  "params": "~15M"},
        "medium": {"d_model": 512,  "n_heads": 8,  "n_layers": 8,  "d_ff": 2048,  "params": "~85M"},
        "large":  {"d_model": 768,  "n_heads": 12, "n_layers": 12, "d_ff": 3072,  "params": "~180M"},
        "xlarge": {"d_model": 1024, "n_heads": 16, "n_layers": 16, "d_ff": 4096,  "params": "~368M"},
    }

class MultiHeadAttention(nn.Module):
    """آلية الانتباه المتعددة — قلب Transformer"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # Query
        self.W_k = nn.Linear(d_model, d_model)  # Key
        self.W_v = nn.Linear(d_model, d_model)  # Value
        self.W_o = nn.Linear(d_model, d_model)  # Output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # تحويل لـ multi-head
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # تطبيق على Values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)

class TransformerBlock(nn.Module):
    """كتلة Transformer واحدة"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),             # GELU أفضل من ReLU للـ Transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm (أفضل من post-norm للتدريب العميق)
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out  # Residual connection
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out    # Residual connection
        
        return x

class BITransformer(nn.Module):
    """نموذج BI-IDE الكامل"""
    
    def __init__(self, config: BITransformerConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output Head
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (يقلل عدد parameters)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"logits": logits, "loss": loss}
```

---

## III. قاعدة البيانات — المخطط الكامل

```sql
-- ═══════════════════════════════════════════
-- PostgreSQL Schema — bi_ide
-- ═══════════════════════════════════════════

-- المستخدمين
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(20) DEFAULT 'developer',  -- admin, developer, viewer
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    last_login      TIMESTAMP
);

-- جلسات التدريب
CREATE TABLE training_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_preset    VARCHAR(20) NOT NULL,
    model_params    BIGINT,
    epochs_total    INT,
    epochs_done     INT DEFAULT 0,
    batch_size      INT,
    learning_rate   FLOAT,
    device          VARCHAR(20),           -- cuda, cpu, mps
    worker_id       VARCHAR(100),
    status          VARCHAR(20) DEFAULT 'queued',  -- queued, running, paused, done, failed
    loss_final      FLOAT,
    accuracy_final  FLOAT,
    throughput_sps  FLOAT,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    config_json     JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- نقاط حفظ النماذج
CREATE TABLE model_checkpoints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_run_id UUID REFERENCES training_runs(id),
    epoch           INT,
    loss            FLOAT,
    accuracy        FLOAT,
    file_path       TEXT,
    file_size_mb    FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- قرارات المجلس
CREATE TABLE council_decisions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question        TEXT NOT NULL,
    decision        VARCHAR(20),           -- approved, rejected, deferred
    confidence      FLOAT,
    votes_json      JSONB,
    reasoning       TEXT,
    shadow_analysis TEXT,
    light_suggestion TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- مقاييس العمال
CREATE TABLE worker_metrics (
    id              SERIAL PRIMARY KEY,
    worker_id       VARCHAR(100) NOT NULL,
    cpu_percent     FLOAT,
    gpu_percent     FLOAT,
    gpu_temp_c      FLOAT,
    ram_percent     FLOAT,
    gpu_vram_used   FLOAT,
    gpu_vram_total  FLOAT,
    is_training     BOOLEAN DEFAULT FALSE,
    measured_at     TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_worker_metrics_time ON worker_metrics(worker_id, measured_at DESC);

-- سجل التعلم
CREATE TABLE learning_log (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(50),           -- user_interaction, auto_learning, internet
    content_type    VARCHAR(50),           -- code, question, correction, pdf
    content         TEXT,
    learned_topics  TEXT[],
    confidence      FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- التنبيهات
CREATE TABLE alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    severity        VARCHAR(20) NOT NULL,  -- info, warning, error, critical
    source          VARCHAR(100),
    message         TEXT,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

---

## IV. Frontend — هيكل المكونات التفصيلي

```
apps/desktop-tauri/src/
├── App.tsx                          # نقطة الدخول
├── main.tsx                         # React entry
├── styles/
│   ├── globals.css                  # متغيرات CSS عالمية
│   ├── themes/
│   │   ├── dark.css                 # ثيم داكن (الافتراضي)
│   │   ├── light.css                # ثيم فاتح
│   │   └── bi-brand.css             # ألوان BI المخصصة
│   └── components/                  # CSS لكل مكون
├── components/
│   ├── layout/
│   │   ├── Layout.tsx               # التخطيط الرئيسي
│   │   ├── Header.tsx               # الرأس العلوي
│   │   ├── Sidebar.tsx              # القائمة الجانبية (586 سطر ✅)
│   │   └── StatusBar.tsx            # شريط الحالة (180 سطر ✅)
│   ├── editor/
│   │   ├── Editor.tsx               # محرر الكود (175 سطر ✅)
│   │   ├── EditorTabs.tsx           # تبويبات الملفات  [جديد]
│   │   ├── MiniMap.tsx              # خريطة مصغرة  [جديد]
│   │   └── DiffViewer.tsx           # عرض الفروقات  [جديد]
│   ├── council/
│   │   ├── CouncilPanel.tsx         # لوحة المجلس (190 سطر ✅)
│   │   ├── CouncilChat.tsx          # محادثة مع المجلس  [جديد]
│   │   ├── VoteDisplay.tsx          # عرض التصويتات  [جديد]
│   │   └── MemberList.tsx           # قائمة الأعضاء  [جديد]
│   ├── training/
│   │   ├── TrainingDashboard.tsx    # لوحة التدريب  [جديد]
│   │   ├── LossChart.tsx            # رسم الـ loss  [جديد]
│   │   ├── GPUMonitor.tsx           # مراقبة GPU  [جديد]
│   │   └── ModelSelector.tsx        # اختيار نموذج  [جديد]
│   ├── workers/
│   │   ├── WorkerGrid.tsx           # شبكة العمال  [جديد]
│   │   ├── WorkerCard.tsx           # بطاقة عامل  [جديد]
│   │   └── ResourceGauge.tsx        # مقياس الموارد  [جديد]
│   ├── terminal/
│   │   └── Terminal.tsx             # الطرفية (202 سطر ✅)
│   ├── files/
│   │   ├── FileExplorer.tsx         # مستكشف الملفات  [جديد]
│   │   └── FileTree.tsx             # شجرة الملفات  [جديد]
│   └── common/
│       ├── Button.tsx               # زر موحد  [جديد]
│       ├── Modal.tsx                # نافذة منبثقة  [جديد]
│       ├── Toast.tsx                # إشعار  [جديد]
│       ├── Chart.tsx                # رسم بياني  [جديد]
│       ├── Loading.tsx              # تحميل  [جديد]
│       └── ErrorBoundary.tsx        # معالجة أخطاء  [جديد]
├── hooks/
│   ├── useWebSocket.ts              # WebSocket hook  [جديد]
│   ├── useTraining.ts               # بيانات التدريب  [جديد]
│   ├── useWorkers.ts                # بيانات العمال  [جديد]
│   ├── useCouncil.ts                # بيانات المجلس  [جديد]
│   └── useAuth.ts                   # مصادقة  [جديد]
├── services/
│   ├── api.ts                       # HTTP client  [جديد]
│   ├── websocket.ts                 # WebSocket client  [جديد]
│   └── tauri.ts                     # Tauri IPC  [جديد]
└── types/
    └── index.ts                     # TypeScript types  [جديد]
```

```typescript
// ═══════════════════════════════════════════
// hooks/useWebSocket.ts — WebSocket للمراقبة الحية
// ═══════════════════════════════════════════

interface RealtimeData {
  workers: WorkerInfo[];
  training: TrainingStatus;
  system: SystemResources;
  alerts: Alert[];
}

export function useWebSocket(url: string): RealtimeData {
  const [data, setData] = useState<RealtimeData>(initialData);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setData(prev => ({ ...prev, ...update }));
    };
    ws.onclose = () => setTimeout(() => ws.reconnect(), 2000);
    return () => ws.close();
  }, [url]);
  
  return data;
}

// ═══════════════════════════════════════════
// components/training/GPUMonitor.tsx
// ═══════════════════════════════════════════

interface GPUMonitorProps {
  workers: WorkerInfo[];
}

export function GPUMonitor({ workers }: GPUMonitorProps) {
  return (
    <div className="gpu-monitor">
      {workers.map(w => (
        <div key={w.worker_id} className="gpu-card">
          <h3>{w.hostname}</h3>
          <GaugeChart value={w.usage.gpu_percent} label="GPU" max={100} />
          <GaugeChart value={w.gpu_vram_gb} label="VRAM" max={w.gpu_vram_total_gb} />
          <span className="temp">{w.gpu_temp_c}°C</span>
          <StatusBadge status={w.status} />
        </div>
      ))}
    </div>
  );
}
```

---

## V. التعلم الموزع — البروتوكول الكامل

```python
# ═══════════════════════════════════════════
# ai/learning/distributed_trainer.py
# ═══════════════════════════════════════════

class DistributedProtocol:
    """
    بروتوكول التدريب الموزع:
    
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ RTX 5090 │     │ Windows  │     │   Mac    │
    │ Primary  │     │ RTX 4050 │     │ M5 Eval  │
    │ Trainer  │     │ Secondary│     │          │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         └────────┬───────┘                │
                  │ AllReduce              │
         ┌───────┴────────┐                │
         │  Orchestrator  │◄───────────────┘
         │   (Hostinger)  │   Evaluation Results
         └────────────────┘
    
    الخطوات:
    1. Orchestrator يقسم البيانات
    2. كل GPU يحسب gradients
    3. AllReduce يجمع ويحسب المتوسط
    4. كل GPU يحدث الأوزان
    5. Mac يقيّم كل N steps
    """
    
    MSG_TYPES = {
        "INIT":        0x01,  # تهيئة العامل
        "DATA_SHARD":  0x02,  # إرسال جزء بيانات
        "GRADIENT":    0x03,  # إرسال gradients
        "WEIGHT_SYNC": 0x04,  # مزامنة أوزان
        "EVAL_REQ":    0x05,  # طلب تقييم
        "EVAL_RES":    0x06,  # نتيجة تقييم
        "HEARTBEAT":   0x07,  # نبضة حياة
        "CHECKPOINT":  0x08,  # حفظ نقطة
        "STOP":        0x09,  # إيقاف
    }
    
    # خوارزمية AllReduce المبسطة
    async def all_reduce(self, local_gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Ring AllReduce:
        1. كل worker يرسل gradients للجار
        2. كل worker يجمع ويمرر
        3. في النهاية كل worker عنده المجموع
        4. نقسم على عدد العمال -> المتوسط
        """
        n_workers = len(self.workers)
        
        # Scatter-Reduce phase
        for chunk_idx in range(n_workers - 1):
            send_to = (self.rank + 1) % n_workers
            recv_from = (self.rank - 1) % n_workers
            
            await self.send_gradients(send_to, local_gradients, chunk_idx)
            received = await self.recv_gradients(recv_from, chunk_idx)
            
            # تجميع
            for key in local_gradients:
                local_gradients[key] += received[key]
        
        # AllGather phase
        for chunk_idx in range(n_workers - 1):
            send_to = (self.rank + 1) % n_workers
            recv_from = (self.rank - 1) % n_workers
            
            await self.send_gradients(send_to, local_gradients, chunk_idx)
            received = await self.recv_gradients(recv_from, chunk_idx)
            local_gradients[chunk_idx] = received[chunk_idx]
        
        # المتوسط
        for key in local_gradients:
            local_gradients[key] /= n_workers
        
        return local_gradients
```

---

## VI. الأمان — الطبقات الأربع

```python
# ═══════════════════════════════════════════
# security/ — نظام أمان متعدد الطبقات
# ═══════════════════════════════════════════

class SecurityStack:
    """
    الطبقات:
    ┌────────────────────────────┐
    │    4. Anomaly Detection    │  ML-based
    │    (كشف الشذوذ بالذكاء)    │
    ├────────────────────────────┤
    │    3. Rate Limiting        │  Token bucket
    │    (حدود الطلبات)          │
    ├────────────────────────────┤
    │    2. Authentication       │  JWT + RBAC
    │    (المصادقة والصلاحيات)    │
    ├────────────────────────────┤
    │    1. Encryption           │  AES-256 + TLS
    │    (التشفير)               │
    └────────────────────────────┘
    """

class JWTAuth:
    def create_token(self, user_id: int, role: str) -> str:
        payload = {
            "sub": user_id, "role": role,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # لمنع إعادة الاستخدام
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    def verify_token(self, token: str) -> Dict:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if self.is_revoked(payload["jti"]):
            raise AuthError("Token revoked")
        return payload

class RateLimiter:
    """Token Bucket Algorithm"""
    def __init__(self, max_tokens: int = 100, refill_rate: float = 10):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens/second
        self.buckets: Dict[str, float] = {}
    
    def allow(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.buckets:
            self.buckets[client_id] = (self.max_tokens, now)
        
        tokens, last_time = self.buckets[client_id]
        elapsed = now - last_time
        tokens = min(self.max_tokens, tokens + elapsed * self.refill_rate)
        
        if tokens >= 1:
            self.buckets[client_id] = (tokens - 1, now)
            return True
        return False

class AnomalyDetector:
    """كشف الشذوذ بالـ ML — Isolation Forest"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.01, random_state=42)
        self.features = []  # request_rate, payload_size, response_time...
    
    def is_anomalous(self, request_features: List[float]) -> bool:
        prediction = self.model.predict([request_features])
        return prediction[0] == -1  # -1 = anomaly
```

---

## VII. خوارزمية التعلم من الإنترنت

```python
# ═══════════════════════════════════════════
# hierarchy/internet_auto_training.py — التعلم من الإنترنت
# ═══════════════════════════════════════════

class InternetLearner:
    """
    خوارزمية التعلم التلقائي من الإنترنت:
    
    1. اختيار مواضيع بناءً على المجلس + المستخدم
    2. بحث ذكي (Google, GitHub, Stack Overflow, Wikipedia)
    3. تصفية وتنقية المحتوى
    4. تقييم الجودة (relevance + accuracy + freshness)
    5. تحويل لبيانات تدريب
    6. تدريب تدريجي (curriculum learning)
    """
    
    SOURCES = {
        "github":         {"url": "api.github.com", "type": "code", "trust": 0.9},
        "stackoverflow":  {"url": "api.stackexchange.com", "type": "qa", "trust": 0.85},
        "arxiv":          {"url": "arxiv.org/api", "type": "papers", "trust": 0.95},
        "mdn":            {"url": "developer.mozilla.org", "type": "docs", "trust": 0.95},
        "python_docs":    {"url": "docs.python.org", "type": "docs", "trust": 0.99},
    }
    
    async def learn_topic(self, topic: str, depth: int = 3):
        # 1. البحث
        raw_data = await self._search_all_sources(topic)
        
        # 2. التصفية
        filtered = self._filter_quality(raw_data, min_score=0.7)
        
        # 3. التحويل لبيانات تدريب
        training_samples = self._convert_to_training_data(filtered)
        
        # 4. الحفظ في قاعدة البيانات
        await self._store_learning(training_samples, topic)
        
        # 5. التدريب التدريجي
        if len(training_samples) >= 100:
            await self.trainer.fine_tune(training_samples, epochs=3)
    
    def _filter_quality(self, data: List[Dict], min_score: float) -> List[Dict]:
        """تقييم الجودة"""
        scored = []
        for item in data:
            score = (
                item.get("source_trust", 0.5) * 0.3 +
                self._relevance_score(item) * 0.3 +
                self._freshness_score(item) * 0.2 +
                self._complexity_score(item) * 0.2
            )
            if score >= min_score:
                item["quality_score"] = score
                scored.append(item)
        return sorted(scored, key=lambda x: -x["quality_score"])
```

---

## VIII. استراتيجية ترحيل البيانات (Data Migration) 🔴

```python
# ═══════════════════════════════════════════
# migration/rollback_plan.py
# ═══════════════════════════════════════════

class MigrationStrategy:
    """
    كل تحديث إصدار يحتاج خطة ترحيل بيانات + rollback.
    الهدف: ≤15 دقيقة للتراجع عن أي تغيير.
    """
    
    DATABASE = {
        "backup_before_migration": True,
        "blue_green_deployment": True,       # جدولين بالتوازي عند v9+
        "rollback_time_sla": "< 15 minutes",
        "migration_tool": "alembic",
        "test_on_staging_first": True,
    }
    
    MODEL_CHECKPOINTS = {
        "backward_compatibility": 3,         # محتفظة لـ 3 إصدارات سابقة
        "auto_conversion": True,             # تحويل تلقائي للـ checkpoints القديمة
        "checkpoint_format": "safetensors",   # أسرع وأأمن من pickle
        "versioned_naming": "{model}_{version}_{epoch}_{loss:.4f}.safetensors",
    }
    
    TRAINING_DATA = {
        "legacy_import": True,               # استيراد 9 ملفات AI القديمة
        "deduplication": True,               # إزالة التكرارات
        "schema_versioning": True,           # كل ملف بيانات عنده version
    }
    
    async def migrate(self, from_version: str, to_version: str):
        # 1. نسخة احتياطية كاملة
        backup_id = await self.full_backup()
        
        # 2. ترحيل قاعدة البيانات
        try:
            await self.run_migrations(from_version, to_version)
            await self.verify_data_integrity()
        except Exception:
            await self.rollback(backup_id)
            raise
        
        # 3. ترحيل checkpoints
        await self.convert_checkpoints(from_version, to_version)
        
        # 4. تحقق نهائي
        assert await self.health_check(), "Migration health check failed"

    async def rollback(self, backup_id: str):
        """تراجع كامل في ≤15 دقيقة"""
        await self.restore_database(backup_id)
        await self.restore_model_files(backup_id)
        await self.restart_all_services()
```

---

## IX. اختبار الأداء تحت الحمل (Load Testing) 🟡

```python
# ═══════════════════════════════════════════
# tests/load/load_test_plan.py
# ═══════════════════════════════════════════

import locust  # أداة اختبار الحمل

LOAD_TEST_SCENARIOS = {
    "concurrent_training_requests": {
        "users": 100,
        "ramp_up": "30s",
        "target": "كل طلب يُقبل أو يُرفض بـ response ≤500ms"
    },
    "websocket_connections": {
        "connections": 1000,
        "duration": "1h",
        "target": "لا يوجد disconnect غير متوقع + latency ≤2s"
    },
    "api_requests_per_second": {
        "rps": 10000,
        "endpoints": ["/health", "/workers", "/training/status"],
        "target": "P95 ≤300ms + 0% 5xx errors"
    },
    "memory_leak_test": {
        "duration": "72_hours",
        "target": "RSS growth ≤5% من الأصلي"
    }
}

class BIIDELoadTest(locust.HttpUser):
    """اختبار حمل API"""
    
    wait_time = locust.between(0.1, 0.5)
    
    @locust.task(10)
    def health_check(self):
        self.client.get("/api/v1/monitoring/system/resources")
    
    @locust.task(5)
    def training_status(self):
        self.client.get("/api/v1/training/status")
    
    @locust.task(3)
    def workers_list(self):
        self.client.get("/api/v1/orchestrator/workers")
    
    @locust.task(1)
    def council_query(self):
        self.client.post("/api/v1/council/query", json={
            "question": "ما أفضل نموذج للتدريب؟",
            "urgency": "normal"
        })

# تشغيل: locust -f tests/load/load_test_plan.py --host=https://bi-iq.com
```

---

## X. SLOs و Error Budgets 🔵

```python
# ═══════════════════════════════════════════
# monitoring/slo.py — أهداف مستوى الخدمة
# ═══════════════════════════════════════════

class ServiceLevelObjectives:
    """
    SLO = الهدف المتفق عليه
    Error Budget = الهامش المسموح للأخطاء
    
    مثال: SLO 99.9% = Error Budget 0.1%
    = 8.76 ساعة downtime مسموح/سنة
    = 43.8 دقيقة/شهر
    """
    
    OBJECTIVES = {
        "api_availability": {
            "slo": 0.999,              # 99.9%
            "error_budget_monthly": "43.8 minutes",
            "measurement": "successful_requests / total_requests",
            "alert_at": 0.998,         # تنبيه عند 99.8%
        },
        "training_completion_rate": {
            "slo": 0.95,               # 95%
            "measurement": "completed_epochs / scheduled_epochs",
            "alert_at": 0.90,
        },
        "worker_reconnect_time": {
            "slo_seconds": 30,         # ≤30 ثانية
            "measurement": "time_from_disconnect_to_reconnect",
            "alert_at_seconds": 45,
        },
        "dashboard_refresh_latency": {
            "slo_ms": 2000,            # ≤2 ثانية
            "measurement": "websocket_message_to_render",
            "alert_at_ms": 3000,
        },
        "training_gpu_utilization": {
            "slo": 0.70,               # ≥70% أثناء التدريب
            "measurement": "avg_gpu_percent_during_training",
            "alert_at": 0.50,
        }
    }
    
    def check_budget(self, metric: str, current_value: float) -> dict:
        obj = self.OBJECTIVES[metric]
        slo = obj["slo"]
        remaining = current_value - slo
        budget_consumed = max(0, (slo - current_value) / (1 - slo)) * 100
        
        return {
            "metric": metric,
            "slo": slo,
            "current": current_value,
            "budget_consumed_percent": budget_consumed,
            "status": "ok" if current_value >= slo else "breached",
            "action": "freeze_deployments" if budget_consumed > 80 else "normal"
        }
```

---

## XI. الوضع بدون إنترنت (Offline Mode) — تفصيلي

```typescript
// ═══════════════════════════════════════════
// apps/desktop-tauri/src/offline/
// ═══════════════════════════════════════════

interface OfflineCapabilities {
  // ─── AI محلي ───
  local_model_inference: {
    enabled: true;
    model_format: "ONNX" | "GGUF";        // أخف من PyTorch
    max_model_size: "2GB";                  // quantized INT4/INT8
    supported_tasks: ["completion", "chat", "code_review"];
    fallback_if_no_model: "rule_based_suggestions";
  };
  
  // ─── بيانات محلية ───
  cached_data: {
    training_data: "last_7_days";
    council_decisions: "last_30";
    worker_history: "last_24_hours";
    project_index: "full";                  // فهرس المشروع كامل
  };
  
  // ─── مزامنة عند العودة ───
  sync_on_reconnect: {
    auto: true;
    conflict_resolution: "server_wins" | "client_wins" | "manual";
    default_strategy: "server_wins";
    queue_max_size: 1000;                   // أقصى عدد عمليات منتظرة
  };
  
  // ─── Git محلي ───
  local_git: {
    enabled: true;
    operations: ["commit", "branch", "diff", "log", "stash"];
    push_pull: "queued_until_online";
  };
  
  // ─── مؤشر الحالة ───
  status_indicator: {
    online: "🟢";
    offline: "🔴";
    syncing: "🟡";
    partial: "🟠";  // بعض الخدمات متاحة
  };
}
```

```python
# ═══════════════════════════════════════════
# services/offline_manager.py — إدارة الوضع بدون إنترنت
# ═══════════════════════════════════════════

class OfflineManager:
    def __init__(self):
        self.queue = OfflineQueue(max_size=1000)
        self.local_model = None
        self.is_online = True
    
    async def on_disconnect(self):
        """عند فقدان الاتصال"""
        self.is_online = False
        self.local_model = await self.load_quantized_model()
        self.notify_ui("offline", "تم التبديل للوضع المحلي")
    
    async def on_reconnect(self):
        """عند عودة الاتصال"""
        self.is_online = True
        pending = self.queue.get_all()
        
        for op in pending:
            try:
                await self.sync_operation(op)
                self.queue.mark_done(op.id)
            except ConflictError as e:
                await self.resolve_conflict(op, e)
        
        self.notify_ui("online", f"تمت مزامنة {len(pending)} عملية")
    
    async def queue_operation(self, operation: Dict):
        """تخزين العملية للمزامنة لاحقاً"""
        if self.is_online:
            return await self.execute_online(operation)
        self.queue.add(operation)
```

---

## XII. التوثيق ككود (Documentation as Code) 🟡

```yaml
# ═══════════════════════════════════════════
# docs/documentation_strategy.yml
# ═══════════════════════════════════════════

documentation:
  api:
    tool: "FastAPI auto-docs (OpenAPI/Swagger)"
    url: "/docs"
    auto_generated: true
    # كل endpoint يتوثق تلقائياً من docstrings + Pydantic schemas
  
  code:
    tool: "mkdocs + mkdocstrings"
    source: "docstrings في Python"
    style: "Google docstring format"
    deploy: "GitHub Pages"
    command: "mkdocs serve"  # تطوير محلي
  
  architecture:
    tool: "Mermaid diagrams"
    model: "C4 (Context, Container, Component, Code)"
    location: "docs/architecture/"
    files:
      - "system_context.md"    # النظام ككل
      - "container_diagram.md" # الخدمات
      - "component_diagram.md" # المكونات الداخلية
      - "data_flow.md"         # تدفق البيانات
  
  runbooks:
    location: "docs/runbooks/"
    format: "markdown"
    required_sections:
      - "المشكلة"
      - "الأعراض"
      - "التشخيص (خطوة بخطوة)"
      - "الحل"
      - "التحقق"
      - "الوقاية"
    examples:
      - "gpu_driver_crash.md"
      - "worker_not_connecting.md"
      - "training_stuck.md"
      - "database_migration_failed.md"
      - "high_memory_usage.md"
  
  changelog:
    tool: "conventional-commits + auto-changelog"
    format: "CHANGELOG.md"
    # كل commit بصيغة: feat(training): add distributed gradient sync
```

---

## XIII. المرونة والتراجع التدريجي (Graceful Degradation)

```python
# ═══════════════════════════════════════════
# services/resilience.py — نظام المرونة
# ═══════════════════════════════════════════

class GracefulDegradation:
    """
    النظام يستمر بالعمل حتى لو سقطت أجزاء منه.
    كل سيناريو فشل → خطة بديلة تلقائية.
    """
    
    FALLBACK_CHAIN = {
        "if_rtx5090_down": {
            "action": "switch_to_windows_rtx4050",
            "impact": "سرعة تدريب أقل (6GB vs 24GB VRAM)",
            "auto": True,
            "notification": "RTX 5090 offline — تم التبديل لـ Windows RTX 4050"
        },
        "if_windows_down": {
            "action": "switch_to_mac_cpu_training",
            "impact": "تدريب أبطأ بـ 10x (CPU only)",
            "auto": True,
        },
        "if_orchestrator_down": {
            "action": "workers_standalone_mode",
            "impact": "لا توزيع مهام — كل عامل يشتغل مستقل",
            "auto": True,
            "recovery": "auto_reconnect_every_30s"
        },
        "if_internet_down": {
            "action": "use_cached_data_and_local_model",
            "impact": "لا تدريب من الإنترنت — وضع offline",
            "auto": True,
        },
        "if_redis_down": {
            "action": "fallback_to_in_memory_cache",
            "impact": "cache يضيع عند restart",
            "auto": True,
        },
        "if_database_down": {
            "action": "read_only_mode_from_cache",
            "impact": "لا كتابة — قراءة فقط من cache",
            "auto": True,
            "alert": "critical"
        },
        "if_gpu_overheat": {
            "threshold_c": 85,
            "action": "throttle_training_50_percent",
            "impact": "سرعة تدريب أقل بالنص",
            "auto": True,
            "resume_at_c": 70
        }
    }
    
    async def handle_failure(self, component: str):
        chain_key = f"if_{component}_down"
        if chain_key not in self.FALLBACK_CHAIN:
            await self.alert("critical", f"No fallback for {component}")
            return
        
        fallback = self.FALLBACK_CHAIN[chain_key]
        
        if fallback["auto"]:
            await self.execute_fallback(fallback["action"])
            await self.notify(fallback.get("notification", f"{component} failed — fallback active"))
        
        # مراقبة الاسترجاع
        asyncio.create_task(self.monitor_recovery(component))
    
    async def monitor_recovery(self, component: str, interval: int = 30):
        """فحص دوري — هل الكمبونت رجع؟"""
        while True:
            await asyncio.sleep(interval)
            if await self.is_healthy(component):
                await self.restore_primary(component)
                await self.notify(f"✅ {component} عاد للعمل — تم استعادة الوضع الطبيعي")
                break
```

---

## XIV. BI-IDE CLI — أداة سطر الأوامر 💡

```python
# ═══════════════════════════════════════════
# cli/bi.py — أداة التحكم السريع
# ═══════════════════════════════════════════

import click

@click.group()
def bi():
    """BI-IDE Command Line Interface"""
    pass

# ─── التدريب ───
@bi.group()
def training():
    """إدارة التدريب"""

@training.command()
@click.option("--preset", default="xlarge", type=click.Choice(["small","medium","large","xlarge"]))
@click.option("--devices", default="all", help="rtx5090,windows,mac,hostinger")
@click.option("--epochs", default=200, type=int)
@click.option("--gpu-percent", default=100, type=int)
def start(preset, devices, epochs, gpu_percent):
    """بدء تدريب"""
    # bi training start --preset large --devices rtx5090,windows
    click.echo(f"🚀 Starting {preset} training on {devices}...")
    response = api.post("/training/start", json={...})
    click.echo(f"✅ Training started: {response['model_params']} params")

@training.command()
def status():
    """حالة التدريب"""
    # bi training status
    data = api.get("/training/status")
    click.echo(f"Epoch: {data['epoch']}/{data['total_epochs']}")
    click.echo(f"Loss: {data['loss']:.4f} | Accuracy: {data['accuracy']:.2f}")
    click.echo(f"GPU: {data['gpu_utilization']}% | Throughput: {data['throughput_sps']:.1f} sps")

@training.command()
def stop():
    """إيقاف التدريب"""
    api.post("/training/stop")
    click.echo("⏹️ Training stopped")

# ─── المجلس ───
@bi.group()
def council():
    """التفاعل مع المجلس"""

@council.command()
@click.argument("question")
@click.option("--urgency", default="normal", type=click.Choice(["low","normal","high","critical"]))
def ask(question, urgency):
    """سؤال المجلس"""
    # bi council ask "كيف أحسن الأداء؟"
    response = api.post("/council/query", json={"question": question, "urgency": urgency})
    click.echo(f"📋 القرار: {response['decision']}")
    click.echo(f"🎯 الثقة: {response['confidence']:.0%}")
    click.echo(f"💡 التحليل: {response['reasoning'][:200]}...")

# ─── العمال ───
@bi.group()
def worker():
    """إدارة العمال"""

@worker.command()
def status():
    """حالة كل العمال"""
    # bi worker status
    workers = api.get("/orchestrator/workers")
    for w in workers["workers"]:
        icon = "🟢" if w["status"] == "online" else "🔴"
        click.echo(f"{icon} {w['worker_id']:25s} | CPU:{w['usage']['cpu_percent']:5.1f}% | GPU:{w['usage']['gpu_percent']:5.1f}%")

# ─── النشر ───
@bi.group()
def deploy():
    """النشر والتحديث"""

@deploy.command()
@click.option("--env", default="production", type=click.Choice(["staging","production"]))
@click.option("--skip-tests", is_flag=True)
def push(env, skip_tests):
    """نشر التحديث"""
    # bi deploy push --env production
    if not skip_tests:
        click.echo("🧪 Running tests...")
        run_tests()
    click.echo(f"🚀 Deploying to {env}...")

# ─── النظام ───
@bi.command()
def status():
    """حالة النظام الكاملة"""
    # bi status
    resources = api.get("/monitoring/system/resources")
    click.echo(f"Workers: {len(resources['workers'])}")
    click.echo(f"Total CPU: {resources['total_cpu_cores']} cores")
    click.echo(f"Total VRAM: {resources['total_gpu_vram_gb']:.1f} GB")
    click.echo(f"Active Trainings: {resources['active_trainings']}")

# التثبيت: pip install -e . → bi training start --preset xlarge
```

---

## XV. ⚠️ تحذيرات وقرارات معمارية مهمة

### التحذيرات

| المخاطرة | السبب | الحل المقترح |
|----------|-------|-------------|
| **Scope Creep في v9-v10** | 8-12 أسبوع قد تصير 6 أشهر | فرض MVP صارم + feature flags + تجميد نطاق كل sprint |
| **Windows Worker** | nssm معقد وغير موثوق عبر SSH | خيار 1: NSSM محلياً. خيار 2: PM2 (Node). خيار 3: WSL2 + systemd |
| **BI Transformer من الصفر** | بناء 368M param model مكلف ومعقد | استخدام HuggingFace كأساس (GPT-2/Llama) + fine-tuning بدل التدريب من صفر |
| **ERP + AI** | دمج ERP كامل مع AI معقد جداً | البدء بتقرير واحد فقط (مبيعات شهرية) ثم التوسع |
| **Federated Learning** | معقد لـ 4 أجهزة فقط | تأجيل لـ v9.5، في v9.0 يكفي simple replication |

### قرارات معمارية

```python
# v9.0 — تبسيط التدريب الموزع
PHASE_9_SIMPLIFIED = {
    "simple_replication": True,       # RTX 5090 = Master يدرب
    "central_orchestrator": True,     # Hostinger يوزع فقط
    "windows": "inference + eval",    # مو تدريب موزع
    "mac": "evaluation + benchmarks", # تقييم فقط
    "federated_learning": "v9.5",     # تأجيل
    "full_distributed": "v10.0",      # لما نثبت البنية
}

# Sprint الأول — 3 مهام فقط (مو 5)
SPRINT_1_FOCUSED = [
    "1. إصلاح Syntax Errors (connect_services.py + security_audit.py)",
    "2. تفعيل auth + middleware + rate_limit في api/app.py",
    "3. إنشاء monitoring/system_monitor.py MVP",
    # النتيجة: نظام صلب يشتغل، بعدها نبني فوقه
]
```

