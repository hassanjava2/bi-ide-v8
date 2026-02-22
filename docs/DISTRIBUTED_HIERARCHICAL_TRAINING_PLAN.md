# Distributed Hierarchical Training Plan

## الهدف
بناء شبكة تخصصات هرمية متشعبة، كل عقدة تخصص تحتوي AI ثنائي التفكير:
1. تحسين التطور الحالي للتخصص.
2. إعادة بناء التخصص من الصفر بكفاءة أعلى.

مع دعم تدريب موزع على أي عدد من الأجهزة عبر Worker Agents، وتخزين حالة التدريب لحظيًا.

## مبدأ مهم (قيّد تقني)
لا يمكن لموقع ويب عادي أن يستهلك موارد أي جهاز زائر مباشرة بدون Agent محلي.
الحل الصحيح:
- الـ API المركزي يدير graph + tasks.
- كل جهاز يشغّل `distributed_worker_agent.py` ليوفّر الموارد فعليًا.

## ما تم تنفيذه الآن
- Specialized Graph service:
  - ملف: `hierarchy/specialized_ai_network.py`
  - graph هرمي seeded (رياضيات -> فيزيائية/إحصاء/تحسين -> تخصصات أدق).
- Distributed worker/task state:
  - `data/learning/distributed-training-state.json`
  - `data/knowledge/specialization-network.json`
- API endpoints جاهزة:
  - `GET /api/v1/network/status`
  - `GET /api/v1/network/graph`
  - `POST /api/v1/network/graph/expand`
  - `POST /api/v1/network/workers/register`
  - `POST /api/v1/network/workers/heartbeat`
  - `POST /api/v1/network/training/enqueue`
  - `POST /api/v1/network/training/claim`
  - `POST /api/v1/network/training/complete`
  - `POST /api/v1/network/think/dual`
- Worker Agent:
  - ملف: `distributed_worker_agent.py`
  - يسجل نفسه، يعمل heartbeat، يطالب tasks، ينفذ training CPU، ويرجع metrics.

## تشغيل سريع
### 1) شغل API
- `D:/bi-ide-v8/.venv/Scripts/python.exe api.py`

### 2) أمر واحد لكل سيرفر مؤجّر (H200 وغيره)
- `powershell -ExecutionPolicy Bypass -File .\start_h200_worker.ps1 -ApiUrl http://<API_HOST>:8000 -WorkerId h200-01 -PollSec 2 -TrainSec 30`
- السكربت يعيد تشغيل الـ worker تلقائيًا إذا توقف، ويستخدم outbox retry حتى لا تضيع نتائج المهام عند انقطاع الشبكة.

### 3) تشغيل orchestrator مستمر لتغذية الـ queue (يفضل على السيرفر المركزي)
- `D:/bi-ide-v8/.venv/Scripts/python.exe continuous_training_orchestrator.py --api http://<API_HOST>:8000 --min-queue 30 --burst 10 --sleep-sec 8 --priority 8`
- هذا يضمن وجود مهام دائمًا حتى تبقى كل السيرفرات تعمل بدون idle.

### 4) أضف task يدويًا (اختياري)
- `POST /api/v1/network/training/enqueue`
- body مثال:
```json
{
  "topic": "تحسين Bayesian inference under drift",
  "node_id": "stats-bayesian",
  "priority": 8
}
```

## خارطة المرحلة القادمة (لأفضل جودة)
1. **Planner Agent**: يحوّل أي موضوع تدريبي إلى تفريعات تخصصية تلقائيًا.
2. **Resource-aware Scheduler**: جدولة حسب CPU/RAM/GPU وcost-aware batching.
3. **Real Training Executors**: ربط task بأنماط تدريب فعلية (PyTorch/LoRA/quantization).
4. **Live Artifact Stream**: رفع artifacts/checkpoints incremental إلى storage مركزي.
5. **Consensus Layer**: دمج مخرجات current-evolver + zero-reinventor لتوصية نهائية.
6. **Quality Gates**: قبول/رفض checkpoint حسب KPIs (loss, latency, evidence rate).

## وعد الجودة
الأساس الحالي deterministic ومتحقق منه syntax/runtime endpoints.
للوصول إلى تدريب إنتاجي ضخم (GPU cluster + realtime sync) نكمل خطوات المرحلة القادمة أعلاه بشكل تدريجي مضبوط.
