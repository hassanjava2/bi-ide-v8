"""
🧠 Training Coordinator — منسق التدريب الموزع

المسؤوليات:
- توزيع مهام التدريب على العقد تلقائياً
- إنشاء مهام تدريب حسب الطبقات الهرمية
- تغذية queue بمهام مستمرة (لا idle)
- مراعاة قدرات كل عقدة (GPU vs CPU)
- Hostinger-aware scheduling (خفيف)
- تعلم ذاتي من النتائج السابقة
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────── Config ────────────

COORDINATOR_INTERVAL_SEC = int(os.getenv("COORDINATOR_INTERVAL_SEC", "30"))
MIN_QUEUE_SIZE = int(os.getenv("MIN_QUEUE_SIZE", "5"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "50"))

# ──────────── Hierarchy Layers ────────────

TRAINING_LAYERS = [
    {
        "name": "council",
        "display": "المجلس الأعلى",
        "description": "قرارات استراتيجية، تحليل، تنبؤات",
        "requires_gpu": True,
        "priority": 9,
        "data_sources": ["data/knowledge/council_decisions.json"],
    },
    {
        "name": "scouts",
        "display": "الكشافة",
        "description": "استكشاف الويب، جلب الأخبار، تحليل السوق",
        "requires_gpu": False,
        "priority": 7,
        "data_sources": ["data/knowledge/scout_reports.json"],
    },
    {
        "name": "erp_accounting",
        "display": "المحاسبة",
        "description": "محاسبة، ميزانيات، تقارير مالية",
        "requires_gpu": True,
        "priority": 8,
        "data_sources": ["data/learning/erp_data.json"],
    },
    {
        "name": "erp_inventory",
        "display": "المخزون",
        "description": "إدارة مخزون، تنبيه نقص، تتبع حركة",
        "requires_gpu": True,
        "priority": 8,
        "data_sources": ["data/learning/inventory_data.json"],
    },
    {
        "name": "code_generation",
        "display": "توليد الكود",
        "description": "كتابة كود، إصلاح أخطاء، copilot",
        "requires_gpu": True,
        "priority": 9,
        "data_sources": ["data/learning/code_samples.json"],
    },
    {
        "name": "copilot",
        "display": "المساعد الذكي",
        "description": "اكمال الكود، اقتراحات، توثيق",
        "requires_gpu": True,
        "priority": 8,
        "data_sources": ["data/learning/copilot_data.json"],
    },
    {
        "name": "balance",
        "display": "مجلس التوازن",
        "description": "مراقبة الأداء، توازن الموارد",
        "requires_gpu": False,
        "priority": 6,
        "data_sources": ["data/learning/balance_logs.json"],
    },
    {
        "name": "meta_team",
        "display": "الفريق الفوقي",
        "description": "تنسيق الفرق، تحسين العمليات",
        "requires_gpu": False,
        "priority": 5,
        "data_sources": ["data/learning/meta_team_data.json"],
    },
    {
        "name": "domain_experts",
        "display": "خبراء المجال",
        "description": "خبرة متعددة المجالات",
        "requires_gpu": True,
        "priority": 7,
        "data_sources": ["data/learning/expert_knowledge.json"],
    },
    {
        "name": "seventh_dimension",
        "display": "البعد السابع",
        "description": "خطط طويلة المدى، رؤية مستقبلية",
        "requires_gpu": False,
        "priority": 4,
        "data_sources": ["data/learning/century_plan.json"],
    },
]


# ──────────── Training Topics ────────────

TRAINING_TOPICS = {
    "council": [
        "تحسين دقة القرارات الاستراتيجية",
        "تحليل المخاطر وإدارتها",
        "تنبؤات السوق المالية",
        "تحسين consensus بين الحكماء",
    ],
    "scouts": [
        "استكشاف أخبار التقنية الحديثة",
        "تحليل تهديدات الأمن السيبراني",
        "رصد فرص الأعمال الجديدة",
        "تحسين كفاءة جمع المعلومات",
    ],
    "erp_accounting": [
        "تحسين تصنيف المعاملات المحاسبية",
        "اكتشاف أنماط الاحتيال المالي",
        "تحسين التقارير المالية الآلية",
    ],
    "erp_inventory": [
        "تنبؤ الطلب على المنتجات",
        "تحسين إدارة المخزون الذكية",
        "اكتشاف أنماط الحركة الموسمية",
    ],
    "code_generation": [
        "تحسين جودة الكود المولد",
        "تعلم أنماط الكود من المشاريع",
        "تحسين اصلاح الأخطاء التلقائي",
        "فهم السياق البرمجي المعقد",
    ],
    "copilot": [
        "تحسين اقتراحات إكمال الكود",
        "تعلم أسلوب المبرمج",
        "تحسين التوثيق الآلي",
    ],
    "balance": [
        "تحسين مراقبة موارد النظام",
        "اكتشاف الاختناقات المبكرة",
    ],
    "meta_team": [
        "تحسين تنسيق الفرق",
        "تحسين كفاءة العمليات",
    ],
    "domain_experts": [
        "تحسين دقة النصائح المتخصصة",
        "توسيع قاعدة المعرفة",
    ],
    "seventh_dimension": [
        "تحسين التنبؤات طويلة المدى",
        "تحليل الاتجاهات العالمية",
    ],
}


# ──────────── Coordinator ────────────

class TrainingCoordinator:
    """
    Manages the training pipeline:
    1. Monitors queue size
    2. Auto-generates training tasks when queue is low
    3. Assigns tasks based on worker capabilities
    4. Tracks training history and improves scheduling
    """

    def __init__(self):
        self.is_running = False
        self.cycle_count = 0
        self.history_file = Path("data/learning/training_history.jsonl")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    async def start(self, orchestrator_state):
        """Start the coordinator loop."""
        self.is_running = True
        self.state = orchestrator_state
        print("🧠 Training Coordinator started")

        while self.is_running:
            try:
                await self._coordination_cycle()
                self.cycle_count += 1
            except Exception as e:
                print(f"⚠️ Coordinator error: {e}")

            await asyncio.sleep(COORDINATOR_INTERVAL_SEC)

    def stop(self):
        self.is_running = False
        print("🧠 Training Coordinator stopped")

    async def _coordination_cycle(self):
        """One coordination cycle."""
        if not hasattr(self, 'state'):
            return

        # 1. Count queued jobs
        queued = [j for j in self.state.jobs.values() if j["status"] == "queued"]
        running = [j for j in self.state.jobs.values() if j["status"] == "running"]
        online_workers = [w for w in self.state.workers.values()
                         if w.get("status") in ("online", "idle", "training")]

        if not online_workers:
            return  # No workers, no point

        # 2. Fill queue if needed
        if len(queued) < MIN_QUEUE_SIZE:
            needed = MIN_QUEUE_SIZE - len(queued)
            new_tasks = self._generate_tasks(needed, online_workers)
            for task in new_tasks:
                self.state.jobs[task["job_id"]] = task

            if new_tasks:
                print(f"🧠 Generated {len(new_tasks)} training tasks "
                      f"(queue: {len(queued)}→{len(queued)+len(new_tasks)}, "
                      f"running: {len(running)}, workers: {len(online_workers)})")

        # 3. Try to assign queued jobs to idle workers
        idle_workers = [w for w in online_workers
                       if not w.get("current_job") and w.get("status") != "throttled"]

        for worker in idle_workers:
            for job in queued:
                if job["status"] != "queued":
                    continue
                if self._worker_matches_job(worker, job):
                    job["status"] = "running"
                    job["assigned_worker"] = worker["worker_id"]
                    job["started_at"] = datetime.now(timezone.utc).isoformat()
                    worker["current_job"] = job["job_id"]
                    print(f"  📋 Assigned: {job['name']} → {worker.get('hostname', worker['worker_id'])}")
                    break

    def _generate_tasks(self, count: int, workers: List[Dict]) -> List[Dict]:
        """Generate training tasks based on available workers."""
        tasks = []
        has_gpu = any(
            w.get("hardware", {}).get("gpu", {}).get("cuda_available", False)
            or w.get("hardware", {}).get("gpu", {}).get("name", "none") != "none"
            for w in workers
        )

        # Select layers based on available hardware
        available_layers = []
        for layer in TRAINING_LAYERS:
            if layer["requires_gpu"] and not has_gpu:
                continue
            available_layers.append(layer)

        if not available_layers:
            available_layers = [l for l in TRAINING_LAYERS if not l["requires_gpu"]]

        for _ in range(count):
            layer = random.choice(available_layers)
            topics = TRAINING_TOPICS.get(layer["name"], ["تدريب عام"])
            topic = random.choice(topics)

            import uuid
            job_id = str(uuid.uuid4())[:12]

            labels = ["gpu"] if layer["requires_gpu"] else ["cpu"]

            task = {
                "job_id": job_id,
                "name": f"{layer['display']}: {topic}",
                "command": "",  # Worker uses default training
                "shell": False,
                "target_labels": labels,
                "priority": layer["priority"],
                "config": {
                    "layer": layer["name"],
                    "topic": topic,
                    "data_sources": layer["data_sources"],
                },
                "layer_name": layer["name"],
                "auto_sync_to_primary": True,
                "status": "queued",
                "assigned_worker": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "artifacts": [],
                "logs": [],
            }
            tasks.append(task)

        return tasks

    def _worker_matches_job(self, worker: Dict, job: Dict) -> bool:
        """Check if a worker can run a specific job."""
        worker_labels = set(worker.get("labels", []))
        job_labels = set(job.get("target_labels", []))

        if not job_labels:
            return True  # No label requirement

        # Check label intersection
        if not worker_labels.intersection(job_labels):
            return False

        # Hostinger: skip GPU jobs
        if "hostinger" in worker_labels and "gpu" in job_labels:
            return False

        return True

    def _record_result(self, job: Dict):
        """Record training result for future scheduling optimization."""
        try:
            record = {
                "job_id": job["job_id"],
                "layer": job.get("layer_name"),
                "worker": job.get("assigned_worker"),
                "status": job["status"],
                "duration": None,
                "metrics": job.get("result"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if job.get("started_at") and job.get("completed_at"):
                # Calculate duration
                pass

            with open(self.history_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        if not hasattr(self, 'state'):
            return {"is_running": self.is_running, "cycle_count": self.cycle_count}

        return {
            "is_running": self.is_running,
            "cycle_count": self.cycle_count,
            "queued_jobs": sum(1 for j in self.state.jobs.values() if j["status"] == "queued"),
            "running_jobs": sum(1 for j in self.state.jobs.values() if j["status"] == "running"),
            "completed_jobs": sum(1 for j in self.state.jobs.values() if j["status"] == "completed"),
            "total_workers": len(self.state.workers),
            "layers_count": len(TRAINING_LAYERS),
        }


# ──────────── Singleton ────────────

training_coordinator = TrainingCoordinator()
