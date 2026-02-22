from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


NETWORK_STATE_FILE = Path("data/learning/distributed-training-state.json")
NETWORK_GRAPH_FILE = Path("data/knowledge/specialization-network.json")
ARTIFACTS_DIR = Path("data/learning/distributed-artifacts")


@dataclass
class SpecializationNode:
    node_id: str
    name: str
    description: str
    parent_id: Optional[str]
    depth: int
    path: str
    ai_current_evolver: str
    ai_zero_reinventor: str
    children: List[str]


@dataclass
class WorkerInfo:
    worker_id: str
    hostname: str
    capabilities: Dict[str, Any]
    status: str
    last_seen: str


@dataclass
class TrainingTask:
    task_id: str
    node_id: str
    topic: str
    priority: int
    status: str
    assigned_worker: Optional[str]
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]]


class SpecializedNetworkService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.nodes: Dict[str, SpecializationNode] = {}
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, TrainingTask] = {}
        self._ensure_dirs()
        self._load_or_seed_graph()
        self._load_state()

    def _ensure_dirs(self) -> None:
        NETWORK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        NETWORK_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _seed_graph(self) -> Dict[str, Dict[str, Any]]:
        seed = {
            "root-math": {
                "node_id": "root-math",
                "name": "الرياضيات",
                "description": "الجذر المعرفي للمنطق والنمذجة",
                "parent_id": None,
                "depth": 0,
                "path": "الرياضيات",
                "ai_current_evolver": "Math-Evolver-v1",
                "ai_zero_reinventor": "Math-Reinventor-v1",
                "children": ["math-physics", "math-statistics", "math-optimization"],
            },
            "math-physics": {
                "node_id": "math-physics",
                "name": "الرياضيات الفيزيائية",
                "description": "نماذج المعادلات التفاضلية والأنظمة الفيزيائية",
                "parent_id": "root-math",
                "depth": 1,
                "path": "الرياضيات/الرياضيات الفيزيائية",
                "ai_current_evolver": "PhysMath-Evolver-v1",
                "ai_zero_reinventor": "PhysMath-Reinventor-v1",
                "children": ["physics-fluid", "physics-quantum"],
            },
            "math-statistics": {
                "node_id": "math-statistics",
                "name": "الإحصاء والاحتمالات",
                "description": "الاستدلال الإحصائي والنمذجة الاحتمالية",
                "parent_id": "root-math",
                "depth": 1,
                "path": "الرياضيات/الإحصاء",
                "ai_current_evolver": "Stats-Evolver-v1",
                "ai_zero_reinventor": "Stats-Reinventor-v1",
                "children": ["stats-bayesian", "stats-time-series"],
            },
            "math-optimization": {
                "node_id": "math-optimization",
                "name": "التحسين",
                "description": "تحسين الأداء والموارد والخوارزميات",
                "parent_id": "root-math",
                "depth": 1,
                "path": "الرياضيات/التحسين",
                "ai_current_evolver": "Opt-Evolver-v1",
                "ai_zero_reinventor": "Opt-Reinventor-v1",
                "children": ["opt-convex", "opt-discrete"],
            },
            "physics-fluid": {
                "node_id": "physics-fluid",
                "name": "ميكانيكا الموائع",
                "description": "نماذج التدفق والمحاكاة",
                "parent_id": "math-physics",
                "depth": 2,
                "path": "الرياضيات/الرياضيات الفيزيائية/ميكانيكا الموائع",
                "ai_current_evolver": "Fluid-Evolver-v1",
                "ai_zero_reinventor": "Fluid-Reinventor-v1",
                "children": [],
            },
            "physics-quantum": {
                "node_id": "physics-quantum",
                "name": "النمذجة الكمية",
                "description": "الاحتمالات الكمية والمحاكاة",
                "parent_id": "math-physics",
                "depth": 2,
                "path": "الرياضيات/الرياضيات الفيزيائية/النمذجة الكمية",
                "ai_current_evolver": "Quantum-Evolver-v1",
                "ai_zero_reinventor": "Quantum-Reinventor-v1",
                "children": [],
            },
            "stats-bayesian": {
                "node_id": "stats-bayesian",
                "name": "الاستدلال البايزي",
                "description": "تحديث المعتقدات عبر البيانات",
                "parent_id": "math-statistics",
                "depth": 2,
                "path": "الرياضيات/الإحصاء/الاستدلال البايزي",
                "ai_current_evolver": "Bayes-Evolver-v1",
                "ai_zero_reinventor": "Bayes-Reinventor-v1",
                "children": [],
            },
            "stats-time-series": {
                "node_id": "stats-time-series",
                "name": "السلاسل الزمنية",
                "description": "التنبؤ والتحليل الزمني",
                "parent_id": "math-statistics",
                "depth": 2,
                "path": "الرياضيات/الإحصاء/السلاسل الزمنية",
                "ai_current_evolver": "TS-Evolver-v1",
                "ai_zero_reinventor": "TS-Reinventor-v1",
                "children": [],
            },
            "opt-convex": {
                "node_id": "opt-convex",
                "name": "التحسين المحدب",
                "description": "حلول عالمية لمسائل محدبة",
                "parent_id": "math-optimization",
                "depth": 2,
                "path": "الرياضيات/التحسين/التحسين المحدب",
                "ai_current_evolver": "Convex-Evolver-v1",
                "ai_zero_reinventor": "Convex-Reinventor-v1",
                "children": [],
            },
            "opt-discrete": {
                "node_id": "opt-discrete",
                "name": "التحسين المتقطع",
                "description": "الجدولة والمسائل التوافقية",
                "parent_id": "math-optimization",
                "depth": 2,
                "path": "الرياضيات/التحسين/التحسين المتقطع",
                "ai_current_evolver": "Discrete-Evolver-v1",
                "ai_zero_reinventor": "Discrete-Reinventor-v1",
                "children": [],
            },
        }
        return seed

    def _load_or_seed_graph(self) -> None:
        if not NETWORK_GRAPH_FILE.exists():
            NETWORK_GRAPH_FILE.write_text(json.dumps(self._seed_graph(), ensure_ascii=False, indent=2), encoding="utf-8")

        raw = json.loads(NETWORK_GRAPH_FILE.read_text(encoding="utf-8") or "{}")
        for node_id, node_data in raw.items():
            self.nodes[node_id] = SpecializationNode(**node_data)

    def _load_state(self) -> None:
        if not NETWORK_STATE_FILE.exists():
            NETWORK_STATE_FILE.write_text(json.dumps({"workers": {}, "tasks": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        raw = json.loads(NETWORK_STATE_FILE.read_text(encoding="utf-8") or "{}")
        for worker_id, worker_data in raw.get("workers", {}).items():
            self.workers[worker_id] = WorkerInfo(**worker_data)
        for task_id, task_data in raw.get("tasks", {}).items():
            self.tasks[task_id] = TrainingTask(**task_data)

    def _persist_state(self) -> None:
        payload = {
            "workers": {worker_id: asdict(worker) for worker_id, worker in self.workers.items()},
            "tasks": {task_id: asdict(task) for task_id, task in self.tasks.items()},
        }
        NETWORK_STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_graph(self) -> None:
        payload = {node_id: asdict(node) for node_id, node in self.nodes.items()}
        NETWORK_GRAPH_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_graph(self) -> Dict[str, Any]:
        with self._lock:
            roots = [node for node in self.nodes.values() if node.parent_id is None]
            return {
                "nodes": [asdict(node) for node in self.nodes.values()],
                "roots": [asdict(root) for root in roots],
                "total_nodes": len(self.nodes),
            }

    def expand_specialization(self, parent_id: str, name: str, description: str) -> Dict[str, Any]:
        with self._lock:
            parent = self.nodes.get(parent_id)
            if not parent:
                raise ValueError("parent not found")

            node_id = f"node-{uuid.uuid4().hex[:10]}"
            path = f"{parent.path}/{name}"
            new_node = SpecializationNode(
                node_id=node_id,
                name=name,
                description=description,
                parent_id=parent_id,
                depth=parent.depth + 1,
                path=path,
                ai_current_evolver=f"{name}-Evolver-v1",
                ai_zero_reinventor=f"{name}-Reinventor-v1",
                children=[],
            )
            self.nodes[node_id] = new_node
            parent.children.append(node_id)
            self._persist_graph()
            return asdict(new_node)

    def register_worker(self, worker_id: str, hostname: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        with self._lock:
            worker = WorkerInfo(
                worker_id=worker_id,
                hostname=hostname,
                capabilities=capabilities or {},
                status="online",
                last_seen=now,
            )
            self.workers[worker_id] = worker
            self._persist_state()
            return asdict(worker)

    def heartbeat_worker(self, worker_id: str, capabilities: Optional[Dict[str, Any]] = None, status: str = "online") -> Dict[str, Any]:
        now = datetime.now().isoformat()
        with self._lock:
            worker = self.workers.get(worker_id)
            if not worker:
                worker = WorkerInfo(
                    worker_id=worker_id,
                    hostname=worker_id,
                    capabilities=capabilities or {},
                    status=status,
                    last_seen=now,
                )
                self.workers[worker_id] = worker
            else:
                if capabilities:
                    worker.capabilities = capabilities
                worker.status = status
                worker.last_seen = now
            self._persist_state()
            return asdict(worker)

    def enqueue_training_task(self, topic: str, node_id: Optional[str] = None, priority: int = 5) -> Dict[str, Any]:
        with self._lock:
            target_node_id = node_id or "root-math"
            if target_node_id not in self.nodes:
                raise ValueError("target node not found")

            now = datetime.now().isoformat()
            task = TrainingTask(
                task_id=f"task-{uuid.uuid4().hex[:12]}",
                node_id=target_node_id,
                topic=topic,
                priority=max(1, min(10, int(priority))),
                status="queued",
                assigned_worker=None,
                created_at=now,
                updated_at=now,
                result=None,
            )
            self.tasks[task.task_id] = task
            self._persist_state()
            return asdict(task)

    def claim_training_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            online_worker = self.workers.get(worker_id)
            if not online_worker or online_worker.status != "online":
                return None

            queued = [task for task in self.tasks.values() if task.status == "queued"]
            if not queued:
                return None

            queued.sort(key=lambda item: (item.priority, item.created_at), reverse=True)
            task = queued[0]
            task.status = "running"
            task.assigned_worker = worker_id
            task.updated_at = datetime.now().isoformat()
            self._persist_state()
            return asdict(task)

    def complete_training_task(
        self,
        task_id: str,
        worker_id: str,
        metrics: Dict[str, Any],
        artifact_name: Optional[str] = None,
        artifact_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError("task not found")
            if task.assigned_worker and task.assigned_worker != worker_id:
                raise ValueError("worker does not own this task")

            persisted_artifact_name = artifact_name
            if artifact_payload is not None:
                safe_name = artifact_name or f"artifact-{task_id}-{worker_id}.json"
                artifact_path = ARTIFACTS_DIR / safe_name
                artifact_path.write_text(
                    json.dumps(artifact_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                persisted_artifact_name = safe_name

            task.status = "completed"
            task.updated_at = datetime.now().isoformat()
            task.result = {
                "worker_id": worker_id,
                "metrics": metrics or {},
                "artifact_name": persisted_artifact_name,
                "artifact_uploaded": artifact_payload is not None,
                "completed_at": task.updated_at,
            }
            self._persist_state()
            return asdict(task)

    def dual_thought(self, node_id: str, prompt: str) -> Dict[str, Any]:
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                raise ValueError("node not found")

            return {
                "node_id": node_id,
                "node_name": node.name,
                "current_evolution_thought": {
                    "ai": node.ai_current_evolver,
                    "focus": f"تحسين تدريجي داخل اختصاص {node.name}",
                    "plan": [
                        f"تحليل baseline الحالي في {node.name}",
                        "تحسين الدقة تدريجياً بنسبة 3-7% لكل دورة",
                        "تقليل استهلاك الموارد عبر pruning/quantization"
                    ],
                    "prompt": prompt,
                },
                "zero_reinvention_thought": {
                    "ai": node.ai_zero_reinventor,
                    "focus": f"إعادة بناء من الصفر لاختصاص {node.name}",
                    "plan": [
                        "تصميم بنية جديدة بالكامل أقل تعقيداً",
                        "اختبار خوارزميات بديلة بكفاءة أعلى",
                        "اختيار pipeline تدريب أقصر بزمن convergence أقل"
                    ],
                    "prompt": prompt,
                }
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            queued = sum(1 for task in self.tasks.values() if task.status == "queued")
            running = sum(1 for task in self.tasks.values() if task.status == "running")
            completed = sum(1 for task in self.tasks.values() if task.status == "completed")
            online_workers = sum(1 for worker in self.workers.values() if worker.status == "online")
            return {
                "nodes": len(self.nodes),
                "workers_total": len(self.workers),
                "workers_online": online_workers,
                "tasks": {
                    "queued": queued,
                    "running": running,
                    "completed": completed,
                    "total": len(self.tasks),
                }
            }


_specialized_network_instance: Optional[SpecializedNetworkService] = None


def get_specialized_network_service() -> SpecializedNetworkService:
    global _specialized_network_instance
    if _specialized_network_instance is None:
        _specialized_network_instance = SpecializedNetworkService()
    return _specialized_network_instance
