"""
🎛️ Orchestrator API — المنسق المركزي V2

يدير:
- Workers (عقد التدريب) → تسجيل + heartbeat + WebSocket حي
- Jobs (مهام التدريب) → إنشاء + توزيع + تتبع
- Brain Auto-Scheduling (NEW)
- Artifacts (checkpoints + نماذج) → رفع + تحميل + مزامنة
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
    Query,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Import brain for auto-scheduling
from brain import brain, BIBrain

# ──────────── Config ────────────

ORCHESTRATOR_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "")
ARTIFACTS_DIR = Path(os.getenv("ORCHESTRATOR_ARTIFACTS_DIR", "data/orchestrator/artifacts"))
JOBS_DIR = Path(os.getenv("ORCHESTRATOR_JOBS_DIR", "data/orchestrator/jobs"))
HOSTINGER_CPU_LIMIT = float(os.getenv("HOSTINGER_CPU_LIMIT", "75"))
HOSTINGER_CPU_WINDOW_SEC = int(os.getenv("HOSTINGER_CPU_WINDOW_SEC", str(3 * 3600)))

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])


# ──────────── Auth ────────────

def verify_token(x_orchestrator_token: str = Header(default="")):
    if ORCHESTRATOR_TOKEN and x_orchestrator_token != ORCHESTRATOR_TOKEN:
        raise HTTPException(401, "Invalid orchestrator token")
    return x_orchestrator_token


# ──────────── Models ────────────

class WorkerRegister(BaseModel):
    worker_id: str = ""
    hostname: str = ""
    labels: List[str] = Field(default_factory=list)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"


class WorkerHeartbeat(BaseModel):
    worker_id: str
    status: str = "online"
    usage: Dict[str, Any] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(default_factory=dict)


class JobCreate(BaseModel):
    name: str
    command: str = ""
    shell: bool = False
    target_labels: List[str] = Field(default_factory=list)
    priority: int = 5
    config: Dict[str, Any] = Field(default_factory=dict)
    layer_name: str = ""
    auto_sync_to_primary: bool = True


class JobBatch(BaseModel):
    jobs: List[JobCreate]


class AutoScheduleRequest(BaseModel):
    """طلب الجدولة التلقائية"""
    priority: str = "medium"  # critical, high, medium, low, idle
    layer_name: str = "general"
    config: Dict[str, Any] = Field(default_factory=dict)


class AutoScheduleResponse(BaseModel):
    """استجابة الجدولة التلقائية"""
    job_id: str
    status: str
    scheduled_at: str
    estimated_start: Optional[str] = None
    assigned_worker: Optional[str] = None
    priority: str
    layer_name: str


class BrainStatusResponse(BaseModel):
    """استجابة حالة الدماغ"""
    is_active: bool
    scheduler_status: str
    jobs_in_queue: int
    active_jobs: int
    completed_jobs_today: int
    average_job_duration_minutes: int
    next_scheduled_job: Optional[Dict[str, Any]] = None


# ──────────── State ────────────

class OrchestratorState:
    """In-memory orchestrator state."""

    def __init__(self):
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.cpu_history: deque = deque(maxlen=360)
        self.startup_time = datetime.now(timezone.utc).isoformat()
        self._load_persisted_jobs()
        
        # Initialize brain
        self.brain: Optional[BIBrain] = None
        self._init_brain()
    
    def _init_brain(self):
        """Initialize brain for auto-scheduling"""
        try:
            self.brain = brain
            logger.info("Brain auto-scheduling initialized")
        except Exception as e:
            logger.warning(f"Brain initialization failed: {e}")

    def _load_persisted_jobs(self):
        """Load any previously persisted jobs."""
        state_file = JOBS_DIR / "jobs_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.jobs = data.get("jobs", {})
                print(f"📂 Loaded {len(self.jobs)} persisted jobs")
            except Exception:
                pass

    def _persist_jobs(self):
        """Persist job state to disk."""
        try:
            state_file = JOBS_DIR / "jobs_state.json"
            state_file.write_text(json.dumps({
                "jobs": self.jobs,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def get_primary_worker(self) -> Optional[Dict[str, Any]]:
        """Get the primary (RTX 5090) worker."""
        for w in self.workers.values():
            if "primary" in w.get("labels", []) or "rtx5090" in w.get("labels", []):
                return w
        return None

    def check_hostinger_throttle(self, cpu_percent: float) -> bool:
        """Check if Hostinger should be throttled."""
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) >= 180:
            avg = sum(self.cpu_history) / len(self.cpu_history)
            if avg > HOSTINGER_CPU_LIMIT:
                return True
        return False


state = OrchestratorState()


# ──────────── Health ────────────

@router.get("/health")
async def orchestrator_health():
    online = [w for w in state.workers.values() if w.get("status") != "offline"]
    primary = state.get_primary_worker()
    return {
        "status": "ok",
        "uptime_since": state.startup_time,
        "workers_total": len(state.workers),
        "workers_online": len(online),
        "primary_connected": primary is not None and primary.get("status") != "offline",
        "jobs_total": len(state.jobs),
        "jobs_running": sum(1 for j in state.jobs.values() if j["status"] == "running"),
        "brain_active": state.brain is not None,
    }


# ──────────── NEW: Brain Auto-Scheduling ────────────

@router.post("/auto-schedule", response_model=AutoScheduleResponse)
async def auto_schedule_job(
    request: AutoScheduleRequest,
    _=Depends(verify_token)
):
    """
    تلقائيًا جدولة مهمة تدريب
    
    Automatically schedule a training job based on priority.
    """
    if not state.brain:
        raise HTTPException(503, "Brain service not available")
    
    try:
        job_id = await state.brain.schedule_training(
            name=f"Auto-scheduled {request.layer_name} training",
            layer_name=request.layer_name,
            priority=request.priority,
            config=request.config
        )
        
        return AutoScheduleResponse(
            job_id=job_id,
            status="scheduled",
            scheduled_at=datetime.now(timezone.utc).isoformat(),
            estimated_start=None,  # Will be determined by scheduler
            assigned_worker=None,
            priority=request.priority,
            layer_name=request.layer_name
        )
        
    except Exception as e:
        logger.error(f"Auto-schedule failed: {e}")
        raise HTTPException(500, f"Failed to schedule job: {str(e)}")


@router.get("/brain/status", response_model=BrainStatusResponse)
async def get_brain_status(_=Depends(verify_token)):
    """
    الحصول على حالة الدماغ والجدولة
    
    Get brain scheduler status.
    """
    if not state.brain:
        return BrainStatusResponse(
            is_active=False,
            scheduler_status="not_initialized",
            jobs_in_queue=0,
            active_jobs=0,
            completed_jobs_today=0,
            average_job_duration_minutes=0
        )
    
    brain_status = state.brain.get_status()
    
    return BrainStatusResponse(
        is_active=brain_status["is_running"],
        scheduler_status="running" if brain_status["is_running"] else "stopped",
        jobs_in_queue=brain_status["scheduler"]["pending_jobs"],
        active_jobs=brain_status["scheduler"]["running_jobs"],
        completed_jobs_today=brain_status["scheduler"]["completed_jobs"],
        average_job_duration_minutes=45,  # Placeholder
        next_scheduled_job=None  # Could be populated from scheduler
    )


@router.get("/brain/jobs")
async def list_brain_jobs(
    status: str = None,
    limit: int = 50,
    _=Depends(verify_token)
):
    """List scheduled brain jobs"""
    if not state.brain:
        raise HTTPException(503, "Brain service not available")
    
    jobs = state.brain.get_jobs(status=status, limit=limit)
    return {"jobs": jobs}


@router.get("/brain/evaluations")
async def list_brain_evaluations(
    model_id: str = None,
    limit: int = 50,
    _=Depends(verify_token)
):
    """List model evaluations"""
    if not state.brain:
        raise HTTPException(503, "Brain service not available")
    
    evaluations = state.brain.get_evaluations(model_id=model_id, limit=limit)
    return {"evaluations": evaluations}


# ──────────── Workers (existing endpoints) ────────────

@router.post("/workers/register")
async def register_worker(req: WorkerRegister, _=Depends(verify_token)):
    wid = req.worker_id or str(uuid.uuid4())[:12]

    # Re-register by hostname if exists
    for existing_id, w in state.workers.items():
        if w["hostname"] == req.hostname and req.hostname:
            wid = existing_id
            break

    worker = {
        "worker_id": wid,
        "hostname": req.hostname,
        "labels": req.labels,
        "hardware": req.hardware,
        "version": req.version,
        "status": "online",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat": time.time(),
        "current_job": None,
    }
    state.workers[wid] = worker

    is_primary = "primary" in req.labels or "rtx5090" in req.labels
    role = "🏆 PRIMARY" if is_primary else "🔧 HELPER"
    gpu_name = req.hardware.get("gpu", {}).get("name", "none")
    print(f"📡 Worker registered: {role} {req.hostname} [{wid}] — {gpu_name}")

    # Broadcast to all connected websockets
    await _broadcast({
        "type": "worker_registered",
        "worker": worker,
    })

    return {"status": "ok", "worker_id": wid, "role": "primary" if is_primary else "helper"}


@router.post("/workers/heartbeat")
async def worker_heartbeat(req: WorkerHeartbeat, _=Depends(verify_token)):
    w = state.workers.get(req.worker_id)
    if not w:
        raise HTTPException(404, "Worker not found. Re-register.")

    w["status"] = req.status
    w["last_heartbeat"] = time.time()

    if req.usage:
        w["usage"] = req.usage
        # Hostinger throttle check
        if "hostinger" in w.get("labels", []) or "orchestrator" in w.get("labels", []):
            cpu = req.usage.get("cpu_percent", 0)
            throttled = state.check_hostinger_throttle(cpu)
            if throttled:
                w["status"] = "throttled"
                return {
                    "status": "ok",
                    "command": "throttle",
                    "message": f"CPU المعدل > {HOSTINGER_CPU_LIMIT}% — قلل التدريب",
                }

    if req.training:
        w["training"] = req.training

    # Check if there's a pending command
    pending = w.pop("_pending_command", None)
    if pending:
        return {"status": "ok", "command": pending["command"], "params": pending.get("params", {})}

    return {"status": "ok"}


@router.get("/workers")
async def list_workers(_=Depends(verify_token)):
    now = time.time()
    workers = []
    for w in state.workers.values():
        w_copy = dict(w)
        # Mark as offline if no heartbeat for 90s
        if now - w.get("last_heartbeat", 0) > 90:
            w_copy["status"] = "offline"
            w["status"] = "offline"
        w_copy["last_heartbeat_ago"] = f"{now - w.get('last_heartbeat', now):.0f}s"
        workers.append(w_copy)
    return {"status": "ok", "workers": workers}


# ──────────── Jobs (existing endpoints) ────────────

@router.post("/jobs")
async def create_job(req: JobCreate, _=Depends(verify_token)):
    job_id = str(uuid.uuid4())[:12]
    job = {
        "job_id": job_id,
        "name": req.name,
        "command": req.command,
        "shell": req.shell,
        "target_labels": req.target_labels,
        "priority": req.priority,
        "config": req.config,
        "layer_name": req.layer_name,
        "auto_sync_to_primary": req.auto_sync_to_primary,
        "status": "queued",
        "assigned_worker": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "artifacts": [],
        "logs": [],
    }
    state.jobs[job_id] = job
    state._persist_jobs()

    # Try auto-assign to a matching worker
    await _try_assign_job(job)

    await _broadcast({"type": "job_created", "job": job})

    return {"status": "ok", "job": job}


@router.get("/jobs")
async def list_jobs(status: str = None, limit: int = 50, _=Depends(verify_token)):
    jobs = list(state.jobs.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"status": "ok", "jobs": jobs[:limit], "total": len(jobs)}


# ──────────── Internal Helpers ────────────

async def _broadcast(msg: Dict[str, Any], exclude: str = ""):
    """Broadcast message to all connected dashboard clients."""
    dead = []
    for ws in _dashboard_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _dashboard_clients.remove(ws) if ws in _dashboard_clients else None


async def _try_assign_job(job: Dict[str, Any]):
    """Try to assign a queued job to a matching online worker."""
    for wid, w in state.workers.items():
        if w.get("status") not in ("online", "idle"):
            continue
        if w.get("current_job"):
            continue
        if job["target_labels"]:
            if not set(job["target_labels"]).intersection(set(w.get("labels", []))):
                continue

        # Send via websocket if connected
        ws = state.websockets.get(wid)
        if ws:
            try:
                await ws.send_json({"type": "new_job", "job": job})
            except Exception:
                pass


_dashboard_clients: List[WebSocket] = []


import logging
logger = logging.getLogger(__name__)
