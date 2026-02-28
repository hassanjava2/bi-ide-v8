"""
🎛️ Orchestrator API — المنسق المركزي

يدير:
- Workers (عقد التدريب) → تسجيل + heartbeat + WebSocket حي
- Jobs (مهام التدريب) → إنشاء + توزيع + تتبع
- Artifacts (checkpoints + نماذج) → رفع + تحميل + مزامنة
- Throttle → Hostinger CPU حماية + RTX 5090 أولوية

يتكامل مع:
- hierarchy/specialized_ai_network.py → SpecializedNetworkService
- api/routes/training_data.py → relay
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

# ──────────── Config ────────────

ORCHESTRATOR_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "")
ARTIFACTS_DIR = Path(os.getenv("ORCHESTRATOR_ARTIFACTS_DIR", "data/orchestrator/artifacts"))
JOBS_DIR = Path(os.getenv("ORCHESTRATOR_JOBS_DIR", "data/orchestrator/jobs"))
HOSTINGER_CPU_LIMIT = float(os.getenv("HOSTINGER_CPU_LIMIT", "75"))
HOSTINGER_CPU_WINDOW_SEC = int(os.getenv("HOSTINGER_CPU_WINDOW_SEC", str(3 * 3600)))  # 3 hours

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


# ──────────── State ────────────

class OrchestratorState:
    """In-memory orchestrator state."""

    def __init__(self):
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.cpu_history: deque = deque(maxlen=360)  # 30s intervals × 360 = 3hrs
        self.startup_time = datetime.now(timezone.utc).isoformat()
        self._load_persisted_jobs()

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
        if len(self.cpu_history) >= 180:  # at least 1.5hrs of data
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
    }


# ──────────── Workers ────────────

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


@router.post("/workers/{worker_id}/command")
async def send_worker_command(worker_id: str, command: str, params: Dict[str, Any] = None,
                              _=Depends(verify_token)):
    w = state.workers.get(worker_id)
    if not w:
        raise HTTPException(404, "Worker not found")

    # Try WebSocket first
    ws = state.websockets.get(worker_id)
    if ws:
        try:
            await ws.send_json({
                "type": "command",
                "command": command,
                "params": params or {},
            })
            return {"status": "ok", "delivery": "websocket"}
        except Exception:
            del state.websockets[worker_id]

    # Fallback: queue for next heartbeat
    w["_pending_command"] = {"command": command, "params": params or {}}
    return {"status": "ok", "delivery": "queued_for_heartbeat"}


@router.delete("/workers/{worker_id}")
async def remove_worker(worker_id: str, _=Depends(verify_token)):
    if worker_id not in state.workers:
        raise HTTPException(404, "Worker not found")
    del state.workers[worker_id]
    ws = state.websockets.pop(worker_id, None)
    if ws:
        try:
            await ws.close()
        except Exception:
            pass
    return {"status": "ok"}


# ──────────── Jobs ────────────

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


@router.post("/jobs/batch")
async def create_jobs_batch(req: JobBatch, _=Depends(verify_token)):
    created = []
    for j in req.jobs:
        result = await create_job(j)
        created.append(result["job"])
    return {"status": "ok", "jobs": created, "count": len(created)}


@router.get("/jobs")
async def list_jobs(status: str = None, limit: int = 50, _=Depends(verify_token)):
    jobs = list(state.jobs.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    # Sort by creation time, newest first
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"status": "ok", "jobs": jobs[:limit], "total": len(jobs)}


@router.get("/jobs/next")
async def get_next_job(worker_id: str, labels: str = "", _=Depends(verify_token)):
    """Worker polls for the next available job matching its labels."""
    worker_labels = set(labels.split(",")) if labels else set()

    # Check if worker is throttled
    w = state.workers.get(worker_id)
    if w and w.get("status") == "throttled":
        return {"status": "ok", "job": None, "reason": "throttled"}

    # Find highest priority queued job matching labels
    candidates = []
    for job in state.jobs.values():
        if job["status"] != "queued":
            continue
        if job["target_labels"]:
            if not worker_labels or not set(job["target_labels"]).intersection(worker_labels):
                continue
        candidates.append(job)

    if not candidates:
        return {"status": "ok", "job": None}

    candidates.sort(key=lambda j: j.get("priority", 5), reverse=True)
    best = candidates[0]

    # Auto-claim
    best["status"] = "running"
    best["assigned_worker"] = worker_id
    best["started_at"] = datetime.now(timezone.utc).isoformat()
    if w:
        w["current_job"] = best["job_id"]
    state._persist_jobs()

    return {"status": "ok", "job": best}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, _=Depends(verify_token)):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": "ok", "job": job}


@router.post("/jobs/{job_id}/claim")
async def claim_job(job_id: str, worker_id: str, _=Depends(verify_token)):
    """Worker claims a queued job."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "queued":
        raise HTTPException(400, f"Job is {job['status']}, not queued")

    w = state.workers.get(worker_id)
    if not w:
        raise HTTPException(404, "Worker not found")

    job["status"] = "running"
    job["assigned_worker"] = worker_id
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    w["current_job"] = job_id
    state._persist_jobs()

    await _broadcast({"type": "job_started", "job": job})
    return {"status": "ok", "job": job}


@router.post("/jobs/{job_id}/complete")
async def complete_job(job_id: str, worker_id: str,
                       metrics: Dict[str, Any] = None,
                       _=Depends(verify_token)):
    """Worker completes a job."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    job["status"] = "completed"
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    job["result"] = metrics or {}
    state._persist_jobs()

    w = state.workers.get(worker_id)
    if w:
        w["current_job"] = None

    await _broadcast({"type": "job_completed", "job": job})
    return {"status": "ok", "job": job}


@router.post("/jobs/{job_id}/fail")
async def fail_job(job_id: str, worker_id: str, error: str = "",
                   _=Depends(verify_token)):
    """Mark a job as failed."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    job["status"] = "failed"
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    job["result"] = {"error": error}
    state._persist_jobs()

    w = state.workers.get(worker_id)
    if w:
        w["current_job"] = None

    await _broadcast({"type": "job_failed", "job": job, "error": error})
    return {"status": "ok"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, _=Depends(verify_token)):
    """Cancel a queued or running job."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] == "running" and job.get("assigned_worker"):
        # Send stop signal to worker
        await send_worker_command(job["assigned_worker"], "stop_job", {"job_id": job_id})

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    state._persist_jobs()

    await _broadcast({"type": "job_cancelled", "job": job})
    return {"status": "ok"}


@router.post("/jobs/{job_id}/log")
async def append_job_log(job_id: str, line: str, _=Depends(verify_token)):
    """Worker appends a log line to a job."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if "logs" not in job:
        job["logs"] = []
    job["logs"].append({
        "time": datetime.now(timezone.utc).isoformat(),
        "line": line,
    })
    # Keep last 500 log lines
    if len(job["logs"]) > 500:
        job["logs"] = job["logs"][-500:]
    return {"status": "ok"}




# ──────────── Artifacts ────────────

@router.post("/jobs/{job_id}/artifacts/upload")
async def upload_artifact(job_id: str, file: UploadFile = File(...),
                          _=Depends(verify_token)):
    """Upload a training artifact (checkpoint, model, etc.)."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    artifact_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"artifact_{artifact_id}"
    filepath = job_dir / filename

    content = await file.read()
    filepath.write_bytes(content)

    checksum = hashlib.sha256(content).hexdigest()[:16]

    artifact_info = {
        "artifact_id": artifact_id,
        "filename": filename,
        "size_bytes": len(content),
        "checksum_sha256": checksum,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "path": str(filepath),
    }

    if "artifacts" not in job:
        job["artifacts"] = []
    job["artifacts"].append(artifact_info)
    state._persist_jobs()

    # Auto-sync to primary if configured
    if job.get("auto_sync_to_primary"):
        asyncio.create_task(_sync_artifact_to_primary(artifact_info, content))

    return {"status": "ok", "artifact": artifact_info}


@router.get("/jobs/{job_id}/artifacts")
async def list_artifacts(job_id: str, _=Depends(verify_token)):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": "ok", "artifacts": job.get("artifacts", [])}


@router.get("/jobs/{job_id}/artifacts/{artifact_id}/download")
async def download_artifact(job_id: str, artifact_id: str, _=Depends(verify_token)):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    for a in job.get("artifacts", []):
        if a["artifact_id"] == artifact_id:
            path = Path(a["path"])
            if path.exists():
                return FileResponse(path, filename=a["filename"])
            raise HTTPException(404, "Artifact file missing")
    raise HTTPException(404, "Artifact not found")


# ──────────── Downloads (Agent Install Scripts) ────────────

@router.get("/download/linux")
async def download_linux_agent():
    """Download Linux/Mac install script."""
    script = _generate_install_script("linux")
    return HTMLResponse(content=script, media_type="text/plain")


@router.get("/download/macos")
async def download_macos_agent():
    """Download macOS install script."""
    script = _generate_install_script("macos")
    return HTMLResponse(content=script, media_type="text/plain")


@router.get("/download/windows")
async def download_windows_agent():
    """Download Windows install script."""
    script = _generate_install_script("windows")
    return HTMLResponse(content=script, media_type="text/plain")


# ──────────── WebSocket ────────────

@router.websocket("/ws/{worker_id}")
async def websocket_endpoint(websocket: WebSocket, worker_id: str):
    """Live WebSocket connection for a worker node."""
    # Verify token from query param
    token = websocket.query_params.get("token", "")
    if ORCHESTRATOR_TOKEN and token != ORCHESTRATOR_TOKEN:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    state.websockets[worker_id] = websocket
    print(f"🔌 WebSocket connected: {worker_id}")

    # Mark worker as online
    w = state.workers.get(worker_id)
    if w:
        w["status"] = "online"
        w["last_heartbeat"] = time.time()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "heartbeat":
                if w:
                    w["last_heartbeat"] = time.time()
                    w["status"] = data.get("status", "online")
                    if data.get("usage"):
                        w["usage"] = data["usage"]
                    if data.get("training"):
                        w["training"] = data["training"]

                    # Hostinger throttle
                    if "hostinger" in w.get("labels", []):
                        cpu = data.get("usage", {}).get("cpu_percent", 0)
                        if state.check_hostinger_throttle(cpu):
                            await websocket.send_json({
                                "type": "command",
                                "command": "throttle",
                                "message": f"CPU avg > {HOSTINGER_CPU_LIMIT}%",
                            })

                await websocket.send_json({"type": "heartbeat_ack"})

            elif msg_type == "job_progress":
                job_id = data.get("job_id")
                if job_id and job_id in state.jobs:
                    job = state.jobs[job_id]
                    job["result"] = data.get("metrics", job.get("result"))
                    await _broadcast({
                        "type": "job_progress",
                        "job_id": job_id,
                        "metrics": data.get("metrics", {}),
                    }, exclude=worker_id)

            elif msg_type == "training_status":
                if w:
                    w["training"] = data.get("training", {})
                await _broadcast({
                    "type": "training_update",
                    "worker_id": worker_id,
                    "training": data.get("training", {}),
                }, exclude=worker_id)

            elif msg_type == "log":
                job_id = data.get("job_id")
                if job_id and job_id in state.jobs:
                    job = state.jobs[job_id]
                    if "logs" not in job:
                        job["logs"] = []
                    job["logs"].append({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "line": data.get("line", ""),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ WebSocket error ({worker_id}): {e}")
    finally:
        state.websockets.pop(worker_id, None)
        if w:
            w["status"] = "offline"
        print(f"🔌 WebSocket disconnected: {worker_id}")


# ──────────── Dashboard WebSocket ────────────

_dashboard_clients: List[WebSocket] = []


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket for the web dashboard to receive live updates."""
    token = websocket.query_params.get("token", "")
    if ORCHESTRATOR_TOKEN and token != ORCHESTRATOR_TOKEN:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _dashboard_clients.append(websocket)
    print("📊 Dashboard client connected")

    try:
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "workers": list(state.workers.values()),
            "jobs": list(state.jobs.values())[-50:],
        })

        while True:
            data = await websocket.receive_json()
            # Dashboard can send commands
            if data.get("type") == "command":
                target = data.get("worker_id")
                cmd = data.get("command")
                if target and cmd:
                    await send_worker_command(target, cmd, data.get("params"))

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _dashboard_clients.remove(websocket) if websocket in _dashboard_clients else None
        print("📊 Dashboard client disconnected")


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


async def _sync_artifact_to_primary(artifact: Dict[str, Any], content: bytes):
    """Sync an artifact to the primary (RTX 5090) node."""
    primary = state.get_primary_worker()
    if not primary:
        print("⚠️ No primary worker connected — artifact queued for sync")
        return

    ws = state.websockets.get(primary["worker_id"])
    if ws:
        try:
            await ws.send_json({
                "type": "sync_artifact",
                "artifact": artifact,
                "size_bytes": len(content),
            })
            print(f"📤 Artifact {artifact['filename']} queued for sync to primary")
        except Exception as e:
            print(f"❌ Failed to notify primary for sync: {e}")


def _generate_install_script(platform: str) -> str:
    """Generate platform-specific install script."""
    server_url = os.getenv("SERVER_URL", "https://bi-iq.com")

    if platform == "windows":
        return f'''# BI-IDE Worker Agent — Windows Installer
# Usage: .\\install.ps1 -ServerUrl "{server_url}" -Token "YOUR_TOKEN" -Labels "gpu,rtx5090"

param(
    [string]$ServerUrl = "{server_url}",
    [string]$Token = "",
    [string]$Labels = "cpu",
    [string]$WorkerId = ""
)

Write-Host "🏗️ Installing BI-IDE Worker Agent..." -ForegroundColor Cyan

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {{
    Write-Host "❌ Python not found. Install Python 3.10+ first." -ForegroundColor Red
    exit 1
}}

# Create worker directory
$installDir = "$env:USERPROFILE\\.bi-ide-worker"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

# Download worker script
Invoke-WebRequest -Uri "$ServerUrl/api/v1/orchestrator/worker-script" -OutFile "$installDir\\bi_worker.py"

# Create start script
$startScript = @"
cd $installDir
python bi_worker.py --server $ServerUrl --token $Token --labels $Labels --worker-id $WorkerId
"@
Set-Content -Path "$installDir\\start_worker.ps1" -Value $startScript

# Create auto-start task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File $installDir\\start_worker.ps1"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "BI-IDE-Worker" -Action $action -Trigger $trigger -RunLevel Highest -Force

Write-Host "✅ Worker installed at $installDir" -ForegroundColor Green
Write-Host "🚀 Starting worker..." -ForegroundColor Cyan
& "$installDir\\start_worker.ps1"
'''

    else:  # linux / macos
        return f'''#!/bin/bash
# BI-IDE Worker Agent — Linux/macOS Installer
# Usage: ./install.sh <server_url> <token> <worker_id> <labels>

SERVER_URL="${{1:-{server_url}}}"
TOKEN="${{2:-}}"
WORKER_ID="${{3:-$(hostname)}}"
LABELS="${{4:-cpu}}"

echo "🏗️ Installing BI-IDE Worker Agent..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install Python 3.10+ first."
    exit 1
fi

# Create worker directory
INSTALL_DIR="$HOME/.bi-ide-worker"
mkdir -p "$INSTALL_DIR"

# Download worker script
curl -fsSL "$SERVER_URL/api/v1/orchestrator/worker-script" -o "$INSTALL_DIR/bi_worker.py"

# Create start script
cat > "$INSTALL_DIR/start_worker.sh" << SCRIPT
#!/bin/bash
cd "$INSTALL_DIR"
while true; do
    python3 bi_worker.py --server "$SERVER_URL" --token "$TOKEN" --labels "$LABELS" --worker-id "$WORKER_ID"
    echo "Worker exited. Restarting in 5s..."
    sleep 5
done
SCRIPT
chmod +x "$INSTALL_DIR/start_worker.sh"

# Setup systemd service (Linux) or launchd (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: launchd
    cat > "$HOME/Library/LaunchAgents/com.bi-ide.worker.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.bi-ide.worker</string>
    <key>ProgramArguments</key>
    <array><string>$INSTALL_DIR/start_worker.sh</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$INSTALL_DIR/worker.log</string>
    <key>StandardErrorPath</key><string>$INSTALL_DIR/worker.err</string>
</dict>
</plist>
PLIST
    launchctl load "$HOME/Library/LaunchAgents/com.bi-ide.worker.plist"
else
    # Linux: systemd
    sudo tee /etc/systemd/system/bi-ide-worker.service << SERVICE
[Unit]
Description=BI-IDE Worker Agent
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$INSTALL_DIR/start_worker.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE
    sudo systemctl daemon-reload
    sudo systemctl enable bi-ide-worker
    sudo systemctl start bi-ide-worker
fi

echo "✅ Worker installed at $INSTALL_DIR"
echo "🚀 Worker is running!"
'''


# ──────────── Version & Auto-Update ────────────

@router.get("/version")
async def get_version():
    """Return current code version for auto-update checks."""
    import subprocess as _sp
    git_hash = "unknown"
    git_date = "unknown"
    try:
        git_hash = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=3, cwd=str(Path(__file__).parent)).stdout.strip()
        git_date = _sp.run(["git", "log", "-1", "--format=%ci"],
                           capture_output=True, text=True, timeout=3, cwd=str(Path(__file__).parent)).stdout.strip()
    except Exception:
        pass
    return {
        "version": "4.0",
        "git_hash": git_hash,
        "git_date": git_date,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "workers_online": sum(1 for w in state.workers.values() if w.get("status") == "online"),
        "total_jobs": state.job_counter,
    }


@router.get("/rtx-files/{filename}")
async def get_rtx_file(filename: str):
    """Serve RTX server files for auto-deploy."""
    allowed = {"rtx4090_server.py", "resource_manager.py"}
    if filename not in allowed:
        raise HTTPException(404, f"File not available: {filename}")
    path = Path("rtx4090_machine") / filename
    if path.exists():
        return FileResponse(path, filename=filename, media_type="text/plain")
    raise HTTPException(404, f"{filename} not found")


# ──────────── Worker Script Download ────────────

@router.get("/worker-script")
async def get_worker_script():
    """Serve the latest bi_worker.py script."""
    worker_path = Path("worker/bi_worker.py")
    if worker_path.exists():
        return FileResponse(worker_path, filename="bi_worker.py", media_type="text/plain")
    raise HTTPException(404, "Worker script not available yet")


# ──────────── Background Tasks ────────────

async def _heartbeat_monitor():
    """Background task: mark workers offline if no heartbeat."""
    while True:
        now = time.time()
        for w in state.workers.values():
            if w.get("status") not in ("offline",) and now - w.get("last_heartbeat", 0) > 90:
                w["status"] = "offline"
                print(f"💀 Worker timeout: {w.get('hostname', w['worker_id'])}")
                await _broadcast({"type": "worker_offline", "worker_id": w["worker_id"]})
        await asyncio.sleep(15)


def start_background_tasks():
    """Start orchestrator background tasks."""
    asyncio.create_task(_heartbeat_monitor())
    print("🎛️ Orchestrator background tasks started")
