"""
Remote Training Orchestrator API
يدير أجهزة العمال (Workers) ويوزع عليهم مهام التدريب عن بعد
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, UploadFile, File as FastAPIFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

STATE_FILE = Path("data/orchestrator/state.json")
ARTIFACTS_DIR = Path("data/orchestrator/artifacts")
HEARTBEAT_TIMEOUT_SEC = int(os.getenv("ORCHESTRATOR_HEARTBEAT_TIMEOUT_SEC", "45"))
SERVER_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "").strip()


class WorkerRegisterRequest(BaseModel):
    name: str
    platform: str
    hostname: str
    python_version: str
    cpu_count: int = 1
    gpu: Optional[str] = None
    labels: List[str] = Field(default_factory=list)


class WorkerHeartbeatRequest(BaseModel):
    current_job_id: Optional[str] = None


class JobCreateRequest(BaseModel):
    name: str
    command: str
    shell: bool = True
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    target_labels: List[str] = Field(default_factory=list)


class JobBatchCreateRequest(JobCreateRequest):
    count: int = 1


class JobStatusUpdateRequest(BaseModel):
    status: str
    return_code: Optional[int] = None
    logs_tail: Optional[str] = None
    error: Optional[str] = None


class OrchestratorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "workers": {},
            "jobs": {},
            "queue": []
        }
        self._load()

    def _load(self):
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text(encoding="utf-8") or "{}")
                if isinstance(data, dict):
                    self.state["workers"] = data.get("workers", {})
                    self.state["jobs"] = data.get("jobs", {})
                    self.state["queue"] = data.get("queue", [])
        except Exception:
            self.state = {"workers": {}, "jobs": {}, "queue": []}

    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def with_lock(self):
        return self.lock


store = OrchestratorState()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _require_token(token: Optional[str]):
    if not SERVER_TOKEN:
        return
    if token != SERVER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid orchestrator token")


def _refresh_worker_online_status(worker: Dict[str, Any]) -> Dict[str, Any]:
    last_seen_str = worker.get("last_seen")
    if not last_seen_str:
        worker["online"] = False
        return worker

    try:
        last_seen = datetime.fromisoformat(last_seen_str.replace("Z", ""))
        worker["online"] = (datetime.utcnow() - last_seen) <= timedelta(seconds=HEARTBEAT_TIMEOUT_SEC)
    except Exception:
        worker["online"] = False

    return worker


def _labels_match(worker_labels: List[str], target_labels: List[str]) -> bool:
    if not target_labels:
        return True
    wl = set(worker_labels)
    return all(label in wl for label in target_labels)


def _build_job(payload: JobCreateRequest, *, restarted_from: Optional[str] = None) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    return {
        "id": job_id,
        "name": payload.name,
        "command": payload.command,
        "shell": payload.shell,
        "args": payload.args,
        "env": payload.env,
        "cwd": payload.cwd,
        "target_labels": payload.target_labels,
        "status": "pending",
        "created_at": _now(),
        "assigned_worker_id": None,
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "logs_tail": None,
        "error": None,
        "stop_requested": False,
        "restarted_from": restarted_from,
        "artifacts": [],
    }


@router.get("/health")
def orchestrator_health():
    with store.with_lock():
        workers = list(store.state["workers"].values())
        online_workers = 0
        for worker in workers:
            if _refresh_worker_online_status(worker).get("online"):
                online_workers += 1

        jobs = store.state["jobs"]

        return {
            "status": "ok",
            "workers_total": len(workers),
            "workers_online": online_workers,
            "jobs_total": len(jobs),
            "jobs_pending": sum(1 for j in jobs.values() if j.get("status") == "pending"),
            "jobs_running": sum(1 for j in jobs.values() if j.get("status") == "running"),
            "jobs_completed": sum(1 for j in jobs.values() if j.get("status") == "completed"),
            "timestamp": _now()
        }


@router.post("/workers/register")
def register_worker(
    payload: WorkerRegisterRequest,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    worker_id = str(uuid.uuid4())
    worker = {
        "id": worker_id,
        "name": payload.name,
        "platform": payload.platform,
        "hostname": payload.hostname,
        "python_version": payload.python_version,
        "cpu_count": payload.cpu_count,
        "gpu": payload.gpu,
        "labels": payload.labels,
        "status": "idle",
        "current_job_id": None,
        "last_seen": _now(),
        "created_at": _now(),
        "online": True
    }

    with store.with_lock():
        store.state["workers"][worker_id] = worker
        store._save()

    return {
        "worker_id": worker_id,
        "poll_interval_sec": 5,
        "heartbeat_timeout_sec": HEARTBEAT_TIMEOUT_SEC
    }


@router.post("/workers/{worker_id}/heartbeat")
def worker_heartbeat(
    worker_id: str,
    payload: WorkerHeartbeatRequest,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        worker = store.state["workers"].get(worker_id)
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker["last_seen"] = _now()
        if payload.current_job_id is not None:
            worker["current_job_id"] = payload.current_job_id
            worker["status"] = "busy" if payload.current_job_id else "idle"

        _refresh_worker_online_status(worker)

        stop_current_job = False
        current_job_id = worker.get("current_job_id")
        if current_job_id:
            current_job = store.state["jobs"].get(current_job_id)
            if current_job and bool(current_job.get("stop_requested")):
                stop_current_job = True

        store._save()

        return {
            "status": "ok",
            "server_time": _now(),
            "stop_current_job": stop_current_job
        }


@router.get("/workers")
def list_workers(
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        workers = []
        for worker in store.state["workers"].values():
            workers.append(_refresh_worker_online_status(dict(worker)))
        return {"workers": workers}


@router.post("/jobs")
def create_job(
    payload: JobCreateRequest,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    job = _build_job(payload)
    job_id = job["id"]

    with store.with_lock():
        store.state["jobs"][job_id] = job
        store.state["queue"].append(job_id)
        store._save()

    return {"status": "queued", "job_id": job_id}


@router.post("/jobs/batch")
def create_jobs_batch(
    payload: JobBatchCreateRequest,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    if payload.count < 1:
        raise HTTPException(status_code=400, detail="count must be >= 1")
    if payload.count > 200:
        raise HTTPException(status_code=400, detail="count must be <= 200")

    queued_ids: List[str] = []
    with store.with_lock():
        for idx in range(payload.count):
            request = JobCreateRequest(
                name=f"{payload.name} #{idx + 1}" if payload.count > 1 else payload.name,
                command=payload.command,
                shell=payload.shell,
                args=payload.args,
                env=payload.env,
                cwd=payload.cwd,
                target_labels=payload.target_labels,
            )
            job = _build_job(request)
            job_id = job["id"]
            store.state["jobs"][job_id] = job
            store.state["queue"].append(job_id)
            queued_ids.append(job_id)

        store._save()

    return {
        "status": "queued",
        "count": payload.count,
        "job_ids": queued_ids,
    }


@router.get("/jobs")
def list_jobs(
    status: Optional[str] = None,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        jobs = list(store.state["jobs"].values())
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return {"jobs": jobs}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.get("status") == "pending":
            job["status"] = "cancelled"
            job["stop_requested"] = True
            if job_id in store.state["queue"]:
                store.state["queue"].remove(job_id)
            job["finished_at"] = _now()
            store._save()
            return {"status": "cancelled"}

        if job.get("status") in {"running", "stopping"}:
            job["stop_requested"] = True
            job["status"] = "stopping"
            store._save()
            return {"status": "stopping", "message": "Stop signal sent to worker"}

        return {"status": "ignored", "reason": f"Job status is {job.get('status')}"}


@router.post("/jobs/{job_id}/restart")
def restart_job(
    job_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        old_job = store.state["jobs"].get(job_id)
        if not old_job:
            raise HTTPException(status_code=404, detail="Job not found")

        payload = JobCreateRequest(
            name=f"{old_job.get('name', 'job')} (restart)",
            command=old_job.get("command", ""),
            shell=bool(old_job.get("shell", True)),
            args=list(old_job.get("args", [])),
            env=dict(old_job.get("env", {})),
            cwd=old_job.get("cwd"),
            target_labels=list(old_job.get("target_labels", [])),
        )

        new_job = _build_job(payload, restarted_from=job_id)
        new_job_id = new_job["id"]

        store.state["jobs"][new_job_id] = new_job
        store.state["queue"].append(new_job_id)
        store._save()

        return {"status": "queued", "job_id": new_job_id, "restarted_from": job_id}


@router.post("/workers/{worker_id}/jobs/next")
def claim_next_job(
    worker_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        worker = store.state["workers"].get(worker_id)
        if not worker:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker_labels = worker.get("labels", [])

        for job_id in list(store.state["queue"]):
            job = store.state["jobs"].get(job_id)
            if not job or job.get("status") != "pending":
                continue
            if not _labels_match(worker_labels, job.get("target_labels", [])):
                continue

            job["status"] = "running"
            job["assigned_worker_id"] = worker_id
            job["started_at"] = _now()

            worker["status"] = "busy"
            worker["current_job_id"] = job_id
            worker["last_seen"] = _now()

            store.state["queue"].remove(job_id)
            store._save()

            return {
                "job": {
                    "id": job["id"],
                    "name": job["name"],
                    "command": job["command"],
                    "shell": job["shell"],
                    "args": job["args"],
                    "env": job["env"],
                    "cwd": job["cwd"]
                }
            }

        worker["last_seen"] = _now()
        worker["status"] = "idle" if not worker.get("current_job_id") else "busy"
        store._save()
        return {"job": None}


@router.post("/jobs/{job_id}/status")
def update_job_status(
    job_id: str,
    payload: JobStatusUpdateRequest,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    allowed = {"running", "completed", "failed", "stopping"}
    if payload.status not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid status. Allowed: {sorted(allowed)}")

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if payload.status == "running" and bool(job.get("stop_requested")):
            job["status"] = "stopping"
        else:
            job["status"] = payload.status
        if payload.return_code is not None:
            job["return_code"] = payload.return_code
        if payload.logs_tail is not None:
            job["logs_tail"] = payload.logs_tail[-12000:]
        if payload.error is not None:
            job["error"] = payload.error

        if payload.status in {"completed", "failed"}:
            job["finished_at"] = _now()
            job["stop_requested"] = False
            worker_id = job.get("assigned_worker_id")
            if worker_id and worker_id in store.state["workers"]:
                worker = store.state["workers"][worker_id]
                worker["current_job_id"] = None
                worker["status"] = "idle"
                worker["last_seen"] = _now()

        store._save()

    return {"status": "ok"}


@router.post("/jobs/{job_id}/artifacts/upload")
async def upload_job_artifact(
    job_id: str,
    file: UploadFile = FastAPIFile(...),
    worker_id: str = Form("unknown"),
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "artifact.bin").name
    artifact_id = str(uuid.uuid4())
    saved_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{artifact_id[:8]}_{original_name}"
    target_path = job_dir / saved_name

    with target_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    size_bytes = target_path.stat().st_size
    meta = {
        "id": artifact_id,
        "worker_id": worker_id,
        "original_name": original_name,
        "saved_name": saved_name,
        "size_bytes": size_bytes,
        "uploaded_at": _now(),
        "path": str(target_path),
    }

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.setdefault("artifacts", []).append(meta)
        store._save()

    return {
        "status": "uploaded",
        "artifact": {
            "id": artifact_id,
            "original_name": original_name,
            "saved_name": saved_name,
            "size_bytes": size_bytes,
        }
    }


@router.get("/jobs/{job_id}/artifacts")
def list_job_artifacts(
    job_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        artifacts = job.get("artifacts", [])
        return {"job_id": job_id, "count": len(artifacts), "artifacts": artifacts}


@router.get("/jobs/{job_id}/artifacts/{artifact_id}/download")
def download_job_artifact(
    job_id: str,
    artifact_id: str,
    x_orchestrator_token: Optional[str] = Header(default=None)
):
    _require_token(x_orchestrator_token)

    with store.with_lock():
        job = store.state["jobs"].get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        artifacts = job.get("artifacts", [])
        artifact = next((a for a in artifacts if a.get("id") == artifact_id), None)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

    artifact_path = Path(str(artifact.get("path", "")))
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on server")

    return FileResponse(path=str(artifact_path), filename=artifact.get("original_name", artifact_path.name))


@router.get("/download/agent.py")
def download_agent_script():
    script_path = Path("remote_worker_agent.py")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Agent script not found")
    return FileResponse(path=str(script_path), filename="remote_worker_agent.py")


@router.get("/download/windows")
def download_windows_installer():
    script_path = Path("agent/install_windows.ps1")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Windows installer script not found")
    return FileResponse(path=str(script_path), filename="install_windows.ps1")


@router.get("/download/linux")
def download_linux_installer():
    script_path = Path("agent/install_linux.sh")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Linux installer script not found")
    return FileResponse(path=str(script_path), filename="install_linux.sh")


@router.get("/download/macos")
def download_macos_installer():
    script_path = Path("agent/install_macos.sh")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="macOS installer script not found")
    return FileResponse(path=str(script_path), filename="install_macos.sh")


@router.get("/mobile", response_class=HTMLResponse)
def mobile_dashboard():
        return """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BI IDE Mobile Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background:#0f172a; color:#e2e8f0; }
        .wrap { max-width: 900px; margin: 0 auto; padding: 12px; }
        .card { background:#111827; border:1px solid #1f2937; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
        h2 { margin: 0 0 8px; font-size: 18px; }
        input, textarea, button, select { width:100%; box-sizing:border-box; padding:10px; margin-top:6px; border-radius:8px; border:1px solid #334155; background:#0b1220; color:#e2e8f0; }
        button { background:#2563eb; border:none; font-weight:700; }
        button.small { width:auto; padding:6px 10px; font-size:12px; margin-inline-start:6px; }
        pre { white-space: pre-wrap; background:#020617; border:1px solid #1e293b; border-radius:10px; padding:10px; max-height:260px; overflow:auto; }
        .row { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
        .pill { display:inline-block; background:#1e293b; padding:4px 8px; border-radius:999px; margin:2px; font-size:12px; }
        .job-item { border:1px solid #263143; border-radius:10px; padding:8px; margin-bottom:8px; }
        .job-actions { margin-top:6px; }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h2>لوحة تحكم BI IDE (موبايل)</h2>
            <label>Orchestrator Token</label>
            <input id="token" type="password" placeholder="ORCHESTRATOR_TOKEN" />
            <div class="row">
                <button id="refreshBtn">تحديث الحالة</button>
                <button id="autoBtn">تفعيل/إيقاف التحديث التلقائي</button>
            </div>
        </div>

        <div class="card">
            <h2>الحالة العامة</h2>
            <div id="health"></div>
        </div>

        <div class="card">
            <h2>الأجهزة (Workers)</h2>
            <div id="workers"></div>
        </div>

        <div class="card">
            <h2>الـ Jobs</h2>
            <div id="jobs"></div>
            <label>آخر Logs</label>
            <pre id="logs">لا توجد بيانات بعد</pre>
        </div>

        <div class="card">
            <h2>إنشاء Job تدريب</h2>
            <label>اسم الـ Job</label>
            <input id="jobName" type="text" placeholder="RTX4090 Training Job" value="RTX4090 Training Job" />
            <label>الأمر (Command)</label>
            <textarea id="jobCommand" rows="3" placeholder="python rtx4090_machine/rtx4090_server.py">python rtx4090_machine/rtx4090_server.py</textarea>
            <label>Labels (comma separated)</label>
            <input id="jobLabels" type="text" placeholder="gpu,rtx4090" value="gpu" />
            <label>عدد المهام</label>
            <input id="jobCount" type="number" min="1" max="200" value="1" />
            <button id="createJobBtn">إنشاء Job الآن</button>
            <pre id="jobCreateOut">جاهز لإنشاء Job</pre>
        </div>

        <div class="card">
            <h2>التحدث مع الطبقات</h2>
            <textarea id="msg" rows="3" placeholder="اكتب رسالتك للمجلس..."></textarea>
            <button id="sendMsg">إرسال</button>
            <pre id="chatOut">لا توجد رسائل بعد</pre>
        </div>
    </div>

    <script>
        let timer = null;
        let lastJobStatusMap = {};
        let audioCtx = null;

        function headers() {
            const token = document.getElementById('token').value.trim();
            const h = { 'Content-Type': 'application/json' };
            if (token) h['X-Orchestrator-Token'] = token;
            return h;
        }

        async function fetchJson(url, options = {}) {
            const res = await fetch(url, options);
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`${res.status} ${txt}`);
            }
            return await res.json();
        }

        async function refreshAll() {
            try {
                const h = headers();
                const [health, workers, jobs] = await Promise.all([
                    fetchJson('/api/v1/orchestrator/health', { headers: h }),
                    fetchJson('/api/v1/orchestrator/workers', { headers: h }),
                    fetchJson('/api/v1/orchestrator/jobs', { headers: h }),
                ]);

                document.getElementById('health').innerHTML = `
                    <span class="pill">Online Workers: ${health.workers_online}/${health.workers_total}</span>
                    <span class="pill">Pending: ${health.jobs_pending}</span>
                    <span class="pill">Running: ${health.jobs_running}</span>
                    <span class="pill">Completed: ${health.jobs_completed}</span>
                `;

                const workersHtml = (workers.workers || []).map(w =>
                    `<div class="pill">${w.name} | ${w.online ? '🟢 online' : '🔴 offline'} | ${w.status} | ${w.gpu || 'no-gpu'}</div>`
                ).join('') || 'لا توجد أجهزة حالياً';
                document.getElementById('workers').innerHTML = workersHtml;

                const jobsList = jobs.jobs || [];
                for (const job of jobsList) {
                    const id = job.id || '';
                    const status = job.status || '';
                    if (!id) continue;
                    const prev = lastJobStatusMap[id];
                    if (prev && prev !== status && ['completed', 'failed', 'stopping'].includes(status)) {
                        notifyStatusChange(job.name || id, status);
                    }
                    lastJobStatusMap[id] = status;
                }

                const jobsHtml = jobsList.slice(0, 12).map(j => {
                    const jid = j.id || '';
                    return `
                        <div class="job-item">
                            <div><b>${j.name}</b></div>
                            <div class="pill">${j.status}</div>
                            <div class="pill">${j.assigned_worker_id || 'unassigned'}</div>
                            <div class="job-actions">
                                <button class="small" onclick="stopJob('${jid}')">إيقاف</button>
                                <button class="small" onclick="restartJob('${jid}')">Restart</button>
                            </div>
                        </div>
                    `;
                }).join('') || 'لا توجد Jobs';
                document.getElementById('jobs').innerHTML = jobsHtml;

                const latestWithLogs = jobsList.find(j => j.logs_tail);
                if (latestWithLogs) {
                    document.getElementById('logs').textContent = latestWithLogs.logs_tail;
                }
            } catch (err) {
                document.getElementById('health').textContent = `خطأ: ${err.message}`;
            }
        }

        function beep() {
            try {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                if (!audioCtx) audioCtx = new AC();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sine';
                osc.frequency.value = 880;
                gain.gain.value = 0.03;
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                setTimeout(() => osc.stop(), 180);
            } catch (e) {
            }
        }

        function notifyStatusChange(jobName, status) {
            const msg = `Job ${jobName} -> ${status}`;
            beep();
            if ('Notification' in window) {
                if (Notification.permission === 'granted') {
                    new Notification('BI IDE Alert', { body: msg });
                }
            }
            document.getElementById('jobCreateOut').textContent = msg;
        }

        async function sendCouncilMessage() {
            const message = document.getElementById('msg').value.trim();
            if (!message) return;
            try {
                const out = await fetchJson('/api/v1/council/message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, user_id: 'mobile_user', alert_level: 'GREEN' })
                });
                document.getElementById('chatOut').textContent = `${out.council_member || 'المجلس'}:\n${out.response || ''}`;
            } catch (err) {
                document.getElementById('chatOut').textContent = `خطأ: ${err.message}`;
            }
        }

        async function createTrainingJob() {
            const name = document.getElementById('jobName').value.trim();
            const command = document.getElementById('jobCommand').value.trim();
            const labelsRaw = document.getElementById('jobLabels').value.trim();
            const countRaw = parseInt(document.getElementById('jobCount').value || '1', 10);
            const count = Number.isFinite(countRaw) ? Math.max(1, Math.min(200, countRaw)) : 1;
            const labels = labelsRaw ? labelsRaw.split(',').map(v => v.trim()).filter(Boolean) : [];

            if (!name || !command) {
                document.getElementById('jobCreateOut').textContent = 'الاسم والأمر مطلوبين';
                return;
            }

            try {
                const h = headers();
                const endpoint = count > 1 ? '/api/v1/orchestrator/jobs/batch' : '/api/v1/orchestrator/jobs';
                const out = await fetchJson(endpoint, {
                    method: 'POST',
                    headers: h,
                    body: JSON.stringify({
                        name,
                        command,
                        shell: true,
                        target_labels: labels,
                        count,
                    })
                });

                if (count > 1) {
                    document.getElementById('jobCreateOut').textContent = `تم إنشاء ${out.count || count} Jobs بنجاح`;
                } else {
                    document.getElementById('jobCreateOut').textContent = `تم إنشاء Job بنجاح\nID: ${out.job_id || 'unknown'}`;
                }
                refreshAll();
            } catch (err) {
                document.getElementById('jobCreateOut').textContent = `خطأ: ${err.message}`;
            }
        }

        async function stopJob(jobId) {
            if (!jobId) return;
            try {
                const h = headers();
                const out = await fetchJson(`/api/v1/orchestrator/jobs/${jobId}/cancel`, {
                    method: 'POST',
                    headers: h,
                    body: JSON.stringify({})
                });
                document.getElementById('jobCreateOut').textContent = `Stop: ${JSON.stringify(out)}`;
                refreshAll();
            } catch (err) {
                document.getElementById('jobCreateOut').textContent = `Stop error: ${err.message}`;
            }
        }

        async function restartJob(jobId) {
            if (!jobId) return;
            try {
                const h = headers();
                const out = await fetchJson(`/api/v1/orchestrator/jobs/${jobId}/restart`, {
                    method: 'POST',
                    headers: h,
                    body: JSON.stringify({})
                });
                document.getElementById('jobCreateOut').textContent = `Restart queued. New ID: ${out.job_id || 'unknown'}`;
                refreshAll();
            } catch (err) {
                document.getElementById('jobCreateOut').textContent = `Restart error: ${err.message}`;
            }
        }

        window.stopJob = stopJob;
        window.restartJob = restartJob;

        document.getElementById('refreshBtn').addEventListener('click', refreshAll);
        document.getElementById('sendMsg').addEventListener('click', sendCouncilMessage);
        document.getElementById('createJobBtn').addEventListener('click', createTrainingJob);
        document.getElementById('autoBtn').addEventListener('click', () => {
            if (timer) {
                clearInterval(timer);
                timer = null;
                return;
            }
            timer = setInterval(refreshAll, 5000);
            refreshAll();
        });

        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().catch(() => {});
        }

        refreshAll();
    </script>
</body>
</html>
        """
