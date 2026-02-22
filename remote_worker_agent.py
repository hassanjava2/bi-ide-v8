#!/usr/bin/env python3
"""
Remote Worker Agent
يرتبط مع Orchestrator API ويسحب مهام التدريب وينفذها محلياً
"""

from __future__ import annotations

import argparse
from glob import glob
import json
import os
import platform
import threading
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception:
    requests = None

DEFAULT_POLL_SEC = 5


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Orchestrator-Token"] = token

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def _detect_gpu() -> Optional[str]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _register_worker(base_url: str, token: Optional[str], name: str, labels: list[str]) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/orchestrator/workers/register"
    payload = {
        "name": name,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "gpu": _detect_gpu(),
        "labels": labels,
    }
    return _http_json("POST", url, payload=payload, token=token)


def _heartbeat(base_url: str, worker_id: str, token: Optional[str], current_job_id: Optional[str]) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/orchestrator/workers/{worker_id}/heartbeat"
    return _http_json("POST", url, payload={"current_job_id": current_job_id}, token=token)


def _claim_job(base_url: str, worker_id: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    url = f"{base_url}/api/v1/orchestrator/workers/{worker_id}/jobs/next"
    response = _http_json("POST", url, payload={}, token=token)
    return response.get("job")


def _update_job_status(
    base_url: str,
    job_id: str,
    token: Optional[str],
    status: str,
    return_code: Optional[int] = None,
    logs_tail: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    url = f"{base_url}/api/v1/orchestrator/jobs/{job_id}/status"
    payload = {
        "status": status,
        "return_code": return_code,
        "logs_tail": logs_tail,
        "error": error,
    }
    _http_json("POST", url, payload=payload, token=token)


def _upload_artifact(
    base_url: str,
    job_id: str,
    worker_id: str,
    token: Optional[str],
    file_path: Path,
) -> bool:
    if not file_path.exists() or not file_path.is_file():
        return False

    if requests is None:
        return False

    url = f"{base_url}/api/v1/orchestrator/jobs/{job_id}/artifacts/upload"
    headers: Dict[str, str] = {}
    if token:
        headers["X-Orchestrator-Token"] = token

    try:
        with file_path.open("rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data={"worker_id": worker_id},
                files={"file": (file_path.name, f, "application/octet-stream")},
                timeout=180,
            )
        return response.status_code == 200
    except Exception:
        return False


def _auto_upload_training_artifacts(
    base_url: str,
    job_id: str,
    worker_id: str,
    token: Optional[str],
    job: Dict[str, Any],
    job_started_at: float,
) -> int:
    cwd_value = job.get("cwd")
    base_dir = Path(cwd_value).expanduser() if cwd_value else Path.cwd()
    if not base_dir.exists():
        base_dir = Path.cwd()

    configured = os.getenv(
        "ORCHESTRATOR_UPLOAD_GLOBS",
        "checkpoints/**/*.pt,checkpoints/**/*.json,learning_data/checkpoints/**/*.pt,learning_data/checkpoints/**/*.json"
    )
    globs = [pattern.strip() for pattern in configured.split(",") if pattern.strip()]

    candidates: Dict[str, Path] = {}
    for pattern in globs:
        full_pattern = str((base_dir / pattern).resolve())
        for p in glob(full_pattern, recursive=True):
            path = Path(p)
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except Exception:
                continue
            if stat.st_size <= 0:
                continue
            if stat.st_mtime + 5 < job_started_at:
                continue
            candidates[str(path.resolve())] = path

    uploaded = 0
    for path in sorted(candidates.values(), key=lambda p: p.stat().st_mtime):
        if _upload_artifact(base_url, job_id, worker_id, token, path):
            uploaded += 1

    return uploaded


def _run_job(
    job: Dict[str, Any],
    on_progress: Optional[Callable[[str], None]] = None,
    on_tick: Optional[Callable[[], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    update_interval_sec: int = 3,
) -> tuple[int, str, Optional[str]]:
    command = job.get("command")
    shell = bool(job.get("shell", True))
    args = job.get("args") or []
    env = job.get("env") or {}
    cwd = job.get("cwd")

    if not command:
        return 1, "", "Job command is empty"

    run_env = os.environ.copy()
    run_env.update({str(k): str(v) for k, v in env.items()})

    if cwd:
        resolved = Path(cwd).expanduser()
        if not resolved.exists():
            return 1, "", f"cwd does not exist: {resolved}"
        cwd = str(resolved)

    try:
        if shell:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        else:
            full_cmd = [command] + [str(item) for item in args]
            process = subprocess.Popen(
                full_cmd,
                shell=False,
                cwd=cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

        lines: List[str] = []
        lock = threading.Lock()

        def _reader(pipe, stream_name: str):
            if pipe is None:
                return
            for line in iter(pipe.readline, ""):
                cleaned = line.rstrip("\n")
                if cleaned:
                    with lock:
                        lines.append(f"[{stream_name}] {cleaned}")
                        if len(lines) > 2000:
                            del lines[: len(lines) - 2000]
            pipe.close()

        stdout_thread = threading.Thread(target=_reader, args=(process.stdout, "STDOUT"), daemon=True)
        stderr_thread = threading.Thread(target=_reader, args=(process.stderr, "STDERR"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        last_push = 0.0
        while True:
            rc = process.poll()
            now = time.time()

            if should_stop and should_stop() and rc is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    with lock:
                        lines.append("[SYSTEM] Stop signal received from orchestrator. Process terminated.")
                except Exception:
                    pass

            if now - last_push >= max(1, update_interval_sec):
                with lock:
                    tail = "\n".join(lines)[-12000:]
                if on_progress:
                    on_progress(tail)
                if on_tick:
                    on_tick()
                last_push = now

            if rc is not None:
                break

            time.sleep(1)

        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        with lock:
            final_tail = "\n".join(lines)[-12000:]

        return process.returncode or 0, final_tail, None
    except Exception as exc:
        return 1, "", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="BI IDE Remote Worker Agent")
    parser.add_argument("--server", required=True, help="Orchestrator base URL, e.g. https://your-server.com")
    parser.add_argument("--name", default=platform.node() or "worker", help="Worker display name")
    parser.add_argument("--labels", default="", help="Comma-separated labels, e.g. gpu,rtx4090")
    parser.add_argument("--token", default=os.getenv("ORCHESTRATOR_TOKEN", ""), help="Orchestrator token")
    parser.add_argument("--poll", type=int, default=DEFAULT_POLL_SEC, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle then exit")
    args = parser.parse_args()

    base_url = args.server.rstrip("/")
    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    token = args.token.strip() or None

    print(f"[agent] connecting to {base_url}")
    registration = _register_worker(base_url, token, args.name, labels)

    worker_id = registration["worker_id"]
    poll_interval = int(registration.get("poll_interval_sec", args.poll))

    print(f"[agent] registered worker_id={worker_id}")
    current_job: Optional[str] = None

    while True:
        try:
            _heartbeat(base_url, worker_id, token, current_job)
            job = _claim_job(base_url, worker_id, token)

            if not job:
                if args.once:
                    print("[agent] no jobs in queue")
                    return 0
                time.sleep(max(1, poll_interval))
                continue

            job_id = str(job["id"])
            current_job = job_id
            print(f"[agent] executing job={job_id} name={job.get('name')}")
            stop_requested = False
            job_started_at = time.time()

            def _push_progress(logs_tail: str):
                try:
                    _update_job_status(base_url, job_id, token, status="running", logs_tail=logs_tail)
                except Exception:
                    pass

            def _tick_heartbeat():
                nonlocal stop_requested
                try:
                    hb = _heartbeat(base_url, worker_id, token, current_job)
                    stop_requested = bool(hb.get("stop_current_job"))
                except Exception:
                    pass

            def _should_stop() -> bool:
                return stop_requested

            _update_job_status(base_url, job_id, token, status="running", logs_tail="Job started")
            return_code, logs_tail, error = _run_job(
                job,
                on_progress=_push_progress,
                on_tick=_tick_heartbeat,
                should_stop=_should_stop,
                update_interval_sec=3,
            )

            uploaded_count = _auto_upload_training_artifacts(
                base_url=base_url,
                job_id=job_id,
                worker_id=worker_id,
                token=token,
                job=job,
                job_started_at=job_started_at,
            )
            if uploaded_count > 0:
                logs_tail = f"{logs_tail}\n\n[SYSTEM] Auto-uploaded artifacts: {uploaded_count}".strip()

            if error:
                _update_job_status(
                    base_url,
                    job_id,
                    token,
                    status="failed",
                    return_code=return_code,
                    logs_tail=logs_tail,
                    error=error,
                )
                print(f"[agent] job failed: {error}")
            elif return_code == 0:
                _update_job_status(
                    base_url,
                    job_id,
                    token,
                    status="completed",
                    return_code=return_code,
                    logs_tail=logs_tail,
                )
                print("[agent] job completed")
            else:
                stop_error = "Stopped by user" if stop_requested else "Process returned non-zero exit code"
                _update_job_status(
                    base_url,
                    job_id,
                    token,
                    status="failed",
                    return_code=return_code,
                    logs_tail=logs_tail,
                    error=stop_error,
                )
                print(f"[agent] job failed with return_code={return_code}")

            current_job = None
            if args.once:
                return 0

        except KeyboardInterrupt:
            print("\n[agent] stopped by user")
            return 0
        except Exception as exc:
            print(f"[agent] error: {exc}")
            if args.once:
                return 1
            time.sleep(max(3, poll_interval))


if __name__ == "__main__":
    sys.exit(main())
