"""
Distributed Worker Agent
يشغّل عامل تدريب على أي جهاز ويربطه تلقائياً بالـ API.
"""

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


def _outbox_path(worker_id: str) -> Path:
    safe_worker = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in worker_id)
    root = Path("data/learning/worker-outbox")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_worker}.json"


def _load_outbox(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "[]")
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    return []


def _save_outbox(path: Path, items: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _flush_outbox(api: str, outbox_path: Path) -> int:
    queued = _load_outbox(outbox_path)
    if not queued:
        return 0

    pending: List[Dict[str, Any]] = []
    sent = 0
    for item in queued:
        try:
            response = requests.post(
                f"{api}/api/v1/network/training/complete",
                json=item,
                timeout=30,
            )
            if response.status_code == 200:
                sent += 1
            else:
                pending.append(item)
        except Exception:
            pending.append(item)

    _save_outbox(outbox_path, pending)
    return sent


def cpu_heavy_training(topic: str, seconds: int) -> dict:
    """محاكاة تدريب يستهلك CPU بشكل محسوب"""
    started = time.time()
    checksum = 0
    loops = 0
    payload = (topic or "training").encode("utf-8")

    while time.time() - started < max(1, seconds):
        for byte in payload:
            checksum = ((checksum << 5) - checksum + byte + loops) & 0xFFFFFFFF
            loops += 1

    duration = time.time() - started
    return {
        "topic": topic,
        "duration_sec": round(duration, 2),
        "iterations": loops,
        "checksum": int(checksum),
        "finished_at": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.getenv("BI_IDE_API_URL", "http://localhost:8000"))
    parser.add_argument("--worker-id", default=os.getenv("WORKER_ID", socket.gethostname()))
    parser.add_argument("--poll-sec", type=int, default=int(os.getenv("WORKER_POLL_SEC", "3")))
    parser.add_argument("--train-sec", type=int, default=int(os.getenv("WORKER_TRAIN_SEC", "5")))
    args = parser.parse_args()

    api = args.api.rstrip("/")
    worker_id = args.worker_id
    outbox_path = _outbox_path(worker_id)

    capabilities = {
        "cpu_count": os.cpu_count() or 1,
        "hostname": socket.gethostname(),
        "platform": os.name,
        "mode": "auto-worker-agent",
    }

    requests.post(
        f"{api}/api/v1/network/workers/register",
        json={
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "capabilities": capabilities,
        },
        timeout=20,
    )

    print(f"✅ Worker connected: {worker_id} -> {api}")

    while True:
        try:
            flushed = _flush_outbox(api=api, outbox_path=outbox_path)
            if flushed:
                print(f"📤 Flushed {flushed} pending completion updates")

            requests.post(
                f"{api}/api/v1/network/workers/heartbeat",
                json={
                    "worker_id": worker_id,
                    "status": "online",
                    "capabilities": capabilities,
                },
                timeout=10,
            )

            claim = requests.post(
                f"{api}/api/v1/network/training/claim",
                json={"worker_id": worker_id},
                timeout=20,
            )

            if claim.status_code == 200:
                payload = claim.json()
                task = payload.get("task")
                if task:
                    topic = task.get("topic", "")
                    metrics = cpu_heavy_training(topic=topic, seconds=args.train_sec)

                    artifact_dir = Path("data/learning/distributed-artifacts")
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    artifact_name = f"artifact-{task['task_id']}-{worker_id}.json"
                    artifact_path = artifact_dir / artifact_name
                    artifact_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

                    completion_payload = {
                        "task_id": task["task_id"],
                        "worker_id": worker_id,
                        "metrics": metrics,
                        "artifact_name": artifact_name,
                        "artifact_payload": metrics,
                    }

                    completion_response = requests.post(
                        f"{api}/api/v1/network/training/complete",
                        json=completion_payload,
                        timeout=30,
                    )
                    if completion_response.status_code != 200:
                        pending = _load_outbox(outbox_path)
                        pending.append(completion_payload)
                        _save_outbox(outbox_path, pending)
                        print(f"⏳ Completion queued in outbox for retry: {task['task_id']}")

                    print(f"🏁 Completed {task['task_id']} ({topic})")

        except Exception as error:
            print(f"⚠️ Worker loop error: {error}")

        time.sleep(max(1, args.poll_sec))


if __name__ == "__main__":
    main()
