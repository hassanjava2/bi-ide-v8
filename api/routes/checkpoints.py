"""
Checkpoint Routes - نقاط النهاية لمزامنة نقاط الحفظ
"""

import os
import asyncio
import shutil
from datetime import datetime
from pathlib import Path

import requests as http_requests
from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter(prefix="/api/v1/checkpoints", tags=["checkpoints"])

# Config
AI_CORE_HOST = os.getenv("AI_CORE_HOST", None)
AI_CORE_PORT = os.getenv("AI_CORE_PORT", "8080")
AI_CORE_PORTS = [p.strip() for p in os.getenv("AI_CORE_PORTS", AI_CORE_PORT).split(",") if p.strip()]

CHECKPOINTS_DIR = Path("learning_data/checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

AUTO_SYNC_CHECKPOINTS = os.getenv("AUTO_SYNC_CHECKPOINTS", "1").lower() in ("1", "true", "yes", "on")
AUTO_SYNC_INTERVAL_SEC = int(os.getenv("AUTO_SYNC_INTERVAL_SEC", "60"))
MIN_CHECKPOINT_SIZE_MB = float(os.getenv("MIN_CHECKPOINT_SIZE_MB", "1"))
MIN_CHECKPOINT_SIZE_BYTES = int(MIN_CHECKPOINT_SIZE_MB * 1024 * 1024)

checkpoint_sync_task = None
last_checkpoint_sync = {
    "status": "idle", "timestamp": None,
    "synced": 0, "skipped": 0, "failed": 0, "total": 0, "message": "not started",
}


def _get_ai_core_base_urls():
    if not AI_CORE_HOST:
        return []
    return [f"http://{AI_CORE_HOST}:{port}" for port in (AI_CORE_PORTS or [AI_CORE_PORT])]


def _sync_checkpoints_once(force: bool = False):
    if not AI_CORE_HOST:
        return {"status": "error", "message": "RTX 4090 not configured", "synced": 0, "skipped": 0, "failed": 0, "total": 0}

    base_urls = _get_ai_core_base_urls()
    if not base_urls:
        return {"status": "error", "message": "no AI core base urls", "synced": 0, "skipped": 0, "failed": 0, "total": 0}

    try:
        chosen_base_url = None
        response = None
        for base_url in base_urls:
            try:
                candidate = http_requests.get(f"{base_url}/checkpoints/list", timeout=20)
                if candidate.status_code == 200:
                    response = candidate
                    chosen_base_url = base_url
                    break
            except Exception:
                continue

        if not response or not chosen_base_url:
            return {
                "status": "error",
                "message": f"failed to list remote checkpoints on ports: {','.join(AI_CORE_PORTS)}",
                "synced": 0, "skipped": 0, "failed": 0, "total": 0,
                "timestamp": datetime.now().isoformat(),
            }

        remote_checkpoints = response.json().get("checkpoints", [])
        synced = skipped = failed = 0

        for ckpt in remote_checkpoints:
            layer_name = ckpt.get("layer")
            filename = ckpt.get("file")
            if not layer_name or not filename:
                failed += 1
                continue
            layer_dir = CHECKPOINTS_DIR / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            file_path = layer_dir / filename

            if file_path.exists() and file_path.stat().st_size >= MIN_CHECKPOINT_SIZE_BYTES and not force:
                skipped += 1
                continue

            ckpt_response = http_requests.get(
                f"{chosen_base_url}/checkpoints/download/{layer_name}/{filename}", timeout=60,
            )
            if ckpt_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(ckpt_response.content)
                synced += 1
            else:
                failed += 1

        return {
            "status": "success", "source": chosen_base_url,
            "synced": synced, "skipped": skipped, "failed": failed,
            "total": len(remote_checkpoints), "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "synced": 0, "skipped": 0, "failed": 0, "total": 0, "timestamp": datetime.now().isoformat()}


async def checkpoint_sync_loop():
    """Background auto-sync loop."""
    global last_checkpoint_sync
    while True:
        try:
            result = await asyncio.to_thread(_sync_checkpoints_once, False)
            last_checkpoint_sync = result
            if result.get("status") == "success":
                print(f"🔄 Auto-sync: synced={result.get('synced', 0)}, skipped={result.get('skipped', 0)}")
            else:
                print(f"⚠️ Auto-sync failed: {result.get('message', 'unknown')}")
        except Exception as e:
            last_checkpoint_sync = {
                "status": "error", "timestamp": datetime.now().isoformat(),
                "synced": 0, "skipped": 0, "failed": 0, "total": 0, "message": str(e),
            }
        await asyncio.sleep(max(15, AUTO_SYNC_INTERVAL_SEC))


async def start_checkpoint_sync():
    """Called at startup to begin checkpoint sync if enabled."""
    global checkpoint_sync_task, last_checkpoint_sync
    if AI_CORE_HOST and AUTO_SYNC_CHECKPOINTS:
        print(f"🔄 Checkpoint auto-sync enabled (every {AUTO_SYNC_INTERVAL_SEC}s)")
        initial = await asyncio.to_thread(_sync_checkpoints_once, False)
        last_checkpoint_sync = initial
        checkpoint_sync_task = asyncio.create_task(checkpoint_sync_loop())


async def stop_checkpoint_sync():
    global checkpoint_sync_task
    if checkpoint_sync_task:
        checkpoint_sync_task.cancel()
        try:
            await checkpoint_sync_task
        except asyncio.CancelledError:
            pass


# ──── Routes ────


@router.post("/upload/{layer_name}")
async def upload_checkpoint(layer_name: str, file: UploadFile = File(...)):
    """استقبال checkpoint"""
    layer_dir = CHECKPOINTS_DIR / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    file_path = layer_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "saved", "path": str(file_path), "size": file_path.stat().st_size}


@router.get("")
async def list_checkpoints():
    """قائمة الـ checkpoints"""
    checkpoints = []
    for layer_dir in CHECKPOINTS_DIR.iterdir():
        if layer_dir.is_dir():
            for ckpt in layer_dir.glob("*.pt"):
                checkpoints.append({
                    "layer": layer_dir.name,
                    "file": ckpt.name,
                    "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
                    "modified": ckpt.stat().st_mtime,
                })
    return {"total": len(checkpoints), "location": str(CHECKPOINTS_DIR), "checkpoints": checkpoints}


@router.post("/sync-from-rtx4090")
async def sync_checkpoints_from_rtx4090(force: bool = False):
    try:
        return await asyncio.to_thread(_sync_checkpoints_once, force)
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/sync-status")
async def get_checkpoint_sync_status():
    return {
        "enabled": AUTO_SYNC_CHECKPOINTS,
        "interval_sec": AUTO_SYNC_INTERVAL_SEC,
        "min_checkpoint_size_mb": MIN_CHECKPOINT_SIZE_MB,
        "ai_core_host": AI_CORE_HOST,
        "ai_core_ports": AI_CORE_PORTS,
        "last_sync": last_checkpoint_sync,
    }
