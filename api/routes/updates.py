"""
Update manifest and reporting endpoints for desktop auto-update.

The desktop app checks /api/v1/updates/manifest every 5 minutes.
It compares current_version with latest_version and serves download URL.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/updates", tags=["updates"])

# ─── Version configuration ─────────────────────────────────────
# Update this when releasing a new version
LATEST_VERSION = "8.0.5"
RELEASE_NOTES = """
## v8.0.5
- ✅ إصلاح الشاشة الزرقاء (visible:true)
- ✅ فصل AI Assistant عن المجلس (الحكماء بالمجلس فقط)
- ✅ نظام تحديث أوتوماتيكي مفعّل
- ✅ 4 routers جديدة (RTX, Network, Brain, Notifications)
- ✅ LoRA training model (17MB)
- ✅ 51 test passing
"""
DOWNLOAD_URL = "https://bi-iq.com/releases/installers/BI-IDE%20Desktop_8.0.5_aarch64.dmg"
CRITICAL = False
SIZE_MB = 13.0


# ─── Update log ─────────────────────────────────────────────────
UPDATE_LOG_DIR = Path(os.getenv("UPDATE_LOG_DIR", "logs/updates"))
UPDATE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _compare_versions(current: str, latest: str) -> bool:
    """Return True if latest > current (semver comparison)."""
    try:
        cur_parts = [int(x) for x in current.strip().split(".")]
        lat_parts = [int(x) for x in latest.strip().split(".")]
        return lat_parts > cur_parts
    except (ValueError, AttributeError):
        return False


@router.get("/manifest")
async def update_manifest(
    device_id: str = Query("unknown"),
    current_version: str = Query("0.0.0"),
    channel: str = Query("stable"),
) -> Dict[str, Any]:
    """
    Check if an update is available.
    Desktop calls this every 5 minutes.
    """
    has_update = _compare_versions(current_version, LATEST_VERSION)

    result: Dict[str, Any] = {
        "has_update": has_update,
        "current_version": current_version,
        "latest_version": LATEST_VERSION,
        "channel": channel,
    }

    if has_update:
        result.update({
            "version": LATEST_VERSION,
            "critical": CRITICAL,
            "size_mb": SIZE_MB,
            "estimated_download_size_mb": SIZE_MB,
            "download_url": DOWNLOAD_URL,
            "release_notes": RELEASE_NOTES.strip(),
        })

    return result


class UpdateReportRequest(BaseModel):
    device_id: str = "unknown"
    version_from: str = ""
    version_to: str = ""
    status: str = ""
    error_message: Optional[str] = None
    timestamp: Optional[int] = None


@router.post("/report")
async def report_update_status(request: UpdateReportRequest) -> Dict[str, str]:
    """
    Receive update status reports from desktop clients.
    Logs to file for monitoring.
    """
    log_entry = {
        "device_id": request.device_id,
        "version_from": request.version_from,
        "version_to": request.version_to,
        "status": request.status,
        "error_message": request.error_message,
        "timestamp": request.timestamp or int(datetime.now().timestamp() * 1000),
        "received_at": datetime.now().isoformat(),
    }

    # Append to log file
    log_file = UPDATE_LOG_DIR / "update_reports.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {"status": "ok"}
