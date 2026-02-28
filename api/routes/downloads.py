"""
Downloads Routes - تنزيل ملفات التنصيب حسب النظام
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth import get_current_user

router = APIRouter(prefix="/api/v1/downloads", tags=["downloads"])


def _installers_root() -> Path:
    configured = os.getenv("INSTALLERS_DIR", "releases/installers")
    return Path(configured).resolve()


def _detect_platform(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith(".dmg"):
        return "macos"
    if lower.endswith(".msi") or lower.endswith(".exe"):
        return "windows"
    if lower.endswith(".zip") and ("x64" in lower or "x86" in lower or "win" in lower):
        return "windows"
    if lower.endswith(".deb") or lower.endswith(".rpm") or lower.endswith(".appimage"):
        return "linux"
    return "other"


def _detect_arch(file_name: str) -> Optional[str]:
    lower = file_name.lower()
    if "arm64" in lower or "aarch64" in lower:
        return "arm64"
    if "x64" in lower or "amd64" in lower or "x86_64" in lower:
        return "x64"
    return None


def _collect_installers() -> List[Dict]:
    root = _installers_root()
    if not root.exists() or not root.is_dir():
        return []

    allowed_ext = {".dmg", ".msi", ".exe", ".deb", ".rpm", ".appimage", ".zip", ".tar.gz"}
    installers: List[Dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        name = path.name
        lower = name.lower()
        suffix = path.suffix.lower()
        is_tar_gz = lower.endswith(".tar.gz")
        if suffix not in allowed_ext and not is_tar_gz:
            continue

        rel = path.relative_to(root).as_posix()
        platform = _detect_platform(name)

        installers.append(
            {
                "id": rel,
                "name": name,
                "platform": platform,
                "arch": _detect_arch(name),
                "size_bytes": path.stat().st_size,
                "version": os.getenv("DESKTOP_APP_VERSION", "0.1.0"),
            }
        )

    return installers


@router.get("/installers")
async def list_installers(
    current_user: Dict = Depends(get_current_user),
):
    installers = _collect_installers()
    return {
        "items": installers,
        "count": len(installers),
    }


@router.get("/installers/{installer_id:path}")
async def download_installer(
    installer_id: str,
    current_user: Dict = Depends(get_current_user),
):
    root = _installers_root()
    installers = _collect_installers()
    selected = next((item for item in installers if item["id"] == installer_id), None)

    if not selected:
        raise HTTPException(status_code=404, detail="Installer not found")

    file_path = (root / selected["id"]).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Installer file missing")

    return FileResponse(
        path=str(file_path),
        filename=selected["name"],
        media_type="application/octet-stream",
    )
