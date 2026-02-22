"""
System Status Routes - نقاط النهاية لحالة النظام
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/status")
async def get_full_system_status():
    """حالة النظام الكاملة"""
    # Import dependencies at call-time to avoid circular imports
    try:
        from core.config import settings
        from core.database import db_manager
        from core.cache import cache_manager
        core_available = True
    except ImportError:
        core_available = False

    try:
        from council_ai import smart_council
        council_available = smart_council is not None
    except Exception:
        council_available = False

    try:
        from api.routes.council import get_live_metrics_snapshot, RTX4090_URL, _check_rtx4090
        rtx_available = _check_rtx4090()
        live_metrics = get_live_metrics_snapshot()
    except Exception:
        rtx_available = False
        live_metrics = None

    # Learning stats
    learning_stats = None
    if core_available:
        try:
            learning_stats = await db_manager.get_learning_stats()
        except Exception:
            pass

    # Cache stats
    cache_stats = None
    if core_available:
        try:
            cache_stats = await cache_manager.get_stats()
        except Exception:
            pass

    # Hierarchy status
    hierarchy_status = None
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy:
            hierarchy_status = ai_hierarchy.get_full_status()
    except Exception:
        pass

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
        "services": {
            "smart_council": council_available,
            "rtx4090": rtx_available,
            "core_modules": core_available,
        },
        "hierarchy": hierarchy_status,
        "learning": learning_stats,
        "cache": cache_stats,
        "council_metrics": live_metrics,
    }


@router.get("/rtx4090/status")
async def get_rtx4090_status():
    """حالة اتصال RTX 4090 (مع retry mechanism)"""
    try:
        from api.routes.council import (
            RTX4090_URL, check_rtx4090_with_retry, 
            send_rtx4090_request_with_retry
        )
        connected = check_rtx4090_with_retry(timeout=5.0)
        payload = {
            "connected": connected,
            "url": RTX4090_URL,
            "timestamp": datetime.now().isoformat(),
        }
        if connected:
            r = send_rtx4090_request_with_retry(
                endpoint="/health",
                json_data={},
                timeout=5.0,
            )
            if r is not None and r.status_code == 200:
                payload["remote"] = r.json()
                payload["retry_config"] = {
                    "max_retries": int(os.getenv("RTX4090_MAX_RETRIES", "3")),
                    "retry_delay": float(os.getenv("RTX4090_RETRY_DELAY", "1.0")),
                    "backoff": float(os.getenv("RTX4090_RETRY_BACKOFF", "2.0")),
                }
        return payload
    except Exception:
        return {"connected": False, "timestamp": datetime.now().isoformat()}


@router.get("/system/config")
async def get_system_config():
    """System configuration (sanitized)"""
    try:
        from core.config import settings
        from api.routes.council import _check_rtx4090
        return {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "features": {
                "smart_council": settings.ENABLE_SMART_COUNCIL,
                "autonomous_learning": settings.ENABLE_AUTONOMOUS_LEARNING,
                "council_discussions": settings.ENABLE_COUNCIL_DISCUSSIONS,
                "code_analysis": settings.ENABLE_CODE_ANALYSIS,
            },
            "rtx4090": {
                "host": settings.RTX4090_HOST,
                "port": settings.RTX4090_PORT,
                "connected": _check_rtx4090(),
            },
        }
    except ImportError:
        return {"app_name": "BI IDE", "version": "8.0.0", "core_modules": False}
