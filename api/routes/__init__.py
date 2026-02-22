"""
Health & System Routes - نقاط النهاية للصحة والمراقبة
"""

from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for Docker/K8s"""
    try:
        from core.config import settings
        from core.cache import cache_manager
        core_available = True
    except ImportError:
        core_available = False

    # Check optional services
    services = {"api": "ok"}

    try:
        from council_ai import smart_council
        services["smart_council"] = "available" if smart_council else "unavailable"
    except Exception:
        services["smart_council"] = "unavailable"

    if core_available:
        try:
            cache_stats = await cache_manager.get_stats()
            services["cache"] = "ok"
        except Exception:
            services["cache"] = "degraded"

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
        "services": services,
    }

    all_ok = all(v in ("ok", "available") for v in services.values())
    if not all_ok:
        health_status["status"] = "degraded"
        return JSONResponse(content=health_status, status_code=503)

    return health_status


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(
            content="# Prometheus metrics not available\n",
            media_type="text/plain",
        )
