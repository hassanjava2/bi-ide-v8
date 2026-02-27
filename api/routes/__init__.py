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

    # Check AI Hierarchy
    try:
        from hierarchy import ai_hierarchy
        services["ai_hierarchy"] = "available" if ai_hierarchy else "unavailable"
    except Exception:
        services["ai_hierarchy"] = "unavailable"

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

    # Only consider critical services for health status
    # ai_hierarchy is the new system, smart_council is legacy (optional)
    critical_services = ["api", "ai_hierarchy", "cache"]
    all_ok = all(
        services.get(svc) in ("ok", "available") 
        for svc in critical_services 
        if svc in services
    )
    
    if not all_ok:
        health_status["status"] = "degraded"
        return JSONResponse(content=health_status, status_code=503)

    return health_status


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes with service initialization verification.
    Returns 200 when critical services are initialized.
    Some services (cache, ide, erp) are optional and don't block readiness.
    """
    from fastapi.responses import JSONResponse
    
    checks = {
        "database": False,
        "cache": False,
        "ai_hierarchy": False,
        "ide_service": False,
        "erp_service": False,
    }
    
    # Check Database - Critical
    try:
        from core.database import db_manager
        # Check if database is initialized (engine exists)
        if db_manager.async_engine is not None:
            checks["database"] = True
        elif db_manager.database_url:  # Database URL configured but not initialized yet
            # Try to initialize
            try:
                await db_manager.initialize()
                checks["database"] = True
            except Exception:
                pass
    except Exception:
        pass
    
    # Check Cache - Optional (doesn't block readiness)
    try:
        from core.cache import cache_manager
        if cache_manager.redis_client:
            checks["cache"] = True
    except Exception:
        pass
    
    # Check AI Hierarchy - Critical (but can work without explicit initialization)
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy:
            # Hierarchy is available even if not explicitly initialized
            checks["ai_hierarchy"] = True
    except Exception:
        pass
    
    # Check IDE Service - Optional
    try:
        from api.routes.ide import get_ide_service
        ide_svc = get_ide_service()
        checks["ide_service"] = ide_svc is not None
    except Exception:
        pass
    
    # Check ERP Service - Optional
    try:
        from api.routes.erp import get_erp_service
        erp_svc = get_erp_service()
        checks["erp_service"] = erp_svc is not None
    except Exception:
        pass
    
    # Only database and ai_hierarchy are critical for basic operation
    critical_services = ["database", "ai_hierarchy"]
    all_ready = all(checks[svc] for svc in critical_services)
    
    response = {
        "ready": all_ready,
        "timestamp": datetime.now().isoformat(),
        "services": checks,
    }
    
    status_code = 200 if all_ready else 503
    return JSONResponse(content=response, status_code=status_code)


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
