"""
Health & System Routes - نقاط النهاية للصحة والمراقبة
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(tags=["health"])


async def _check_database() -> tuple[bool, str, float]:
    """فحص قاعدة البيانات فعلياً - يعيد (success, message, latency_ms)"""
    import time
    start = time.time()
    
    try:
        from core.database import db_manager
        from sqlalchemy import text
        
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            value = result.scalar()
            
            if value == 1:
                latency = (time.time() - start) * 1000
                return True, "connected", latency
            else:
                return False, "unexpected result", (time.time() - start) * 1000
                
    except Exception as e:
        latency = (time.time() - start) * 1000
        return False, str(e)[:100], latency


async def _check_redis() -> tuple[bool, str, float]:
    """فحص Redis فعلياً"""
    import time
    start = time.time()
    
    try:
        import redis.asyncio as redis_lib
        from core.config import settings
        
        r = redis_lib.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
        
        latency = (time.time() - start) * 1000
        return True, "connected", latency
        
    except Exception as e:
        latency = (time.time() - start) * 1000
        return False, str(e)[:100], latency


async def _check_ai_hierarchy() -> tuple[bool, str, float]:
    """فحص AI Hierarchy"""
    import time
    start = time.time()
    
    try:
        from hierarchy import ai_hierarchy
        
        if ai_hierarchy and hasattr(ai_hierarchy, 'council'):
            latency = (time.time() - start) * 1000
            return True, "active", latency
        else:
            latency = (time.time() - start) * 1000
            return False, "not initialized", latency
            
    except Exception as e:
        latency = (time.time() - start) * 1000
        return False, str(e)[:100], latency


@router.get("/health")
async def health_check():
    """
    Health check endpoint with REAL service verification.
    
    Checks:
    - Database connectivity (CRITICAL)
    - Redis/Cache connectivity (CRITICAL)
    - AI Hierarchy availability
    """
    # فحص الخدمات فعلياً بشكل متوازي
    db_ok, db_msg, db_latency = await _check_database()
    redis_ok, redis_msg, redis_latency = await _check_redis()
    hierarchy_ok, hierarchy_msg, hierarchy_latency = await _check_ai_hierarchy()
    
    services = {
        "api": {"status": "ok", "latency_ms": 0},
        "database": {
            "status": "ok" if db_ok else "down",
            "message": db_msg,
            "latency_ms": round(db_latency, 2)
        },
        "redis": {
            "status": "ok" if redis_ok else "down",
            "message": redis_msg,
            "latency_ms": round(redis_latency, 2)
        },
        "ai_hierarchy": {
            "status": "ok" if hierarchy_ok else "degraded",
            "message": hierarchy_msg,
            "latency_ms": round(hierarchy_latency, 2)
        },
    }
    
    # الخدمات الحرجة: DB + Redis
    critical_services_ok = db_ok and redis_ok
    
    health_status = {
        "status": "healthy" if critical_services_ok else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
        "services": services,
    }
    
    status_code = 200 if critical_services_ok else 503
    
    return JSONResponse(content=health_status, status_code=status_code)


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes.
    Returns 200 when critical services are initialized.
    """
    checks = {
        "api": True,
        "database": False,
        "redis": False,
        "ai_hierarchy": False,
    }
    
    # Check Database - Critical
    db_ok, _, _ = await _check_database()
    checks["database"] = db_ok
    
    # Check Redis - Critical
    redis_ok, _, _ = await _check_redis()
    checks["redis"] = redis_ok
    
    # Check AI Hierarchy
    hierarchy_ok, _, _ = await _check_ai_hierarchy()
    checks["ai_hierarchy"] = hierarchy_ok
    
    # DB + Redis are critical
    all_ready = db_ok and redis_ok
    
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
