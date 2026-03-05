"""
Legacy Route Compatibility - توافقية المسارات القديمة

يضمن عمل v6 API مع v8 بدون تغييرات على العميل
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["legacy-compat"])


# ═══════════════════════════════════════════════════════════════
# v6 Legacy Routes → v8 New Routes Mapping
# ═══════════════════════════════════════════════════════════════

@router.get("/smart_council/status")
async def legacy_council_status():
    """v6: /smart_council/status → v8: /council/status"""
    from hierarchy import ai_hierarchy
    
    return {
        "active": True,
        "mode": ai_hierarchy.active_mode if ai_hierarchy else "normal",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0 (v6-compat)",
        "message": "Smart Council migrated to AI Hierarchy V8"
    }


@router.post("/smart_council/ask")
async def legacy_council_ask(request: Dict[str, Any]):
    """v6: /smart_council/ask → v8: /council/message"""
    message = request.get("message", "")
    
    from hierarchy import ai_hierarchy
    
    if ai_hierarchy:
        response = ai_hierarchy.ask(message)
        return {
            "response": response.get("response", ""),
            "wise_man": response.get("wise_man", "المجلس"),
            "confidence": response.get("confidence", 0.85),
            "version": "8.0.0",
        }
    
    return {
        "response": "AI Hierarchy not initialized",
        "wise_man": "System",
        "confidence": 0.0,
    }


@router.get("/rtx4090/status")
async def legacy_rtx_status():
    """v6: /rtx4090/status → v8: /rtx4090/status (same)"""
    from api.routes.council import RTX4090_URL, _check_rtx4090
    
    return {
        "connected": _check_rtx4090(),
        "url": RTX4090_URL,
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
    }


@router.get("/guardian/status")
async def legacy_guardian_status():
    """v6: /guardian/status → v8: /guardian/status"""
    try:
        from hierarchy import ai_hierarchy
        full_status = ai_hierarchy.get_full_status() if ai_hierarchy else {}
        president = full_status.get("president", {}) if isinstance(full_status, dict) else {}
        
        return {
            "active": True,
            "security_level": "high",
            "veto_power": bool(president.get("veto_power", False)),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }
    except Exception:
        return {
            "active": True,
            "security_level": "degraded",
            "veto_power": False,
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }


@router.get("/wisdom")
async def legacy_wisdom(horizon: str = "century"):
    """v6: /wisdom → v8: /wisdom"""
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy and hasattr(ai_hierarchy, "get_wisdom"):
            wisdom = ai_hierarchy.get_wisdom()
        else:
            wisdom = "Focus on reliability, observability, and incremental delivery."
        
        return {
            "wisdom": wisdom,
            "horizon": horizon,
            "source": "hierarchy",
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }
    except Exception:
        return {
            "wisdom": "Stability first, then scale.",
            "horizon": horizon,
            "source": "fallback",
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }


@router.get("/learning/stats")
async def legacy_learning_stats():
    """v6: /learning/stats → v8: /status"""
    try:
        from core.database import db_manager
        stats = await db_manager.get_learning_stats()
        return {
            "total_samples": stats.get("total_samples", 0),
            "today_samples": stats.get("today_samples", 0),
            "last_training": stats.get("last_training"),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }
    except Exception as e:
        return {
            "total_samples": 0,
            "today_samples": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }


@router.get("/system/health")
async def legacy_system_health():
    """v6: /system/health → v8: /health"""
    from api.routes import health_check
    return await health_check()


@router.post("/training/start")
async def legacy_training_start(request: Optional[Dict[str, Any]] = None):
    """v6: /training/start → v8: /training/start"""
    try:
        from training.v8_modules import get_training_pipeline
        
        pipeline = get_training_pipeline()
        started = pipeline.start_training()
        
        return {
            "started": started,
            "status": "training_initiated",
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }
    except Exception as e:
        return {
            "started": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }


@router.get("/training/status")
async def legacy_training_status():
    """v6: /training/status → v8: /training/status"""
    try:
        from training.v8_modules import get_training_pipeline
        
        pipeline = get_training_pipeline()
        status = pipeline.get_status()
        
        return {
            **status,
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }
    except Exception as e:
        return {
            "state": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0",
        }


# ═══════════════════════════════════════════════════════════════
# Deprecated v6 Routes (return migration message)
# ═══════════════════════════════════════════════════════════════

@router.get("/v6/{path:path}")
async def v6_deprecated_handler(path: str):
    """All other v6 routes → migration message"""
    return {
        "deprecated": True,
        "message": f"v6 endpoint '/{path}' is deprecated. Please use v8 API.",
        "documentation": "/docs",
        "migration_guide": "https://docs.bi-ide.dev/migration/v6-to-v8",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0",
    }
