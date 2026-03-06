#!/usr/bin/env python3
"""
brain_api.py — جسر الدماغ ↔ IDE

FastAPI Router يربط نظام الكبسولات بالتطبيق:
  POST /brain/ask         → سؤال → الكبسولة الصح → جواب
  POST /brain/ask-multi   → سؤال لأكثر من كبسولة
  GET  /brain/status      → حالة النظام كاملة
  GET  /brain/capsules    → قائمة كل الكبسولات
  GET  /brain/tree        → شجرة التطور
  GET  /brain/rankings    → ترتيب حسب القوة
  POST /brain/eval        → امتحان كبسولة
  POST /brain/council     → قرار المجلس

يضاف إلى rtx_api_server.py:
  app.include_router(brain_router)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("brain_api")

# Lazy imports لتجنب circular
_bus = None
_factory = None
_evaluator = None
_council = None
CAPSULES_DIR = Path(__file__).parent.parent / "capsules"


def _get_bus():
    global _bus
    if _bus is None:
        from brain.capsule_bus import CapsuleBus
        _bus = CapsuleBus(CAPSULES_DIR)
    return _bus


def _get_factory():
    global _factory
    if _factory is None:
        from brain.brain_factory import BrainFactory
        _factory = BrainFactory(CAPSULES_DIR)
    return _factory


def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        from brain.self_eval import SelfEval
        _evaluator = SelfEval(CAPSULES_DIR)
    return _evaluator


def _get_council():
    global _council
    if _council is None:
        from brain.council_brain import CouncilBrain
        _council = CouncilBrain(CAPSULES_DIR)
    return _council


def create_brain_router():
    """إنشاء الـ router — يستدعى من rtx_api_server"""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    router = APIRouter(prefix="/brain", tags=["brain-capsules"])

    # ─── Models ──────────────────────────────────────

    class AskRequest(BaseModel):
        question: str = Field(..., min_length=1, max_length=5000)
        capsule_id: Optional[str] = None
        max_tokens: int = 512

    class AskMultiRequest(BaseModel):
        question: str = Field(..., min_length=1, max_length=5000)
        capsule_ids: Optional[List[str]] = None
        top_n: int = 3

    class EvalRequest(BaseModel):
        capsule_id: str

    # ─── Endpoints ──────────────────────────────────

    @router.post("/ask")
    async def brain_ask(request: AskRequest):
        """سؤال → الكبسولة المناسبة → جواب"""
        t0 = time.time()
        bus = _get_bus()

        result = bus.ask(request.question, request.capsule_id)
        result["processing_ms"] = int((time.time() - t0) * 1000)

        return result

    @router.post("/ask-multi")
    async def brain_ask_multi(request: AskMultiRequest):
        """سؤال لأكثر من كبسولة — جمع الآراء"""
        t0 = time.time()
        bus = _get_bus()

        results = bus.ask_multi(
            request.question,
            request.capsule_ids,
            request.top_n,
        )

        return {
            "results": results,
            "processing_ms": int((time.time() - t0) * 1000),
        }

    @router.get("/status")
    async def brain_status():
        """حالة النظام الكاملة"""
        factory = _get_factory()
        bus = _get_bus()

        status = factory.get_status()
        bus_status = bus.get_status()

        return {
            **status,
            "ready_capsules": bus_status["ready_capsules"],
            "total_queries": bus_status["total_queries"],
            "registry_size": bus_status["registry_size"],
        }

    @router.get("/capsules")
    async def brain_capsules():
        """قائمة كل الكبسولات مع معلوماتها"""
        factory = _get_factory()
        capsules = factory.get_all_capsules()
        return {"capsules": capsules, "total": len(capsules)}

    @router.get("/tree")
    async def brain_tree():
        """شجرة التطور"""
        factory = _get_factory()
        return factory.get_tree()

    @router.get("/rankings")
    async def brain_rankings():
        """ترتيب الكبسولات حسب القوة"""
        evaluator = _get_evaluator()
        return {"rankings": evaluator.get_rankings()}

    @router.post("/eval")
    async def brain_eval(request: EvalRequest):
        """امتحان كبسولة"""
        evaluator = _get_evaluator()
        result = evaluator.evaluate_capsule(request.capsule_id)
        return result

    @router.post("/eval-all")
    async def brain_eval_all():
        """امتحان كل الكبسولات"""
        evaluator = _get_evaluator()
        results = evaluator.evaluate_all()
        return {"results": results}

    @router.post("/council")
    async def brain_council():
        """قرار المجلس"""
        council = _get_council()
        decisions = council.decide()
        executed = council.execute_decisions(decisions)
        return {
            "decisions": decisions,
            "executed": executed,
        }

    @router.get("/route")
    async def brain_route(question: str):
        """شوف أي كبسولة راح تجاوب (بدون جواب فعلي)"""
        bus = _get_bus()
        capsule_id = bus.route(question)
        return {"question": question, "routed_to": capsule_id}

    return router


# للاختبار المباشر
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Brain API", description="BI-IDE Brain System")
    app.include_router(create_brain_router())

    uvicorn.run(app, host="0.0.0.0", port=8091)
