"""
Brain Router - واجهة دماغ BI
Provides endpoints for the BI Brain orchestrator.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brain", tags=["الدماغ | Brain"])


class ScheduleTrainingRequest(BaseModel):
    """طلب جدولة تدريب"""
    name: str
    layer_name: str = "general"
    priority: str = "medium"
    config: Optional[Dict[str, Any]] = None


class EvaluateModelRequest(BaseModel):
    """طلب تقييم نموذج"""
    model_id: str
    model_name: str
    metrics: Dict[str, float]


class ConsultRequest(BaseModel):
    """طلب استشارة من الهرمية"""
    question: str


def _get_brain():
    """Get brain singleton"""
    from brain.bi_brain import brain
    return brain


@router.get(
    "/status",
    summary="حالة الدماغ | Brain Status"
)
async def brain_status():
    """
    الحصول على حالة دماغ BI الحالية.
    Get current BI Brain status.
    """
    brain = _get_brain()
    status = brain.get_status()
    
    # Add hierarchy connection status
    hierarchy_status = "connected" if brain.hierarchy else "disconnected"
    status["hierarchy_connection"] = hierarchy_status
    
    return status


@router.post(
    "/schedule",
    summary="جدولة تدريب | Schedule Training"
)
async def schedule_training(request: ScheduleTrainingRequest):
    """
    جدولة مهمة تدريب جديدة.
    Schedule a new training job.
    """
    brain = _get_brain()
    job_id = await brain.schedule_training(
        name=request.name,
        layer_name=request.layer_name,
        priority=request.priority,
        config=request.config,
    )
    return {"status": "scheduled", "job_id": job_id}


@router.post(
    "/evaluate",
    summary="تقييم نموذج | Evaluate Model"
)
async def evaluate_model(request: EvaluateModelRequest):
    """
    تقييم نموذج والتحقق من جاهزيته للنشر.
    Evaluate a model and check deployment readiness.
    """
    brain = _get_brain()
    result = await brain.evaluate_model(
        model_id=request.model_id,
        model_name=request.model_name,
        metrics=request.metrics,
    )
    return result


@router.get(
    "/jobs",
    summary="قائمة المهام | Job List"
)
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    الحصول على قائمة مهام التدريب.
    Get list of training jobs.
    """
    brain = _get_brain()
    jobs = brain.get_jobs(status=status, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.get(
    "/evaluations",
    summary="قائمة التقييمات | Evaluation List"
)
async def list_evaluations(model_id: Optional[str] = None, limit: int = 50):
    """
    الحصول على قائمة تقييمات النماذج.
    Get list of model evaluations.
    """
    brain = _get_brain()
    evaluations = brain.get_evaluations(model_id=model_id, limit=limit)
    return {"evaluations": evaluations, "count": len(evaluations)}


@router.post(
    "/consult",
    summary="استشارة الهرمية | Consult Hierarchy"
)
async def consult_hierarchy(request: ConsultRequest):
    """
    استشارة النظام الهرمي لاتخاذ قرار.
    Consult the hierarchy system for a decision.
    """
    brain = _get_brain()
    result = await brain.consult_hierarchy(request.question)
    return result


@router.post(
    "/start",
    summary="تشغيل الدماغ | Start Brain"
)
async def start_brain():
    """
    تشغيل دماغ BI (الجدولة + المراقبة).
    Start BI Brain (scheduling + monitoring).
    """
    brain = _get_brain()
    if brain.is_running:
        return {"status": "already_running"}
    
    await brain.start()
    return {"status": "started"}


@router.post(
    "/stop",
    summary="إيقاف الدماغ | Stop Brain"
)
async def stop_brain():
    """
    إيقاف دماغ BI.
    Stop BI Brain.
    """
    brain = _get_brain()
    if not brain.is_running:
        return {"status": "already_stopped"}
    
    brain.stop()
    return {"status": "stopped"}
