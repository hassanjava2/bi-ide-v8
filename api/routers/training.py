"""
روتر التدريب - Training Router

يوفر نقاط النهاية لإدارة التدريب الموزع.
Provides endpoints for distributed training management.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User
from services.training_service import training_service, TrainingStatus as ServiceTrainingStatus

router = APIRouter(prefix="/training", tags=["التدريب | Training"])


class TrainingStatus(str, Enum):
    """حالة التدريب | Training status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(str, Enum):
    """حالة النموذج | Model status"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class DeviceType(str, Enum):
    """نوع الجهاز | Device type"""
    GPU = "gpu"
    CPU = "cpu"
    TPU = "tpu"


# نماذج Pydantic - Pydantic Models
class DeviceMetrics(BaseModel):
    """نموذج مقاييس الجهاز | Device metrics model"""
    device_id: str
    device_type: DeviceType
    status: TrainingStatus
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    temperature: Optional[float] = None
    active_task: Optional[str] = None


class TrainingMetrics(BaseModel):
    """نموذج مقاييس التدريب | Training metrics model"""
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    samples_per_second: float
    estimated_time_remaining: int


class ModelInfo(BaseModel):
    """نموذج معلومات النموذج | Model info model"""
    id: str
    name: str
    version: str
    status: ModelStatus
    architecture: str
    parameters_count: int
    dataset_size: int
    trained_epochs: int
    created_at: datetime
    deployed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None


class TrainingConfig(BaseModel):
    """نموذج إعدادات التدريب | Training config model"""
    model_id: Optional[str] = None
    dataset_path: str
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: str = "adam"
    distributed: bool = True
    device_ids: Optional[List[str]] = None


class TrainingHistoryItem(BaseModel):
    """نموذج عنصر سجل التدريب | Training history item model"""
    id: str
    model_name: str
    status: TrainingStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_epochs: int
    best_accuracy: float
    devices_used: List[str]


class StartTrainingResponse(BaseModel):
    """نموذج استجابة بدء التدريب | Start training response model"""
    job_id: str
    status: TrainingStatus
    message: str
    devices_assigned: List[str]


class OverallTrainingStatus(BaseModel):
    """نموذج حالة التدريب الإجمالية | Overall training status model"""
    active_jobs: int
    queued_jobs: int
    total_devices: int
    active_devices: int
    avg_cluster_utilization: float
    jobs: List[Dict[str, Any]]


# WebSocket connections
ws_connections: List[WebSocket] = []


async def get_training_history_from_db(limit: int = 50) -> List[Dict[str, Any]]:
    """Get real training history from database"""
    try:
        from core.database import db_manager
        history = await db_manager.get_training_history(limit)
        return history
    except Exception as e:
        logger.error(f"Error fetching training history from DB: {e}")
        # Fallback to training_service if DB fails
        try:
            jobs = await training_service.get_all_jobs()
            history = []
            for job in jobs:
                if job.status.value in ["completed", "failed", "cancelled"]:
                    history.append({
                        "id": job.job_id,
                        "model_name": job.model_name,
                        "status": job.status.value,
                        "started_at": job.created_at,
                        "completed_at": getattr(job, 'completed_at', None),
                        "total_epochs": job.config.get("epochs", 0),
                        "best_accuracy": (job.metrics or {}).get("accuracy", 0.0),
                        "devices_used": job.config.get("devices", ["local"]),
                    })
            history.sort(key=lambda x: x["started_at"], reverse=True)
            return history[:limit]
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return []

import logging
logger = logging.getLogger(__name__)


@router.get(
    "/status",
    response_model=OverallTrainingStatus,
    status_code=status.HTTP_200_OK,
    summary="حالة التدريب | Training status"
)
async def get_training_status(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على حالة التدريب عبر جميع الأجهزة.
    Get training status across all devices.
    """
    jobs = await training_service.get_all_jobs()
    active_jobs = sum(1 for j in jobs if j.status == ServiceTrainingStatus.RUNNING)
    queued_jobs = sum(1 for j in jobs if j.status in {ServiceTrainingStatus.PENDING, ServiceTrainingStatus.PAUSED})

    active_devices = 1 if active_jobs > 0 else 0
    avg_util = 70.0 if active_jobs > 0 else 10.0
    
    return OverallTrainingStatus(
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
        total_devices=1,
        active_devices=active_devices,
        avg_cluster_utilization=round(avg_util, 2),
        jobs=[
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "config": j.config,
                "started_at": j.created_at,
                "progress": {
                    "current_epoch": (j.metrics or {}).get("epoch", 0),
                    "total_epochs": j.config.get("epochs", 0),
                },
            }
            for j in jobs
        ]
    )


@router.post(
    "/start",
    response_model=StartTrainingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="بدء التدريب | Start training"
)
async def start_training(
    config: TrainingConfig,
    current_user: User = Depends(get_current_active_user)
):
    """
    بدء مهمة تدريب جديدة.
    Start a new training job.
    """
    devices = config.device_ids or ["local-gpu-01"]
    job_id = f"job-{int(datetime.utcnow().timestamp())}"

    # Start in training service
    await training_service.start_training(
        job_id=job_id,
        model_name=config.model_id or "bi-ide-model",
        config=config.dict(),
    )
    
    # Also store in db_manager for consistent history
    try:
        from core.database import db_manager
        await db_manager.store_training_job(
            job_id=job_id,
            model_name=config.model_id or "bi-ide-model",
            status="running",
            config=config.dict(),
            devices_used=devices
        )
    except Exception as e:
        logger.warning(f"Failed to store job in db_manager: {e}")
    
    return StartTrainingResponse(
        job_id=job_id,
        status=TrainingStatus.RUNNING,
        message="تم بدء التدريب بنجاح | Training started successfully",
        devices_assigned=devices
    )


@router.post(
    "/stop",
    status_code=status.HTTP_200_OK,
    summary="إيقاف التدريب | Stop training"
)
async def stop_training(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    إيقاف مهمة تدريب.
    Stop a training job.
    """
    job = await training_service.get_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المهمة غير موجودة | Job not found"
        )

    await training_service.stop_training(job_id)
    updated_job = await training_service.get_status(job_id)
    
    # Set completed_at for history tracking
    if updated_job:
        updated_job.completed_at = datetime.utcnow()
    
    return {
        "job_id": job_id,
        "status": TrainingStatus.COMPLETED,
        "message": "تم إيقاف التدريب | Training stopped"
    }


@router.get(
    "/metrics",
    response_model=TrainingMetrics,
    status_code=status.HTTP_200_OK,
    summary="مقاييس التدريب | Training metrics"
)
async def get_metrics(
    job_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على مقاييس التدريب.
    Get training metrics.
    """
    target_job_id = job_id
    if not target_job_id:
        jobs = await training_service.get_all_jobs(limit=1)
        if jobs:
            target_job_id = jobs[0].job_id

    metrics = {}
    if target_job_id:
        metrics = await training_service.get_metrics(target_job_id) or {}

    return TrainingMetrics(
        epoch=metrics.get("epoch", 0),
        total_epochs=metrics.get("total_epochs", 0),
        loss=float(metrics.get("loss", 0.0)),
        accuracy=float(metrics.get("accuracy", 0.0)),
        val_loss=float(metrics.get("loss", 0.0)),
        val_accuracy=float(metrics.get("accuracy", 0.0)),
        learning_rate=float(metrics.get("learning_rate", 0.001)),
        samples_per_second=float(metrics.get("samples_per_second", 0.0)),
        estimated_time_remaining=int(metrics.get("estimated_time_remaining", 0)),
    )


@router.get(
    "/models",
    response_model=List[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="قائمة النماذج | List models"
)
async def list_models(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على قائمة النماذج المدربة.
    Get list of trained models.
    """
    models = await training_service.list_models()
    return [
        ModelInfo(
            id=m.model_id,
            name=m.name,
            version=m.version,
            status=ModelStatus.DEPLOYED if m.is_deployed else ModelStatus.READY,
            architecture="Transformer",
            parameters_count=int(m.metadata.get("parameters", 0)),
            dataset_size=int(m.metadata.get("dataset_size", 0)),
            trained_epochs=int(m.metadata.get("trained_epochs", 0)),
            created_at=m.created_at,
            deployed_at=m.created_at if m.is_deployed else None,
            metrics={"accuracy": m.accuracy},
        )
        for m in models
    ]


@router.post(
    "/models/{model_id}/deploy",
    status_code=status.HTTP_200_OK,
    summary="نشر النموذج | Deploy model"
)
async def deploy_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    نشر نموذج للإنتاج.
    Deploy a model to production.
    """
    deployed = await training_service.deploy_model(model_id)
    if not deployed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="النموذج غير موجود | Model not found"
        )

    return {
        "model_id": model_id,
        "status": ModelStatus.DEPLOYED,
        "deployed_at": datetime.utcnow(),
        "message": "تم نشر النموذج بنجاح | Model deployed successfully"
    }


@router.get(
    "/history",
    response_model=List[TrainingHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="سجل التدريب | Training history"
)
async def get_training_history(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على سجل التدريب.
    Get training history from training service.
    """
    history = await get_training_history_from_db(limit)
    return [TrainingHistoryItem(**h) for h in history]


# WebSocket للتحديثات الفورية
@router.websocket("/ws/updates")
async def training_updates_websocket(websocket: WebSocket):
    """
    WebSocket لتحديثات التدريب الفورية.
    WebSocket for real-time training updates.
    """
    await websocket.accept()
    ws_connections.append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "متصل بتحديثات التدريب | Connected to training updates"
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                job_id = data.get("job_id")
                await websocket.send_json({
                    "type": "subscribed",
                    "job_id": job_id,
                    "status": "active"
                })
    
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
