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


fake_history = []

# WebSocket connections
ws_connections: List[WebSocket] = []


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
    jobs = list(training_service._jobs.values())
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

    await training_service.start_training(
        job_id=job_id,
        model_name=config.model_id or "bi-ide-model",
        config=config.dict(),
    )
    
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

    fake_history.append({
        "id": job_id,
        "model_name": updated_job.model_name,
        "status": TrainingStatus.COMPLETED,
        "started_at": updated_job.created_at,
        "completed_at": datetime.utcnow(),
        "total_epochs": updated_job.config.get("epochs", 0),
        "best_accuracy": (updated_job.metrics or {}).get("accuracy", 0.0),
        "devices_used": ["local-gpu-01"],
    })
    
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
    if not target_job_id and training_service._jobs:
        target_job_id = next(reversed(training_service._jobs.keys()))

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
    Get training history.
    """
    return [TrainingHistoryItem(**h) for h in fake_history[:limit]]


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
