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


# قواعد بيانات وهمية - Fake Databases
fake_devices = {
    "gpu-01": {
        "device_id": "gpu-01",
        "device_type": DeviceType.GPU,
        "status": TrainingStatus.IDLE,
        "cpu_percent": 15.0,
        "memory_percent": 30.0,
        "gpu_utilization": 0.0,
        "gpu_memory_used": 1024.0,
        "temperature": 45.0,
        "active_task": None
    },
    "gpu-02": {
        "device_id": "gpu-02",
        "device_type": DeviceType.GPU,
        "status": TrainingStatus.IDLE,
        "cpu_percent": 20.0,
        "memory_percent": 25.0,
        "gpu_utilization": 0.0,
        "gpu_memory_used": 512.0,
        "temperature": 42.0,
        "active_task": None
    },
    "cpu-01": {
        "device_id": "cpu-01",
        "device_type": DeviceType.CPU,
        "status": TrainingStatus.IDLE,
        "cpu_percent": 10.0,
        "memory_percent": 20.0,
        "temperature": 35.0,
        "active_task": None
    }
}

fake_models = {
    "model-001": {
        "id": "model-001",
        "name": "Code Generator v1",
        "version": "1.0.0",
        "status": ModelStatus.READY,
        "architecture": "Transformer",
        "parameters_count": 700000000,
        "dataset_size": 500000,
        "trained_epochs": 50,
        "created_at": datetime.utcnow(),
        "deployed_at": None,
        "metrics": {
            "bleu_score": 0.85,
            "perplexity": 12.3
        }
    }
}

fake_jobs = {}
fake_history = []
fake_job_counter = 1

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
    active_jobs = sum(
        1 for j in fake_jobs.values()
        if j["status"] == TrainingStatus.RUNNING
    )
    queued_jobs = sum(
        1 for j in fake_jobs.values()
        if j["status"] in [TrainingStatus.IDLE, TrainingStatus.PAUSED]
    )
    
    active_devices = sum(
        1 for d in fake_devices.values()
        if d["status"] == TrainingStatus.RUNNING
    )
    
    # حساب متوسط الاستخدام | Calculate average utilization
    total_util = sum(d.get("gpu_utilization", d["cpu_percent"]) for d in fake_devices.values())
    avg_util = total_util / len(fake_devices) if fake_devices else 0
    
    return OverallTrainingStatus(
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
        total_devices=len(fake_devices),
        active_devices=active_devices,
        avg_cluster_utilization=round(avg_util, 2),
        jobs=list(fake_jobs.values())
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
    global fake_job_counter
    
    # اختيار الأجهزة | Select devices
    if config.device_ids:
        devices = config.device_ids
    else:
        # استخدام جميع وحدات GPU المتاحة | Use all available GPUs
        devices = [
            d_id for d_id, d in fake_devices.items()
            if d["device_type"] == DeviceType.GPU and d["status"] == TrainingStatus.IDLE
        ]
    
    if not devices:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا توجد أجهزة متاحة | No devices available"
        )
    
    # تحديث حالة الأجهزة | Update device status
    for device_id in devices:
        if device_id in fake_devices:
            fake_devices[device_id]["status"] = TrainingStatus.RUNNING
    
    job_id = f"job-{fake_job_counter:04d}"
    fake_job_counter += 1
    
    fake_jobs[job_id] = {
        "job_id": job_id,
        "status": TrainingStatus.RUNNING,
        "config": config.dict(),
        "devices": devices,
        "started_at": datetime.utcnow(),
        "started_by": current_user.id,
        "progress": {
            "current_epoch": 0,
            "total_epochs": config.epochs
        }
    }
    
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
    if job_id not in fake_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المهمة غير موجودة | Job not found"
        )
    
    job = fake_jobs[job_id]
    
    # تحرير الأجهزة | Release devices
    for device_id in job.get("devices", []):
        if device_id in fake_devices:
            fake_devices[device_id]["status"] = TrainingStatus.IDLE
            fake_devices[device_id]["active_task"] = None
    
    job["status"] = TrainingStatus.COMPLETED
    job["completed_at"] = datetime.utcnow()
    
    # إضافة إلى السجل | Add to history
    fake_history.append({
        "id": job_id,
        "model_name": job["config"].get("model_id", "unknown"),
        "status": TrainingStatus.COMPLETED,
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "total_epochs": job["config"].get("epochs", 0),
        "best_accuracy": 0.0,  # سيتم تحديثه | Will be updated
        "devices_used": job.get("devices", [])
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
    # إرجاع مقاييس وهمية | Return fake metrics
    return TrainingMetrics(
        epoch=5,
        total_epochs=10,
        loss=0.234,
        accuracy=0.876,
        val_loss=0.245,
        val_accuracy=0.865,
        learning_rate=0.001,
        samples_per_second=125.5,
        estimated_time_remaining=1800
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
    return [ModelInfo(**m) for m in fake_models.values()]


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
    if model_id not in fake_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="النموذج غير موجود | Model not found"
        )
    
    model = fake_models[model_id]
    
    if model["status"] == ModelStatus.DEPLOYED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="النموذج منشور مسبقاً | Model already deployed"
        )
    
    model["status"] = ModelStatus.DEPLOYED
    model["deployed_at"] = datetime.utcnow()
    
    return {
        "model_id": model_id,
        "status": ModelStatus.DEPLOYED,
        "deployed_at": model["deployed_at"],
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
