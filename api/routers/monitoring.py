"""
روتر المراقبة - Monitoring Router

يوفر نقاط النهاية لمراقبة النظام والعمال.
Provides endpoints for system and worker monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User

router = APIRouter(prefix="/monitoring", tags=["المراقبة | Monitoring"])


class WorkerStatus(str, Enum):
    """حالة العامل | Worker status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"


class AlertSeverity(str, Enum):
    """شدة التنبيه | Alert severity"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """حالة التنبيه | Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# نماذج Pydantic - Pydantic Models
class ResourceMetrics(BaseModel):
    """نموذج مقاييس الموارد | Resource metrics model"""
    worker_id: str
    hostname: str
    cpu_percent: float = Field(..., ge=0, le=100)
    cpu_cores: int
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float = Field(..., ge=0, le=100)
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float = Field(..., ge=0, le=100)
    gpu_count: int = 0
    gpu_metrics: Optional[List[Dict[str, Any]]] = None
    network_io_mb: Dict[str, float]
    timestamp: datetime


class GPUInfo(BaseModel):
    """نموذج معلومات GPU | GPU info model"""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_celsius: float
    power_draw_watts: float


class WorkerInfo(BaseModel):
    """نموذج معلومات العامل | Worker info model"""
    id: str
    hostname: str
    status: WorkerStatus
    ip_address: str
    platform: str
    python_version: str
    capabilities: List[str]
    current_task: Optional[str] = None
    last_seen: datetime
    uptime_seconds: int
    resources: ResourceMetrics


class Alert(BaseModel):
    """نموذج التنبيه | Alert model"""
    id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    source: str  # worker_id or system
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[int] = None
    resolved_at: Optional[datetime] = None


class LogEntry(BaseModel):
    """نموذج سجل النظام | System log entry model"""
    timestamp: datetime
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    source: str
    message: str
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SystemResourcesResponse(BaseModel):
    """نموذج استجابة موارد النظام | System resources response model"""
    cluster_summary: Dict[str, Any]
    workers: List[ResourceMetrics]
    updated_at: datetime


class AlertAcknowledgeRequest(BaseModel):
    """نموذج طلب تأكيد التنبيه | Alert acknowledge request model"""
    alert_id: str
    comment: Optional[str] = None


# قواعد بيانات وهمية - Fake Databases
fake_workers = {
    "worker-01": {
        "id": "worker-01",
        "hostname": "gpu-node-01",
        "status": WorkerStatus.ONLINE,
        "ip_address": "192.168.1.101",
        "platform": "Linux-5.15.0",
        "python_version": "3.11.4",
        "capabilities": ["gpu", "training", "inference"],
        "current_task": "training-job-001",
        "last_seen": datetime.utcnow(),
        "uptime_seconds": 86400,
        "resources": {
            "worker_id": "worker-01",
            "hostname": "gpu-node-01",
            "cpu_percent": 45.2,
            "cpu_cores": 16,
            "memory_total_gb": 64.0,
            "memory_used_gb": 32.5,
            "memory_percent": 50.8,
            "disk_total_gb": 1000.0,
            "disk_used_gb": 450.0,
            "disk_percent": 45.0,
            "gpu_count": 2,
            "gpu_metrics": [
                {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "memory_total_mb": 40960,
                    "memory_used_mb": 28672,
                    "utilization_percent": 85.0,
                    "temperature_celsius": 72,
                    "power_draw_watts": 280
                },
                {
                    "index": 1,
                    "name": "NVIDIA A100",
                    "memory_total_mb": 40960,
                    "memory_used_mb": 20480,
                    "utilization_percent": 65.0,
                    "temperature_celsius": 68,
                    "power_draw_watts": 250
                }
            ],
            "network_io_mb": {"sent": 1250.5, "received": 980.3},
            "timestamp": datetime.utcnow()
        }
    },
    "worker-02": {
        "id": "worker-02",
        "hostname": "cpu-node-01",
        "status": WorkerStatus.IDLE,
        "ip_address": "192.168.1.102",
        "platform": "Linux-5.15.0",
        "python_version": "3.11.4",
        "capabilities": ["cpu", "inference", "data_processing"],
        "current_task": None,
        "last_seen": datetime.utcnow(),
        "uptime_seconds": 172800,
        "resources": {
            "worker_id": "worker-02",
            "hostname": "cpu-node-01",
            "cpu_percent": 15.0,
            "cpu_cores": 32,
            "memory_total_gb": 128.0,
            "memory_used_gb": 24.0,
            "memory_percent": 18.8,
            "disk_total_gb": 2000.0,
            "disk_used_gb": 800.0,
            "disk_percent": 40.0,
            "gpu_count": 0,
            "gpu_metrics": None,
            "network_io_mb": {"sent": 450.2, "received": 380.1},
            "timestamp": datetime.utcnow()
        }
    }
}

fake_alerts = {
    "alert-001": {
        "id": "alert-001",
        "severity": AlertSeverity.WARNING,
        "status": AlertStatus.ACTIVE,
        "title": "استخدام GPU مرتفع | High GPU Usage",
        "message": "استخدام GPU على worker-01 تجاوز 80%",
        "source": "worker-01",
        "created_at": datetime.utcnow() - timedelta(hours=2),
        "acknowledged_at": None,
        "acknowledged_by": None,
        "resolved_at": None
    },
    "alert-002": {
        "id": "alert-002",
        "severity": AlertSeverity.INFO,
        "status": AlertStatus.ACKNOWLEDGED,
        "title": "تحديث النظام متاح | System update available",
        "message": "يوجد تحديث أمني متاح للنظام",
        "source": "system",
        "created_at": datetime.utcnow() - timedelta(days=1),
        "acknowledged_at": datetime.utcnow() - timedelta(hours=12),
        "acknowledged_by": 1,
        "resolved_at": None
    }
}

fake_logs = [
    {
        "timestamp": datetime.utcnow() - timedelta(minutes=5),
        "level": "INFO",
        "source": "worker-01",
        "message": "بدأت مهمة التدريب training-job-001",
        "worker_id": "worker-01",
        "metadata": {"job_id": "training-job-001"}
    },
    {
        "timestamp": datetime.utcnow() - timedelta(minutes=10),
        "level": "WARNING",
        "source": "monitoring",
        "message": "استخدام GPU مرتفع على worker-01",
        "worker_id": "worker-01",
        "metadata": {"gpu_utilization": 85.0}
    },
    {
        "timestamp": datetime.utcnow() - timedelta(minutes=30),
        "level": "ERROR",
        "source": "worker-02",
        "message": "فشل في الاتصال بقاعدة البيانات",
        "worker_id": "worker-02",
        "metadata": {"error": "ConnectionTimeout"}
    }
]

# WebSocket connections
ws_connections: List[WebSocket] = []


@router.get(
    "/system/resources",
    response_model=SystemResourcesResponse,
    status_code=status.HTTP_200_OK,
    summary="موارد النظام | System resources"
)
async def get_system_resources(
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على موارد CPU/GPU/RAM لجميع العمال.
    Get CPU/GPU/RAM resources for all workers.
    """
    workers_resources = [w["resources"] for w in fake_workers.values()]
    
    # حساب الإحصائيات | Calculate summary
    total_cpu = sum(r["cpu_percent"] for r in workers_resources) / len(workers_resources)
    total_memory = sum(r["memory_percent"] for r in workers_resources) / len(workers_resources)
    total_gpus = sum(r.get("gpu_count", 0) for r in workers_resources)
    
    return SystemResourcesResponse(
        cluster_summary={
            "workers_online": len([w for w in fake_workers.values() if w["status"] == WorkerStatus.ONLINE]),
            "workers_total": len(fake_workers),
            "avg_cpu_percent": round(total_cpu, 2),
            "avg_memory_percent": round(total_memory, 2),
            "total_gpus": total_gpus,
            "active_training_jobs": len([w for w in fake_workers.values() if w["current_task"]])
        },
        workers=workers_resources,
        updated_at=datetime.utcnow()
    )


@router.get(
    "/workers",
    response_model=List[WorkerInfo],
    status_code=status.HTTP_200_OK,
    summary="حالة العمال | Worker status"
)
async def list_workers(
    status: Optional[WorkerStatus] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على حالة جميع العمال.
    Get status of all workers.
    """
    workers = list(fake_workers.values())
    
    if status:
        workers = [w for w in workers if w["status"] == status]
    
    # تحويل البيانات | Transform data
    result = []
    for w in workers:
        worker_data = {**w}
        worker_data["resources"] = ResourceMetrics(**w["resources"])
        result.append(WorkerInfo(**worker_data))
    
    return result


@router.get(
    "/workers/{worker_id}",
    response_model=WorkerInfo,
    status_code=status.HTTP_200_OK,
    summary="تفاصيل العامل | Worker details"
)
async def get_worker(
    worker_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على تفاصيل عامل محدد.
    Get details of a specific worker.
    """
    if worker_id not in fake_workers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="العامل غير موجود | Worker not found"
        )
    
    worker = fake_workers[worker_id]
    worker_data = {**worker}
    worker_data["resources"] = ResourceMetrics(**worker["resources"])
    
    return WorkerInfo(**worker_data)


@router.get(
    "/alerts",
    response_model=List[Alert],
    status_code=status.HTTP_200_OK,
    summary="التنبيهات النشطة | Active alerts"
)
async def list_alerts(
    severity: Optional[AlertSeverity] = None,
    status: Optional[AlertStatus] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على التنبيهات النشطة.
    Get active alerts.
    """
    alerts = list(fake_alerts.values())
    
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if status:
        alerts = [a for a in alerts if a["status"] == status]
    
    return [Alert(**a) for a in sorted(
        alerts,
        key=lambda x: x["created_at"],
        reverse=True
    )]


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=Alert,
    status_code=status.HTTP_200_OK,
    summary="تأكيد التنبيه | Acknowledge alert"
)
async def acknowledge_alert(
    alert_id: str,
    comment: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    تأكيد التنبيه.
    Acknowledge an alert.
    """
    if alert_id not in fake_alerts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="التنبيه غير موجود | Alert not found"
        )
    
    alert = fake_alerts[alert_id]
    
    if alert["status"] != AlertStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="التنبيه ليس نشطاً | Alert is not active"
        )
    
    alert["status"] = AlertStatus.ACKNOWLEDGED
    alert["acknowledged_at"] = datetime.utcnow()
    alert["acknowledged_by"] = current_user.id
    
    return Alert(**alert)


@router.get(
    "/logs",
    response_model=List[LogEntry],
    status_code=status.HTTP_200_OK,
    summary="سجلات النظام | System logs"
)
async def get_logs(
    level: Optional[str] = None,
    worker_id: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على سجلات النظام.
    Get system logs.
    """
    logs = fake_logs
    
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]
    if worker_id:
        logs = [l for l in logs if l.get("worker_id") == worker_id]
    
    return [LogEntry(**l) for l in sorted(
        logs[:limit],
        key=lambda x: x["timestamp"],
        reverse=True
    )]


# WebSocket للمراقبة الفورية
@router.websocket("/ws/realtime")
async def realtime_monitoring_websocket(websocket: WebSocket):
    """
    WebSocket للمراقبة الفورية.
    WebSocket for real-time monitoring.
    """
    await websocket.accept()
    ws_connections.append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "متصل بالمراقبة الفورية | Connected to real-time monitoring",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # إرسال تحديثات دورية | Send periodic updates
        while True:
            # محاكاة بيانات فورية | Simulate real-time data
            update = {
                "type": "metrics_update",
                "timestamp": datetime.utcnow().isoformat(),
                "workers": [
                    {
                        "worker_id": w["id"],
                        "status": w["status"],
                        "cpu_percent": w["resources"]["cpu_percent"],
                        "memory_percent": w["resources"]["memory_percent"],
                        "current_task": w["current_task"]
                    }
                    for w in fake_workers.values()
                ]
            }
            
            await websocket.send_json(update)
            await asyncio.sleep(5)  # تحديث كل 5 ثواني
            
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
    except Exception as e:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
