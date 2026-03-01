"""
روتر المراقبة - Monitoring Router

يوفر نقاط النهاية لمراقبة النظام والعمال.
Provides endpoints for system and worker monitoring.

V2:
- No fake in-memory stores
- Real system snapshot via psutil
- Persistent alerts/logs via JSON files
"""

import asyncio
import json
import platform
import socket
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

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
    source: str
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class LogEntry(BaseModel):
    """نموذج سجل النظام | System log entry model"""
    timestamp: datetime
    level: str
    source: str
    message: str
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SystemResourcesResponse(BaseModel):
    """نموذج استجابة موارد النظام | System resources response model"""
    cluster_summary: Dict[str, Any]
    workers: List[ResourceMetrics]
    updated_at: datetime


DATA_DIR = Path("data/monitoring")
ALERTS_FILE = DATA_DIR / "alerts.json"
LOGS_FILE = DATA_DIR / "logs.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> datetime:
    return datetime.utcnow()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_json(file_path: Path, default):
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(file_path: Path, payload) -> None:
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_alert(alert: Alert) -> Dict[str, Any]:
    return {
        "id": alert.id,
        "severity": alert.severity.value,
        "status": alert.status.value,
        "title": alert.title,
        "message": alert.message,
        "source": alert.source,
        "created_at": alert.created_at.isoformat(),
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "acknowledged_by": alert.acknowledged_by,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
    }


def _deserialize_alert(data: Dict[str, Any]) -> Alert:
    return Alert(
        id=data["id"],
        severity=AlertSeverity(data["severity"]),
        status=AlertStatus(data["status"]),
        title=data["title"],
        message=data["message"],
        source=data["source"],
        created_at=_parse_dt(data.get("created_at")) or _now(),
        acknowledged_at=_parse_dt(data.get("acknowledged_at")),
        acknowledged_by=data.get("acknowledged_by"),
        resolved_at=_parse_dt(data.get("resolved_at")),
    )


def _load_alerts() -> List[Alert]:
    raw = _load_json(ALERTS_FILE, default=[])
    return [_deserialize_alert(item) for item in raw]


def _save_alerts(alerts: List[Alert]) -> None:
    _save_json(ALERTS_FILE, [_serialize_alert(a) for a in alerts])


def _load_logs() -> List[LogEntry]:
    raw = _load_json(LOGS_FILE, default=[])
    output: List[LogEntry] = []
    for item in raw:
        output.append(
            LogEntry(
                timestamp=_parse_dt(item.get("timestamp")) or _now(),
                level=item.get("level", "INFO"),
                source=item.get("source", "monitoring"),
                message=item.get("message", ""),
                worker_id=item.get("worker_id"),
                metadata=item.get("metadata"),
            )
        )
    return output


def _append_log(level: str, source: str, message: str, worker_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    logs = _load_json(LOGS_FILE, default=[])
    logs.append(
        {
            "timestamp": _now().isoformat(),
            "level": level,
            "source": source,
            "message": message,
            "worker_id": worker_id,
            "metadata": metadata or {},
        }
    )
    logs = logs[-1000:]
    _save_json(LOGS_FILE, logs)


def _system_snapshot() -> WorkerInfo:
    hostname = socket.gethostname()
    ip_address = "127.0.0.1"
    try:
        ip_address = socket.gethostbyname(hostname)
    except Exception:
        pass

    if psutil is None:
        resources = ResourceMetrics(
            worker_id=hostname,
            hostname=hostname,
            cpu_percent=0.0,
            cpu_cores=1,
            memory_total_gb=0.0,
            memory_used_gb=0.0,
            memory_percent=0.0,
            disk_total_gb=0.0,
            disk_used_gb=0.0,
            disk_percent=0.0,
            gpu_count=0,
            gpu_metrics=None,
            network_io_mb={"sent": 0.0, "received": 0.0},
            timestamp=_now(),
        )
    else:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")
        net = psutil.net_io_counters()
        resources = ResourceMetrics(
            worker_id=hostname,
            hostname=hostname,
            cpu_percent=float(psutil.cpu_percent(interval=0.1)),
            cpu_cores=int(psutil.cpu_count(logical=True) or 1),
            memory_total_gb=round(vm.total / (1024 ** 3), 2),
            memory_used_gb=round(vm.used / (1024 ** 3), 2),
            memory_percent=float(vm.percent),
            disk_total_gb=round(du.total / (1024 ** 3), 2),
            disk_used_gb=round(du.used / (1024 ** 3), 2),
            disk_percent=float(du.percent),
            gpu_count=0,
            gpu_metrics=None,
            network_io_mb={
                "sent": round(net.bytes_sent / (1024 ** 2), 2),
                "received": round(net.bytes_recv / (1024 ** 2), 2),
            },
            timestamp=_now(),
        )

    status = WorkerStatus.BUSY if resources.cpu_percent >= 70 else WorkerStatus.IDLE

    return WorkerInfo(
        id=hostname,
        hostname=hostname,
        status=status,
        ip_address=ip_address,
        platform=platform.platform(),
        python_version=platform.python_version(),
        capabilities=["monitoring", "local-system"],
        current_task=None,
        last_seen=_now(),
        uptime_seconds=int(time.time() - psutil.boot_time()) if psutil else 0,
        resources=resources,
    )


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
    worker = _system_snapshot()
    return SystemResourcesResponse(
        cluster_summary={
            "workers_online": 1,
            "workers_total": 1,
            "avg_cpu_percent": worker.resources.cpu_percent,
            "avg_memory_percent": worker.resources.memory_percent,
            "total_gpus": worker.resources.gpu_count,
            "active_training_jobs": 1 if worker.status == WorkerStatus.BUSY else 0,
        },
        workers=[worker.resources],
        updated_at=_now(),
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
    worker = _system_snapshot()
    workers = [worker]
    if status:
        workers = [w for w in workers if w.status == status]
    return workers


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
    worker = _system_snapshot()
    if worker.id != worker_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="العامل غير موجود | Worker not found")
    return worker


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
    alerts = _load_alerts()

    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    if status:
        alerts = [a for a in alerts if a.status == status]

    return sorted(alerts, key=lambda x: x.created_at, reverse=True)


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
    alerts = _load_alerts()
    target = next((a for a in alerts if a.id == alert_id), None)

    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="التنبيه غير موجود | Alert not found")

    if target.status != AlertStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="التنبيه ليس نشطاً | Alert is not active")

    target.status = AlertStatus.ACKNOWLEDGED
    target.acknowledged_at = _now()
    target.acknowledged_by = str(current_user.id)

    _save_alerts(alerts)
    _append_log(
        level="INFO",
        source="monitoring",
        message=f"Alert acknowledged: {alert_id}",
        metadata={"comment": comment} if comment else {},
    )

    return target


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
    logs = _load_logs()

    if level:
        logs = [l for l in logs if l.level.upper() == level.upper()]
    if worker_id:
        logs = [l for l in logs if l.worker_id == worker_id]

    logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)
    return logs[:limit]


@router.post(
    "/alerts/create",
    response_model=Alert,
    status_code=status.HTTP_201_CREATED,
    summary="إنشاء تنبيه | Create alert"
)
async def create_alert(
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str = "system",
    current_user: User = Depends(get_current_active_user),
):
    alerts = _load_alerts()
    alert = Alert(
        id=f"alert-{uuid.uuid4().hex[:8]}",
        severity=severity,
        status=AlertStatus.ACTIVE,
        title=title,
        message=message,
        source=source,
        created_at=_now(),
    )
    alerts.append(alert)
    _save_alerts(alerts)
    _append_log(level="WARNING" if severity != AlertSeverity.INFO else "INFO", source="monitoring", message=title)
    return alert


@router.websocket("/ws/realtime")
async def realtime_monitoring_websocket(websocket: WebSocket):
    """WebSocket for real-time monitoring updates."""
    await websocket.accept()
    ws_connections.append(websocket)

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "message": "متصل بالمراقبة الفورية | Connected to real-time monitoring",
                "timestamp": _now().isoformat(),
            }
        )

        while True:
            worker = _system_snapshot()
            update = {
                "type": "metrics_update",
                "timestamp": _now().isoformat(),
                "workers": [
                    {
                        "worker_id": worker.id,
                        "status": worker.status.value,
                        "cpu_percent": worker.resources.cpu_percent,
                        "memory_percent": worker.resources.memory_percent,
                        "current_task": worker.current_task,
                    }
                ],
            }
            await websocket.send_json(update)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
    except Exception:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
