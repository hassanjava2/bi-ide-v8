"""
Health Dashboard - صفحة صحة النظام
Real-time health monitoring dashboard with web interface support
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import uuid4

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """حالات الصحة - Health statuses"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """صحة المكون - Component health"""
    name: str
    status: HealthStatus
    latency_ms: float
    last_check: float
    message: str
    metadata: Dict[str, Any]


@dataclass
class SystemHealth:
    """صحة النظام - System health"""
    timestamp: float
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    alerts: List[Dict[str, Any]]


class HealthDashboard:
    """
    لوحة صحة النظام
    Health Dashboard
    
    توفر مراقبة فورية لصحة النظام مع دعم واجهة الويب
    Provides real-time system health monitoring with web interface support
    """
    
    # عتبات الحالة - Status thresholds
    HEALTHY_THRESHOLD_MS = 100
    WARNING_THRESHOLD_MS = 500
    CRITICAL_THRESHOLD_MS = 1000
    
    # المكونات الافتراضية - Default components
    DEFAULT_COMPONENTS = ['api', 'database', 'redis', 'workers', 'gpu']
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        database_url: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        worker_urls: Optional[List[str]] = None,
        check_interval: float = 2.0,
        websocket_manager: Optional[Any] = None
    ):
        """
        تهيئة لوحة الصحة
        Initialize health dashboard
        
        Args:
            api_url: رابط API
            database_url: رابط قاعدة البيانات
            redis_url: رابط Redis
            worker_urls: روابط العمال
            check_interval: فاصل التحقق بالثواني
            websocket_manager: مدير WebSocket
        """
        self.api_url = api_url
        self.database_url = database_url
        self.redis_url = redis_url
        self.worker_urls = worker_urls or []
        self.check_interval = check_interval
        self.websocket_manager = websocket_manager
        
        self._component_health: Dict[str, ComponentHealth] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[SystemHealth], None]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._history: List[SystemHealth] = []
        self._max_history = 1000
        
        # تهيئة المكونات - Initialize components
        for component in self.DEFAULT_COMPONENTS:
            self._component_health[component] = ComponentHealth(
                name=component,
                status=HealthStatus.UNKNOWN,
                latency_ms=0.0,
                last_check=0.0,
                message="Not checked yet",
                metadata={}
            )
    
    async def start(self) -> None:
        """بدء المراقبة الدورية - Start periodic monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health dashboard started")
    
    async def stop(self) -> None:
        """إيقاف المراقبة - Stop monitoring"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health dashboard stopped")
    
    def add_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """إضافة دالة استدعاء عند تحديث الصحة - Add callback on health update"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def _monitoring_loop(self) -> None:
        """حلقة المراقبة الدورية - Monitoring loop"""
        while self._running:
            try:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_components(self) -> None:
        """التحقق من جميع المكونات - Check all components"""
        tasks = [
            self._check_api(),
            self._check_database(),
            self._check_redis(),
            self._check_workers(),
            self._check_gpu()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # إنشاء حالة النظام - Create system health
        overall_status = self._calculate_overall_status()
        system_health = SystemHealth(
            timestamp=time.time(),
            overall_status=overall_status,
            components=self._component_health.copy(),
            alerts=self._alerts.copy()
        )
        
        # تخزين في التاريخ - Store in history
        self._history.append(system_health)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # إرسال عبر WebSocket - Send via WebSocket
        if self.websocket_manager:
            await self._send_websocket_update(system_health)
        
        # استدعاء الدوال المسجلة - Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(system_health)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _check_api(self) -> None:
        """التحقق من API - Check API"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = self._latency_to_status(latency_ms)
                        message = "API is healthy"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"API returned status {response.status}"
                    
                    self._component_health['api'] = ComponentHealth(
                        name='api',
                        status=status,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        message=message,
                        metadata={'status_code': response.status}
                    )
        except Exception as e:
            self._component_health['api'] = ComponentHealth(
                name='api',
                status=HealthStatus.CRITICAL,
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
                message=f"API check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _check_database(self) -> None:
        """التحقق من قاعدة البيانات - Check database"""
        start_time = time.time()
        try:
            # محاولة الاتصال بقاعدة البيانات عبر API
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health/database",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        status = self._latency_to_status(latency_ms)
                        message = "Database is healthy"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"Database check failed with status {response.status}"
                    
                    self._component_health['database'] = ComponentHealth(
                        name='database',
                        status=status,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        message=message,
                        metadata={'status_code': response.status}
                    )
        except Exception as e:
            self._component_health['database'] = ComponentHealth(
                name='database',
                status=HealthStatus.CRITICAL,
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
                message=f"Database check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _check_redis(self) -> None:
        """التحقق من Redis - Check Redis"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health/redis",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = self._latency_to_status(latency_ms)
                        message = "Redis is healthy"
                    else:
                        status = HealthStatus.WARNING
                        message = f"Redis check returned status {response.status}"
                    
                    self._component_health['redis'] = ComponentHealth(
                        name='redis',
                        status=status,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        message=message,
                        metadata={'status_code': response.status}
                    )
        except Exception as e:
            self._component_health['redis'] = ComponentHealth(
                name='redis',
                status=HealthStatus.CRITICAL,
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
                message=f"Redis check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _check_workers(self) -> None:
        """التحقق من العمال - Check workers"""
        start_time = time.time()
        healthy_workers = 0
        total_workers = len(self.worker_urls) + 1  # +1 للعامل المحلي
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/workers/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        workers = data.get('workers', [])
                        healthy_workers = sum(1 for w in workers if w.get('healthy', False))
                        
                        if healthy_workers == total_workers:
                            status = HealthStatus.HEALTHY
                            message = f"All {total_workers} workers are healthy"
                        elif healthy_workers >= total_workers / 2:
                            status = HealthStatus.WARNING
                            message = f"{healthy_workers}/{total_workers} workers are healthy"
                        else:
                            status = HealthStatus.CRITICAL
                            message = f"Only {healthy_workers}/{total_workers} workers are healthy"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"Workers check failed with status {response.status}"
                    
                    self._component_health['workers'] = ComponentHealth(
                        name='workers',
                        status=status,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        message=message,
                        metadata={
                            'healthy_workers': healthy_workers,
                            'total_workers': total_workers
                        }
                    )
        except Exception as e:
            self._component_health['workers'] = ComponentHealth(
                name='workers',
                status=HealthStatus.CRITICAL,
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
                message=f"Workers check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    async def _check_gpu(self) -> None:
        """التحقق من GPU - Check GPU"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health/gpu",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        gpus = data.get('gpus', [])
                        
                        if gpus:
                            avg_temp = sum(g.get('temperature', 0) for g in gpus) / len(gpus)
                            if avg_temp > 85:
                                status = HealthStatus.WARNING
                                message = f"GPU temperature high: {avg_temp:.1f}°C"
                            else:
                                status = HealthStatus.HEALTHY
                                message = f"{len(gpus)} GPU(s) are healthy"
                        else:
                            status = HealthStatus.WARNING
                            message = "No GPUs detected"
                    else:
                        status = HealthStatus.WARNING
                        message = "GPU check not available"
                    
                    self._component_health['gpu'] = ComponentHealth(
                        name='gpu',
                        status=status,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        message=message,
                        metadata={'gpu_count': len(gpus) if 'gpus' in locals() else 0}
                    )
        except Exception as e:
            self._component_health['gpu'] = ComponentHealth(
                name='gpu',
                status=HealthStatus.WARNING,
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
                message=f"GPU check failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _latency_to_status(self, latency_ms: float) -> HealthStatus:
        """تحويل زمن الاستجابة إلى حالة - Convert latency to status"""
        if latency_ms < self.HEALTHY_THRESHOLD_MS:
            return HealthStatus.HEALTHY
        elif latency_ms < self.WARNING_THRESHOLD_MS:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_overall_status(self) -> HealthStatus:
        """حساب الحالة العامة - Calculate overall status"""
        statuses = [c.status for c in self._component_health.values()]
        
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        elif any(s == HealthStatus.WARNING for s in statuses):
            return HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def _send_websocket_update(self, health: SystemHealth) -> None:
        """إرسال تحديث عبر WebSocket - Send update via WebSocket"""
        if not self.websocket_manager:
            return
        
        try:
            data = {
                'type': 'health_update',
                'timestamp': health.timestamp,
                'overall_status': health.overall_status.value,
                'components': {
                    name: {
                        'name': c.name,
                        'status': c.status.value,
                        'latency_ms': c.latency_ms,
                        'message': c.message
                    }
                    for name, c in health.components.items()
                }
            }
            await self.websocket_manager.broadcast(data)
        except Exception as e:
            logger.error(f"Error sending WebSocket update: {e}")
    
    def get_system_health(self) -> SystemHealth:
        """
        الحصول على صحة النظام الحالية
        Get current system health
        
        Returns:
            SystemHealth: صحة النظام
        """
        return SystemHealth(
            timestamp=time.time(),
            overall_status=self._calculate_overall_status(),
            components=self._component_health.copy(),
            alerts=self._alerts.copy()
        )
    
    def get_component_status(self, component_name: str) -> Optional[ComponentHealth]:
        """
        الحصول على حالة مكون محدد
        Get status of a specific component
        
        Args:
            component_name: اسم المكون
            
        Returns:
            Optional[ComponentHealth]: حالة المكون أو None
        """
        return self._component_health.get(component_name)
    
    def get_health_history(self, limit: int = 100) -> List[SystemHealth]:
        """
        الحصول على تاريخ الصحة
        Get health history
        
        Args:
            limit: الحد الأقصى للنتائج
            
        Returns:
            List[SystemHealth]: قائمة حالات الصحة
        """
        return self._history[-limit:]
    
    def generate_dashboard_html(self) -> str:
        """
        إنشاء HTML للوحة المعلومات
        Generate dashboard HTML
        
        Returns:
            str: HTML للوحة المعلومات
        """
        health = self.get_system_health()
        
        # تحديد لون الحالة - Determine status color
        status_colors = {
            HealthStatus.HEALTHY: '#28a745',
            HealthStatus.WARNING: '#ffc107',
            HealthStatus.CRITICAL: '#dc3545',
            HealthStatus.UNKNOWN: '#6c757d'
        }
        
        status_color = status_colors.get(health.overall_status, '#6c757d')
        status_arabic = {
            HealthStatus.HEALTHY: 'صحي',
            HealthStatus.WARNING: 'تحذير',
            HealthStatus.CRITICAL: 'حرج',
            HealthStatus.UNKNOWN: 'غير معروف'
        }.get(health.overall_status, health.overall_status.value)
        
        html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BI-IDE Health Dashboard - لوحة صحة النظام</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }}
        .overall-status {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .status-indicator {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: white;
            background-color: {status_color};
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        .status-text {{
            font-size: 1.5rem;
            color: {status_color};
            font-weight: bold;
        }}
        .components-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .component-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        .component-card:hover {{
            transform: translateY(-5px);
        }}
        .component-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .component-name {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
        }}
        .component-status {{
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-size: 0.9rem;
            font-weight: bold;
        }}
        .status-healthy {{ background-color: #28a745; }}
        .status-warning {{ background-color: #ffc107; color: #333; }}
        .status-critical {{ background-color: #dc3545; }}
        .status-unknown {{ background-color: #6c757d; }}
        .component-details {{
            color: #666;
            font-size: 0.95rem;
        }}
        .component-details p {{
            margin: 8px 0;
        }}
        .latency {{
            font-family: monospace;
            background: #f8f9fa;
            padding: 3px 8px;
            border-radius: 4px;
        }}
        .last-update {{
            text-align: center;
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
        }}
        .alerts-section {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .alerts-section h2 {{
            margin-bottom: 15px;
            color: #333;
        }}
        .alert-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-right: 4px solid;
        }}
        .alert-critical {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
        }}
        .no-alerts {{
            text-align: center;
            color: #28a745;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 BI-IDE Health Dashboard<br><small>لوحة صحة النظام</small></h1>
        
        <div class="overall-status">
            <div class="status-indicator">
                {'✓' if health.overall_status == HealthStatus.HEALTHY else '⚠' if health.overall_status == HealthStatus.WARNING else '✗'}
            </div>
            <div class="status-text">{status_arabic}</div>
            <p style="margin-top: 10px; color: #666;">
                Overall System Status - الحالة العامة للنظام
            </p>
        </div>
        
        <div class="components-grid">
"""
        
        # إضافة بطاقات المكونات - Add component cards
        for name, component in health.components.items():
            status_class = f"status-{component.status.value}"
            status_text = {
                HealthStatus.HEALTHY: 'صحي',
                HealthStatus.WARNING: 'تحذير',
                HealthStatus.CRITICAL: 'حرج',
                HealthStatus.UNKNOWN: 'غير معروف'
            }.get(component.status, component.status.value)
            
            html += f"""
            <div class="component-card">
                <div class="component-header">
                    <span class="component-name">{name.upper()}</span>
                    <span class="component-status {status_class}">{status_text}</span>
                </div>
                <div class="component-details">
                    <p><strong>الرسالة:</strong> {component.message}</p>
                    <p><strong>زمن الاستجابة:</strong> <span class="latency">{component.latency_ms:.2f} ms</span></p>
                    <p><strong>آخر فحص:</strong> {datetime.fromtimestamp(component.last_check).strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
"""
        
        html += """
        </div>
        
        <div class="alerts-section">
            <h2>🚨 Active Alerts - التنبيهات النشطة</h2>
"""
        
        if health.alerts:
            for alert in health.alerts:
                level = alert.get('level', 'info')
                alert_class = f"alert-{level}"
                html += f"""
            <div class="alert-item {alert_class}">
                <strong>{alert.get('title', 'Alert')}</strong>
                <p>{alert.get('message', '')}</p>
            </div>
"""
        else:
            html += '<div class="no-alerts">✓ لا توجد تنبيهات نشطة - No active alerts</div>'
        
        html += f"""
        </div>
        
        <p class="last-update">
            آخر تحديث: {datetime.fromtimestamp(health.timestamp).strftime('%Y-%m-%d %H:%M:%S')} |
            Auto-refresh every {self.check_interval} seconds
        </p>
    </div>
    
    <script>
        // تحديث تلقائي كل {self.check_interval} ثانية
        setTimeout(() => {{
            location.reload();
        }}, {int(self.check_interval * 1000)});
    </script>
</body>
</html>
"""
        
        return html
    
    def get_health_json(self) -> Dict[str, Any]:
        """
        الحصول على صحة النظام بصيغة JSON
        Get system health as JSON
        
        Returns:
            Dict[str, Any]: صحة النظام بصيغة قاموس
        """
        health = self.get_system_health()
        return {
            'timestamp': health.timestamp,
            'overall_status': health.overall_status.value,
            'components': {
                name: {
                    'name': c.name,
                    'status': c.status.value,
                    'latency_ms': c.latency_ms,
                    'last_check': c.last_check,
                    'message': c.message,
                    'metadata': c.metadata
                }
                for name, c in health.components.items()
            },
            'alerts': health.alerts
        }
