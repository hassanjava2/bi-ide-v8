"""
مدير التنبيهات - Alert Manager
==============================
إدارة التنبيهات والإشعارات للنظام
Alert and notification management for the system
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """مستويات التنبيه - Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """قنوات الإشعار - Notification channels"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SLACK = "slack"


@dataclass
class Alert:
    """تنبيه - Alert"""
    id: str
    title: str
    message: str
    level: AlertLevel
    source: str
    worker_id: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    is_resolved: bool
    channels: List[NotificationChannel]
    metadata: Optional[Dict[str, Any]]


@dataclass
class AlertThreshold:
    """عتبة التنبيه - Alert threshold"""
    name: str
    metric: str
    operator: str  # gt, lt, eq, gte, lte
    value: float
    level: AlertLevel
    cooldown_seconds: int = 300  # 5 دقائق افتراضياً
    channels: List[NotificationChannel] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [NotificationChannel.WEBSOCKET]


class AlertManager:
    """
    مدير التنبيهات
    Alert manager
    
    يدير التنبيهات والإشعارات عبر قنوات متعددة
    Manages alerts and notifications across multiple channels
    
    العتبات الافتراضية:
    - درجة حرارة GPU > 85°C
    - العامل متوقف > 30 ثانية
    - استخدام القرص > 90%
    """
    
    DEFAULT_THRESHOLDS = [
        AlertThreshold(
            name="gpu_temperature_high",
            metric="gpu_temperature",
            operator="gt",
            value=85.0,
            level=AlertLevel.WARNING,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.SLACK]
        ),
        AlertThreshold(
            name="worker_down",
            metric="worker_last_seen_seconds",
            operator="gt",
            value=30.0,
            level=AlertLevel.ERROR,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL]
        ),
        AlertThreshold(
            name="disk_usage_high",
            metric="disk_percent",
            operator="gt",
            value=90.0,
            level=AlertLevel.CRITICAL,
            channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL, NotificationChannel.SLACK]
        ),
        AlertThreshold(
            name="ram_usage_high",
            metric="ram_percent",
            operator="gt",
            value=95.0,
            level=AlertLevel.WARNING,
            channels=[NotificationChannel.WEBSOCKET]
        ),
        AlertThreshold(
            name="cpu_usage_high",
            metric="cpu_percent",
            operator="gt",
            value=95.0,
            level=AlertLevel.WARNING,
            channels=[NotificationChannel.WEBSOCKET]
        )
    ]
    
    def __init__(
        self,
        websocket_manager: Optional[Any] = None,
        email_config: Optional[Dict[str, Any]] = None,
        slack_webhook_url: Optional[str] = None,
        thresholds: Optional[List[AlertThreshold]] = None
    ):
        """
        تهيئة مدير التنبيهات
        Initialize alert manager
        
        Args:
            websocket_manager: مدير WebSocket للإشعارات الفورية
            email_config: إعدادات البريد الإلكتروني
            slack_webhook_url: رابط Webhook الخاص بـ Slack
            thresholds: قائمة عتبات التنبيه المخصصة
        """
        self.websocket_manager = websocket_manager
        self.email_config = email_config or {}
        self.slack_webhook_url = slack_webhook_url
        
        self._alerts: Dict[str, Alert] = {}
        self._active_alerts: Set[str] = set()
        self._thresholds: Dict[str, AlertThreshold] = {}
        self._last_alert_time: Dict[str, datetime] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = asyncio.Lock()
        
        # إضافة العتبات الافتراضية
        for threshold in (thresholds or self.DEFAULT_THRESHOLDS):
            self._thresholds[threshold.name] = threshold
    
    async def create_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel,
        source: str,
        worker_id: Optional[str] = None,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        إنشاء تنبيه جديد
        Create a new alert
        
        Args:
            title: عنوان التنبيه
            message: نص التنبيه
            level: مستوى التنبيه
            source: مصدر التنبيه
            worker_id: معرف العامل المرتبط
            channels: قنوات الإشعار
            metadata: بيانات إضافية
            
        Returns:
            Alert: التنبيه المنشأ
        """
        alert_id = str(uuid4())
        
        if channels is None:
            channels = [NotificationChannel.WEBSOCKET]
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            level=level,
            source=source,
            worker_id=worker_id,
            created_at=datetime.utcnow(),
            resolved_at=None,
            is_resolved=False,
            channels=channels,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self._alerts[alert_id] = alert
            self._active_alerts.add(alert_id)
        
        logger.warning(f"Alert created: [{level.value}] {title}")
        
        # إرسال الإشعارات
        await self.notify(alert)
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        return alert
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> Optional[Alert]:
        """
        حل تنبيه
        Resolve an alert
        
        Args:
            alert_id: معرف التنبيه
            resolution_message: رسالة الحل
            
        Returns:
            Optional[Alert]: التنبيه المحلول أو None
        """
        async with self._lock:
            if alert_id not in self._alerts:
                return None
            
            alert = self._alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = datetime.utcnow()
            
            if alert_id in self._active_alerts:
                self._active_alerts.remove(alert_id)
        
        logger.info(f"Alert resolved: {alert_id}")
        
        # إشعار بحل التنبيه
        if self.websocket_manager:
            await self._send_websocket_notification({
                'type': 'alert_resolved',
                'alert_id': alert_id,
                'message': resolution_message or f"Alert '{alert.title}' has been resolved",
                'resolved_at': alert.resolved_at.isoformat()
            })
        
        return alert
    
    async def resolve_alerts_by_source(
        self,
        source: str,
        worker_id: Optional[str] = None
    ) -> int:
        """
        حل جميع التنبيهات من مصدر معين
        Resolve all alerts from a specific source
        
        Args:
            source: مصدر التنبيهات
            worker_id: معرف العامل (اختياري)
            
        Returns:
            int: عدد التنبيهات المحلولة
        """
        resolved_count = 0
        
        async with self._lock:
            for alert_id in list(self._active_alerts):
                alert = self._alerts[alert_id]
                if alert.source == source:
                    if worker_id is None or alert.worker_id == worker_id:
                        alert.is_resolved = True
                        alert.resolved_at = datetime.utcnow()
                        self._active_alerts.remove(alert_id)
                        resolved_count += 1
        
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} alerts from source: {source}")
        
        return resolved_count
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        worker_id: Optional[str] = None
    ) -> List[Alert]:
        """
        الحصول على التنبيهات النشطة
        Get active alerts
        
        Args:
            level: تصفية حسب المستوى
            worker_id: تصفية حسب العامل
            
        Returns:
            List[Alert]: قائمة التنبيهات النشطة
        """
        alerts = []
        
        for alert_id in self._active_alerts:
            alert = self._alerts[alert_id]
            
            if level and alert.level != level:
                continue
            
            if worker_id and alert.worker_id != worker_id:
                continue
            
            alerts.append(alert)
        
        # ترتيب حسب الأولوية والوقت
        level_priority = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.ERROR: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3
        }
        alerts.sort(key=lambda a: (level_priority[a.level], a.created_at), reverse=False)
        
        return alerts
    
    def get_alert_history(
        self,
        limit: int = 100,
        include_resolved: bool = True
    ) -> List[Alert]:
        """
        الحصول على تاريخ التنبيهات
        Get alert history
        
        Args:
            limit: الحد الأقصى للنتائج
            include_resolved: تضمين التنبيهات المحلولة
            
        Returns:
            List[Alert]: قائمة التنبيهات
        """
        alerts = list(self._alerts.values())
        
        if not include_resolved:
            alerts = [a for a in alerts if not a.is_resolved]
        
        # ترتيب حسب وقت الإنشاء
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        
        return alerts[:limit]
    
    async def notify(self, alert: Alert) -> None:
        """
        إرسال إشعار عبر القنوات المحددة
        Send notification via specified channels
        
        Args:
            alert: التنبيه للإشعار
        """
        tasks = []
        
        for channel in alert.channels:
            if channel == NotificationChannel.WEBSOCKET:
                tasks.append(self._send_websocket_alert(alert))
            elif channel == NotificationChannel.EMAIL:
                tasks.append(self._send_email_alert(alert))
            elif channel == NotificationChannel.SLACK:
                tasks.append(self._send_slack_alert(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_websocket_alert(self, alert: Alert) -> None:
        """إرسال تنبيه عبر WebSocket - Send alert via WebSocket"""
        if not self.websocket_manager:
            return
        
        await self._send_websocket_notification({
            'type': 'alert',
            'alert': {
                'id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'level': alert.level.value,
                'source': alert.source,
                'worker_id': alert.worker_id,
                'created_at': alert.created_at.isoformat()
            }
        })
    
    async def _send_websocket_notification(self, data: Dict[str, Any]) -> None:
        """إرسال إشعار WebSocket عام - Send generic WebSocket notification"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast(data)
            except Exception as e:
                logger.error(f"Error sending WebSocket notification: {e}")
    
    async def _send_email_alert(self, alert: Alert) -> None:
        """إرسال تنبيه عبر البريد الإلكتروني - Send alert via email"""
        if not self.email_config:
            return
        
        try:
            msg = MIMEText(f"""
Alert: {alert.title}
Level: {alert.level.value.upper()}
Source: {alert.source}
Worker: {alert.worker_id or 'N/A'}
Time: {alert.created_at.isoformat()}

{alert.message}

---
BI-IDE Monitoring System
            """)
            
            msg['Subject'] = f"[{alert.level.value.upper()}] BI-IDE Alert: {alert.title}"
            msg['From'] = self.email_config.get('from', 'alerts@bi-ide.local')
            msg['To'] = ', '.join(self.email_config.get('to', []))
            
            # إرسال البريد في Thread منفصل
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email_sync, msg)
            
            logger.info(f"Email alert sent for: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_email_sync(self, msg: MIMEText) -> None:
        """إرسال البريد الإلكتروني بشكل متزامن - Send email synchronously"""
        smtp_host = self.email_config.get('smtp_host', 'localhost')
        smtp_port = self.email_config.get('smtp_port', 587)
        smtp_user = self.email_config.get('smtp_user')
        smtp_password = self.email_config.get('smtp_password')
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
    
    async def _send_slack_alert(self, alert: Alert) -> None:
        """إرسال تنبيه عبر Slack - Send alert via Slack"""
        if not self.slack_webhook_url:
            return
        
        # تحديد اللون حسب المستوى
        color_map = {
            AlertLevel.INFO: '#36a64f',
            AlertLevel.WARNING: '#ff9900',
            AlertLevel.ERROR: '#ff0000',
            AlertLevel.CRITICAL: '#990000'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(alert.level, '#808080'),
                'title': f"[{alert.level.value.upper()}] {alert.title}",
                'text': alert.message,
                'fields': [
                    {
                        'title': 'Source',
                        'value': alert.source,
                        'short': True
                    },
                    {
                        'title': 'Worker',
                        'value': alert.worker_id or 'N/A',
                        'short': True
                    },
                    {
                        'title': 'Time',
                        'value': alert.created_at.isoformat(),
                        'short': True
                    }
                ],
                'footer': 'BI-IDE Monitoring',
                'ts': alert.created_at.timestamp()
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for: {alert.id}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    async def check_thresholds(
        self,
        metric_name: str,
        metric_value: float,
        worker_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        التحقق من العتبات وإنشاء التنبيهات
        Check thresholds and create alerts
        
        Args:
            metric_name: اسم المقياس
            metric_value: قيمة المقياس
            worker_id: معرف العامل
            context: سياق إضافي
            
        Returns:
            List[Alert]: قائمة التنبيهات المنشأة
        """
        alerts = []
        
        for threshold in self._thresholds.values():
            if threshold.metric != metric_name:
                continue
            
            # التحقق من فترة التهدئة
            cooldown_key = f"{threshold.name}:{worker_id or 'global'}"
            last_time = self._last_alert_time.get(cooldown_key)
            
            if last_time:
                elapsed = (datetime.utcnow() - last_time).total_seconds()
                if elapsed < threshold.cooldown_seconds:
                    continue
            
            # التحقق من الشرط
            triggered = False
            
            if threshold.operator == 'gt' and metric_value > threshold.value:
                triggered = True
            elif threshold.operator == 'lt' and metric_value < threshold.value:
                triggered = True
            elif threshold.operator == 'gte' and metric_value >= threshold.value:
                triggered = True
            elif threshold.operator == 'lte' and metric_value <= threshold.value:
                triggered = True
            elif threshold.operator == 'eq' and metric_value == threshold.value:
                triggered = True
            
            if triggered:
                alert = await self.create_alert(
                    title=f"{threshold.name}: {metric_name} = {metric_value:.2f}",
                    message=f"Threshold '{threshold.name}' triggered: {metric_name} ({metric_value:.2f}) {threshold.operator} {threshold.value}",
                    level=threshold.level,
                    source='threshold_check',
                    worker_id=worker_id,
                    channels=threshold.channels,
                    metadata={
                        'threshold': threshold.name,
                        'metric': metric_name,
                        'value': metric_value,
                        'operator': threshold.operator,
                        'threshold_value': threshold.value,
                        **(context or {})
                    }
                )
                alerts.append(alert)
                self._last_alert_time[cooldown_key] = datetime.utcnow()
        
        return alerts
    
    def add_threshold(self, threshold: AlertThreshold) -> None:
        """إضافة عتبة جديدة - Add new threshold"""
        self._thresholds[threshold.name] = threshold
        logger.info(f"Alert threshold added: {threshold.name}")
    
    def remove_threshold(self, name: str) -> bool:
        """إزالة عتبة - Remove threshold"""
        if name in self._thresholds:
            del self._thresholds[name]
            logger.info(f"Alert threshold removed: {name}")
            return True
        return False
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """إضافة دالة استدعاء للتنبيهات - Add alert callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
