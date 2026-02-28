"""
خدمة الإشعارات
Notification Service for multi-channel notifications
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """قنوات الإشعارات"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    IN_APP = "in_app"
    PUSH = "push"


class NotificationPriority(Enum):
    """أولويات الإشعارات"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """نموذج الإشعار"""
    notification_id: str
    user_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    is_read: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس"""
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "channel": self.channel.value,
            "priority": self.priority.value,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class WebSocketConnection:
    """نموذج اتصال WebSocket"""
    user_id: str
    socket_id: str
    connected_at: datetime
    send_callback: Optional[Callable[[str], None]] = None
    is_active: bool = True


class NotificationService:
    """
    خدمة الإشعارات
    
    تدير إرسال الإشعارات عبر قنوات متعددة: WebSocket، البريد، والتطبيق
    """
    
    def __init__(self):
        """تهيئة خدمة الإشعارات"""
        self._notifications: Dict[str, Notification] = {}
        self._user_notifications: Dict[str, List[str]] = {}
        self._websocket_connections: Dict[str, WebSocketConnection] = {}
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}
        
        # إعداد معالجات القنوات
        self._setup_channel_handlers()
        
        logger.info("تم تهيئة خدمة الإشعارات")
    
    def _setup_channel_handlers(self) -> None:
        """إعداد معالجات القنوات"""
        self._channel_handlers = {
            NotificationChannel.WEBSOCKET: self._send_websocket,
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.IN_APP: self._send_in_app,
            NotificationChannel.PUSH: self._send_push
        }
    
    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Notification]:
        """
        إرسال إشعار
        
        المعاملات:
            user_id: معرف المستخدم
            title: عنوان الإشعار
            message: نص الإشعار
            channel: قناة الإرسال
            priority: الأولوية
            metadata: بيانات إضافية
            
        العائد:
            Notification: الإشعار المنشأ أو None
        """
        try:
            notification_id = f"notif_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            notification = Notification(
                notification_id=notification_id,
                user_id=user_id,
                title=title,
                message=message,
                channel=channel,
                priority=priority,
                metadata=metadata or {}
            )
            
            # تخزين الإشعار
            self._notifications[notification_id] = notification
            
            if user_id not in self._user_notifications:
                self._user_notifications[user_id] = []
            self._user_notifications[user_id].append(notification_id)
            
            # إرسال عبر القناة المحددة
            handler = self._channel_handlers.get(channel)
            if handler:
                await handler(notification)
            
            logger.info(f"تم إرسال إشعار: {notification_id} للمستخدم: {user_id}")
            return notification
            
        except Exception as e:
            logger.error(f"خطأ في إرسال الإشعار: {e}")
            return None
    
    async def send_bulk_notification(
        self,
        user_ids: List[str],
        title: str,
        message: str,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        إرسال إشعار لمجموعة مستخدمين
        
        المعاملات:
            user_ids: قائمة معرفات المستخدمين
            title: العنوان
            message: النص
            channel: القناة
            priority: الأولوية
            metadata: البيانات الإضافية
            
        العائد:
            List[str]: قائمة معرفات الإشعارات المرسلة
        """
        notification_ids = []
        
        for user_id in user_ids:
            notification = await self.send_notification(
                user_id=user_id,
                title=title,
                message=message,
                channel=channel,
                priority=priority,
                metadata=metadata
            )
            if notification:
                notification_ids.append(notification.notification_id)
        
        return notification_ids
    
    async def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
        channel: Optional[NotificationChannel] = None
    ) -> List[Notification]:
        """
        الحصول على إشعارات المستخدم
        
        المعاملات:
            user_id: معرف المستخدم
            unread_only: فقط غير المقروءة
            limit: الحد الأقصى
            channel: تصفية حسب القناة
            
        العائد:
            List[Notification]: قائمة الإشعارات
        """
        try:
            notification_ids = self._user_notifications.get(user_id, [])
            notifications = []
            
            for nid in reversed(notification_ids):  # الأحدث أولاً
                if len(notifications) >= limit:
                    break
                
                notif = self._notifications.get(nid)
                if not notif:
                    continue
                
                if unread_only and notif.is_read:
                    continue
                
                if channel and notif.channel != channel:
                    continue
                
                notifications.append(notif)
            
            return notifications
            
        except Exception as e:
            logger.error(f"خطأ في جلب الإشعارات: {e}")
            return []
    
    async def mark_read(
        self,
        user_id: str,
        notification_id: Optional[str] = None,
        mark_all: bool = False
    ) -> int:
        """
        تحديد إشعار كمقروء
        
        المعاملات:
            user_id: معرف المستخدم
            notification_id: معرف الإشعار (اختياري إذا mark_all=True)
            mark_all: تحديد الكل كمقروء
            
        العائد:
            int: عدد الإشعارات المحدثة
        """
        try:
            count = 0
            
            if mark_all:
                notification_ids = self._user_notifications.get(user_id, [])
                for nid in notification_ids:
                    notif = self._notifications.get(nid)
                    if notif and not notif.is_read:
                        notif.is_read = True
                        count += 1
            elif notification_id:
                notif = self._notifications.get(notification_id)
                if notif and notif.user_id == user_id and not notif.is_read:
                    notif.is_read = True
                    count = 1
            
            logger.info(f"تم تحديد {count} إشعار كمقروء للمستخدم: {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"خطأ في تحديد الإشعار كمقروء: {e}")
            return 0
    
    async def register_websocket(
        self,
        user_id: str,
        socket_id: str,
        send_callback: Callable[[str], None]
    ) -> bool:
        """
        تسجيل اتصال WebSocket
        
        المعاملات:
            user_id: معرف المستخدم
            socket_id: معرف الاتصال
            send_callback: دالة الإرسال
            
        العائد:
            bool: True إذا نجح التسجيل
        """
        try:
            connection = WebSocketConnection(
                user_id=user_id,
                socket_id=socket_id,
                connected_at=datetime.now(),
                send_callback=send_callback,
                is_active=True
            )
            
            self._websocket_connections[socket_id] = connection
            logger.info(f"تم تسجيل WebSocket: {socket_id} للمستخدم: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تسجيل WebSocket: {e}")
            return False
    
    async def unregister_websocket(self, socket_id: str) -> bool:
        """
        إلغاء تسجيل اتصال WebSocket
        
        المعاملات:
            socket_id: معرف الاتصال
            
        العائد:
            bool: True إذا نجح الإلغاء
        """
        try:
            if socket_id in self._websocket_connections:
                self._websocket_connections[socket_id].is_active = False
                del self._websocket_connections[socket_id]
                logger.info(f"تم إلغاء تسجيل WebSocket: {socket_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء تسجيل WebSocket: {e}")
            return False
    
    async def get_unread_count(self, user_id: str) -> int:
        """
        الحصول على عدد الإشعارات غير المقروءة
        
        المعاملات:
            user_id: معرف المستخدم
            
        العائد:
            int: العدد
        """
        try:
            notification_ids = self._user_notifications.get(user_id, [])
            count = 0
            
            for nid in notification_ids:
                notif = self._notifications.get(nid)
                if notif and not notif.is_read:
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"خطأ في حساب الإشعارات غير المقروءة: {e}")
            return 0
    
    async def delete_notification(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """
        حذف إشعار
        
        المعاملات:
            user_id: معرف المستخدم
            notification_id: معرف الإشعار
            
        العائد:
            bool: True إذا نجح الحذف
        """
        try:
            notif = self._notifications.get(notification_id)
            if not notif or notif.user_id != user_id:
                return False
            
            del self._notifications[notification_id]
            
            if user_id in self._user_notifications:
                self._user_notifications[user_id].remove(notification_id)
            
            logger.info(f"تم حذف الإشعار: {notification_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حذف الإشعار: {e}")
            return False
    
    # معالجات القنوات
    async def _send_websocket(self, notification: Notification) -> bool:
        """إرسال عبر WebSocket"""
        try:
            user_connections = [
                conn for conn in self._websocket_connections.values()
                if conn.user_id == notification.user_id and conn.is_active
            ]
            
            if not user_connections:
                logger.debug(f"لا يوجد اتصال WebSocket للمستخدم: {notification.user_id}")
                return False
            
            message = json.dumps(notification.to_dict())
            
            for conn in user_connections:
                if conn.send_callback:
                    try:
                        conn.send_callback(message)
                    except Exception as e:
                        logger.error(f"خطأ في إرسال WebSocket: {e}")
                        conn.is_active = False
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال WebSocket: {e}")
            return False
    
    async def _send_email(self, notification: Notification) -> bool:
        """إرسال عبر البريد الإلكتروني"""
        try:
            # محاكاة إرسال بريد
            logger.info(f"[EMAIL] إلى {notification.user_id}: {notification.title}")
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال البريد: {e}")
            return False
    
    async def _send_in_app(self, notification: Notification) -> bool:
        """إرسال داخل التطبيق"""
        try:
            # الإشعارات داخل التطبيق مخزنة فقط
            logger.info(f"[IN_APP] للمستخدم {notification.user_id}: {notification.title}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال إشعار داخلي: {e}")
            return False
    
    async def _send_push(self, notification: Notification) -> bool:
        """إرسال Push Notification"""
        try:
            # محاكاة إرسال Push
            logger.info(f"[PUSH] للمستخدم {notification.user_id}: {notification.title}")
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال Push: {e}")
            return False
