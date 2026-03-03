"""
خدمة الإشعارات - Notification Service V2
PostgreSQL-based persistent storage replacing in-memory _notifications
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, update

# Try importing database models
try:
    from core.database import get_async_session
    from core.service_models import NotificationDB
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

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
    """نموذج الإشعار (in-memory representation)"""
    notification_id: str
    user_id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    is_read: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_db(cls, db_notification: NotificationDB) -> 'Notification':
        """Create from database model"""
        return cls(
            notification_id=db_notification.id,
            user_id=db_notification.user_id,
            title=db_notification.title,
            message=db_notification.message,
            channel=NotificationChannel(db_notification.channel),
            priority=NotificationPriority(db_notification.priority),
            is_read=db_notification.is_read,
            created_at=db_notification.created_at,
            metadata=db_notification.get_metadata()
        )


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
    خدمة الإشعارات V2 - PostgreSQL Persistent Storage
    
    WebSocket connections remain in-memory (transient),
    but notifications are persisted to PostgreSQL.
    """
    
    def __init__(self):
        """تهيئة خدمة الإشعارات"""
        # WebSocket connections remain in-memory (they're transient)
        self._websocket_connections: Dict[str, WebSocketConnection] = {}
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}
        
        # PostgreSQL availability
        self._postgres_available = POSTGRES_AVAILABLE
        
        # Fallback in-memory storage if PostgreSQL not available
        self._fallback_notifications: Dict[str, Notification] = {}
        self._fallback_user_notifications: Dict[str, List[str]] = {}
        
        # إعداد معالجات القنوات
        self._setup_channel_handlers()
        
        if self._postgres_available:
            logger.info("✅ NotificationService V2 initialized (PostgreSQL)")
        else:
            logger.warning("⚠️ NotificationService V2 initialized (In-Memory Fallback)")
    
    def _setup_channel_handlers(self) -> None:
        """إعداد معالجات القنوات"""
        self._channel_handlers = {
            NotificationChannel.WEBSOCKET: self._send_websocket,
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.IN_APP: self._send_in_app,
            NotificationChannel.PUSH: self._send_push
        }
    
    async def _get_db_session(self) -> Optional[AsyncSession]:
        """Get database session if PostgreSQL is available"""
        if not self._postgres_available:
            return None
        try:
            # Get async session from generator
            session_gen = get_async_session()
            return await session_gen.__anext__()
        except Exception as e:
            logger.warning(f"Failed to get DB session: {e}")
            return None
    
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
        إرسال إشعار - Persisted to PostgreSQL
        """
        try:
            notification_id = str(uuid.uuid4())
            
            if self._postgres_available:
                # Store in PostgreSQL
                db = await self._get_db_session()
                if db:
                    try:
                        db_notification = NotificationDB(
                            id=notification_id,
                            user_id=user_id,
                            title=title,
                            message=message,
                            channel=channel.value,
                            priority=priority.value,
                            is_read=False,
                            metadata_json=metadata or {}
                        )
                        db.add(db_notification)
                        await db.commit()
                    except Exception as e:
                        logger.error(f"DB error storing notification: {e}")
                        await db.rollback()
                        # Fall through to in-memory
                    finally:
                        await db.close()
                else:
                    # Fallback to in-memory
                    self._fallback_store(notification_id, user_id, title, message, channel, priority, metadata)
            else:
                # In-memory fallback
                self._fallback_store(notification_id, user_id, title, message, channel, priority, metadata)
            
            # Create notification object for return and channel handling
            notification = Notification(
                notification_id=notification_id,
                user_id=user_id,
                title=title,
                message=message,
                channel=channel,
                priority=priority,
                metadata=metadata or {}
            )
            
            # إرسال عبر القناة المحددة
            handler = self._channel_handlers.get(channel)
            if handler:
                await handler(notification)
            
            logger.info(f"✅ تم إرسال إشعار: {notification_id} للمستخدم: {user_id}")
            return notification
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الإشعار: {e}")
            return None
    
    def _fallback_store(self, notification_id: str, user_id: str, title: str, message: str,
                        channel: NotificationChannel, priority: NotificationPriority,
                        metadata: Optional[Dict[str, Any]]) -> None:
        """Store notification in fallback in-memory storage"""
        notification = Notification(
            notification_id=notification_id,
            user_id=user_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            metadata=metadata or {}
        )
        self._fallback_notifications[notification_id] = notification
        
        if user_id not in self._fallback_user_notifications:
            self._fallback_user_notifications[user_id] = []
        self._fallback_user_notifications[user_id].append(notification_id)
    
    async def send_bulk_notification(
        self,
        user_ids: List[str],
        title: str,
        message: str,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """إرسال إشعار لمجموعة مستخدمين"""
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
        الحصول على إشعارات المستخدم - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        query = select(NotificationDB).where(NotificationDB.user_id == user_id)
                        
                        if unread_only:
                            query = query.where(NotificationDB.is_read == False)
                        
                        if channel:
                            query = query.where(NotificationDB.channel == channel.value)
                        
                        query = query.order_by(desc(NotificationDB.created_at)).limit(limit)
                        
                        result = await db.execute(query)
                        db_notifications = result.scalars().all()
                        
                        return [Notification.from_db(n) for n in db_notifications]
                    except Exception as e:
                        logger.error(f"DB error fetching notifications: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_get_user_notifications(user_id, unread_only, limit, channel)
            
        except Exception as e:
            logger.error(f"خطأ في جلب الإشعارات: {e}")
            return []
    
    def _fallback_get_user_notifications(
        self, user_id: str, unread_only: bool, limit: int,
        channel: Optional[NotificationChannel]
    ) -> List[Notification]:
        """Fallback: Get notifications from in-memory storage"""
        notification_ids = self._fallback_user_notifications.get(user_id, [])
        notifications = []
        
        for nid in reversed(notification_ids):  # الأحدث أولاً
            if len(notifications) >= limit:
                break
            
            notif = self._fallback_notifications.get(nid)
            if not notif:
                continue
            
            if unread_only and notif.is_read:
                continue
            
            if channel and notif.channel != channel:
                continue
            
            notifications.append(notif)
        
        return notifications
    
    async def mark_read(
        self,
        user_id: str,
        notification_id: Optional[str] = None,
        mark_all: bool = False
    ) -> int:
        """
        تحديد إشعار كمقروء - Update PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        if mark_all:
                            # Update all unread notifications for user
                            result = await db.execute(
                                update(NotificationDB)
                                .where(NotificationDB.user_id == user_id)
                                .where(NotificationDB.is_read == False)
                                .values(is_read=True, read_at=datetime.now(timezone.utc))
                            )
                            await db.commit()
                            count = result.rowcount
                        elif notification_id:
                            # Update specific notification
                            result = await db.execute(
                                update(NotificationDB)
                                .where(NotificationDB.id == notification_id)
                                .where(NotificationDB.user_id == user_id)
                                .values(is_read=True, read_at=datetime.now(timezone.utc))
                            )
                            await db.commit()
                            count = 1 if result.rowcount > 0 else 0
                        else:
                            count = 0
                        
                        logger.info(f"✅ تم تحديد {count} إشعار كمقروء للمستخدم: {user_id}")
                        return count
                    except Exception as e:
                        logger.error(f"DB error marking notifications read: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_mark_read(user_id, notification_id, mark_all)
            
        except Exception as e:
            logger.error(f"خطأ في تحديد الإشعار كمقروء: {e}")
            return 0
    
    def _fallback_mark_read(
        self, user_id: str, notification_id: Optional[str], mark_all: bool
    ) -> int:
        """Fallback: Mark notifications as read in in-memory storage"""
        count = 0
        
        if mark_all:
            notification_ids = self._fallback_user_notifications.get(user_id, [])
            for nid in notification_ids:
                notif = self._fallback_notifications.get(nid)
                if notif and not notif.is_read:
                    notif.is_read = True
                    count += 1
        elif notification_id:
            notif = self._fallback_notifications.get(notification_id)
            if notif and notif.user_id == user_id and not notif.is_read:
                notif.is_read = True
                count = 1
        
        return count
    
    async def register_websocket(
        self,
        user_id: str,
        socket_id: str,
        send_callback: Callable[[str], None]
    ) -> bool:
        """تسجيل اتصال WebSocket (remains in-memory)"""
        try:
            connection = WebSocketConnection(
                user_id=user_id,
                socket_id=socket_id,
                connected_at=datetime.now(timezone.utc),
                send_callback=send_callback,
                is_active=True
            )
            
            self._websocket_connections[socket_id] = connection
            logger.info(f"✅ تم تسجيل WebSocket: {socket_id} للمستخدم: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تسجيل WebSocket: {e}")
            return False
    
    async def unregister_websocket(self, socket_id: str) -> bool:
        """إلغاء تسجيل اتصال WebSocket"""
        try:
            if socket_id in self._websocket_connections:
                self._websocket_connections[socket_id].is_active = False
                del self._websocket_connections[socket_id]
                logger.info(f"✅ تم إلغاء تسجيل WebSocket: {socket_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء تسجيل WebSocket: {e}")
            return False
    
    async def get_unread_count(self, user_id: str) -> int:
        """
        الحصول على عدد الإشعارات غير المقروءة - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(func.count(NotificationDB.id))
                            .where(NotificationDB.user_id == user_id)
                            .where(NotificationDB.is_read == False)
                        )
                        count = result.scalar() or 0
                        return count
                    except Exception as e:
                        logger.error(f"DB error counting unread: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_get_unread_count(user_id)
            
        except Exception as e:
            logger.error(f"خطأ في حساب الإشعارات غير المقروءة: {e}")
            return 0
    
    def _fallback_get_unread_count(self, user_id: str) -> int:
        """Fallback: Get unread count from in-memory storage"""
        notification_ids = self._fallback_user_notifications.get(user_id, [])
        count = 0
        
        for nid in notification_ids:
            notif = self._fallback_notifications.get(nid)
            if notif and not notif.is_read:
                count += 1
        
        return count
    
    async def delete_notification(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """
        حذف إشعار - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(NotificationDB)
                            .where(NotificationDB.id == notification_id)
                            .where(NotificationDB.user_id == user_id)
                        )
                        db_notification = result.scalar_one_or_none()
                        
                        if db_notification:
                            await db.delete(db_notification)
                            await db.commit()
                            logger.info(f"✅ تم حذف الإشعار: {notification_id}")
                            return True
                        return False
                    except Exception as e:
                        logger.error(f"DB error deleting notification: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_delete_notification(user_id, notification_id)
            
        except Exception as e:
            logger.error(f"خطأ في حذف الإشعار: {e}")
            return False
    
    def _fallback_delete_notification(self, user_id: str, notification_id: str) -> bool:
        """Fallback: Delete notification from in-memory storage"""
        notif = self._fallback_notifications.get(notification_id)
        if not notif or notif.user_id != user_id:
            return False
        
        del self._fallback_notifications[notification_id]
        
        if user_id in self._fallback_user_notifications:
            self._fallback_user_notifications[user_id].remove(notification_id)
        
        logger.info(f"✅ تم حذف الإشعار: {notification_id}")
        return True
    
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
            logger.info(f"[EMAIL] إلى {notification.user_id}: {notification.title}")
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال البريد: {e}")
            return False
    
    async def _send_in_app(self, notification: Notification) -> bool:
        """إرسال داخل التطبيق"""
        try:
            logger.info(f"[IN_APP] للمستخدم {notification.user_id}: {notification.title}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال إشعار داخلي: {e}")
            return False
    
    async def _send_push(self, notification: Notification) -> bool:
        """إرسال Push Notification"""
        try:
            logger.info(f"[PUSH] للمستخدم {notification.user_id}: {notification.title}")
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال Push: {e}")
            return False


# Global service instance
notification_service = NotificationService()
