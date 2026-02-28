"""
خدمة المزامنة
Sync Service for cross-device data synchronization
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """حالات المزامنة"""
    IDLE = "idle"
    SYNCING = "syncing"
    OFFLINE = "offline"
    CONFLICT = "conflict"
    ERROR = "error"
    PAUSED = "paused"


class ConflictResolution(Enum):
    """استراتيجيات حل التعارض"""
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"
    MANUAL = "manual"
    TIMESTAMP = "timestamp"


@dataclass
class SyncItem:
    """نموذج عنصر للمزامنة"""
    item_id: str
    user_id: str
    data: Dict[str, Any]
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    device_id: Optional[str] = None
    deleted: bool = False
    
    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """حساب المجموع الاختباري للبيانات"""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def update_data(self, new_data: Dict[str, Any], device_id: str) -> None:
        """تحديث البيانات"""
        self.data = new_data
        self.version += 1
        self.modified_at = datetime.now()
        self.device_id = device_id
        self.checksum = self._calculate_checksum()


@dataclass
class Conflict:
    """نموذج تعارض"""
    conflict_id: str
    item_id: str
    user_id: str
    local_item: SyncItem
    remote_item: SyncItem
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[ConflictResolution] = None
    resolved_data: Optional[Dict[str, Any]] = None


@dataclass
class Device:
    """نموذج جهاز"""
    device_id: str
    user_id: str
    name: str
    device_type: str  # desktop, mobile, web
    last_sync: Optional[datetime] = None
    is_online: bool = True
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncState:
    """نموذج حالة المزامنة"""
    user_id: str
    status: SyncStatus
    last_sync_at: Optional[datetime] = None
    pending_changes: int = 0
    conflicts_count: int = 0
    connected_devices: int = 0
    message: Optional[str] = None


class SyncService:
    """
    خدمة المزامنة
    
    تدير مزامنة البيانات عبر الأجهزة مع معالجة حالات عدم الاتصال والتعارضات
    """
    
    def __init__(self):
        """تهيئة خدمة المزامنة"""
        self._items: Dict[str, SyncItem] = {}
        self._user_items: Dict[str, Set[str]] = {}
        self._conflicts: Dict[str, Conflict] = {}
        self._devices: Dict[str, Device] = {}
        self._pending_changes: Dict[str, List[SyncItem]] = {}
        self._online_status: Dict[str, bool] = {}
        
        # قفل للعمليات المتزامنة
        self._sync_lock = asyncio.Lock()
        
        logger.info("تم تهيئة خدمة المزامنة")
    
    async def sync_data(
        self,
        user_id: str,
        device_id: str,
        local_changes: List[Dict[str, Any]],
        strategy: ConflictResolution = ConflictResolution.TIMESTAMP
    ) -> Dict[str, Any]:
        """
        مزامنة البيانات
        
        المعاملات:
            user_id: معرف المستخدم
            device_id: معرف الجهاز
            local_changes: التغييرات المحلية
            strategy: استراتيجية حل التعارض
            
        العائد:
            Dict: نتيجة المزامنة
        """
        async with self._sync_lock:
            try:
                logger.info(f"بدء مزامنة للمستخدم: {user_id} من الجهاز: {device_id}")
                
                # تحديث حالة الجهاز
                await self._update_device_sync_time(device_id)
                
                synced_items = []
                conflicts = []
                
                for change in local_changes:
                    item_id = change.get("item_id")
                    
                    if not item_id:
                        continue
                    
                    # التحقق من وجود عنصر محلي
                    existing_item = self._items.get(item_id)
                    
                    if existing_item:
                        # التحقق من التعارض
                        if existing_item.version != change.get("version", 0):
                            conflict = await self._detect_conflict(
                                item_id, user_id, existing_item, change, device_id
                            )
                            if conflict:
                                conflicts.append(conflict.conflict_id)
                                
                                # محاولة حل التعارض تلقائياً
                                if strategy != ConflictResolution.MANUAL:
                                    await self.resolve_conflict(
                                        conflict.conflict_id, strategy
                                    )
                                continue
                        
                        # تحديث العنصر
                        existing_item.update_data(change.get("data", {}), device_id)
                        synced_items.append(item_id)
                    else:
                        # إنشاء عنصر جديد
                        new_item = SyncItem(
                            item_id=item_id,
                            user_id=user_id,
                            data=change.get("data", {}),
                            device_id=device_id
                        )
                        
                        self._items[item_id] = new_item
                        
                        if user_id not in self._user_items:
                            self._user_items[user_id] = set()
                        self._user_items[user_id].add(item_id)
                        
                        synced_items.append(item_id)
                
                # الحصول على التغييرات البعيدة
                remote_changes = await self._get_remote_changes(user_id, device_id)
                
                result = {
                    "success": True,
                    "synced_items": synced_items,
                    "remote_changes": [self._item_to_dict(item) for item in remote_changes],
                    "conflicts": conflicts,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"اكتملت المزامنة للمستخدم: {user_id}")
                return result
                
            except Exception as e:
                logger.error(f"خطأ في المزامنة: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        strategy: ConflictResolution,
        manual_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        حل تعارض
        
        المعاملات:
            conflict_id: معرف التعارض
            strategy: استراتيجية الحل
            manual_data: بيانات يدوية (للاستراتيجية MANUAL)
            
        العائد:
            bool: True إذا نجح الحل
        """
        try:
            if conflict_id not in self._conflicts:
                logger.warning(f"التعارض غير موجود: {conflict_id}")
                return False
            
            conflict = self._conflicts[conflict_id]
            
            if strategy == ConflictResolution.LOCAL_WINS:
                resolved_data = conflict.local_item.data
            elif strategy == ConflictResolution.REMOTE_WINS:
                resolved_data = conflict.remote_item.data
            elif strategy == ConflictResolution.TIMESTAMP:
                if conflict.local_item.modified_at > conflict.remote_item.modified_at:
                    resolved_data = conflict.local_item.data
                else:
                    resolved_data = conflict.remote_item.data
            elif strategy == ConflictResolution.MERGE:
                resolved_data = self._merge_data(
                    conflict.local_item.data,
                    conflict.remote_item.data
                )
            elif strategy == ConflictResolution.MANUAL:
                if manual_data is None:
                    logger.warning("لم يتم توفير بيانات يدوية")
                    return False
                resolved_data = manual_data
            else:
                return False
            
            # تحديث العنصر بالبيانات المحلولة
            item = self._items.get(conflict.item_id)
            if item:
                item.data = resolved_data
                item.version += 1
                item.modified_at = datetime.now()
                item.checksum = item._calculate_checksum()
            
            # تحديث حالة التعارض
            conflict.resolved = True
            conflict.resolution = strategy
            conflict.resolved_data = resolved_data
            
            logger.info(f"تم حل التعارض: {conflict_id} باستخدام: {strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حل التعارض: {e}")
            return False
    
    async def get_sync_status(self, user_id: str) -> SyncState:
        """
        الحصول على حالة المزامنة
        
        المعاملات:
            user_id: معرف المستخدم
            
        العائد:
            SyncState: حالة المزامنة
        """
        try:
            is_online = self._online_status.get(user_id, True)
            pending = len(self._pending_changes.get(user_id, []))
            
            user_conflicts = [
                c for c in self._conflicts.values()
                if c.user_id == user_id and not c.resolved
            ]
            
            user_devices = [
                d for d in self._devices.values()
                if d.user_id == user_id and d.is_online
            ]
            
            last_sync = None
            for device in user_devices:
                if device.last_sync:
                    if last_sync is None or device.last_sync > last_sync:
                        last_sync = device.last_sync
            
            if not is_online:
                status = SyncStatus.OFFLINE
            elif pending > 0:
                status = SyncStatus.SYNCING
            elif len(user_conflicts) > 0:
                status = SyncStatus.CONFLICT
            else:
                status = SyncStatus.IDLE
            
            return SyncState(
                user_id=user_id,
                status=status,
                last_sync_at=last_sync,
                pending_changes=pending,
                conflicts_count=len(user_conflicts),
                connected_devices=len(user_devices),
                message=self._get_status_message(status)
            )
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على حالة المزامنة: {e}")
            return SyncState(
                user_id=user_id,
                status=SyncStatus.ERROR,
                message=str(e)
            )
    
    async def set_online_status(self, user_id: str, is_online: bool) -> bool:
        """
        تعيين حالة الاتصال
        
        المعاملات:
            user_id: معرف المستخدم
            is_online: حالة الاتصال
            
        العائد:
            bool: True إذا نجحت العملية
        """
        try:
            self._online_status[user_id] = is_online
            
            if is_online:
                # محاولة مزامنة التغييرات المعلقة
                await self._sync_pending_changes(user_id)
            
            logger.info(f"تم تغيير حالة الاتصال للمستخدم {user_id}: {is_online}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تعيين حالة الاتصال: {e}")
            return False
    
    async def register_device(
        self,
        device_id: str,
        user_id: str,
        name: str,
        device_type: str,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        تسجيل جهاز جديد
        
        المعاملات:
            device_id: معرف الجهاز
            user_id: معرف المستخدم
            name: اسم الجهاز
            device_type: نوع الجهاز
            capabilities: إمكانيات الجهاز
            
        العائد:
            bool: True إذا نجح التسجيل
        """
        try:
            device = Device(
                device_id=device_id,
                user_id=user_id,
                name=name,
                device_type=device_type,
                is_online=True,
                capabilities=capabilities or {}
            )
            
            self._devices[device_id] = device
            logger.info(f"تم تسجيل جهاز: {name} ({device_type}) للمستخدم: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تسجيل الجهاز: {e}")
            return False
    
    async def unregister_device(self, device_id: str) -> bool:
        """
        إلغاء تسجيل جهاز
        
        المعاملات:
            device_id: معرف الجهاز
            
        العائد:
            bool: True إذا نجح الإلغاء
        """
        try:
            if device_id in self._devices:
                device = self._devices[device_id]
                device.is_online = False
                logger.info(f"تم إلغاء تسجيل الجهاز: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء تسجيل الجهاز: {e}")
            return False
    
    async def get_conflicts(self, user_id: str) -> List[Conflict]:
        """
        الحصول على التعارضات للمستخدم
        
        المعاملات:
            user_id: معرف المستخدم
            
        العائد:
            List[Conflict]: قائمة التعارضات
        """
        return [
            c for c in self._conflicts.values()
            if c.user_id == user_id and not c.resolved
        ]
    
    # دوال مساعدة
    async def _detect_conflict(
        self,
        item_id: str,
        user_id: str,
        local_item: SyncItem,
        remote_change: Dict[str, Any],
        device_id: str
    ) -> Optional[Conflict]:
        """اكتشاف تعارض"""
        try:
            conflict_id = f"conflict_{item_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # إنشاء عنصر مؤقت للتغيير البعيد
            remote_item = SyncItem(
                item_id=item_id,
                user_id=user_id,
                data=remote_change.get("data", {}),
                version=remote_change.get("version", 0),
                device_id=device_id,
                modified_at=datetime.fromisoformat(
                    remote_change.get("modified_at", datetime.now().isoformat())
                )
            )
            
            conflict = Conflict(
                conflict_id=conflict_id,
                item_id=item_id,
                user_id=user_id,
                local_item=local_item,
                remote_item=remote_item
            )
            
            self._conflicts[conflict_id] = conflict
            logger.warning(f"تم اكتشاف تعارض: {conflict_id}")
            return conflict
            
        except Exception as e:
            logger.error(f"خطأ في اكتشاف التعارض: {e}")
            return None
    
    async def _get_remote_changes(
        self,
        user_id: str,
        device_id: str
    ) -> List[SyncItem]:
        """الحصول على التغييرات البعيدة"""
        changes = []
        
        item_ids = self._user_items.get(user_id, set())
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item and item.device_id != device_id and not item.deleted:
                changes.append(item)
        
        return changes
    
    async def _update_device_sync_time(self, device_id: str) -> None:
        """تحديث وقت آخر مزامنة للجهاز"""
        if device_id in self._devices:
            self._devices[device_id].last_sync = datetime.now()
    
    async def _sync_pending_changes(self, user_id: str) -> None:
        """مزامنة التغييرات المعلقة"""
        pending = self._pending_changes.get(user_id, [])
        
        for item in pending:
            # إعادة محاولة مزامنة كل عنصر
            logger.info(f"مزامنة تغيير معلق: {item.item_id}")
        
        self._pending_changes[user_id] = []
    
    def _merge_data(
        self,
        local_data: Dict[str, Any],
        remote_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """دمج بيانات متعارضة"""
        merged = {**remote_data}
        
        for key, value in local_data.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
        
        return merged
    
    def _item_to_dict(self, item: SyncItem) -> Dict[str, Any]:
        """تحويل عنصر إلى قاموس"""
        return {
            "item_id": item.item_id,
            "data": item.data,
            "version": item.version,
            "modified_at": item.modified_at.isoformat(),
            "checksum": item.checksum,
            "device_id": item.device_id,
            "deleted": item.deleted
        }
    
    def _get_status_message(self, status: SyncStatus) -> str:
        """الحصول على رسالة الحالة"""
        messages = {
            SyncStatus.IDLE: "جاهز للمزامنة",
            SyncStatus.SYNCING: "جاري المزامنة...",
            SyncStatus.OFFLINE: "غير متصل",
            SyncStatus.CONFLICT: "يوجد تعارضات تحتاج للحل",
            SyncStatus.ERROR: "حدث خطأ",
            SyncStatus.PAUSED: "المزامنة متوقفة"
        }
        return messages.get(status, "حالة غير معروفة")
