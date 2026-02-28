"""
خدمة النسخ الاحتياطي
Backup Service for data backup and restore
"""

import logging
import asyncio
import json
import hashlib
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """أنواع النسخ الاحتياطي"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """حالات النسخ الاحتياطي"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupManifest:
    """نموذج بيانات النسخة الاحتياطية"""
    files: List[str] = field(default_factory=list)
    total_size: int = 0
    file_hashes: Dict[str, str] = field(default_factory=dict)
    parent_backup: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "total_size": self.total_size,
            "file_hashes": self.file_hashes,
            "parent_backup": self.parent_backup
        }


@dataclass
class Backup:
    """نموذج نسخة احتياطية"""
    backup_id: str
    user_id: str
    name: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    manifest: BackupManifest = field(default_factory=BackupManifest)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    storage_path: Optional[str] = None


@dataclass
class BackupSchedule:
    """نموذج جدولة النسخ الاحتياطي"""
    schedule_id: str
    user_id: str
    name: str
    backup_type: BackupType
    cron_expression: str
    retention_days: int = 30
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class BackupService:
    """
    خدمة النسخ الاحتياطي
    
    تدير إنشاء واستعادة النسخ الاحتياطية مع دعم النسخ التزايدي
    """
    
    def __init__(self, backup_storage_path: str = "./backups"):
        """
        تهيئة خدمة النسخ الاحتياطي
        
        المعاملات:
            backup_storage_path: مسار تخزين النسخ الاحتياطية
        """
        self._backups: Dict[str, Backup] = {}
        self._user_backups: Dict[str, List[str]] = {}
        self._schedules: Dict[str, BackupSchedule] = {}
        self._storage_path = Path(backup_storage_path)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # إنشاء مجلد التخزين
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"تم تهيئة خدمة النسخ الاحتياطي: {backup_storage_path}")
    
    async def create_backup(
        self,
        user_id: str,
        name: str,
        data_sources: List[Dict[str, Any]],
        backup_type: BackupType = BackupType.FULL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Backup:
        """
        إنشاء نسخة احتياطية
        
        المعاملات:
            user_id: معرف المستخدم
            name: اسم النسخة
            data_sources: مصادر البيانات
            backup_type: نوع النسخة
            metadata: بيانات إضافية
            
        العائد:
            Backup: معلومات النسخة الاحتياطية
        """
        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup = Backup(
                backup_id=backup_id,
                user_id=user_id,
                name=name,
                backup_type=backup_type,
                status=BackupStatus.PENDING,
                created_at=datetime.now(),
                metadata=metadata or {}
            )
            
            self._backups[backup_id] = backup
            
            if user_id not in self._user_backups:
                self._user_backups[user_id] = []
            self._user_backups[user_id].append(backup_id)
            
            # بدء النسخ في مهمة منفصلة
            task = asyncio.create_task(
                self._perform_backup(backup_id, data_sources)
            )
            self._running_tasks[backup_id] = task
            
            logger.info(f"تم بدء نسخة احتياطية: {backup_id}")
            return backup
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء النسخة الاحتياطية: {e}")
            raise
    
    async def restore_backup(
        self,
        backup_id: str,
        target_path: Optional[str] = None,
        selective_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        استعادة نسخة احتياطية
        
        المعاملات:
            backup_id: معرف النسخة
            target_path: مسار الاستعادة
            selective_files: ملفات محددة للاستعادة
            
        العائد:
            Dict: نتيجة الاستعادة
        """
        try:
            if backup_id not in self._backups:
                raise ValueError(f"النسخة الاحتياطية غير موجودة: {backup_id}")
            
            backup = self._backups[backup_id]
            
            if backup.status != BackupStatus.COMPLETED:
                raise RuntimeError("النسخة الاحتياطية غير مكتملة")
            
            logger.info(f"بدء استعادة النسخة: {backup_id}")
            
            # محاكاة عملية الاستعادة
            restored_files = []
            files_to_restore = selective_files or backup.manifest.files
            
            for file_path in files_to_restore:
                if file_path in backup.manifest.files:
                    # محاكاة استعادة الملف
                    await asyncio.sleep(0.1)
                    restored_files.append(file_path)
                    logger.debug(f"تم استعادة: {file_path}")
            
            result = {
                "success": True,
                "backup_id": backup_id,
                "restored_files": restored_files,
                "total_files": len(restored_files),
                "target_path": target_path or "default_restore_path",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"تم استعادة النسخة: {backup_id}")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في استعادة النسخة: {e}")
            return {
                "success": False,
                "backup_id": backup_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def list_backups(
        self,
        user_id: str,
        backup_type: Optional[BackupType] = None,
        include_deleted: bool = False
    ) -> List[Backup]:
        """
        قائمة النسخ الاحتياطية
        
        المعاملات:
            user_id: معرف المستخدم
            backup_type: تصفية حسب النوع
            include_deleted: تضمين المحذوفة
            
        العائد:
            List[Backup]: قائمة النسخ الاحتياطية
        """
        try:
            backup_ids = self._user_backups.get(user_id, [])
            backups = []
            
            for bid in backup_ids:
                backup = self._backups.get(bid)
                if not backup:
                    continue
                
                if backup_type and backup.backup_type != backup_type:
                    continue
                
                if not include_deleted and backup.status == BackupStatus.CANCELLED:
                    continue
                
                backups.append(backup)
            
            # ترتيب حسب الأحدث
            backups.sort(key=lambda x: x.created_at, reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"خطأ في جلب النسخ الاحتياطية: {e}")
            return []
    
    async def schedule_backup(
        self,
        user_id: str,
        name: str,
        backup_type: BackupType,
        cron_expression: str,
        data_sources: List[Dict[str, Any]],
        retention_days: int = 30
    ) -> BackupSchedule:
        """
        جدولة نسخ احتياطي
        
        المعاملات:
            user_id: معرف المستخدم
            name: اسم الجدولة
            backup_type: نوع النسخة
            cron_expression: تعبير cron
            data_sources: مصادر البيانات
            retention_days: أيام الاحتفاظ
            
        العائد:
            BackupSchedule: معلومات الجدولة
        """
        try:
            schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # حساب وقت التشغيل التالي (تبسيط)
            next_run = datetime.now() + timedelta(hours=1)
            
            schedule = BackupSchedule(
                schedule_id=schedule_id,
                user_id=user_id,
                name=name,
                backup_type=backup_type,
                cron_expression=cron_expression,
                retention_days=retention_days,
                next_run=next_run
            )
            
            self._schedules[schedule_id] = schedule
            
            logger.info(f"تم إنشاء جدولة: {schedule_id}")
            return schedule
            
        except Exception as e:
            logger.error(f"خطأ في جدولة النسخ الاحتياطي: {e}")
            raise
    
    async def cancel_backup(self, backup_id: str) -> bool:
        """
        إلغاء نسخة احتياطية قيد التشغيل
        
        المعاملات:
            backup_id: معرف النسخة
            
        العائد:
            bool: True إذا نجح الإلغاء
        """
        try:
            if backup_id not in self._backups:
                return False
            
            backup = self._backups[backup_id]
            
            if backup.status != BackupStatus.RUNNING:
                logger.warning(f"لا يمكن إلغاء النسخة بحالة: {backup.status}")
                return False
            
            # إلغاء المهمة
            if backup_id in self._running_tasks:
                self._running_tasks[backup_id].cancel()
                del self._running_tasks[backup_id]
            
            backup.status = BackupStatus.CANCELLED
            backup.completed_at = datetime.now()
            
            logger.info(f"تم إلغاء النسخة الاحتياطية: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء النسخة: {e}")
            return False
    
    async def delete_backup(self, backup_id: str) -> bool:
        """
        حذف نسخة احتياطية
        
        المعاملات:
            backup_id: معرف النسخة
            
        العائد:
            bool: True إذا نجح الحذف
        """
        try:
            if backup_id not in self._backups:
                return False
            
            backup = self._backups[backup_id]
            
            # حذف الملفات إذا وجدت
            if backup.storage_path:
                storage_path = Path(backup.storage_path)
                if storage_path.exists():
                    # محاكاة حذف الملفات
                    logger.info(f"حذف ملفات النسخة: {backup_id}")
            
            # إزالة من القوائم
            del self._backups[backup_id]
            
            user_id = backup.user_id
            if user_id in self._user_backups and backup_id in self._user_backups[user_id]:
                self._user_backups[user_id].remove(backup_id)
            
            logger.info(f"تم حذف النسخة الاحتياطية: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حذف النسخة: {e}")
            return False
    
    async def get_backup_details(self, backup_id: str) -> Optional[Backup]:
        """
        الحصول على تفاصيل نسخة احتياطية
        
        المعاملات:
            backup_id: معرف النسخة
            
        العائد:
            Backup أو None
        """
        return self._backups.get(backup_id)
    
    async def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        التحقق من سلامة نسخة احتياطية
        
        المعاملات:
            backup_id: معرف النسخة
            
        العائد:
            Dict: نتيجة التحقق
        """
        try:
            backup = self._backups.get(backup_id)
            if not backup:
                return {"valid": False, "error": "النسخة غير موجودة"}
            
            if backup.status != BackupStatus.COMPLETED:
                return {"valid": False, "error": "النسخة غير مكتملة"}
            
            # التحقق من الملفات
            verified_files = 0
            failed_files = 0
            
            for file_path, expected_hash in backup.manifest.file_hashes.items():
                # محاكاة التحقق
                await asyncio.sleep(0.05)
                verified_files += 1
            
            return {
                "valid": failed_files == 0,
                "backup_id": backup_id,
                "verified_files": verified_files,
                "failed_files": failed_files,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"خطأ في التحقق من النسخة: {e}")
            return {"valid": False, "error": str(e)}
    
    async def cleanup_old_backups(
        self,
        user_id: str,
        retention_days: int = 30
    ) -> int:
        """
        تنظيف النسخ الاحتياطية القديمة
        
        المعاملات:
            user_id: معرف المستخدم
            retention_days: أيام الاحتفاظ
            
        العائد:
            int: عدد النسخ المحذوفة
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            backup_ids = self._user_backups.get(user_id, [])
            
            deleted_count = 0
            for bid in list(backup_ids):
                backup = self._backups.get(bid)
                if backup and backup.created_at < cutoff_date:
                    if await self.delete_backup(bid):
                        deleted_count += 1
            
            logger.info(f"تم حذف {deleted_count} نسخة قديمة للمستخدم: {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"خطأ في تنظيف النسخ القديمة: {e}")
            return 0
    
    # دوال داخلية
    async def _perform_backup(
        self,
        backup_id: str,
        data_sources: List[Dict[str, Any]]
    ) -> None:
        """تنفيذ عملية النسخ الاحتياطي"""
        try:
            backup = self._backups[backup_id]
            backup.status = BackupStatus.RUNNING
            
            manifest = BackupManifest()
            total_size = 0
            
            # معالجة كل مصدر بيانات
            for source in data_sources:
                source_type = source.get("type", "files")
                source_path = source.get("path", "")
                
                # محاكاة نسخ البيانات
                await asyncio.sleep(0.2)
                
                # إضافة إلى البيان
                files = source.get("files", [])
                for file_path in files:
                    manifest.files.append(file_path)
                    file_size = source.get("size", 1024)
                    total_size += file_size
                    manifest.file_hashes[file_path] = self._calculate_hash(file_path)
            
            manifest.total_size = total_size
            
            # إذا كانت تزايدية، ربطها بالنسخة الكاملة السابقة
            if backup.backup_type == BackupType.INCREMENTAL:
                parent_backup = await self._find_parent_backup(backup.user_id)
                if parent_backup:
                    manifest.parent_backup = parent_backup
            
            backup.manifest = manifest
            backup.size_bytes = total_size
            backup.status = BackupStatus.COMPLETED
            backup.completed_at = datetime.now()
            backup.storage_path = str(self._storage_path / backup_id)
            
            logger.info(f"اكتملت النسخة الاحتياطية: {backup_id}")
            
        except asyncio.CancelledError:
            logger.info(f"تم إلغاء النسخة: {backup_id}")
            raise
        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.error_message = str(e)
            backup.completed_at = datetime.now()
            logger.error(f"فشلت النسخة الاحتياطية {backup_id}: {e}")
    
    async def _find_parent_backup(self, user_id: str) -> Optional[str]:
        """البحث عن نسخة كاملة سابقة"""
        backup_ids = self._user_backups.get(user_id, [])
        
        for bid in reversed(backup_ids):
            backup = self._backups.get(bid)
            if backup and backup.backup_type == BackupType.FULL:
                if backup.status == BackupStatus.COMPLETED:
                    return bid
        
        return None
    
    def _calculate_hash(self, file_path: str) -> str:
        """حساب hash للملف"""
        # محاكاة حساب hash
        return hashlib.md5(file_path.encode()).hexdigest()
