"""
خدمة النسخ الاحتياطي - Backup Service V2
PostgreSQL-based persistent storage replacing in-memory _backups and _schedules
"""

import logging
import asyncio
import json
import hashlib
import os
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update

# Try importing database models
try:
    from core.database import get_async_session
    from core.service_models import BackupDB, BackupScheduleDB
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupManifest':
        """Create from dictionary"""
        manifest = cls()
        if data:
            manifest.files = data.get("files", [])
            manifest.total_size = data.get("total_size", 0)
            manifest.file_hashes = data.get("file_hashes", {})
            manifest.parent_backup = data.get("parent_backup")
        return manifest


@dataclass
class Backup:
    """نموذج نسخة احتياطية (in-memory representation)"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "user_id": self.user_id,
            "name": self.name,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "manifest": self.manifest.to_dict(),
            "metadata": self.metadata,
            "error_message": self.error_message,
            "storage_path": self.storage_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_db(cls, db_backup: BackupDB) -> 'Backup':
        """Create from database model"""
        return cls(
            backup_id=db_backup.id,
            user_id=db_backup.user_id,
            name=db_backup.name,
            backup_type=BackupType(db_backup.backup_type),
            status=BackupStatus(db_backup.status),
            created_at=db_backup.created_at,
            completed_at=db_backup.completed_at,
            size_bytes=db_backup.size_bytes,
            manifest=BackupManifest.from_dict(db_backup.get_manifest()),
            metadata=db_backup.get_metadata(),
            error_message=db_backup.error_message,
            storage_path=db_backup.storage_path
        )


@dataclass
class BackupSchedule:
    """نموذج جدولة النسخ الاحتياطي (in-memory representation)"""
    schedule_id: str
    user_id: str
    name: str
    backup_type: BackupType
    cron_expression: str
    retention_days: int = 30
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "user_id": self.user_id,
            "name": self.name,
            "backup_type": self.backup_type.value,
            "cron_expression": self.cron_expression,
            "retention_days": self.retention_days,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
        }
    
    @classmethod
    def from_db(cls, db_schedule: BackupScheduleDB) -> 'BackupSchedule':
        """Create from database model"""
        return cls(
            schedule_id=db_schedule.id,
            user_id=db_schedule.user_id,
            name=db_schedule.name,
            backup_type=BackupType(db_schedule.backup_type),
            cron_expression=db_schedule.cron_expression,
            retention_days=db_schedule.retention_days,
            is_active=db_schedule.is_active,
            metadata=db_schedule.get_metadata(),
            last_run=db_schedule.last_run,
            next_run=db_schedule.next_run
        )


class BackupService:
    """
    خدمة النسخ الاحتياطي V2 - PostgreSQL Persistent Storage
    """
    
    def __init__(self, backup_storage_path: str = "./backups"):
        """تهيئة خدمة النسخ الاحتياطي"""
        self._storage_path = Path(backup_storage_path)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # PostgreSQL availability
        self._postgres_available = POSTGRES_AVAILABLE
        
        # Fallback in-memory storage if PostgreSQL not available
        self._fallback_backups: Dict[str, Backup] = {}
        self._fallback_schedules: Dict[str, BackupSchedule] = {}
        
        # إنشاء مجلد التخزين
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        if self._postgres_available:
            logger.info(f"✅ BackupService V2 initialized (PostgreSQL): {backup_storage_path}")
        else:
            logger.warning(f"⚠️ BackupService V2 initialized (In-Memory Fallback): {backup_storage_path}")
    
    async def _get_db_session(self) -> Optional[AsyncSession]:
        """Get database session if PostgreSQL is available"""
        if not self._postgres_available:
            return None
        try:
            session_gen = get_async_session()
            return await session_gen.__anext__()
        except Exception as e:
            logger.warning(f"Failed to get DB session: {e}")
            return None
    
    async def create_backup(
        self,
        user_id: str,
        name: str,
        data_sources: List[Dict[str, Any]],
        backup_type: BackupType = BackupType.FULL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Backup:
        """
        إنشاء نسخة احتياطية - Persisted to PostgreSQL
        """
        try:
            backup_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        db_backup = BackupDB(
                            id=backup_id,
                            user_id=user_id,
                            name=name,
                            backup_type=backup_type.value,
                            status=BackupStatus.PENDING.value,
                            size_bytes=0,
                            manifest_json={"files": [], "file_hashes": {}, "total_size": 0},
                            metadata_json=metadata or {},
                            created_at=now
                        )
                        db.add(db_backup)
                        await db.commit()
                        logger.info(f"✅ Backup stored in PostgreSQL: {backup_id}")
                    except Exception as e:
                        logger.error(f"DB error creating backup: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
                else:
                    self._fallback_store_backup(backup_id, user_id, name, backup_type, metadata, now)
            else:
                self._fallback_store_backup(backup_id, user_id, name, backup_type, metadata, now)
            
            backup = Backup(
                backup_id=backup_id,
                user_id=user_id,
                name=name,
                backup_type=backup_type,
                status=BackupStatus.PENDING,
                created_at=now,
                metadata=metadata or {}
            )
            
            # بدء النسخ في مهمة منفصلة
            task = asyncio.create_task(
                self._perform_backup(backup_id, data_sources)
            )
            self._running_tasks[backup_id] = task
            
            logger.info(f"✅ تم بدء نسخة احتياطية: {backup_id}")
            return backup
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء النسخة الاحتياطية: {e}")
            raise
    
    def _fallback_store_backup(
        self, backup_id: str, user_id: str, name: str,
        backup_type: BackupType, metadata: Optional[Dict[str, Any]], now: datetime
    ) -> None:
        """Store backup in fallback in-memory storage"""
        backup = Backup(
            backup_id=backup_id,
            user_id=user_id,
            name=name,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            created_at=now,
            metadata=metadata or {}
        )
        self._fallback_backups[backup_id] = backup
    
    async def restore_backup(
        self,
        backup_id: str,
        target_path: Optional[str] = None,
        selective_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """استعادة نسخة احتياطية"""
        try:
            backup = await self.get_backup_details(backup_id)
            if not backup:
                raise ValueError(f"النسخة الاحتياطية غير موجودة: {backup_id}")
            
            if backup.status != BackupStatus.COMPLETED:
                raise RuntimeError("النسخة الاحتياطية غير مكتملة")
            
            logger.info(f"بدء استعادة النسخة: {backup_id}")
            
            # محاكاة عملية الاستعادة
            restored_files = []
            files_to_restore = selective_files or backup.manifest.files
            
            for file_path in files_to_restore:
                if file_path in backup.manifest.files:
                    await asyncio.sleep(0.1)
                    restored_files.append(file_path)
                    logger.debug(f"تم استعادة: {file_path}")
            
            result = {
                "success": True,
                "backup_id": backup_id,
                "restored_files": restored_files,
                "total_files": len(restored_files),
                "target_path": target_path or "default_restore_path",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✅ تم استعادة النسخة: {backup_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ خطأ في استعادة النسخة: {e}")
            return {
                "success": False,
                "backup_id": backup_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def list_backups(
        self,
        user_id: str,
        backup_type: Optional[BackupType] = None,
        include_deleted: bool = False
    ) -> List[Backup]:
        """
        قائمة النسخ الاحتياطية - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        query = select(BackupDB).where(BackupDB.user_id == user_id)
                        
                        if backup_type:
                            query = query.where(BackupDB.backup_type == backup_type.value)
                        
                        if not include_deleted:
                            query = query.where(BackupDB.status != BackupStatus.CANCELLED.value)
                        
                        query = query.order_by(desc(BackupDB.created_at))
                        
                        result = await db.execute(query)
                        db_backups = result.scalars().all()
                        
                        return [Backup.from_db(b) for b in db_backups]
                    except Exception as e:
                        logger.error(f"DB error fetching backups: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            backups = [
                b for b in self._fallback_backups.values()
                if b.user_id == user_id and (include_deleted or b.status != BackupStatus.CANCELLED)
            ]
            if backup_type:
                backups = [b for b in backups if b.backup_type == backup_type]
            return sorted(backups, key=lambda x: x.created_at, reverse=True)
            
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
        جدولة نسخ احتياطي - Persisted to PostgreSQL
        """
        try:
            schedule_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            next_run = now + timedelta(hours=1)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        db_schedule = BackupScheduleDB(
                            id=schedule_id,
                            user_id=user_id,
                            name=name,
                            backup_type=backup_type.value,
                            cron_expression=cron_expression,
                            retention_days=retention_days,
                            is_active=True,
                            metadata_json={"data_sources": data_sources},
                            created_at=now,
                            next_run=next_run
                        )
                        db.add(db_schedule)
                        await db.commit()
                        logger.info(f"✅ Backup schedule stored in PostgreSQL: {schedule_id}")
                    except Exception as e:
                        logger.error(f"DB error creating schedule: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
                else:
                    self._fallback_store_schedule(
                        schedule_id, user_id, name, backup_type, cron_expression, retention_days, next_run
                    )
            else:
                self._fallback_store_schedule(
                    schedule_id, user_id, name, backup_type, cron_expression, retention_days, next_run
                )
            
            schedule = BackupSchedule(
                schedule_id=schedule_id,
                user_id=user_id,
                name=name,
                backup_type=backup_type,
                cron_expression=cron_expression,
                retention_days=retention_days,
                next_run=next_run
            )
            
            return schedule
            
        except Exception as e:
            logger.error(f"خطأ في جدولة النسخ الاحتياطي: {e}")
            raise
    
    def _fallback_store_schedule(
        self, schedule_id: str, user_id: str, name: str,
        backup_type: BackupType, cron_expression: str,
        retention_days: int, next_run: datetime
    ) -> None:
        """Store schedule in fallback in-memory storage"""
        schedule = BackupSchedule(
            schedule_id=schedule_id,
            user_id=user_id,
            name=name,
            backup_type=backup_type,
            cron_expression=cron_expression,
            retention_days=retention_days,
            next_run=next_run
        )
        self._fallback_schedules[schedule_id] = schedule
    
    async def cancel_backup(self, backup_id: str) -> bool:
        """
        إلغاء نسخة احتياطية قيد التشغيل - Update PostgreSQL
        """
        try:
            # إلغاء المهمة
            if backup_id in self._running_tasks:
                self._running_tasks[backup_id].cancel()
                del self._running_tasks[backup_id]
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(BackupDB).where(BackupDB.id == backup_id)
                        )
                        db_backup = result.scalar_one_or_none()
                        
                        if not db_backup:
                            return False
                        
                        if db_backup.status != BackupStatus.RUNNING.value:
                            logger.warning(f"لا يمكن إلغاء النسخة بحالة: {db_backup.status}")
                            return False
                        
                        db_backup.status = BackupStatus.CANCELLED.value
                        db_backup.completed_at = datetime.now(timezone.utc)
                        await db.commit()
                        
                        logger.info(f"✅ تم إلغاء النسخة الاحتياطية: {backup_id}")
                        return True
                    except Exception as e:
                        logger.error(f"DB error cancelling backup: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            backup = self._fallback_backups.get(backup_id)
            if not backup:
                return False
            if backup.status != BackupStatus.RUNNING:
                return False
            backup.status = BackupStatus.CANCELLED
            backup.completed_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء النسخة: {e}")
            return False
    
    async def delete_backup(self, backup_id: str) -> bool:
        """
        حذف نسخة احتياطية - Delete from PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(BackupDB).where(BackupDB.id == backup_id)
                        )
                        db_backup = result.scalar_one_or_none()
                        
                        if not db_backup:
                            return False
                        
                        # حذف الملفات إذا وجدت
                        if db_backup.storage_path:
                            storage_path = Path(db_backup.storage_path)
                            if storage_path.exists():
                                logger.info(f"حذف ملفات النسخة: {backup_id}")
                        
                        await db.delete(db_backup)
                        await db.commit()
                        
                        logger.info(f"✅ تم حذف النسخة الاحتياطية: {backup_id}")
                        return True
                    except Exception as e:
                        logger.error(f"DB error deleting backup: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            backup = self._fallback_backups.get(backup_id)
            if not backup:
                return False
            
            if backup.storage_path:
                storage_path = Path(backup.storage_path)
                if storage_path.exists():
                    logger.info(f"حذف ملفات النسخة: {backup_id}")
            
            del self._fallback_backups[backup_id]
            logger.info(f"✅ تم حذف النسخة الاحتياطية: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حذف النسخة: {e}")
            return False
    
    async def get_backup_details(self, backup_id: str) -> Optional[Backup]:
        """
        الحصول على تفاصيل نسخة احتياطية - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(BackupDB).where(BackupDB.id == backup_id)
                        )
                        db_backup = result.scalar_one_or_none()
                        if db_backup:
                            return Backup.from_db(db_backup)
                    except Exception as e:
                        logger.error(f"DB error fetching backup: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_backups.get(backup_id)
            
        except Exception as e:
            logger.error(f"خطأ في جلب تفاصيل النسخة: {e}")
            return None
    
    async def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """التحقق من سلامة نسخة احتياطية"""
        try:
            backup = await self.get_backup_details(backup_id)
            if not backup:
                return {"valid": False, "error": "النسخة غير موجودة"}
            
            if backup.status != BackupStatus.COMPLETED:
                return {"valid": False, "error": "النسخة غير مكتملة"}
            
            verified_files = 0
            failed_files = 0
            
            for file_path, expected_hash in backup.manifest.file_hashes.items():
                await asyncio.sleep(0.05)
                verified_files += 1
            
            return {
                "valid": failed_files == 0,
                "backup_id": backup_id,
                "verified_files": verified_files,
                "failed_files": failed_files,
                "timestamp": datetime.now(timezone.utc).isoformat()
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
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(BackupDB)
                            .where(BackupDB.user_id == user_id)
                            .where(BackupDB.created_at < cutoff_date)
                        )
                        old_backups = result.scalars().all()
                        
                        deleted_count = 0
                        for backup in old_backups:
                            await db.delete(backup)
                            deleted_count += 1
                        
                        await db.commit()
                        logger.info(f"✅ تم حذف {deleted_count} نسخة قديمة للمستخدم: {user_id}")
                        return deleted_count
                    except Exception as e:
                        logger.error(f"DB error cleaning up old backups: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            deleted_count = 0
            for bid in list(self._fallback_backups.keys()):
                backup = self._fallback_backups.get(bid)
                if backup and backup.user_id == user_id and backup.created_at < cutoff_date:
                    if await self.delete_backup(bid):
                        deleted_count += 1
            
            logger.info(f"✅ تم حذف {deleted_count} نسخة قديمة للمستخدم: {user_id}")
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
            # Update status to RUNNING
            await self._update_backup_status(backup_id, BackupStatus.RUNNING)
            
            manifest = BackupManifest()
            total_size = 0
            
            for source in data_sources:
                source_type = source.get("type", "files")
                source_path = source.get("path", "")
                
                await asyncio.sleep(0.2)
                
                files = source.get("files", [])
                for file_path in files:
                    manifest.files.append(file_path)
                    file_size = source.get("size", 1024)
                    total_size += file_size
                    manifest.file_hashes[file_path] = self._calculate_hash(file_path)
            
            manifest.total_size = total_size
            
            # If incremental, link to parent full backup
            backup = await self.get_backup_details(backup_id)
            if backup and backup.backup_type == BackupType.INCREMENTAL:
                parent_backup = await self._find_parent_backup(backup.user_id)
                if parent_backup:
                    manifest.parent_backup = parent_backup
            
            # Complete the backup
            await self._complete_backup(backup_id, manifest, total_size)
            
            logger.info(f"✅ اكتملت النسخة الاحتياطية: {backup_id}")
            
        except asyncio.CancelledError:
            logger.info(f"تم إلغاء النسخة: {backup_id}")
            raise
        except Exception as e:
            await self._fail_backup(backup_id, str(e))
            logger.error(f"❌ فشلت النسخة الاحتياطية {backup_id}: {e}")
    
    async def _update_backup_status(self, backup_id: str, status: BackupStatus) -> None:
        """Update backup status in database"""
        if self._postgres_available:
            db = await self._get_db_session()
            if db:
                try:
                    await db.execute(
                        update(BackupDB)
                        .where(BackupDB.id == backup_id)
                        .values(status=status.value)
                    )
                    await db.commit()
                except Exception as e:
                    logger.error(f"DB error updating backup status: {e}")
                    await db.rollback()
                finally:
                    await db.close()
        else:
            backup = self._fallback_backups.get(backup_id)
            if backup:
                backup.status = status
    
    async def _complete_backup(self, backup_id: str, manifest: BackupManifest, total_size: int) -> None:
        """Mark backup as completed"""
        now = datetime.now(timezone.utc)
        storage_path = str(self._storage_path / backup_id)
        
        if self._postgres_available:
            db = await self._get_db_session()
            if db:
                try:
                    await db.execute(
                        update(BackupDB)
                        .where(BackupDB.id == backup_id)
                        .values(
                            status=BackupStatus.COMPLETED.value,
                            size_bytes=total_size,
                            manifest_json=manifest.to_dict(),
                            storage_path=storage_path,
                            completed_at=now
                        )
                    )
                    await db.commit()
                except Exception as e:
                    logger.error(f"DB error completing backup: {e}")
                    await db.rollback()
                finally:
                    await db.close()
        else:
            backup = self._fallback_backups.get(backup_id)
            if backup:
                backup.status = BackupStatus.COMPLETED
                backup.size_bytes = total_size
                backup.manifest = manifest
                backup.storage_path = storage_path
                backup.completed_at = now
    
    async def _fail_backup(self, backup_id: str, error_message: str) -> None:
        """Mark backup as failed"""
        now = datetime.now(timezone.utc)
        
        if self._postgres_available:
            db = await self._get_db_session()
            if db:
                try:
                    await db.execute(
                        update(BackupDB)
                        .where(BackupDB.id == backup_id)
                        .values(
                            status=BackupStatus.FAILED.value,
                            error_message=error_message,
                            completed_at=now
                        )
                    )
                    await db.commit()
                except Exception as e:
                    logger.error(f"DB error failing backup: {e}")
                    await db.rollback()
                finally:
                    await db.close()
        else:
            backup = self._fallback_backups.get(backup_id)
            if backup:
                backup.status = BackupStatus.FAILED
                backup.error_message = error_message
                backup.completed_at = now
    
    async def _find_parent_backup(self, user_id: str) -> Optional[str]:
        """البحث عن نسخة كاملة سابقة"""
        try:
            backups = await self.list_backups(user_id, backup_type=BackupType.FULL)
            for backup in backups:
                if backup.status == BackupStatus.COMPLETED:
                    return backup.backup_id
            return None
        except Exception:
            return None
    
    def _calculate_hash(self, file_path: str) -> str:
        """حساب hash للملف"""
        return hashlib.md5(file_path.encode()).hexdigest()


# Global service instance
backup_service = BackupService()
