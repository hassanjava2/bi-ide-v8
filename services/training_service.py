"""
خدمة إدارة مهام التدريب - Training Service V2
PostgreSQL-based persistent storage replacing in-memory _jobs and _models
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update

# Try importing database models
try:
    from core.database import get_async_session
    from core.service_models import TrainingJobDB, TrainedModelDB
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """حالات التدريب المختلفة"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """نموذج مهمة التدريب (in-memory representation)"""
    job_id: str
    model_name: str
    status: TrainingStatus
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "config": self.config,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_db(cls, db_job: TrainingJobDB) -> 'TrainingJob':
        """Create from database model"""
        return cls(
            job_id=db_job.job_id,
            model_name=db_job.model_name,
            status=TrainingStatus(db_job.status),
            created_at=db_job.created_at,
            updated_at=db_job.updated_at,
            config=db_job.get_config(),
            metrics=db_job.get_metrics(),
            error_message=db_job.error_message,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at
        )


@dataclass
class ModelInfo:
    """نموذج معلومات النموذج (in-memory representation)"""
    model_id: str
    name: str
    version: str
    created_at: datetime
    accuracy: float
    is_deployed: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "accuracy": self.accuracy,
            "is_deployed": self.is_deployed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_db(cls, db_model: TrainedModelDB) -> 'ModelInfo':
        """Create from database model"""
        return cls(
            model_id=db_model.model_id,
            name=db_model.name,
            version=db_model.version,
            created_at=db_model.created_at,
            accuracy=db_model.accuracy or 0.0,
            is_deployed=db_model.is_deployed,
            metadata=db_model.get_metadata()
        )


class TrainingService:
    """
    خدمة إدارة مهام التدريب V2 - PostgreSQL Persistent Storage
    """
    
    def __init__(self):
        """تهيئة خدمة التدريب"""
        # Running tasks remain in-memory (they're ephemeral)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # PostgreSQL availability
        self._postgres_available = POSTGRES_AVAILABLE
        
        # Fallback in-memory storage if PostgreSQL not available
        self._fallback_jobs: Dict[str, TrainingJob] = {}
        self._fallback_models: Dict[str, ModelInfo] = {}
        
        if self._postgres_available:
            logger.info("✅ TrainingService V2 initialized (PostgreSQL)")
        else:
            logger.warning("⚠️ TrainingService V2 initialized (In-Memory Fallback)")
    
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
    
    async def start_training(
        self,
        job_id: str,
        model_name: str,
        config: Dict[str, Any]
    ) -> TrainingJob:
        """
        بدء مهمة تدريب جديدة - Persisted to PostgreSQL
        """
        try:
            if await self.get_job(job_id):
                raise ValueError(f"المهمة {job_id} موجودة مسبقاً")
            
            now = datetime.now(timezone.utc)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        db_job = TrainingJobDB(
                            job_id=job_id,
                            model_name=model_name,
                            status=TrainingStatus.PENDING.value,
                            config_json=config,
                            metrics_json={},
                            created_at=now,
                            updated_at=now
                        )
                        db.add(db_job)
                        await db.commit()
                        logger.info(f"✅ Training job stored in PostgreSQL: {job_id}")
                    except Exception as e:
                        logger.error(f"DB error creating training job: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
                else:
                    self._fallback_store_job(job_id, model_name, config, now)
            else:
                self._fallback_store_job(job_id, model_name, config, now)
            
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                status=TrainingStatus.PENDING,
                created_at=now,
                updated_at=now,
                config=config,
                metrics={}
            )
            
            return job
            
        except Exception as e:
            logger.error(f"خطأ في بدء التدريب: {e}")
            raise
    
    def _fallback_store_job(self, job_id: str, model_name: str, config: Dict[str, Any], now: datetime) -> None:
        """Store job in fallback in-memory storage"""
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            status=TrainingStatus.PENDING,
            created_at=now,
            updated_at=now,
            config=config,
            metrics={}
        )
        self._fallback_jobs[job_id] = job
    
    async def update_job_status(
        self,
        job_id: str,
        status: TrainingStatus,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        تحديث حالة المهمة - Update PostgreSQL
        """
        try:
            now = datetime.now(timezone.utc)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        update_values = {
                            "status": status.value,
                            "updated_at": now
                        }
                        
                        if metrics:
                            update_values["metrics_json"] = metrics
                        if error_message:
                            update_values["error_message"] = error_message
                        if status == TrainingStatus.RUNNING:
                            update_values["started_at"] = now
                        if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                            update_values["completed_at"] = now
                        
                        await db.execute(
                            update(TrainingJobDB)
                            .where(TrainingJobDB.job_id == job_id)
                            .values(**update_values)
                        )
                        await db.commit()
                        logger.info(f"✅ Updated job {job_id} status to {status.value}")
                        return True
                    except Exception as e:
                        logger.error(f"DB error updating job status: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_update_job_status(job_id, status, metrics, error_message, now)
            
        except Exception as e:
            logger.error(f"خطأ في تحديث حالة المهمة: {e}")
            return False
    
    def _fallback_update_job_status(
        self, job_id: str, status: TrainingStatus,
        metrics: Optional[Dict[str, Any]], error_message: Optional[str], now: datetime
    ) -> bool:
        """Fallback: Update job status in in-memory storage"""
        job = self._fallback_jobs.get(job_id)
        if not job:
            return False
        
        job.status = status
        job.updated_at = now
        if metrics:
            job.metrics = metrics
        if error_message:
            job.error_message = error_message
        if status == TrainingStatus.RUNNING:
            job.started_at = now
        if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
            job.completed_at = now
        
        return True
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        الحصول على معلومات مهمة - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(TrainingJobDB).where(TrainingJobDB.job_id == job_id)
                        )
                        db_job = result.scalar_one_or_none()
                        if db_job:
                            return TrainingJob.from_db(db_job)
                    except Exception as e:
                        logger.error(f"DB error fetching job: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_jobs.get(job_id)
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات المهمة: {e}")
            return None
    
    async def get_all_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        limit: int = 100
    ) -> List[TrainingJob]:
        """
        الحصول على قائمة المهام - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        query = select(TrainingJobDB).order_by(desc(TrainingJobDB.created_at)).limit(limit)
                        
                        if status:
                            query = query.where(TrainingJobDB.status == status.value)
                        
                        result = await db.execute(query)
                        db_jobs = result.scalars().all()
                        
                        return [TrainingJob.from_db(j) for j in db_jobs]
                    except Exception as e:
                        logger.error(f"DB error fetching jobs: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            jobs = list(self._fallback_jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"خطأ في جلب قائمة المهام: {e}")
            return []
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        إلغاء مهمة - Update PostgreSQL
        """
        # Cancel running task if any
        if job_id in self._running_tasks:
            task = self._running_tasks[job_id]
            task.cancel()
            del self._running_tasks[job_id]
        
        # Update status
        return await self.update_job_status(job_id, TrainingStatus.CANCELLED)
    
    async def save_model(
        self,
        model_id: str,
        name: str,
        version: str,
        accuracy: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        حفظ معلومات النموذج - Persisted to PostgreSQL
        """
        try:
            now = datetime.now(timezone.utc)
            
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        # Check if model exists
                        result = await db.execute(
                            select(TrainedModelDB).where(TrainedModelDB.model_id == model_id)
                        )
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            # Update existing
                            existing.name = name
                            existing.version = version
                            existing.accuracy = accuracy
                            existing.set_metadata(metadata or {})
                        else:
                            # Create new
                            db_model = TrainedModelDB(
                                model_id=model_id,
                                name=name,
                                version=version,
                                accuracy=accuracy,
                                is_deployed=False,
                                metadata_json=metadata or {},
                                created_at=now
                            )
                            db.add(db_model)
                        
                        await db.commit()
                        logger.info(f"✅ Model saved to PostgreSQL: {model_id}")
                    except Exception as e:
                        logger.error(f"DB error saving model: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
                else:
                    self._fallback_store_model(model_id, name, version, accuracy, metadata, now)
            else:
                self._fallback_store_model(model_id, name, version, accuracy, metadata, now)
            
            model = ModelInfo(
                model_id=model_id,
                name=name,
                version=version,
                created_at=now,
                accuracy=accuracy,
                is_deployed=False,
                metadata=metadata or {}
            )
            
            return model
            
        except Exception as e:
            logger.error(f"خطأ في حفظ النموذج: {e}")
            raise
    
    def _fallback_store_model(
        self, model_id: str, name: str, version: str,
        accuracy: float, metadata: Optional[Dict[str, Any]], now: datetime
    ) -> None:
        """Store model in fallback in-memory storage"""
        model = ModelInfo(
            model_id=model_id,
            name=name,
            version=version,
            created_at=now,
            accuracy=accuracy,
            is_deployed=False,
            metadata=metadata or {}
        )
        self._fallback_models[model_id] = model
    
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        الحصول على معلومات النموذج - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(TrainedModelDB).where(TrainedModelDB.model_id == model_id)
                        )
                        db_model = result.scalar_one_or_none()
                        if db_model:
                            return ModelInfo.from_db(db_model)
                    except Exception as e:
                        logger.error(f"DB error fetching model: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            return self._fallback_models.get(model_id)
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات النموذج: {e}")
            return None
    
    async def get_all_models(
        self,
        deployed_only: bool = False,
        limit: int = 100
    ) -> List[ModelInfo]:
        """
        الحصول على قائمة النماذج - From PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        query = select(TrainedModelDB).order_by(desc(TrainedModelDB.created_at)).limit(limit)
                        
                        if deployed_only:
                            query = query.where(TrainedModelDB.is_deployed == True)
                        
                        result = await db.execute(query)
                        db_models = result.scalars().all()
                        
                        return [ModelInfo.from_db(m) for m in db_models]
                    except Exception as e:
                        logger.error(f"DB error fetching models: {e}")
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            models = list(self._fallback_models.values())
            if deployed_only:
                models = [m for m in models if m.is_deployed]
            return sorted(models, key=lambda m: m.created_at, reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"خطأ في جلب قائمة النماذج: {e}")
            return []
    
    async def deploy_model(self, model_id: str) -> bool:
        """
        نشر نموذج - Update PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        await db.execute(
                            update(TrainedModelDB)
                            .where(TrainedModelDB.model_id == model_id)
                            .values(is_deployed=True)
                        )
                        await db.commit()
                        logger.info(f"✅ Model deployed: {model_id}")
                        return True
                    except Exception as e:
                        logger.error(f"DB error deploying model: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            model = self._fallback_models.get(model_id)
            if model:
                model.is_deployed = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"خطأ في نشر النموذج: {e}")
            return False
    
    async def undeploy_model(self, model_id: str) -> bool:
        """
        إلغاء نشر نموذج - Update PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        await db.execute(
                            update(TrainedModelDB)
                            .where(TrainedModelDB.model_id == model_id)
                            .values(is_deployed=False)
                        )
                        await db.commit()
                        logger.info(f"✅ Model undeployed: {model_id}")
                        return True
                    except Exception as e:
                        logger.error(f"DB error undeploying model: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            model = self._fallback_models.get(model_id)
            if model:
                model.is_deployed = False
                return True
            return False
            
        except Exception as e:
            logger.error(f"خطأ في إلغاء نشر النموذج: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """
        حذف نموذج - Delete from PostgreSQL
        """
        try:
            if self._postgres_available:
                db = await self._get_db_session()
                if db:
                    try:
                        result = await db.execute(
                            select(TrainedModelDB).where(TrainedModelDB.model_id == model_id)
                        )
                        db_model = result.scalar_one_or_none()
                        
                        if db_model:
                            await db.delete(db_model)
                            await db.commit()
                            logger.info(f"✅ Model deleted: {model_id}")
                            return True
                        return False
                    except Exception as e:
                        logger.error(f"DB error deleting model: {e}")
                        await db.rollback()
                    finally:
                        await db.close()
            
            # Fallback to in-memory
            if model_id in self._fallback_models:
                del self._fallback_models[model_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"خطأ في حذف النموذج: {e}")
            return False


# Global service instance
training_service = TrainingService()
