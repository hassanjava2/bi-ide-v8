"""
Service Models - نماذج الخدمات
SQLAlchemy models for notifications, backups, and training jobs
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Boolean, DateTime, Integer, BigInteger, Text, Float, JSON
from sqlalchemy.dialects.postgresql import JSONB

from core.database import Base


JSON_COMPAT = JSONB().with_variant(JSON(), "sqlite")


class NotificationDB(Base):
    """Notification database model"""
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(100), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    channel = Column(String(20), default="in_app", nullable=False)
    priority = Column(String(20), default="medium", nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)
    metadata_json = Column(JSON_COMPAT, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<NotificationDB(id={self.id}, user_id={self.user_id}, title={self.title})>"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict"""
        return self.metadata_json or {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata dict"""
        self.metadata_json = metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "notification_id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "channel": self.channel,
            "priority": self.priority,
            "is_read": self.is_read,
            "metadata": self.get_metadata(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
        }


class BackupDB(Base):
    """Backup database model"""
    __tablename__ = "backups"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    backup_type = Column(String(20), default="full", nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    size_bytes = Column(BigInteger, default=0, nullable=False)
    storage_path = Column(Text, nullable=True)
    manifest_json = Column(JSON_COMPAT, default=dict)
    metadata_json = Column(JSON_COMPAT, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<BackupDB(id={self.id}, user_id={self.user_id}, name={self.name})>"
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get manifest dict"""
        return self.manifest_json or {}
    
    def set_manifest(self, manifest: Dict[str, Any]):
        """Set manifest dict"""
        self.manifest_json = manifest
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict"""
        return self.metadata_json or {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata dict"""
        self.metadata_json = metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "backup_id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "backup_type": self.backup_type,
            "status": self.status,
            "size_bytes": self.size_bytes,
            "storage_path": self.storage_path,
            "manifest": self.get_manifest(),
            "metadata": self.get_metadata(),
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class BackupScheduleDB(Base):
    """Backup schedule database model"""
    __tablename__ = "backup_schedules"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    backup_type = Column(String(20), default="full", nullable=False)
    cron_expression = Column(String(100), nullable=False)
    retention_days = Column(Integer, default=30, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata_json = Column(JSON_COMPAT, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_run = Column(DateTime(timezone=True), nullable=True)
    next_run = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<BackupScheduleDB(id={self.id}, user_id={self.user_id}, name={self.name})>"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict"""
        return self.metadata_json or {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata dict"""
        self.metadata_json = metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "schedule_id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "backup_type": self.backup_type,
            "cron_expression": self.cron_expression,
            "retention_days": self.retention_days,
            "is_active": self.is_active,
            "metadata": self.get_metadata(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
        }


class TrainingJobDB(Base):
    """Training job database model"""
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    config_json = Column(JSON_COMPAT, default=dict)
    metrics_json = Column(JSON_COMPAT, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<TrainingJobDB(id={self.id}, job_id={self.job_id}, model_name={self.model_name})>"
    
    def get_config(self) -> Dict[str, Any]:
        """Get config dict"""
        return self.config_json or {}
    
    def set_config(self, config: Dict[str, Any]):
        """Set config dict"""
        self.config_json = config
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics dict"""
        return self.metrics_json or {}
    
    def set_metrics(self, metrics: Dict[str, Any]):
        """Set metrics dict"""
        self.metrics_json = metrics
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status,
            "config": self.get_config(),
            "metrics": self.get_metrics(),
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TrainedModelDB(Base):
    """Trained model database model"""
    __tablename__ = "trained_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    accuracy = Column(Float, nullable=True)
    is_deployed = Column(Boolean, default=False, nullable=False)
    metadata_json = Column(JSON_COMPAT, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<TrainedModelDB(id={self.id}, model_id={self.model_id}, name={self.name})>"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dict"""
        return self.metadata_json or {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata dict"""
        self.metadata_json = metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "accuracy": self.accuracy,
            "is_deployed": self.is_deployed,
            "metadata": self.get_metadata(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
