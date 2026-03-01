"""
Database Models - SQLAlchemy ORM
Standardized models for BI-IDE v8 PostgreSQL schema
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# Enums
class UserStatus(str, PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class DecisionStatus(str, PyEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELIBERATING = "deliberating"
    NEEDS_REVIEW = "needs_review"


class TrainingStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(str, PyEnum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class WorkerStatus(str, PyEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    THROTTLED = "throttled"
    BUSY = "busy"


class AlertSeverity(str, PyEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================
# User Models
# ============================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    decisions = relationship("CouncilDecision", back_populates="proposer")
    training_jobs = relationship("TrainingJob", back_populates="creator")


class TokenBlacklist(Base):
    __tablename__ = "token_blacklist"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ============================================
# Council Models
# ============================================

class CouncilMember(Base):
    __tablename__ = "council_members"
    
    id = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    expertise = Column(ARRAY(String), default=list)
    is_active = Column(Boolean, default=True, index=True)
    current_focus = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    votes = relationship("CouncilVote", back_populates="member")


class CouncilDecision(Base):
    __tablename__ = "council_decisions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(50), unique=True, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    status = Column(String(20), default=DecisionStatus.PENDING.value, index=True)
    votes = Column(JSONB, default=dict)
    confidence = Column(Float, default=0.0)
    consensus_score = Column(Float, default=0.0)
    evidence = Column(JSONB, default=list)
    proposed_by = Column(Integer, ForeignKey("users.id"))
    decided_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    proposer = relationship("User", back_populates="decisions")
    vote_records = relationship("CouncilVote", back_populates="decision")
    
    __table_args__ = (
        Index('idx_council_decisions_status_created', 'status', 'created_at'),
    )


class CouncilVote(Base):
    __tablename__ = "council_votes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(50), ForeignKey("council_decisions.decision_id", ondelete="CASCADE"))
    member_id = Column(String(10), ForeignKey("council_members.id"))
    vote = Column(String(20), nullable=False)  # approve, reject, abstain
    comment = Column(Text)
    voted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    decision = relationship("CouncilDecision", back_populates="vote_records")
    member = relationship("CouncilMember", back_populates="votes")
    
    __table_args__ = (
        UniqueConstraint('decision_id', 'member_id', name='unique_decision_member_vote'),
        Index('idx_council_votes_decision', 'decision_id'),
    )


# ============================================
# Training Models
# ============================================

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(String(50), primary_key=True)
    job_name = Column(String(200), nullable=False)
    model_name = Column(String(100))
    status = Column(String(20), default=TrainingStatus.PENDING.value, index=True)
    config = Column(JSONB, default=dict)
    metrics = Column(JSONB, default=dict)
    error_message = Column(Text)
    assigned_worker = Column(String(50), ForeignKey("workers.id"))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    worker = relationship("Worker", back_populates="jobs", foreign_keys=[assigned_worker])
    creator = relationship("User", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job", uselist=False)
    
    __table_args__ = (
        Index('idx_training_jobs_status', 'status'),
        Index('idx_training_jobs_created', 'created_at'),
    )


class Worker(Base):
    __tablename__ = "workers"
    
    id = Column(String(50), primary_key=True)
    hostname = Column(String(100), nullable=False)
    status = Column(String(20), default=WorkerStatus.OFFLINE.value, index=True)
    labels = Column(ARRAY(String), default=list)
    hardware = Column(JSONB, default=dict)
    resources = Column(JSONB, default=dict)
    current_job_id = Column(String(50), ForeignKey("training_jobs.id"))
    last_heartbeat = Column(DateTime(timezone=True))
    registered_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    jobs = relationship("TrainingJob", back_populates="worker", foreign_keys="TrainingJob.assigned_worker")
    current_job = relationship("TrainingJob", foreign_keys=[current_job_id])
    
    __table_args__ = (
        Index('idx_workers_status', 'status'),
        Index('idx_workers_labels', 'labels', postgresql_using='gin'),
    )


class Model(Base):
    __tablename__ = "models"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), default="1.0.0")
    status = Column(String(20), default=ModelStatus.TRAINING.value, index=True)
    architecture = Column(String(50))
    parameters_count = Column(Integer)
    dataset_size = Column(Integer)
    trained_epochs = Column(Integer, default=0)
    job_id = Column(String(50), ForeignKey("training_jobs.id"))
    metrics = Column(JSONB, default=dict)
    storage_path = Column(Text)
    is_deployed = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="model")
    
    __table_args__ = (
        Index('idx_models_status', 'status'),
        Index('idx_models_deployed', 'is_deployed'),
    )


# ============================================
# Monitoring Models
# ============================================

class SystemMetric(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    labels = Column(JSONB, default=dict)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index('idx_system_metrics_name_time', 'metric_name', 'timestamp'),
    )


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    source = Column(String(50))
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index('idx_alerts_severity', 'severity'),
        Index('idx_alerts_resolved', 'is_resolved'),
    )


# ============================================
# Brain System Models
# ============================================

class BrainSchedule(Base):
    __tablename__ = "brain_schedules"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_name = Column(String(100), nullable=False)
    layer_name = Column(String(50))
    priority = Column(Integer, default=5)
    config = Column(JSONB, default=dict)
    cron_expression = Column(String(100))
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime(timezone=True))
    next_run = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BrainEvaluation(Base):
    __tablename__ = "brain_evaluations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), ForeignKey("models.id"))
    job_id = Column(String(50), ForeignKey("training_jobs.id"))
    evaluation_type = Column(String(50))  # pre_deploy, periodic, benchmark
    metrics = Column(JSONB, default=dict)
    passed_threshold = Column(Boolean, default=False)
    improvement_delta = Column(Float)
    evaluated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    model = relationship("Model")
    job = relationship("TrainingJob")


# ============================================
# Knowledge Base Models
# ============================================

class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"
    
    id = Column(String(50), primary_key=True)
    category = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    # Note: embedding column requires pgvector extension
    # embedding = Column(Vector(1536))
    source = Column(String(100))
    confidence = Column(Float, default=0.0)
    metadata_json = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index('idx_knowledge_category', 'category'),
    )


# ============================================
# Data Pipeline Models
# ============================================

class DataCleaningRun(Base):
    __tablename__ = "data_cleaning_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False)
    records_in = Column(Integer, nullable=False)
    records_out = Column(Integer, nullable=False)
    duplicates_removed = Column(Integer, default=0)
    noise_removed = Column(Integer, default=0)
    validation_errors = Column(JSONB, default=list)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
