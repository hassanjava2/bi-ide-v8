"""
Database Layer - طبقة قاعدة البيانات
Supports PostgreSQL and SQLite
"""
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/bi_ide.db"  # Default to SQLite for local dev
)

Base = declarative_base()


# Models
class KnowledgeEntry(Base):
    """Knowledge base entries"""
    __tablename__ = "knowledge_entries"
    
    id = Column(String, primary_key=True)
    category = Column(String, index=True)
    content = Column(Text)
    embedding = Column(JSON)  # Vector embedding
    source = Column(String)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)


class LearningExperience(Base):
    """Learning experiences from observation"""
    __tablename__ = "learning_experiences"
    
    id = Column(String, primary_key=True)
    experience_type = Column(String, index=True)  # code, erp, decision, error
    context = Column(JSON)
    action = Column(Text)
    outcome = Column(Text)
    reward = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class CouncilDiscussion(Base):
    """Council meeting discussions"""
    __tablename__ = "council_discussions"
    
    id = Column(String, primary_key=True)
    topic = Column(Text)
    wise_men_input = Column(JSON)  # Dict of wise_man -> opinion
    consensus_score = Column(Float)
    final_decision = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String, index=True)
    value = Column(Float)
    labels = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database manager with async support"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
    async def initialize(self):
        """Initialize database connection"""
        # Sync engine for migrations
        sync_url = self.database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
        self.engine = create_engine(sync_url, echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Async engine for operations
        self.async_engine = create_async_engine(self.database_url, echo=False)
        self.AsyncSessionLocal = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        print(f"Database initialized: {self.database_url}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get async database session"""
        if not self.AsyncSessionLocal:
            await self.initialize()
        
        session = self.AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()
    
    async def store_knowledge(self, entry_id: str, category: str, content: str,
                             embedding: List[float] = None, source: str = None,
                             confidence: float = 0.0, metadata: Dict = None):
        """Store knowledge entry"""
        async with self.get_session() as session:
            from sqlalchemy import select
            
            # Check if exists
            result = await session.execute(
                select(KnowledgeEntry).where(KnowledgeEntry.id == entry_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update
                existing.content = content
                existing.embedding = embedding
                existing.confidence = confidence
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                entry = KnowledgeEntry(
                    id=entry_id,
                    category=category,
                    content=content,
                    embedding=embedding,
                    source=source,
                    confidence=confidence,
                    metadata_json=metadata or {}
                )
                session.add(entry)
    
    async def get_knowledge(self, category: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve knowledge entries"""
        async with self.get_session() as session:
            from sqlalchemy import select
            
            query = select(KnowledgeEntry)
            if category:
                query = query.where(KnowledgeEntry.category == category)
            query = query.order_by(KnowledgeEntry.created_at.desc()).limit(limit)
            
            result = await session.execute(query)
            entries = result.scalars().all()
            
            return [
                {
                    "id": e.id,
                    "category": e.category,
                    "content": e.content,
                    "confidence": e.confidence,
                    "created_at": e.created_at.isoformat()
                }
                for e in entries
            ]
    
    async def store_learning_experience(self, exp_id: str, exp_type: str,
                                       context: Dict, action: str,
                                       outcome: str, reward: float):
        """Store learning experience"""
        async with self.get_session() as session:
            exp = LearningExperience(
                id=exp_id,
                experience_type=exp_type,
                context=context,
                action=action,
                outcome=outcome,
                reward=reward
            )
            session.add(exp)
    
    async def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        async with self.get_session() as session:
            from sqlalchemy import func, select
            
            # Count by type
            result = await session.execute(
                select(LearningExperience.experience_type, func.count())
                .group_by(LearningExperience.experience_type)
            )
            type_counts = dict(result.all())
            
            # Average reward
            result = await session.execute(
                select(func.avg(LearningExperience.reward))
            )
            avg_reward = result.scalar() or 0.0
            
            return {
                "total_experiences": sum(type_counts.values()),
                "by_type": type_counts,
                "average_reward": round(avg_reward, 3)
            }
    
    async def store_metric(self, name: str, value: float, labels: Dict = None):
        """Store system metric"""
        async with self.get_session() as session:
            metric = SystemMetrics(
                metric_name=name,
                value=value,
                labels=labels or {}
            )
            session.add(metric)
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()


# Global instance
db_manager = DatabaseManager()
