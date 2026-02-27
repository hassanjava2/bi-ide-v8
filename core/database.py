"""
Database Layer - طبقة قاعدة البيانات
Supports PostgreSQL and SQLite
"""
import asyncio
import os
import json
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# SQLAlchemy 2.0 style Base class
class Base(DeclarativeBase):
    pass

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./data/bi_ide.db"  # Default to SQLite for local dev
)

# Base is now defined above using SQLAlchemy 2.0 DeclarativeBase


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
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
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
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CouncilDiscussion(Base):
    """Council meeting discussions"""
    __tablename__ = "council_discussions"
    
    id = Column(String, primary_key=True)
    topic = Column(Text)
    wise_men_input = Column(JSON)  # Dict of wise_man -> opinion
    consensus_score = Column(Float)
    final_decision = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String, index=True)
    value = Column(Float)
    labels = Column(JSON)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DatabaseManager:
    """Database manager with async support"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None

        # Lifecycle coordination (prevents init/close races during tests)
        self._init_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._init_future: Optional[asyncio.Future] = None
        
    async def initialize(self):
        """Initialize database connection"""
        # Coalesce concurrent initialize() calls into one.
        created_by_me = False
        async with self._init_lock:
            if self.AsyncSessionLocal and self.async_engine and self.engine:
                return

            if self._init_future is None:
                loop = asyncio.get_running_loop()
                self._init_future = loop.create_future()
                created_by_me = True

            init_future = self._init_future

        if not created_by_me:
            # Someone else is initializing; wait for completion.
            await init_future
            return

        if init_future.done():
            return

        # Sync engine for migrations
        sync_url = self.database_url.replace("+aiosqlite", "").replace("+asyncpg", "")

        # SQLite in-memory is fragile with multiple connections (sync + async) and background startup tasks.
        # For tests/dev that set :memory:, use a per-process temporary file for stability.
        if sync_url.startswith("sqlite") and ":memory:" in sync_url:
            tmp_dir = Path(os.getenv("BI_IDE_TMP_DB_DIR") or tempfile.gettempdir())
            tmp_dir.mkdir(parents=True, exist_ok=True)
            db_path = tmp_dir / f"bi_ide_test_{os.getpid()}_{time.time_ns()}.db"
            try:
                if db_path.exists():
                    db_path.unlink()
            except Exception:
                pass
            sync_url = f"sqlite:///{db_path.as_posix()}"
            self.database_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

        db_connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

        def _sync_connect_args(url: str) -> Dict[str, Any]:
            if url.startswith("sqlite"):
                # Add timeout to prevent "database is locked" errors
                args: Dict[str, Any] = {
                    "check_same_thread": False,
                    "timeout": 30,  # 30 seconds timeout
                }
                return args
            if url.startswith("postgresql"):
                # psycopg2 respects connect_timeout (seconds)
                return {"connect_timeout": db_connect_timeout}
            return {}

        def _async_connect_args(url: str) -> Dict[str, Any]:
            if url.startswith("sqlite"):
                # Add timeout to prevent "database is locked" errors
                args: Dict[str, Any] = {
                    "check_same_thread": False,
                    "timeout": 30,  # 30 seconds timeout
                }
                return args
            if url.startswith("postgresql"):
                # asyncpg uses 'timeout' (seconds)
                return {"timeout": db_connect_timeout}
            return {}

        def _init_sync_engine_and_create_tables():
            # Ensure all tables are registered on Base.metadata before create_all.
            # Importing modules here avoids circular imports at module load time.
            try:
                import core.user_models  # noqa: F401
            except Exception:
                pass
            try:
                import community.models  # noqa: F401
            except Exception:
                pass
            try:
                import erp.models.database_models  # noqa: F401
            except Exception:
                pass

            self.engine = create_engine(
                sync_url,
                echo=False,
                connect_args=_sync_connect_args(sync_url),
                pool_pre_ping=True,
            )
            # NOTE: create_all may perform network I/O (e.g., Postgres) and can block.
            Base.metadata.create_all(self.engine)

        try:
            # Run potentially-blocking DB initialization off the event loop.
            await asyncio.to_thread(_init_sync_engine_and_create_tables)

            # Async engine for operations
            self.async_engine = create_async_engine(
                self.database_url,
                echo=False,
                connect_args=_async_connect_args(self.database_url),
                pool_pre_ping=True,
            )
            self.AsyncSessionLocal = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            print(f"Database initialized: {self.database_url}")

            if not init_future.done():
                init_future.set_result(True)
        except Exception as e:
            if not init_future.done():
                init_future.set_exception(e)
            # Allow a future re-attempt if init failed
            async with self._init_lock:
                if self._init_future is init_future:
                    self._init_future = None
            raise
    
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
                existing.updated_at = datetime.now(timezone.utc)
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
        async with self._close_lock:
            # If initialization is in-flight, wait for it to complete before disposing engines.
            init_future = self._init_future
            if init_future is not None and not init_future.done():
                try:
                    await init_future
                except Exception:
                    # Even if init failed, continue with best-effort cleanup.
                    pass

            if self.async_engine:
                await self.async_engine.dispose()
            if self.engine:
                self.engine.dispose()


# Global instance
db_manager = DatabaseManager()


# FastAPI dependency for database sessions
async def get_db() -> AsyncSession:
    """FastAPI dependency for database sessions"""
    async with db_manager.get_session() as session:
        yield session


# Alias for compatibility
get_async_session = get_db
