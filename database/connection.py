"""
Database Connection Module
PostgreSQL connection pool and session management for BI-IDE v8
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text

# Import models to ensure they're registered
from .models import Base


# Database URL configuration
DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/bi_ide"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

# Convert sync URL to async if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


class DatabaseConnection:
    """
    Manages PostgreSQL database connections with connection pooling.
    
    Usage:
        db = DatabaseConnection()
        await db.initialize()
        
        async with db.get_session() as session:
            result = await session.execute(...)
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session_maker: Optional[async_sessionmaker] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        if self._initialized:
            return
            
        async with self._init_lock:
            if self._initialized:
                return
                
            # Connection pool settings
            pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
            max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
            pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
            
            self._async_engine = create_async_engine(
                self.database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
            )
            
            self._async_session_maker = async_sessionmaker(
                self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
            
            self._initialized = True
            print(f"✅ Database connection pool initialized (pool_size={pool_size})")
    
    async def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        if not self._initialized:
            await self.initialize()
            
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables verified")
    
    async def drop_tables(self) -> None:
        """Drop all tables (use with caution!)."""
        if not self._initialized:
            await self.initialize()
            
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("⚠️ Database tables dropped")
    
    async def health_check(self) -> dict:
        """Check database health and return status."""
        if not self._initialized:
            return {"status": "not_initialized", "healthy": False}
        
        try:
            async with self.get_session() as session:
                start_time = asyncio.get_event_loop().time()
                await session.execute(text("SELECT 1"))
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                
                return {
                    "status": "healthy",
                    "healthy": True,
                    "latency_ms": round(latency, 2),
                    "pool": {
                        "size": self._async_engine.pool.size(),
                        "checked_in": self._async_engine.pool.checkedin(),
                        "checked_out": self._async_engine.pool.checkedout(),
                    }
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic commit/rollback."""
        if not self._initialized:
            await self.initialize()
            
        session = self._async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()
    
    async def close(self) -> None:
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._initialized = False
            print("✅ Database connection pool closed")
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the async engine (for advanced usage)."""
        return self._async_engine
    
    @property
    def is_initialized(self) -> bool:
        """Check if the database is initialized."""
        return self._initialized


# Global instance
db_connection = DatabaseConnection()


# FastAPI dependency
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with db_connection.get_session() as session:
        yield session


# Utility functions
async def init_database(database_url: str = None) -> DatabaseConnection:
    """
    Initialize the database with tables.
    
    Usage:
        db = await init_database()
    """
    db = DatabaseConnection(database_url)
    await db.initialize()
    await db.create_tables()
    return db


async def close_database() -> None:
    """Close the global database connection."""
    await db_connection.close()
