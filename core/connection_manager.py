"""
Connection Manager - مدير الاتصالات
Manage all database and cache connections
إدارة جميع اتصالات قاعدة البيانات والتخزين المؤقت
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class ConnectionStats:
    """إحصائيات الاتصال - Connection statistics"""
    
    def __init__(self):
        self.created_at = datetime.now(timezone.utc)
        self.last_check: Optional[datetime] = None
        self.check_count = 0
        self.failure_count = 0
        self.reconnect_count = 0
        self.total_queries = 0
        self.active_connections = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "created_at": self.created_at.isoformat(),
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "check_count": self.check_count,
            "failure_count": self.failure_count,
            "reconnect_count": self.reconnect_count,
            "total_queries": self.total_queries,
            "active_connections": self.active_connections,
            "uptime_seconds": (datetime.now(timezone.utc) - self.created_at).total_seconds()
        }


class DatabaseConnectionPool:
    """
    مدير تجمع اتصالات قاعدة البيانات
    Database connection pool manager
    """
    
    def __init__(
        self,
        database_url: str = None,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://bi_ide:password@localhost:5432/bi_ide"
        )
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", pool_size))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", max_overflow))
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        self.engine = None
        self.session_factory = None
        self.stats = ConnectionStats()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the database connection pool"""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # إنشاء محرك قاعدة البيانات - Create database engine
                self.engine = create_async_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_timeout=self.pool_timeout,
                    pool_recycle=self.pool_recycle,
                    pool_pre_ping=True,  # التحقق من الاتصال قبل الاستخدام
                    echo=self.echo,
                )
                
                # إنشاء مصنع الجلسات - Create session factory
                self.session_factory = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autocommit=False,
                    autoflush=False
                )
                
                # التحقق من الاتصال - Verify connection
                await self._verify_connection()
                
                self._initialized = True
                self.stats.created_at = datetime.now(timezone.utc)
                
                # بدء مهمة التحقق الصحي - Start health check task
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
                
                logger.info(
                    f"✅ Database pool initialized | "
                    f"حجم التجمع: {self.pool_size}, "
                    f"الحد الأقصى: {self.max_overflow}"
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize database pool: {e}")
                raise
                
    async def _verify_connection(self) -> None:
        """التحقق من صحة الاتصال - Verify connection health"""
        async with self.engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.scalar()
            
    async def _health_check_loop(self) -> None:
        """حلقة التحقق الصحي المستمرة - Continuous health check loop"""
        while True:
            try:
                await asyncio.sleep(30)  # فحص كل 30 ثانية
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ Health check error: {e}")
                
    async def health_check(self) -> Tuple[bool, str]:
        """
        فحص صحة الاتصال
        Check connection health
        """
        self.stats.check_count += 1
        self.stats.last_check = datetime.now(timezone.utc)
        
        try:
            start_time = time.time()
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.scalar()
                latency_ms = (time.time() - start_time) * 1000
                
            # تحديث الإحصائيات - Update stats
            pool_status = self.engine.pool.status()
            self.stats.active_connections = pool_status.checkedin + pool_status.checkedout
            
            logger.debug(
                f"💚 DB health OK | latency: {latency_ms:.1f}ms | "
                f"connections: {self.stats.active_connections}"
            )
            return True, f"healthy (latency: {latency_ms:.1f}ms)"
            
        except Exception as e:
            self.stats.failure_count += 1
            logger.error(f"❌ Database health check failed: {e}")
            
            # محاولة إعادة الاتصال - Try to reconnect
            await self._try_reconnect()
            return False, str(e)
            
    async def _try_reconnect(self) -> bool:
        """محاولة إعادة الاتصال - Attempt to reconnect"""
        try:
            logger.info("🔄 Attempting database reconnection...")
            await self.close()
            await asyncio.sleep(1)
            await self.initialize()
            self.stats.reconnect_count += 1
            logger.info("✅ Database reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database reconnection failed: {e}")
            return False
            
    @asynccontextmanager
    async def get_session(self):
        """
        الحصول على جلسة قاعدة بيانات
        Get database session context manager
        """
        if not self._initialized:
            await self.initialize()
            
        session = self.session_factory()
        start_time = time.time()
        
        try:
            yield session
            await session.commit()
            self.stats.total_queries += 1
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Database session error: {e}")
            raise
            
        finally:
            await session.close()
            
    async def close(self) -> None:
        """إغلاق جميع الاتصالات - Close all connections"""
        async with self._lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                    
            if self.engine:
                await self.engine.dispose()
                
            self._initialized = False
            logger.info("📴 Database pool closed")


class RedisConnectionPool:
    """
    مدير تجمع اتصالات Redis
    Redis connection pool manager
    """
    
    def __init__(
        self,
        redis_url: str = None,
        max_connections: int = 100,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        
        self.client: Optional[aioredis.Redis] = None
        self.stats = ConnectionStats()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # إنشاء عميل Redis - Create Redis client
                self.client = aioredis.from_url(
                    self.redis_url,
                    max_connections=self.max_connections,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    health_check_interval=self.health_check_interval,
                    decode_responses=True
                )
                
                # التحقق من الاتصال - Verify connection
                await self.client.ping()
                
                self._initialized = True
                self.stats.created_at = datetime.now(timezone.utc)
                
                # بدء مهمة التحقق الصحي - Start health check task
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
                
                logger.info(
                    f"✅ Redis pool initialized | "
                    f"URL: {self.redis_url.split('@')[-1]}, "
                    f"max connections: {self.max_connections}"
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Redis pool: {e}")
                raise
                
    async def _health_check_loop(self) -> None:
        """حلقة التحقق الصحي المستمرة - Continuous health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ Redis health check error: {e}")
                
    async def health_check(self) -> Tuple[bool, str]:
        """
        فحص صحة اتصال Redis
        Check Redis connection health
        """
        self.stats.check_count += 1
        self.stats.last_check = datetime.now(timezone.utc)
        
        try:
            start_time = time.time()
            await self.client.ping()
            latency_ms = (time.time() - start_time) * 1000
            
            # الحصول على معلومات الخادم - Get server info
            info = await self.client.info()
            used_memory = info.get('used_memory_human', 'N/A')
            connected_clients = info.get('connected_clients', 0)
            
            self.stats.active_connections = connected_clients
            
            logger.debug(
                f"💚 Redis health OK | latency: {latency_ms:.1f}ms | "
                f"memory: {used_memory} | clients: {connected_clients}"
            )
            return True, f"healthy (latency: {latency_ms:.1f}ms, memory: {used_memory})"
            
        except Exception as e:
            self.stats.failure_count += 1
            logger.error(f"❌ Redis health check failed: {e}")
            
            # محاولة إعادة الاتصال - Try to reconnect
            await self._try_reconnect()
            return False, str(e)
            
    async def _try_reconnect(self) -> bool:
        """محاولة إعادة الاتصال - Attempt to reconnect"""
        try:
            logger.info("🔄 Attempting Redis reconnection...")
            await self.close()
            await asyncio.sleep(1)
            await self.initialize()
            self.stats.reconnect_count += 1
            logger.info("✅ Redis reconnected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Redis reconnection failed: {e}")
            return False
            
    async def get_client(self) -> aioredis.Redis:
        """الحصول على عميل Redis - Get Redis client"""
        if not self._initialized:
            await self.initialize()
        return self.client
        
    async def close(self) -> None:
        """إغلاق اتصال Redis - Close Redis connection"""
        async with self._lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                    
            if self.client:
                await self.client.close()
                
            self._initialized = False
            logger.info("📴 Redis pool closed")


class ConnectionManager:
    """
    مدير الاتصالات الرئيسي
    Main connection manager for all services
    """
    
    def __init__(self):
        self.db_pool = DatabaseConnectionPool()
        self.redis_pool = RedisConnectionPool()
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize all connection pools"""
        logger.info("🚀 Initializing connection manager...")
        
        try:
            await self.db_pool.initialize()
        except Exception as e:
            logger.warning(f"⚠️ Database initialization skipped: {e}")
            
        try:
            await self.redis_pool.initialize()
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization skipped: {e}")
            
        logger.info("✅ Connection manager initialized")
        
    async def health_check(self) -> Dict[str, Any]:
        """
        فحص صحة جميع الاتصالات
        Check health of all connections
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": {"healthy": False, "message": "not initialized"},
            "redis": {"healthy": False, "message": "not initialized"},
            "overall_healthy": True
        }
        
        # فحص قاعدة البيانات - Check database
        if self.db_pool._initialized:
            healthy, message = await self.db_pool.health_check()
            results["database"] = {"healthy": healthy, "message": message}
            if not healthy:
                results["overall_healthy"] = False
                
        # فحص Redis - Check Redis
        if self.redis_pool._initialized:
            healthy, message = await self.redis_pool.health_check()
            results["redis"] = {"healthy": healthy, "message": message}
            if not healthy:
                results["overall_healthy"] = False
                
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات جميع الاتصالات - Get all connection stats"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": self.db_pool.stats.to_dict() if self.db_pool._initialized else None,
            "redis": self.redis_pool.stats.to_dict() if self.redis_pool._initialized else None
        }
        
    async def wait_for_shutdown(self) -> None:
        """الانتظار حتى إيقاف التشغيل - Wait for shutdown signal"""
        await self._shutdown_event.wait()
        
    async def shutdown(self) -> None:
        """إيقاف تشغيل أنيق - Graceful shutdown"""
        logger.info("🛑 Initiating graceful shutdown...")
        self._shutdown_event.set()
        
        await self.db_pool.close()
        await self.redis_pool.close()
        
        logger.info("✅ All connections closed gracefully")


# نسخة عامة واحدة - Global singleton instance
connection_manager = ConnectionManager()


# دوال مساعدة - Helper functions
async def get_db_session():
    """الحصول على جلسة قاعدة بيانات - Get database session"""
    return connection_manager.db_pool.get_session()


async def get_redis_client() -> aioredis.Redis:
    """الحصول على عميل Redis - Get Redis client"""
    return await connection_manager.redis_pool.get_client()


async def check_all_connections() -> Dict[str, Any]:
    """التحقق من جميع الاتصالات - Check all connections"""
    return await connection_manager.health_check()
