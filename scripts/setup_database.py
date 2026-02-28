"""
Database Setup - إعداد قاعدة البيانات
Verify and setup database connections
"""

import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

import aioredis
import asyncpg
import sqlalchemy
from sqlalchemy import text, create_engine
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStatus:
    """حالة الاتصال - Connection status"""
    component: str
    is_connected: bool
    latency_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetupReport:
    """تقرير الإعداد - Setup report"""
    success: bool
    timestamp: float
    statuses: List[ConnectionStatus]
    errors: List[str]
    recommendations: List[str]


class DatabaseSetup:
    """
    إعداد قاعدة البيانات
    Database Setup
    
    يتحقق من الاتصالات ويعمل ترحيلات Alembic ويختبر Redis
    Verifies connections, runs Alembic migrations, and tests Redis
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        alembic_ini: str = "alembic.ini"
    ):
        """
        تهيئة إعداد قاعدة البيانات
        Initialize database setup
        
        Args:
            database_url: رابط قاعدة البيانات
            redis_url: رابط Redis
            alembic_ini: مسار ملف alembic.ini
        """
        self.database_url = database_url or self._get_database_url_from_env()
        self.redis_url = redis_url
        self.alembic_ini = alembic_ini
        self._statuses: List[ConnectionStatus] = []
        self._errors: List[str] = []
        self._recommendations: List[str] = []
    
    def _get_database_url_from_env(self) -> str:
        """الحصول على رابط قاعدة البيانات من متغيرات البيئة"""
        import os
        return os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/bi_ide'
        )
    
    async def run_setup(self) -> SetupReport:
        """
        تشغيل الإعداد الكامل
        Run full setup
        
        Returns:
            SetupReport: تقرير الإعداد
        """
        logger.info("Starting database setup...")
        self._statuses = []
        self._errors = []
        self._recommendations = []
        
        start_time = time.time()
        
        # التحقق من PostgreSQL
        await self._check_postgresql()
        
        # تشغيل ترحيلات Alembic
        await self._run_alembic_migrations()
        
        # التحقق من Redis
        await self._check_redis()
        
        # اختبار وظيفة الذاكرة المؤقتة
        await self._test_cache()
        
        # إنشاء الجداول إذا لم تكن موجودة
        await self._create_tables_if_not_exist()
        
        success = all(s.is_connected for s in self._statuses)
        
        report = SetupReport(
            success=success,
            timestamp=time.time(),
            statuses=self._statuses,
            errors=self._errors,
            recommendations=self._recommendations
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Database setup completed in {elapsed_ms:.2f}ms")
        
        return report
    
    async def _check_postgresql(self) -> None:
        """التحقق من اتصال PostgreSQL - Check PostgreSQL connection"""
        logger.info("Checking PostgreSQL connection...")
        start_time = time.time()
        
        try:
            # محاولة الاتصال باستخدام asyncpg
            conn = await asyncpg.connect(self.database_url, timeout=10)
            
            # تنفيذ استعلام بسيط
            result = await conn.fetchval("SELECT version()")
            await conn.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # تحليل معلومات الإصدار
            version_info = result.split()[1] if result else "unknown"
            
            self._statuses.append(ConnectionStatus(
                component="PostgreSQL",
                is_connected=True,
                latency_ms=latency_ms,
                message=f"Connected successfully (PostgreSQL {version_info})",
                details={'version': result}
            ))
            
            logger.info(f"PostgreSQL connection successful ({latency_ms:.2f}ms)")
            
        except asyncpg.PostgresError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="PostgreSQL",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"PostgreSQL error: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"PostgreSQL: {str(e)}")
            self._recommendations.append(
                "تأكد من تشغيل خادم PostgreSQL وصحة بيانات الاعتماد"
            )
            logger.error(f"PostgreSQL connection failed: {e}")
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="PostgreSQL",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Connection failed: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"PostgreSQL: {str(e)}")
            logger.error(f"PostgreSQL connection failed: {e}")
    
    async def _run_alembic_migrations(self) -> None:
        """تشغيل ترحيلات Alembic - Run Alembic migrations"""
        logger.info("Running Alembic migrations...")
        start_time = time.time()
        
        try:
            # تشغيل Alembic upgrade في عملية فرعية
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "alembic", "upgrade", "head",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/Users/bi/Documents/bi-ide-v8"
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if process.returncode == 0:
                self._statuses.append(ConnectionStatus(
                    component="Alembic Migrations",
                    is_connected=True,
                    latency_ms=latency_ms,
                    message="Migrations completed successfully",
                    details={'output': stdout.decode()[:500]}
                ))
                logger.info("Alembic migrations completed successfully")
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self._statuses.append(ConnectionStatus(
                    component="Alembic Migrations",
                    is_connected=False,
                    latency_ms=latency_ms,
                    message=f"Migration failed: {error_msg[:200]}",
                    details={'returncode': process.returncode}
                ))
                self._errors.append(f"Alembic: {error_msg[:200]}")
                logger.error(f"Alembic migrations failed: {error_msg}")
                
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Alembic Migrations",
                is_connected=False,
                latency_ms=latency_ms,
                message="Migration timeout (60s)",
                details={}
            ))
            self._errors.append("Alembic: Migration timeout")
            logger.error("Alembic migrations timed out")
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Alembic Migrations",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Migration error: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"Alembic: {str(e)}")
            logger.error(f"Alembic migrations error: {e}")
    
    async def _check_redis(self) -> None:
        """التحقق من اتصال Redis - Verify Redis connection"""
        logger.info("Checking Redis connection...")
        start_time = time.time()
        
        try:
            # إنشاء اتصال Redis
            redis = aioredis.from_url(
                self.redis_url,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # اختبار الاتصال بـ ping
            pong = await redis.ping()
            
            # الحصول على معلومات الخادم
            info = await redis.info()
            
            await redis.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            if pong:
                version = info.get('redis_version', 'unknown')
                self._statuses.append(ConnectionStatus(
                    component="Redis",
                    is_connected=True,
                    latency_ms=latency_ms,
                    message=f"Connected successfully (Redis {version})",
                    details={
                        'version': version,
                        'used_memory': info.get('used_memory_human', 'N/A'),
                        'connected_clients': info.get('connected_clients', 0)
                    }
                ))
                logger.info(f"Redis connection successful ({latency_ms:.2f}ms)")
            else:
                self._statuses.append(ConnectionStatus(
                    component="Redis",
                    is_connected=False,
                    latency_ms=latency_ms,
                    message="Redis ping failed",
                    details={}
                ))
                self._errors.append("Redis: Ping failed")
                logger.error("Redis ping failed")
                
        except aioredis.ConnectionError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Redis",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Connection error: {str(e)}",
                details={'error_type': 'ConnectionError'}
            ))
            self._errors.append(f"Redis: {str(e)}")
            self._recommendations.append(
                "تأكد من تشغيل خادم Redis على المنفذ 6379"
            )
            logger.error(f"Redis connection failed: {e}")
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Redis",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Connection failed: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"Redis: {str(e)}")
            logger.error(f"Redis connection failed: {e}")
    
    async def _test_cache(self) -> None:
        """اختبار وظيفة الذاكرة المؤقتة - Test cache functionality"""
        logger.info("Testing cache functionality...")
        start_time = time.time()
        
        try:
            redis = aioredis.from_url(
                self.redis_url,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # اختبار SET
            test_key = "bi_ide_setup_test"
            test_value = f"test_{time.time()}"
            await redis.set(test_key, test_value, ex=60)
            
            # اختبار GET
            retrieved_value = await redis.get(test_key)
            
            # اختبار DELETE
            await redis.delete(test_key)
            
            await redis.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            if retrieved_value and retrieved_value.decode() == test_value:
                self._statuses.append(ConnectionStatus(
                    component="Cache Functionality",
                    is_connected=True,
                    latency_ms=latency_ms,
                    message="Cache operations working correctly",
                    details={'operations': ['SET', 'GET', 'DELETE']}
                ))
                logger.info("Cache functionality test passed")
            else:
                self._statuses.append(ConnectionStatus(
                    component="Cache Functionality",
                    is_connected=False,
                    latency_ms=latency_ms,
                    message="Cache value mismatch",
                    details={'expected': test_value, 'got': retrieved_value}
                ))
                self._errors.append("Cache: Value mismatch")
                logger.error("Cache functionality test failed: value mismatch")
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Cache Functionality",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Cache test failed: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"Cache: {str(e)}")
            logger.error(f"Cache functionality test failed: {e}")
    
    async def _create_tables_if_not_exist(self) -> None:
        """إنشاء الجداول إذا لم تكن موجودة - Create tables if not exist"""
        logger.info("Checking/creating tables...")
        start_time = time.time()
        
        try:
            # التحقق من وجود الجداول الأساسية
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # قائمة الجداول المطلوبة
                required_tables = [
                    'users', 'projects', 'tasks', 'workers',
                    'training_jobs', 'models', 'metrics'
                ]
                
                # التحقق من وجود الجداول
                existing_tables = []
                for table in required_tables:
                    result = conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table}'
                        );
                    """))
                    if result.scalar():
                        existing_tables.append(table)
                
                missing_tables = set(required_tables) - set(existing_tables)
                
                latency_ms = (time.time() - start_time) * 1000
                
                if not missing_tables:
                    self._statuses.append(ConnectionStatus(
                        component="Database Tables",
                        is_connected=True,
                        latency_ms=latency_ms,
                        message=f"All {len(required_tables)} required tables exist",
                        details={
                            'existing_tables': existing_tables,
                            'missing_tables': []
                        }
                    ))
                    logger.info("All required tables exist")
                else:
                    self._statuses.append(ConnectionStatus(
                        component="Database Tables",
                        is_connected=True,
                        latency_ms=latency_ms,
                        message=f"{len(existing_tables)}/{len(required_tables)} tables exist",
                        details={
                            'existing_tables': existing_tables,
                            'missing_tables': list(missing_tables)
                        }
                    ))
                    self._recommendations.append(
                        f"الجداول المفقودة: {', '.join(missing_tables)}. "
                        "سيتم إنشاؤها بواسطة ترحيلات Alembic."
                    )
                    logger.warning(f"Missing tables: {missing_tables}")
                    
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._statuses.append(ConnectionStatus(
                component="Database Tables",
                is_connected=False,
                latency_ms=latency_ms,
                message=f"Table check failed: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            self._errors.append(f"Tables: {str(e)}")
            logger.error(f"Table check failed: {e}")
    
    def print_report(self, report: SetupReport) -> None:
        """
        طباعة تقرير الإعداد
        Print setup report
        
        Args:
            report: تقرير الإعداد
        """
        print("\n" + "=" * 60)
        print("📊 Database Setup Report - تقرير إعداد قاعدة البيانات")
        print("=" * 60)
        
        status_icon = "✅" if report.success else "❌"
        print(f"\n{status_icon} Overall Status: {'SUCCESS' if report.success else 'FAILED'}")
        print(f"⏱️  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        
        print("\n📋 Component Statuses:")
        print("-" * 60)
        for status in report.statuses:
            icon = "✅" if status.is_connected else "❌"
            print(f"{icon} {status.component}:")
            print(f"   Status: {'Connected' if status.is_connected else 'Disconnected'}")
            print(f"   Latency: {status.latency_ms:.2f}ms")
            print(f"   Message: {status.message}")
            print()
        
        if report.errors:
            print("\n❌ Errors:")
            print("-" * 60)
            for error in report.errors:
                print(f"  • {error}")
        
        if report.recommendations:
            print("\n💡 Recommendations:")
            print("-" * 60)
            for rec in report.recommendations:
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)


async def main():
    """الدالة الرئيسية - Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    setup = DatabaseSetup()
    report = await setup.run_setup()
    setup.print_report(report)
    
    # إرجاع رمز الخروج
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
