#!/usr/bin/env python3
"""
Start Services - تشغيل الخدمات
Start all BI-IDE services in correct order
تشغيل جميع خدمات BI-IDE بالترتيب الصحيح
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# إعداد التسجيل - Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """حالة الخدمة - Service status"""
    name: str
    name_ar: str
    enabled: bool = True
    running: bool = False
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None
    health_check_url: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "name_ar": self.name_ar,
            "enabled": self.enabled,
            "running": self.running,
            "pid": self.pid,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "error": self.error_message,
            "health_url": self.health_check_url
        }


class ServiceManager:
    """
    مدير تشغيل الخدمات
    Service startup manager
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceStatus] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        # تهيئة حالات الخدمات - Initialize service statuses
        self._init_services()
        
    def _init_services(self):
        """تهيئة حالات الخدمات - Initialize service statuses"""
        self.services = {
            "postgres": ServiceStatus(
                name="PostgreSQL",
                name_ar="قاعدة البيانات",
                health_check_url=None  # لا يمكن فحصها عبر HTTP
            ),
            "redis": ServiceStatus(
                name="Redis",
                name_ar="التخزين المؤقت",
                health_check_url=None
            ),
            "api": ServiceStatus(
                name="API Server",
                name_ar="خادم API",
                health_check_url="http://localhost:8000/health"
            ),
            "worker": ServiceStatus(
                name="Background Worker",
                name_ar="عامل الخلفية",
                health_check_url=None
            ),
            "monitoring": ServiceStatus(
                name="Monitoring Stack",
                name_ar="نظام المراقبة",
                health_check_url="http://localhost:9090/-/healthy"
            )
        }
        
    async def check_postgres(self) -> Tuple[bool, str]:
        """التحقق من PostgreSQL - Check PostgreSQL"""
        try:
            # التحقق إذا كانت PostgreSQL تعمل
            result = subprocess.run(
                ["pg_isready", "-h", "localhost", "-p", "5432"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "running"
            return False, result.stdout or result.stderr
        except FileNotFoundError:
            # محاولة عبر docker
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=postgres", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "Up" in result.stdout:
                    return True, "running (docker)"
                return False, "not running (docker)"
            except Exception as e:
                return False, f"check failed: {e}"
        except Exception as e:
            return False, f"check failed: {e}"
            
    async def check_redis(self) -> Tuple[bool, str]:
        """التحقق من Redis - Check Redis"""
        try:
            result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "PONG" in result.stdout:
                return True, "running"
            return False, result.stdout or result.stderr
        except FileNotFoundError:
            # محاولة عبر docker
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=redis", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "Up" in result.stdout:
                    return True, "running (docker)"
                return False, "not running (docker)"
            except Exception as e:
                return False, f"check failed: {e}"
        except Exception as e:
            return False, f"check failed: {e}"
            
    async def check_http_health(self, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """التحقق من صحة HTTP - Check HTTP health"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return True, f"healthy (status: {response.status})"
                    return False, f"unhealthy (status: {response.status})"
        except ImportError:
            # استخدام curl كبديل
            try:
                result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.stdout.strip() == "200":
                    return True, "healthy"
                return False, f"status: {result.stdout}"
            except Exception as e:
                return False, str(e)
        except Exception as e:
            return False, str(e)
            
    async def start_postgres(self) -> bool:
        """تشغيل PostgreSQL - Start PostgreSQL"""
        logger.info("🐘 Starting PostgreSQL...")
        
        # التحقق إذا كانت تعمل بالفعل
        running, msg = await self.check_postgres()
        if running:
            logger.info(f"  ✅ PostgreSQL already running ({msg})")
            self.services["postgres"].running = True
            return True
            
        # محاولة التشغيل عبر docker-compose
        try:
            result = subprocess.run(
                ["docker-compose", "up", "-d", "postgres"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                # انتظار التشغيل
                for i in range(30):
                    await asyncio.sleep(1)
                    running, msg = await self.check_postgres()
                    if running:
                        logger.info("  ✅ PostgreSQL started")
                        self.services["postgres"].running = True
                        return True
                        
            logger.error(f"  ❌ Failed to start PostgreSQL: {result.stderr}")
            self.services["postgres"].error_message = result.stderr
            return False
        except Exception as e:
            logger.error(f"  ❌ PostgreSQL start error: {e}")
            self.services["postgres"].error_message = str(e)
            return False
            
    async def start_redis(self) -> bool:
        """تشغيل Redis - Start Redis"""
        logger.info("🔴 Starting Redis...")
        
        # التحقق إذا كانت تعمل بالفعل
        running, msg = await self.check_redis()
        if running:
            logger.info(f"  ✅ Redis already running ({msg})")
            self.services["redis"].running = True
            return True
            
        # محاولة التشغيل عبر docker-compose
        try:
            result = subprocess.run(
                ["docker-compose", "up", "-d", "redis"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                # انتظار التشغيل
                for i in range(30):
                    await asyncio.sleep(1)
                    running, msg = await self.check_redis()
                    if running:
                        logger.info("  ✅ Redis started")
                        self.services["redis"].running = True
                        return True
                        
            logger.error(f"  ❌ Failed to start Redis: {result.stderr}")
            self.services["redis"].error_message = result.stderr
            return False
        except Exception as e:
            logger.error(f"  ❌ Redis start error: {e}")
            self.services["redis"].error_message = str(e)
            return False
            
    async def start_api(self, workers: int = 4, port: int = 8000) -> bool:
        """تشغيل خادم API - Start API server"""
        logger.info(f"🚀 Starting API server on port {port}...")
        
        # التحقق إذا كان يعمل بالفعل
        if self.services["api"].health_check_url:
            running, msg = await self.check_http_health(self.services["api"].health_check_url)
            if running:
                logger.info(f"  ✅ API server already running")
                self.services["api"].running = True
                return True
                
        try:
            # تشغيل خادم API
            env = os.environ.copy()
            env["PORT"] = str(port)
            
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "api.app:app",
                    "--host", "0.0.0.0",
                    "--port", str(port),
                    "--workers", str(workers)
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            self.processes["api"] = process
            self.services["api"].pid = process.pid
            self.services["api"].start_time = datetime.now(timezone.utc)
            
            # انتظار التشغيل
            for i in range(30):
                await asyncio.sleep(1)
                running, msg = await self.check_http_health(
                    self.services["api"].health_check_url,
                    timeout=5
                )
                if running:
                    logger.info(f"  ✅ API server started (PID: {process.pid})")
                    self.services["api"].running = True
                    return True
                    
                # التحقق من أن العملية لا تزال تعمل
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    logger.error(f"  ❌ API server crashed: {stderr}")
                    self.services["api"].error_message = stderr
                    return False
                    
            logger.error("  ❌ API server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"  ❌ API server start error: {e}")
            self.services["api"].error_message = str(e)
            return False
            
    async def start_worker(self) -> bool:
        """تشغيل عامل الخلفية - Start background worker"""
        logger.info("⚙️ Starting background worker...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "core.tasks"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            self.processes["worker"] = process
            self.services["worker"].pid = process.pid
            self.services["worker"].start_time = datetime.now(timezone.utc)
            
            # انتظار قليلاً للتحقق من بدء التشغيل
            await asyncio.sleep(3)
            
            if process.poll() is None:
                logger.info(f"  ✅ Worker started (PID: {process.pid})")
                self.services["worker"].running = True
                return True
            else:
                stderr = process.stderr.read().decode() if process.stderr else ""
                logger.error(f"  ❌ Worker failed to start: {stderr}")
                self.services["worker"].error_message = stderr
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Worker start error: {e}")
            self.services["worker"].error_message = str(e)
            return False
            
    async def health_check_all(self) -> Dict[str, Dict]:
        """فحص صحة جميع الخدمات - Health check all services"""
        results = {}
        
        # فحص PostgreSQL
        if self.services["postgres"].enabled:
            healthy, msg = await self.check_postgres()
            results["postgres"] = {
                "healthy": healthy,
                "message": msg,
                "name_ar": self.services["postgres"].name_ar
            }
            self.services["postgres"].running = healthy
            
        # فحص Redis
        if self.services["redis"].enabled:
            healthy, msg = await self.check_redis()
            results["redis"] = {
                "healthy": healthy,
                "message": msg,
                "name_ar": self.services["redis"].name_ar
            }
            self.services["redis"].running = healthy
            
        # فحص API
        if self.services["api"].enabled and self.services["api"].health_check_url:
            healthy, msg = await self.check_http_health(
                self.services["api"].health_check_url
            )
            results["api"] = {
                "healthy": healthy,
                "message": msg,
                "name_ar": self.services["api"].name_ar
            }
            self.services["api"].running = healthy
            
        return results
        
    async def start_all(
        self,
        skip_db: bool = False,
        skip_redis: bool = False,
        api_workers: int = 4,
        api_port: int = 8000
    ) -> bool:
        """تشغيل جميع الخدمات - Start all services"""
        logger.info("=" * 60)
        logger.info("🚀 Starting BI-IDE Services | تشغيل خدمات BI-IDE")
        logger.info("=" * 60)
        
        success = True
        
        # 1. تشغيل PostgreSQL
        if not skip_db:
            if not await self.start_postgres():
                success = False
        else:
            logger.info("🐘 PostgreSQL skipped (--skip-db)")
            self.services["postgres"].enabled = False
            
        # 2. تشغيل Redis
        if not skip_redis:
            if not await self.start_redis():
                success = False
        else:
            logger.info("🔴 Redis skipped (--skip-redis)")
            self.services["redis"].enabled = False
            
        # 3. تشغيل API
        if not await self.start_api(workers=api_workers, port=api_port):
            success = False
            
        # 4. تشغيل Worker
        if not await self.start_worker():
            success = False
            
        # 5. فحص صحة جميع الخدمات
        logger.info("\n🔍 Running health checks...")
        health_results = await self.health_check_all()
        
        for service, result in health_results.items():
            status = "✅" if result["healthy"] else "❌"
            logger.info(f"  {status} {result['name_ar']}: {result['message']}")
            
        # طباعة ملخص
        await self.print_status()
        
        return success
        
    async def print_status(self):
        """طباعة حالة الخدمات - Print service status"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Service Status | حالة الخدمات")
        logger.info("=" * 60)
        
        for service_id, status in self.services.items():
            if not status.enabled:
                icon = "⏭️"
                state = "skipped"
            elif status.running:
                icon = "🟢"
                state = f"running (PID: {status.pid})" if status.pid else "running"
            else:
                icon = "🔴"
                state = "stopped"
                if status.error_message:
                    state += f" - {status.error_message[:50]}"
                    
            logger.info(f"{icon} {status.name} ({status.name_ar}): {state}")
            
    async def stop_all(self):
        """إيقاف جميع الخدمات - Stop all services"""
        logger.info("\n🛑 Stopping all services...")
        
        for service_id, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"  Stopping {service_id} (PID: {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.processes.clear()
        
        for service in self.services.values():
            service.running = False
            service.pid = None
            
        logger.info("✅ All services stopped")


def signal_handler(manager: ServiceManager):
    """معالج الإشارات - Signal handler"""
    def handler(signum, frame):
        logger.info("\n🛑 Shutdown signal received...")
        asyncio.create_task(manager.stop_all())
        sys.exit(0)
    return handler


async def main():
    """الدالة الرئيسية - Main function"""
    parser = argparse.ArgumentParser(
        description="Start BI-IDE services | تشغيل خدمات BI-IDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples | أمثلة:
  python scripts/start_services.py
  python scripts/start_services.py --skip-db --skip-redis
  python scripts/start_services.py --api-port 8080 --api-workers 2
        """
    )
    
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip starting PostgreSQL | تخطي تشغيل قاعدة البيانات"
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip starting Redis | تخطي تشغيل Redis"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port | منفذ خادم API"
    )
    parser.add_argument(
        "--api-workers",
        type=int,
        default=4,
        help="Number of API workers | عدد عمال API"
    )
    parser.add_argument(
        "--health-check-only",
        action="store_true",
        help="Only run health checks | تشغيل فحوصات الصحة فقط"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format | إخراج بتنسيق JSON"
    )
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    # معالج الإشارات - Signal handler
    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(manager.stop_all()))
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(manager.stop_all()))
    
    if args.health_check_only:
        # فحص الصحة فقط
        results = await manager.health_check_all()
        
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            logger.info("\n🔍 Health Check Results | نتائج فحص الصحة")
            logger.info("=" * 60)
            for service, result in results.items():
                status = "✅" if result["healthy"] else "❌"
                logger.info(f"{status} {result['name_ar']}: {result['message']}")
                
        # رمز الخروج - Exit code
        all_healthy = all(r["healthy"] for r in results.values())
        sys.exit(0 if all_healthy else 1)
    else:
        # تشغيل جميع الخدمات
        success = await manager.start_all(
            skip_db=args.skip_db,
            skip_redis=args.skip_redis,
            api_workers=args.api_workers,
            api_port=args.api_port
        )
        
        if args.json:
            status_dict = {
                k: v.to_dict() for k, v in manager.services.items()
            }
            print(json.dumps(status_dict, indent=2, ensure_ascii=False))
            
        if success:
            logger.info("\n✅ All services started successfully!")
            logger.info("Press Ctrl+C to stop | اضغط Ctrl+C للإيقاف")
            
            # الانتظار حتى إشارة الإيقاف
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await manager.stop_all()
        else:
            logger.error("\n❌ Some services failed to start")
            await manager.stop_all()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
