"""
Health Check System - BI-IDE v8
نظام فحص الصحة المتكامل للخدمات
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum

from core.config import settings
from core.cache import cache_manager
from core.logging_config import logger


class HealthStatus(Enum):
    """حالة الصحة"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """حالة صحة خدمة واحدة"""
    name: str
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp.isoformat() + "Z",
            "message": self.message,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """تقرير الصحة الكامل"""
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat() + "Z",
            "services": {name: svc.to_dict() for name, svc in self.services.items()},
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


class HealthChecker:
    """
    فاحص الصحة - يتحقق من جميع الخدمات
    Production-ready مع دعم async والتخزين المؤقت
    """
    
    # حدود الوقت (ثواني)
    TIMEOUT_DATABASE = 5.0
    TIMEOUT_REDIS = 3.0
    TIMEOUT_RTX4090 = 10.0
    TIMEOUT_API = 5.0
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_ttl = 30  # ثواني
        self._cache_key = "health_check:last_result"
        
    async def check_database(self) -> ServiceHealth:
        """
        فحص قاعدة البيانات
        
        Returns:
            ServiceHealth: حالة قاعدة البيانات
        """
        start_time = time.perf_counter()
        
        try:
            from core.database import db_manager
            
            # محاولة الاتصال وتنفيذ استعلام بسيط
            if hasattr(db_manager, 'check_connection'):
                is_connected = await db_manager.check_connection()
            else:
                # Fallback: محاولة الحصول على session
                async with db_manager.get_session() as session:
                    result = await session.execute("SELECT 1")
                    is_connected = result.scalar() == 1
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return ServiceHealth(
                name="database",
                status=HealthStatus.HEALTHY if is_connected else HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message="Connected" if is_connected else "Connection failed",
                details={
                    "type": "postgresql" if "postgresql" in settings.DATABASE_URL else "sqlite",
                    "connected": is_connected,
                }
            )
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Database health check failed: {e}")
            return ServiceHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def check_redis(self) -> ServiceHealth:
        """
        فحص Redis/Cache
        
        Returns:
            ServiceHealth: حالة Redis
        """
        start_time = time.perf_counter()
        
        try:
            if not cache_manager.redis_client:
                response_time = (time.perf_counter() - start_time) * 1000
                return ServiceHealth(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    message="Using in-memory cache (Redis not available)",
                    details={"mode": "in-memory"}
                )
            
            # Ping Redis
            await asyncio.to_thread(cache_manager.redis_client.ping)
            
            # Get stats
            info = await asyncio.to_thread(lambda: cache_manager.redis_client.info())
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return ServiceHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Connected",
                details={
                    "version": info.get("redis_version"),
                    "memory_used": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "mode": "redis",
                }
            )
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Redis health check failed: {e}")
            return ServiceHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def check_rtx4090(self) -> ServiceHealth:
        """
        فحص RTX 4090 Inference Server
        
        Returns:
            ServiceHealth: حالة RTX 4090
        """
        start_time = time.perf_counter()
        
        try:
            import aiohttp
            
            url = f"{settings.rtx4090_url}/status"
            timeout = aiohttp.ClientTimeout(total=self.TIMEOUT_RTX4090)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return ServiceHealth(
                            name="rtx4090",
                            status=HealthStatus.HEALTHY,
                            response_time_ms=response_time,
                            message="Connected and responsive",
                            details={
                                "status_code": response.status,
                                "gpu": data.get("gpu", "Unknown"),
                                "device": data.get("device", "Unknown"),
                                "running": data.get("running", False),
                            }
                        )
                    else:
                        return ServiceHealth(
                            name="rtx4090",
                            status=HealthStatus.DEGRADED,
                            response_time_ms=response_time,
                            message=f"HTTP {response.status}",
                            details={"status_code": response.status}
                        )
                        
        except asyncio.TimeoutError:
            response_time = (time.perf_counter() - start_time) * 1000
            return ServiceHealth(
                name="rtx4090",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message="Connection timeout",
                details={"timeout_seconds": self.TIMEOUT_RTX4090}
            )
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"RTX 4090 health check failed: {e}")
            return ServiceHealth(
                name="rtx4090",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def check_api(self) -> ServiceHealth:
        """
        فحص API المحلي
        
        Returns:
            ServiceHealth: حالة API
        """
        start_time = time.perf_counter()
        
        try:
            import aiohttp
            
            url = f"http://localhost:{settings.PORT}/api/v1/system/status"
            timeout = aiohttp.ClientTimeout(total=self.TIMEOUT_API)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        return ServiceHealth(
                            name="api",
                            status=HealthStatus.HEALTHY,
                            response_time_ms=response_time,
                            message="API is responsive",
                            details={
                                "status_code": response.status,
                                "port": settings.PORT,
                            }
                        )
                    else:
                        return ServiceHealth(
                            name="api",
                            status=HealthStatus.DEGRADED,
                            response_time_ms=response_time,
                            message=f"HTTP {response.status}",
                            details={"status_code": response.status}
                        )
                        
        except asyncio.TimeoutError:
            response_time = (time.perf_counter() - start_time) * 1000
            return ServiceHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message="Connection timeout",
            )
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"API health check failed: {e}")
            return ServiceHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Error: {str(e)}",
                details={"error_type": type(e).__name__}
            )
    
    async def get_overall_status(
        self,
        use_cache: bool = True
    ) -> HealthReport:
        """
        الحصول على حالة الصحة الشاملة (JSON شامل)
        
        Args:
            use_cache: استخدام النتيجة المخزنة مؤقتاً إذا كانت متوفرة
            
        Returns:
            HealthReport: تقرير الصحة الكامل
        """
        # التحقق من الكاش
        if use_cache and self.cache_enabled:
            cached = await cache_manager.get(self._cache_key, prefix="health")
            if cached:
                logger.debug("Using cached health check result")
                # إعادة بناء HealthReport من الكاش
                services = {
                    name: ServiceHealth(
                        name=data["name"],
                        status=HealthStatus(data["status"]),
                        response_time_ms=data["response_time_ms"],
                        message=data.get("message", ""),
                        details=data.get("details", {}),
                    )
                    for name, data in cached.get("services", {}).items()
                }
                return HealthReport(
                    overall_status=HealthStatus(cached["overall_status"]),
                    services=services,
                    metadata=cached.get("metadata", {}),
                )
        
        # تنفيذ جميع الفحوصات بشكل متوازي
        start_time = time.perf_counter()
        
        results = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_rtx4090(),
            self.check_api(),
            return_exceptions=True
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # معالجة النتائج
        services = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check raised exception: {result}")
                continue
            services[result.name] = result
        
        # تحديد الحالة الشاملة
        overall_status = HealthStatus.HEALTHY
        for service in services.values():
            if service.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif service.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        report = HealthReport(
            overall_status=overall_status,
            services=services,
            metadata={
                "total_check_time_ms": round(total_time, 2),
                "services_count": len(services),
                "healthy_count": sum(1 for s in services.values() if s.status == HealthStatus.HEALTHY),
                "degraded_count": sum(1 for s in services.values() if s.status == HealthStatus.DEGRADED),
                "unhealthy_count": sum(1 for s in services.values() if s.status == HealthStatus.UNHEALTHY),
                "environment": settings.ENVIRONMENT,
                "app_version": settings.APP_VERSION,
            }
        )
        
        # تخزين في الكاش
        if self.cache_enabled:
            await cache_manager.set(
                self._cache_key,
                report.to_dict(),
                ttl=self.cache_ttl,
                prefix="health"
            )
        
        return report
    
    async def check_specific_services(
        self,
        services: List[str]
    ) -> Dict[str, ServiceHealth]:
        """
        فحص خدمات محددة فقط
        
        Args:
            services: قائمة أسماء الخدمات ["database", "redis", "rtx4090", "api"]
            
        Returns:
            Dict[str, ServiceHealth]: نتائج الفحص
        """
        check_map = {
            "database": self.check_database,
            "redis": self.check_redis,
            "rtx4090": self.check_rtx4090,
            "api": self.check_api,
        }
        
        tasks = []
        names = []
        
        for service_name in services:
            if service_name in check_map:
                tasks.append(check_map[service_name]())
                names.append(service_name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            name: result if not isinstance(result, Exception) else ServiceHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                message=f"Exception: {result}"
            )
            for name, result in zip(names, results)
        }
    
    def is_healthy(self, report: Optional[HealthReport] = None) -> bool:
        """
        التحقق السريع من أن النظام صحي
        
        Args:
            report: تقرير الصحة (اختياري - يتم الحصول عليه إذا لم يُقدم)
            
        Returns:
            bool: True إذا كان النظام صحياً
        """
        if report is None:
            report = asyncio.run(self.get_overall_status(use_cache=True))
        return report.overall_status == HealthStatus.HEALTHY


# Global instance
health_checker = HealthChecker()


# Convenience functions
async def get_health_status() -> Dict[str, Any]:
    """الحصول على حالة الصحة (دالة مساعدة)"""
    report = await health_checker.get_overall_status()
    return report.to_dict()


def get_health_status_sync() -> Dict[str, Any]:
    """الحصول على حالة الصحة (متزامن)"""
    return asyncio.run(get_health_status())


async def quick_health_check() -> bool:
    """فحص صحة سريع"""
    report = await health_checker.get_overall_status(use_cache=True)
    return report.overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


if __name__ == "__main__":
    # Test health check
    print("=" * 60)
    print("BI-IDE v8 Health Check System")
    print("=" * 60)
    
    report = get_health_status_sync()
    
    print(f"\nOverall Status: {report['overall_status'].upper()}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"\nServices ({report['metadata']['services_count']} total):")
    print("-" * 60)
    
    for name, service in report['services'].items():
        status_icon = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "unknown": "❓",
        }.get(service['status'], "❓")
        
        print(f"{status_icon} {name.upper()}")
        print(f"   Status: {service['status']}")
        print(f"   Response: {service['response_time_ms']}ms")
        print(f"   Message: {service['message']}")
        if service['details']:
            print(f"   Details: {service['details']}")
        print()
    
    print("-" * 60)
    print(f"Metadata: {report['metadata']}")
