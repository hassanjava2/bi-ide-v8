"""
Network Monitor - BI-IDE v8
مراقبة الشبكة وتتبع Metrics
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from collections import deque
import statistics

from core.config import settings
from core.cache import cache_manager
from core.logging_config import logger


@dataclass
class LatencyMetrics:
    """مقاييس التأخير"""
    current: float = 0.0
    min_val: float = float('inf')
    max_val: float = 0.0
    avg: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    history: List[float] = field(default_factory=lambda: [])
    
    def update(self, value: float, max_history: int = 100):
        """تحديث المقاييس"""
        self.current = value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.history.append(value)
        
        if len(self.history) > max_history:
            self.history.pop(0)
        
        self.avg = statistics.mean(self.history)
        if len(self.history) >= 2:
            sorted_history = sorted(self.history)
            p95_idx = int(len(sorted_history) * 0.95)
            p99_idx = int(len(sorted_history) * 0.99)
            self.p95 = sorted_history[min(p95_idx, len(sorted_history) - 1)]
            self.p99 = sorted_history[min(p99_idx, len(sorted_history) - 1)]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "current": round(self.current, 2),
            "min": round(self.min_val, 2) if self.min_val != float('inf') else 0,
            "max": round(self.max_val, 2),
            "avg": round(self.avg, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class ServiceMetrics:
    """مقاييس خدمة واحدة"""
    name: str
    url: str
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    packet_loss: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 100.0
    last_check: Optional[datetime] = None
    status: str = "unknown"
    bandwidth_in: float = 0.0  # bytes/sec
    bandwidth_out: float = 0.0  # bytes/sec
    
    def record_request(self, success: bool, latency_ms: float, bytes_in: int = 0, bytes_out: int = 0):
        """تسجيل طلب"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        
        self.success_rate = ((self.total_requests - self.failed_requests) / self.total_requests) * 100
        self.latency.update(latency_ms)
        self.packet_loss = (self.failed_requests / self.total_requests) * 100
        self.last_check = datetime.now(timezone.utc)
        self.status = "up" if success else "down"
        
        # Bandwidth estimation (simple)
        if bytes_in > 0:
            self.bandwidth_in = bytes_in / (latency_ms / 1000) if latency_ms > 0 else 0
        if bytes_out > 0:
            self.bandwidth_out = bytes_out / (latency_ms / 1000) if latency_ms > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "latency": self.latency.to_dict(),
            "packet_loss_percent": round(self.packet_loss, 2),
            "success_rate_percent": round(self.success_rate, 2),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "last_check": self.last_check.isoformat() + "Z" if self.last_check else None,
            "bandwidth": {
                "in_bytes_per_sec": round(self.bandwidth_in, 2),
                "out_bytes_per_sec": round(self.bandwidth_out, 2),
            }
        }


class NetworkMonitor:
    """
    مراقب الشبكة - يتتبع latency و packet loss و bandwidth
    يخزن Metrics في Redis/Cache
    """
    
    def __init__(
        self,
        check_interval: int = 30,
        max_history: int = 1000,
        cache_prefix: str = "network_monitor"
    ):
        self.check_interval = check_interval
        self.max_history = max_history
        self.cache_prefix = cache_prefix
        self.services: Dict[str, ServiceMetrics] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
        
    def register_service(
        self,
        name: str,
        url: str,
        check_endpoint: Optional[str] = None
    ):
        """
        تسجيل خدمة للمراقبة
        
        Args:
            name: اسم الخدمة
            url: URL الخدمة
            check_endpoint: نقطة نهاية للفحص (اختياري)
        """
        full_url = f"{url}{check_endpoint}" if check_endpoint else url
        self.services[name] = ServiceMetrics(name=name, url=full_url)
        logger.info(f"Registered service for monitoring: {name} ({full_url})")
    
    def unregister_service(self, name: str):
        """إلغاء تسجيل خدمة"""
        if name in self.services:
            del self.services[name]
            logger.info(f"Unregistered service: {name}")
    
    async def _check_service(self, name: str) -> Dict[str, Any]:
        """فحص خدمة واحدة"""
        service = self.services[name]
        start_time = time.perf_counter()
        
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(service.url) as response:
                    latency = (time.perf_counter() - start_time) * 1000
                    content = await response.read()
                    
                    success = response.status < 500
                    service.record_request(
                        success=success,
                        latency_ms=latency,
                        bytes_in=len(content),
                        bytes_out=0
                    )
                    
                    return {
                        "name": name,
                        "success": success,
                        "latency_ms": latency,
                        "status_code": response.status,
                    }
                    
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            service.record_request(
                success=False,
                latency_ms=latency
            )
            return {
                "name": name,
                "success": False,
                "latency_ms": latency,
                "error": str(e),
            }
    
    async def _check_all_services(self):
        """فحص جميع الخدمات"""
        tasks = [self._check_service(name) for name in self.services.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # تخزين النتائج في الكاش
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Service check failed: {result}")
                continue
            
            service_name = result.get("name")
            if service_name:
                await self._store_metrics(service_name)
        
        # تنفيذ الـ callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.get_all_metrics())
                else:
                    callback(self.get_all_metrics())
            except Exception as e:
                logger.error(f"Monitor callback error: {e}")
    
    async def _store_metrics(self, service_name: str):
        """تخزين المقاييس في الكاش"""
        if service_name not in self.services:
            return
        
        service = self.services[service_name]
        metrics = service.to_dict()
        
        # تخزين آخر قيمة
        await cache_manager.set(
            f"{service_name}:latest",
            metrics,
            ttl=self.check_interval * 2,
            prefix=self.cache_prefix
        )
        
        # إضافة للتاريخ
        history_key = f"{service_name}:history"
        history = await cache_manager.get(history_key, prefix=self.cache_prefix) or []
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency": metrics["latency"]["current"],
            "status": metrics["status"],
        })
        
        # الاحتفاظ بآخر max_history قيم
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        
        await cache_manager.set(
            history_key,
            history,
            ttl=86400,  # يوم واحد
            prefix=self.cache_prefix
        )
    
    async def _monitor_loop(self):
        """حلقة المراقبة"""
        while self._running:
            try:
                await self._check_all_services()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def start(self):
        """بدء المراقبة"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info(f"Network monitor started (interval: {self.check_interval}s)")
    
    def stop(self):
        """إيقاف المراقبة"""
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
            logger.info("Network monitor stopped")
    
    def add_callback(self, callback: Callable):
        """إضافة callback يُستدعى بعد كل فحص"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """إزالة callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_metrics(self, service_name: str) -> Optional[Dict[str, Any]]:
        """الحصول على مقاييس خدمة"""
        if service_name in self.services:
            return self.services[service_name].to_dict()
        return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """الحصول على جميع المقاييس"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "services": {
                name: service.to_dict()
                for name, service in self.services.items()
            },
            "summary": {
                "total_services": len(self.services),
                "up_services": sum(1 for s in self.services.values() if s.status == "up"),
                "down_services": sum(1 for s in self.services.values() if s.status == "down"),
                "avg_latency_ms": round(
                    statistics.mean([s.latency.current for s in self.services.values() if s.latency.current > 0])
                    if self.services else 0, 2
                ),
            }
        }
    
    async def get_historical_metrics(
        self,
        service_name: str,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """الحصول على المقاييس التاريخية"""
        history = await cache_manager.get(
            f"{service_name}:history",
            prefix=self.cache_prefix
        ) or []
        
        if since:
            since_str = since.isoformat()
            history = [h for h in history if h.get("timestamp", "") >= since_str]
        
        return history
    
    def get_service_status(self, service_name: str) -> str:
        """الحصول على حالة خدمة"""
        if service_name in self.services:
            return self.services[service_name].status
        return "unknown"
    
    def is_service_healthy(self, service_name: str) -> bool:
        """التحقق من صحة خدمة"""
        return self.get_service_status(service_name) == "up"


# Global instance
network_monitor = NetworkMonitor()


# Convenience functions
def register_default_services():
    """تسجيل الخدمات الافتراضية"""
    network_monitor.register_service(
        "rtx4090",
        settings.rtx4090_url,
        "/status"
    )
    network_monitor.register_service(
        "local_api",
        f"http://localhost:{settings.PORT}",
        "/api/v1/system/status"
    )


async def get_network_metrics() -> Dict[str, Any]:
    """الحصول على مقاييس الشبكة"""
    return network_monitor.get_all_metrics()


def get_network_metrics_sync() -> Dict[str, Any]:
    """الحصول على مقاييس الشبكة (متزامن)"""
    return network_monitor.get_all_metrics()


if __name__ == "__main__":
    # Test network monitor
    print("=" * 60)
    print("BI-IDE v8 Network Monitor")
    print("=" * 60)
    
    # Register services
    register_default_services()
    
    # Run single check
    async def test_monitor():
        print("\nRunning service checks...")
        await network_monitor._check_all_services()
        
        metrics = network_monitor.get_all_metrics()
        print(f"\nTimestamp: {metrics['timestamp']}")
        print(f"Total Services: {metrics['summary']['total_services']}")
        print(f"Up: {metrics['summary']['up_services']}")
        print(f"Down: {metrics['summary']['down_services']}")
        print(f"Avg Latency: {metrics['summary']['avg_latency_ms']}ms")
        
        print("\n--- Service Details ---")
        for name, service_metrics in metrics['services'].items():
            status_icon = "✅" if service_metrics['status'] == 'up' else "❌"
            print(f"\n{status_icon} {name}")
            print(f"   URL: {service_metrics['url']}")
            print(f"   Status: {service_metrics['status']}")
            print(f"   Latency: {service_metrics['latency']['current']}ms (avg: {service_metrics['latency']['avg']}ms)")
            print(f"   Success Rate: {service_metrics['success_rate_percent']}%")
            print(f"   Packet Loss: {service_metrics['packet_loss_percent']}%")
    
    asyncio.run(test_monitor())
