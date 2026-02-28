"""
مراقب النظام - System Monitor
=============================
مراقبة موارد النظام للعمال المحليين والبعيدين
Monitoring system resources for local and remote workers
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

import psutil
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """مقاييس الموارد - Resource metrics"""
    worker_id: str
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)
    is_healthy: bool = True
    error_message: Optional[str] = None


@dataclass
class GPUInfo:
    """معلومات GPU - GPU information"""
    index: int
    name: str
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_celsius: Optional[float] = None


class SystemMonitor:
    """
    مراقب النظام للعمال
    System monitor for workers
    
    يدعم المراقبة المحلية عبر psutil والمراقبة البعيدة عبر API
    Supports local monitoring via psutil and remote monitoring via API
    """
    
    def __init__(
        self,
        update_interval: float = 2.0,
        remote_workers: Optional[Dict[str, str]] = None
    ):
        """
        تهيئة مراقب النظام
        Initialize system monitor
        
        Args:
            update_interval: الفاصل الزمني للتحديث بالثواني (default: 2.0)
            remote_workers: قاموس معرف العامل -> URL API
        """
        self.update_interval = update_interval
        self.remote_workers = remote_workers or {}
        self._metrics_cache: Dict[str, ResourceMetrics] = {}
        self._callbacks: List[Callable[[ResourceMetrics], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._gpu_available = False
        
        # محاولة استيراد pynvml لمراقبة GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            self._pynvml = pynvml
            logger.info("GPU monitoring enabled via NVML")
        except ImportError:
            logger.warning("pynvml not available, GPU monitoring disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
    
    async def start(self) -> None:
        """بدء المراقبة الدورية - Start periodic monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop(self) -> None:
        """إيقاف المراقبة - Stop monitoring"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    def add_callback(self, callback: Callable[[ResourceMetrics], None]) -> None:
        """إضافة دالة استدعاء عند تحديث المقاييس - Add callback on metrics update"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[ResourceMetrics], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def _monitoring_loop(self) -> None:
        """حلقة المراقبة الدورية - Monitoring loop"""
        while self._running:
            try:
                await self.get_all_resources()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _get_local_gpu_info(self) -> List[Dict[str, Any]]:
        """الحصول على معلومات GPU المحلي - Get local GPU info"""
        gpu_info = []
        
        if not self._gpu_available:
            return gpu_info
        
        try:
            device_count = self._pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # الحصول على اسم GPU
                name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # الاستخدام
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # الذاكرة
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # درجة الحرارة
                temperature = None
                try:
                    temperature = self._pynvml.nvmlDeviceGetTemperature(
                        handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
                
                gpu_info.append({
                    'index': i,
                    'name': name,
                    'utilization_percent': utilization.gpu,
                    'memory_used_gb': memory.used / (1024 ** 3),
                    'memory_total_gb': memory.total / (1024 ** 3),
                    'temperature_celsius': temperature
                })
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return gpu_info
    
    async def get_worker_resources(self, worker_id: str) -> ResourceMetrics:
        """
        الحصول على موارد عامل محدد
        Get resources for a specific worker
        
        Args:
            worker_id: معرف العامل
            
        Returns:
            ResourceMetrics: مقاييس الموارد
        """
        # إذا كان العامل بعيداً
        if worker_id in self.remote_workers:
            return await self._get_remote_resources(worker_id)
        
        # إذا كان العامل محلياً
        return await self._get_local_resources(worker_id)
    
    async def _get_local_resources(self, worker_id: str) -> ResourceMetrics:
        """الحصول على الموارد المحلية - Get local resources"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # RAM
            memory = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # GPU
            gpu_info = self._get_local_gpu_info()
            
            metrics = ResourceMetrics(
                worker_id=worker_id,
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                ram_percent=memory.percent,
                ram_used_gb=memory.used / (1024 ** 3),
                ram_total_gb=memory.total / (1024 ** 3),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 ** 3),
                disk_total_gb=disk.total / (1024 ** 3),
                gpu_info=gpu_info,
                is_healthy=True
            )
            
            self._metrics_cache[worker_id] = metrics
            
            # استدعاء الدوال المسجلة
            for callback in self._callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting local resources: {e}")
            metrics = ResourceMetrics(
                worker_id=worker_id,
                timestamp=time.time(),
                cpu_percent=0.0,
                ram_percent=0.0,
                ram_used_gb=0.0,
                ram_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                is_healthy=False,
                error_message=str(e)
            )
            self._metrics_cache[worker_id] = metrics
            return metrics
    
    async def _get_remote_resources(self, worker_id: str) -> ResourceMetrics:
        """الحصول على الموارد البعيدة - Get remote resources"""
        url = self.remote_workers[worker_id]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/api/monitoring/resources",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        metrics = ResourceMetrics(
                            worker_id=worker_id,
                            timestamp=data.get('timestamp', time.time()),
                            cpu_percent=data.get('cpu_percent', 0.0),
                            ram_percent=data.get('ram_percent', 0.0),
                            ram_used_gb=data.get('ram_used_gb', 0.0),
                            ram_total_gb=data.get('ram_total_gb', 0.0),
                            disk_percent=data.get('disk_percent', 0.0),
                            disk_used_gb=data.get('disk_used_gb', 0.0),
                            disk_total_gb=data.get('disk_total_gb', 0.0),
                            gpu_info=data.get('gpu_info', []),
                            is_healthy=True
                        )
                        self._metrics_cache[worker_id] = metrics
                        return metrics
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error getting remote resources from {worker_id}: {e}")
            metrics = ResourceMetrics(
                worker_id=worker_id,
                timestamp=time.time(),
                cpu_percent=0.0,
                ram_percent=0.0,
                ram_used_gb=0.0,
                ram_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                is_healthy=False,
                error_message=f"Failed to connect: {str(e)}"
            )
            self._metrics_cache[worker_id] = metrics
            return metrics
    
    async def get_all_resources(self) -> Dict[str, ResourceMetrics]:
        """
        الحصول على موارد جميع العمال
        Get resources for all workers
        
        Returns:
            Dict[str, ResourceMetrics]: قاموس معرف العامل -> المقاييس
        """
        workers = ['local'] + list(self.remote_workers.keys())
        
        # جمع الموارد بشكل متوازي
        tasks = [self.get_worker_resources(worker_id) for worker_id in workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_metrics = {}
        for worker_id, result in zip(workers, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting resources for {worker_id}: {result}")
                all_metrics[worker_id] = ResourceMetrics(
                    worker_id=worker_id,
                    timestamp=time.time(),
                    cpu_percent=0.0,
                    ram_percent=0.0,
                    ram_used_gb=0.0,
                    ram_total_gb=0.0,
                    disk_percent=0.0,
                    disk_used_gb=0.0,
                    disk_total_gb=0.0,
                    is_healthy=False,
                    error_message=str(result)
                )
            else:
                all_metrics[worker_id] = result
        
        return all_metrics
    
    def check_health(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        التحقق من صحة العمال
        Check health of workers
        
        Args:
            worker_id: معرف العامل المحدد (None للجميع)
            
        Returns:
            Dict[str, Any]: حالة الصحة
        """
        if worker_id:
            if worker_id in self._metrics_cache:
                metrics = self._metrics_cache[worker_id]
                return {
                    'worker_id': worker_id,
                    'is_healthy': metrics.is_healthy,
                    'last_update': metrics.timestamp,
                    'error': metrics.error_message
                }
            return {
                'worker_id': worker_id,
                'is_healthy': False,
                'error': 'No metrics available'
            }
        
        # التحقق من جميع العمال
        health_status = {}
        for wid, metrics in self._metrics_cache.items():
            health_status[wid] = {
                'worker_id': wid,
                'is_healthy': metrics.is_healthy,
                'last_update': metrics.timestamp,
                'error': metrics.error_message
            }
        
        return health_status
    
    def get_cached_metrics(self, worker_id: str) -> Optional[ResourceMetrics]:
        """الحصول على المقاييس المخزنة - Get cached metrics"""
        return self._metrics_cache.get(worker_id)
