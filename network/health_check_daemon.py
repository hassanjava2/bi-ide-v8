"""
Network Health Check Daemon - خدمة فحص صحة الشبكة
Continuous health monitoring between all devices
"""

import asyncio
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from pathlib import Path

import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """حالات الاتصال - Connection statuses"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class PingResult:
    """نتيجة ping - Ping result"""
    host: str
    timestamp: float
    status: ConnectionStatus
    latency_ms: Optional[float]
    packet_loss: float
    message: str


@dataclass
class ConnectionHistory:
    """تاريخ الاتصال - Connection history"""
    host: str
    total_checks: int = 0
    online_count: int = 0
    offline_count: int = 0
    avg_latency: float = 0.0
    last_seen: Optional[float] = None
    history: List[PingResult] = field(default_factory=list)
    
    def add_result(self, result: PingResult) -> None:
        """إضافة نتيجة إلى التاريخ - Add result to history"""
        self.total_checks += 1
        self.history.append(result)
        
        # الاحتفاظ بآخر 100 نتيجة فقط
        if len(self.history) > 100:
            self.history.pop(0)
        
        if result.status == ConnectionStatus.ONLINE:
            self.online_count += 1
            self.last_seen = result.timestamp
            # تحديث متوسط زمن الاستجابة
            if result.latency_ms:
                self.avg_latency = (
                    (self.avg_latency * (self.online_count - 1) + result.latency_ms)
                    / self.online_count
                )
        elif result.status == ConnectionStatus.OFFLINE:
            self.offline_count += 1


class HealthCheckDaemon:
    """
    خدمة فحص صحة الشبكة
    Network Health Check Daemon
    
    مراقبة مستمرة للصحة بين جميع الأجهزة مع إعادة الاتصال التلقائي
    Continuous health monitoring between all devices with auto-reconnect
    """
    
    # الإعدادات الافتراضية - Default settings
    DEFAULT_INTERVAL = 30  # ثواني
    DEFAULT_TIMEOUT = 5    # ثواني
    DEFAULT_RETRY_ATTEMPTS = 3
    HISTORY_FILE = "connection_history.json"
    
    def __init__(
        self,
        workers: Optional[Dict[str, str]] = None,
        check_interval: float = DEFAULT_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        data_dir: Optional[str] = None,
        reporting_callback: Optional[Callable[[str, PingResult], None]] = None
    ):
        """
        تهيئة خدمة فحص الصحة
        Initialize health check daemon
        
        Args:
            workers: قاموس معرف العامل -> IP/Hostname
            check_interval: فاصل التحقق بالثواني
            timeout: مهلة الاستجابة بالثواني
            retry_attempts: عدد محاولات إعادة المحاولة
            data_dir: دليل تخزين البيانات
            reporting_callback: دالة استدعاء للتقارير
        """
        self.workers = workers or {}
        self.check_interval = check_interval
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.reporting_callback = reporting_callback
        
        # إعداد دليل البيانات
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / self.HISTORY_FILE
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._history: Dict[str, ConnectionHistory] = {}
        self._callbacks: List[Callable[[str, PingResult], None]] = []
        self._reconnect_callbacks: List[Callable[[str], None]] = []
        self._disconnect_callbacks: List[Callable[[str], None]] = []
        self._status_cache: Dict[str, ConnectionStatus] = {}
        
        # تحميل التاريخ
        self._load_history()
    
    async def start(self) -> None:
        """بدء الخدمة - Start daemon"""
        if self._running:
            logger.warning("Health check daemon already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Health check daemon started (interval: {self.check_interval}s)")
    
    async def stop(self) -> None:
        """إيقاف الخدمة - Stop daemon"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # حفظ التاريخ
        await self._save_history()
        logger.info("Health check daemon stopped")
    
    def add_worker(self, worker_id: str, host: str) -> None:
        """
        إضافة عامل جديد للمراقبة
        Add new worker to monitor
        
        Args:
            worker_id: معرف العامل
            host: IP أو hostname
        """
        self.workers[worker_id] = host
        if worker_id not in self._history:
            self._history[worker_id] = ConnectionHistory(host=host)
        logger.info(f"Added worker {worker_id} ({host}) to monitoring")
    
    def remove_worker(self, worker_id: str) -> None:
        """
        إزالة عامل من المراقبة
        Remove worker from monitoring
        
        Args:
            worker_id: معرف العامل
        """
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id} from monitoring")
    
    def add_callback(self, callback: Callable[[str, PingResult], None]) -> None:
        """إضافة دالة استدعاء - Add callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, PingResult], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def on_reconnect(self, callback: Callable[[str], None]) -> None:
        """تسجيل دالة استدعاء لإعادة الاتصال - Register reconnect callback"""
        self._reconnect_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        """تسجيل دالة استدعاء لفقدان الاتصال - Register disconnect callback"""
        self._disconnect_callbacks.append(callback)
    
    async def _monitoring_loop(self) -> None:
        """حلقة المراقبة الدورية - Monitoring loop"""
        while self._running:
            try:
                await self._check_all_workers()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_workers(self) -> None:
        """التحقق من جميع العمال - Check all workers"""
        tasks = [
            self._check_worker(worker_id, host)
            for worker_id, host in self.workers.items()
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_worker(self, worker_id: str, host: str) -> None:
        """
        التحقق من عامل محدد
        Check specific worker
        
        Args:
            worker_id: معرف العامل
            host: IP أو hostname
        """
        result = await self._ping_host(host)
        
        # تحديث التاريخ
        if worker_id not in self._history:
            self._history[worker_id] = ConnectionHistory(host=host)
        
        self._history[worker_id].add_result(result)
        
        # التحقق من تغيير الحالة
        previous_status = self._status_cache.get(worker_id, ConnectionStatus.UNKNOWN)
        
        if result.status != previous_status:
            self._status_cache[worker_id] = result.status
            
            if result.status == ConnectionStatus.ONLINE and previous_status == ConnectionStatus.OFFLINE:
                # إعادة اتصال
                logger.info(f"Worker {worker_id} ({host}) reconnected")
                await self._trigger_reconnect(worker_id)
            elif result.status == ConnectionStatus.OFFLINE and previous_status == ConnectionStatus.ONLINE:
                # فقدان اتصال
                logger.warning(f"Worker {worker_id} ({host}) disconnected")
                await self._trigger_disconnect(worker_id)
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback(worker_id, result)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        # إرسال تقرير
        if self.reporting_callback:
            try:
                self.reporting_callback(worker_id, result)
            except Exception as e:
                logger.error(f"Error in reporting callback: {e}")
    
    async def _ping_host(self, host: str) -> PingResult:
        """
        تنفيذ ping على مضيف
        Ping a host
        
        Args:
            host: IP أو hostname
            
        Returns:
            PingResult: نتيجة ping
        """
        timestamp = time.time()
        
        # محاولة ping مع إعادة المحاولة
        for attempt in range(self.retry_attempts):
            try:
                result = await self._execute_ping(host)
                
                if result['success']:
                    return PingResult(
                        host=host,
                        timestamp=timestamp,
                        status=ConnectionStatus.ONLINE,
                        latency_ms=result['latency'],
                        packet_loss=0.0,
                        message=f"Online (latency: {result['latency']:.2f}ms)"
                    )
                
                # انتظار قبل إعادة المحاولة
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.debug(f"Ping attempt {attempt + 1} failed for {host}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(1)
        
        # جميع المحاولات فشلت
        return PingResult(
            host=host,
            timestamp=timestamp,
            status=ConnectionStatus.OFFLINE,
            latency_ms=None,
            packet_loss=100.0,
            message="Host unreachable after all retry attempts"
        )
    
    async def _execute_ping(self, host: str) -> Dict[str, Any]:
        """
        تنفيذ أمر ping
        Execute ping command
        
        Args:
            host: IP أو hostname
            
        Returns:
            Dict[str, Any]: نتيجة التنفيذ
        """
        loop = asyncio.get_event_loop()
        
        # تحديد الأمر حسب النظام
        if platform.system().lower() == 'windows':
            cmd = ['ping', '-n', '1', '-w', str(int(self.timeout * 1000)), host]
        else:
            cmd = ['ping', '-c', '1', '-W', str(self.timeout), host]
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 2
                )
            )
            
            output = result.stdout + result.stderr
            
            # تحليل النتيجة
            if result.returncode == 0:
                # استخراج زمن الاستجابة
                latency = None
                
                if platform.system().lower() == 'windows':
                    # Windows: time=XXms
                    import re
                    match = re.search(r'time[<=](\d+)ms', output)
                    if match:
                        latency = float(match.group(1))
                else:
                    # Linux/Mac: time=XX.X ms
                    import re
                    match = re.search(r'time=(\d+\.?\d*) ms', output)
                    if match:
                        latency = float(match.group(1))
                
                return {'success': True, 'latency': latency}
            else:
                return {'success': False, 'latency': None}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'latency': None}
        except Exception as e:
            logger.debug(f"Ping execution error: {e}")
            return {'success': False, 'latency': None}
    
    async def _trigger_reconnect(self, worker_id: str) -> None:
        """تفعيل استدعاءات إعادة الاتصال - Trigger reconnect callbacks"""
        for callback in self._reconnect_callbacks:
            try:
                callback(worker_id)
            except Exception as e:
                logger.error(f"Error in reconnect callback: {e}")
    
    async def _trigger_disconnect(self, worker_id: str) -> None:
        """تفعيل استدعاءات فقدان الاتصال - Trigger disconnect callbacks"""
        for callback in self._disconnect_callbacks:
            try:
                callback(worker_id)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
    
    def _load_history(self) -> None:
        """تحميل التاريخ من الملف - Load history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                for worker_id, hist_data in data.items():
                    self._history[worker_id] = ConnectionHistory(
                        host=hist_data.get('host', ''),
                        total_checks=hist_data.get('total_checks', 0),
                        online_count=hist_data.get('online_count', 0),
                        offline_count=hist_data.get('offline_count', 0),
                        avg_latency=hist_data.get('avg_latency', 0.0),
                        last_seen=hist_data.get('last_seen')
                    )
                
                logger.info(f"Loaded history for {len(self._history)} workers")
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    async def _save_history(self) -> None:
        """حفظ التاريخ إلى الملف - Save history to file"""
        try:
            data = {
                worker_id: {
                    'host': hist.host,
                    'total_checks': hist.total_checks,
                    'online_count': hist.online_count,
                    'offline_count': hist.offline_count,
                    'avg_latency': hist.avg_latency,
                    'last_seen': hist.last_seen
                }
                for worker_id, hist in self._history.items()
            }
            
            async with aiofiles.open(self.history_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.info(f"Saved history for {len(self._history)} workers")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def get_worker_status(self, worker_id: str) -> Optional[ConnectionStatus]:
        """
        الحصول على حالة عامل
        Get worker status
        
        Args:
            worker_id: معرف العامل
            
        Returns:
            Optional[ConnectionStatus]: حالة الاتصال
        """
        return self._status_cache.get(worker_id)
    
    def get_all_statuses(self) -> Dict[str, ConnectionStatus]:
        """
        الحصول على حالات جميع العمال
        Get all worker statuses
        
        Returns:
            Dict[str, ConnectionStatus]: قاموس الحالات
        """
        return self._status_cache.copy()
    
    def get_worker_history(self, worker_id: str) -> Optional[ConnectionHistory]:
        """
        الحصول على تاريخ عامل
        Get worker history
        
        Args:
            worker_id: معرف العامل
            
        Returns:
            Optional[ConnectionHistory]: تاريخ الاتصال
        """
        return self._history.get(worker_id)
    
    def get_uptime_percentage(self, worker_id: str) -> float:
        """
        الحصول على نسبة تشغيل العامل
        Get worker uptime percentage
        
        Args:
            worker_id: معرف العامل
            
        Returns:
            float: نسبة التشغيل (0-100)
        """
        hist = self._history.get(worker_id)
        if not hist or hist.total_checks == 0:
            return 0.0
        
        return (hist.online_count / hist.total_checks) * 100
    
    def generate_report(self) -> Dict[str, Any]:
        """
        إنشاء تقرير شامل
        Generate comprehensive report
        
        Returns:
            Dict[str, Any]: تقرير الحالة
        """
        return {
            'timestamp': time.time(),
            'total_workers': len(self.workers),
            'online_workers': sum(
                1 for s in self._status_cache.values()
                if s == ConnectionStatus.ONLINE
            ),
            'workers': {
                worker_id: {
                    'host': host,
                    'status': self._status_cache.get(worker_id, ConnectionStatus.UNKNOWN).value,
                    'uptime_percentage': self.get_uptime_percentage(worker_id),
                    'avg_latency': self._history.get(worker_id, ConnectionHistory(host=host)).avg_latency
                }
                for worker_id, host in self.workers.items()
            }
        }


async def main():
    """الدالة الرئيسية - Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # عمال افتراضيين للاختبار
    workers = {
        'local': '127.0.0.1',
        'google': '8.8.8.8',
        'cloudflare': '1.1.1.1'
    }
    
    daemon = HealthCheckDaemon(
        workers=workers,
        check_interval=10
    )
    
    # إضافة دالة استدعاء للتسجيل
    def on_ping(worker_id: str, result: PingResult):
        status_icon = "✓" if result.status == ConnectionStatus.ONLINE else "✗"
        latency_str = f"({result.latency_ms:.2f}ms)" if result.latency_ms else ""
        print(f"{status_icon} {worker_id}: {result.status.value} {latency_str}")
    
    daemon.add_callback(on_ping)
    daemon.on_reconnect(lambda w: print(f"🔄 {w} reconnected!"))
    daemon.on_disconnect(lambda w: print(f"⚠️  {w} disconnected!"))
    
    # بدء الخدمة
    await daemon.start()
    
    # التشغيل لمدة 60 ثانية
    await asyncio.sleep(60)
    
    # إيقاف الخدمة
    await daemon.stop()
    
    # طباعة التقرير
    report = daemon.generate_report()
    print("\n" + "=" * 50)
    print("Final Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
