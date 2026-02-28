"""
مجمع السجلات - Log Aggregator
=============================
جمع وإدارة السجلات من جميع الخدمات
Collect and manage logs from all services
"""

import asyncio
import gzip
import json
import logging
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator, Callable, Set

import aiofiles
from aiofiles import os as aio_os

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """مستويات السجل - Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """إدخال السجل - Log entry"""
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    worker_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None  # ملف المصدر
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس - Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'service': self.service,
            'message': self.message,
            'worker_id': self.worker_id,
            'trace_id': self.trace_id,
            'metadata': self.metadata,
            'source': self.source,
            'line_number': self.line_number
        }
    
    def to_json(self) -> str:
        """تحويل إلى JSON - Convert to JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """إنشاء من قاموس - Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            level=LogLevel(data['level']),
            service=data['service'],
            message=data['message'],
            worker_id=data.get('worker_id'),
            trace_id=data.get('trace_id'),
            metadata=data.get('metadata'),
            source=data.get('source'),
            line_number=data.get('line_number')
        )


class LogAggregator:
    """
    مجمع السجلات
    Log aggregator
    
    يجمع السجلات من جميع الخدمات ويدعم السجلات المنظمة (JSON)
    Collects logs from all services and supports structured logging (JSON)
    
    سياسة الاستبقاء: 30 يوم
    Retention policy: 30 days
    """
    
    DEFAULT_RETENTION_DAYS = 30
    
    def __init__(
        self,
        log_dir: Path,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_file_size_mb: int = 100,
        buffer_size: int = 1000
    ):
        """
        تهيئة مجمع السجلات
        Initialize log aggregator
        
        Args:
            log_dir: دليل تخزين السجلات
            retention_days: أيام الاستبقاء
            max_file_size_mb: الحد الأقصى لحمل الملف بالميجابايت
            buffer_size: حجم المخزن المؤقت
        """
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.buffer_size = buffer_size
        
        self._buffer: List[LogEntry] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[LogEntry], None]] = []
        self._subscribers: Set[asyncio.Queue] = set()
        
        # التأكد من وجود الدليل
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """تهيئة مجمع السجلات - Initialize log aggregator"""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        # بدء مهمة تنظيف السجلات القديمة
        asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Log aggregator initialized: {self.log_dir}")
    
    async def close(self) -> None:
        """إغلاق مجمع السجلات - Close log aggregator"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # حفظ المخزن المتبقي
        await self._flush_buffer()
        
        logger.info("Log aggregator closed")
    
    async def collect_logs(
        self,
        level: LogLevel,
        service: str,
        message: str,
        worker_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> LogEntry:
        """
        جمع سجل جديد
        Collect a new log entry
        
        Args:
            level: مستوى السجل
            service: اسم الخدمة
            message: رسالة السجل
            worker_id: معرف العامل
            trace_id: معرف التتبع
            metadata: بيانات إضافية
            source: ملف المصدر
            line_number: رقم السطر
            
        Returns:
            LogEntry: إدخال السجل
        """
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            service=service,
            message=message,
            worker_id=worker_id,
            trace_id=trace_id,
            metadata=metadata,
            source=source,
            line_number=line_number
        )
        
        async with self._lock:
            self._buffer.append(entry)
            
            # حفظ إذا امتلأ المخزن
            if len(self._buffer) >= self.buffer_size:
                await self._flush_buffer()
        
        # إشعار المشتركين
        for queue in self._subscribers:
            try:
                await queue.put(entry)
            except Exception:
                pass
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in log callback: {e}")
        
        return entry
    
    async def collect_json_log(self, json_data: str) -> Optional[LogEntry]:
        """
        جمع سجل من JSON
        Collect log from JSON
        
        Args:
            json_data: بيانات JSON
            
        Returns:
            Optional[LogEntry]: إدخال السجل أو None
        """
        try:
            data = json.loads(json_data)
            entry = LogEntry.from_dict(data)
            
            async with self._lock:
                self._buffer.append(entry)
                
                if len(self._buffer) >= self.buffer_size:
                    await self._flush_buffer()
            
            return entry
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON log: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing JSON log: {e}")
            return None
    
    async def _flush_buffer(self) -> None:
        """حفظ المخزن المؤقت إلى الملف - Flush buffer to file"""
        if not self._buffer:
            return
        
        entries = self._buffer.copy()
        self._buffer.clear()
        
        # تجميع حسب اليوم والخدمة
        grouped: Dict[tuple, List[LogEntry]] = {}
        for entry in entries:
            key = (entry.timestamp.date(), entry.service)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entry)
        
        # كتابة إلى الملفات
        for (date, service), service_entries in grouped.items():
            await self._write_to_file(date, service, service_entries)
    
    async def _write_to_file(
        self,
        date: datetime.date,
        service: str,
        entries: List[LogEntry]
    ) -> None:
        """كتابة السجلات إلى ملف - Write logs to file"""
        # إنشاء هيكل الدليل: logs/2024/01/15/service.log
        log_path = self.log_dir / str(date.year) / f"{date.month:02d}"
        await aio_os.makedirs(log_path, exist_ok=True)
        
        log_file = log_path / f"{date.day:02d}_{service}.log"
        
        # التحقق من حجم الملف وتدويره إذا لزم الأمر
        await self._rotate_if_needed(log_file)
        
        # كتابة السجلات
        lines = [entry.to_json() + '\n' for entry in entries]
        
        async with aiofiles.open(log_file, 'a', encoding='utf-8') as f:
            await f.writelines(lines)
    
    async def _rotate_if_needed(self, log_file: Path) -> None:
        """تدوير الملف إذا تجاوز الحجم - Rotate file if size exceeded"""
        try:
            stat = await aio_os.stat(log_file)
            if stat.st_size > self.max_file_size:
                # ضغط الملف القديم
                rotated = log_file.with_suffix('.log.gz')
                counter = 1
                while rotated.exists():
                    rotated = log_file.with_suffix(f'.log.{counter}.gz')
                    counter += 1
                
                # ضغط الملف
                await self._compress_file(log_file, rotated)
                await aio_os.remove(log_file)
                
        except FileNotFoundError:
            pass
    
    async def _compress_file(self, source: Path, destination: Path) -> None:
        """ضغط ملف - Compress file"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._compress_sync, source, destination)
    
    def _compress_sync(self, source: Path, destination: Path) -> None:
        """ضغط ملف بشكل متزامن - Compress file synchronously"""
        with open(source, 'rb') as f_in:
            with gzip.open(destination, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    async def _periodic_flush(self) -> None:
        """الحفظ الدوري للمخزن - Periodic buffer flush"""
        while self._running:
            try:
                await asyncio.sleep(5)  # حفظ كل 5 ثواني
                async with self._lock:
                    await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _cleanup_loop(self) -> None:
        """حلقة تنظيف السجلات القديمة - Old logs cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # تنظيف كل ساعة
                await self._cleanup_old_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """تنظيف السجلات القديمة - Clean up old logs"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        try:
            for year_dir in self.log_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    for log_file in month_dir.iterdir():
                        if not log_file.is_file():
                            continue
                        
                        # استخراج التاريخ من اسم الملف
                        try:
                            stat = await aio_os.stat(log_file)
                            file_time = datetime.fromtimestamp(stat.st_mtime)
                            
                            if file_time < cutoff_date:
                                await aio_os.remove(log_file)
                                logger.debug(f"Removed old log: {log_file}")
                        except Exception as e:
                            logger.error(f"Error checking log file: {e}")
        
        except Exception as e:
            logger.error(f"Error during log cleanup: {e}")
    
    async def search_logs(
        self,
        query: Optional[str] = None,
        level: Optional[LogLevel] = None,
        service: Optional[str] = None,
        worker_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        trace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[LogEntry]:
        """
        البحث في السجلات
        Search logs
        
        Args:
            query: نص البحث
            level: تصفية حسب المستوى
            service: تصفية حسب الخدمة
            worker_id: تصفية حسب العامل
            start_time: وقت البدء
            end_time: وقت الانتهاء
            trace_id: معرف التتبع
            limit: الحد الأقصى للنتائج
            offset: الإزاحة
            
        Returns:
            List[LogEntry]: قائمة السجلات
        """
        results = []
        
        # تحديد الملفات للبحث
        log_files = await self._get_log_files(start_time, end_time)
        
        for log_file in log_files:
            entries = await self._read_log_file(log_file)
            
            for entry in entries:
                # التصفية
                if level and entry.level != level:
                    continue
                
                if service and entry.service != service:
                    continue
                
                if worker_id and entry.worker_id != worker_id:
                    continue
                
                if trace_id and entry.trace_id != trace_id:
                    continue
                
                if start_time and entry.timestamp < start_time:
                    continue
                
                if end_time and entry.timestamp > end_time:
                    continue
                
                if query and query.lower() not in entry.message.lower():
                    continue
                
                results.append(entry)
                
                if len(results) >= limit + offset:
                    break
            
            if len(results) >= limit + offset:
                break
        
        # تطبيق الإزاحة والحد
        return results[offset:offset + limit]
    
    async def _get_log_files(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[Path]:
        """الحصول على ملفات السجلات - Get log files"""
        files = []
        
        try:
            for year_dir in self.log_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                year = int(year_dir.name)
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    month = int(month_dir.name)
                    
                    for log_file in month_dir.iterdir():
                        if not log_file.is_file():
                            continue
                        
                        # التحقق من نطاق التاريخ
                        if start_time or end_time:
                            try:
                                stat = await aio_os.stat(log_file)
                                file_time = datetime.fromtimestamp(stat.st_mtime)
                                
                                if start_time and file_time < start_time:
                                    continue
                                
                                if end_time and file_time > end_time:
                                    continue
                            except Exception:
                                pass
                        
                        files.append(log_file)
            
            # ترتيب حسب الوقت (الأحدث أولاً)
            files.sort(reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting log files: {e}")
        
        return files
    
    async def _read_log_file(self, log_file: Path) -> List[LogEntry]:
        """قراءة ملف السجل - Read log file"""
        entries = []
        
        try:
            # إذا كان مضغوطاً
            if log_file.suffix == '.gz':
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(None, self._read_gzip_sync, log_file)
            else:
                async with aiofiles.open(log_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
            
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    entry = LogEntry.from_dict(data)
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error parsing log line: {e}")
        
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
        
        return entries
    
    def _read_gzip_sync(self, log_file: Path) -> str:
        """قراءة ملف مضغوط بشكل متزامن - Read gzip file synchronously"""
        with gzip.open(log_file, 'rt', encoding='utf-8') as f:
            return f.read()
    
    async def tail_logs(
        self,
        lines: int = 100,
        level: Optional[LogLevel] = None,
        service: Optional[str] = None
    ) -> AsyncIterator[LogEntry]:
        """
        متابعة السجلات في الوقت الفعلي
        Tail logs in real-time
        
        Args:
            lines: عدد الأسطر الأولية
            level: تصفية حسب المستوى
            service: تصفية حسب الخدمة
            
        Yields:
            LogEntry: إدخالات السجل
        """
        queue: asyncio.Queue[LogEntry] = asyncio.Queue()
        self._subscribers.add(queue)
        
        try:
            # إرسال السجلات الحالية أولاً
            current_logs = await self.search_logs(
                level=level,
                service=service,
                limit=lines
            )
            for entry in reversed(current_logs):
                yield entry
            
            # متابعة السجلات الجديدة
            while self._running:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    if level and entry.level != level:
                        continue
                    
                    if service and entry.service != service:
                        continue
                    
                    yield entry
                    
                except asyncio.TimeoutError:
                    continue
        
        finally:
            self._subscribers.discard(queue)
    
    def add_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """إضافة دالة استدعاء للسجلات - Add log callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_services(self) -> List[str]:
        """الحصول على قائمة الخدمات - Get list of services"""
        services = set()
        
        log_files = await self._get_log_files(None, None)
        
        for log_file in log_files:
            # استخراج اسم الخدمة من اسم الملف
            match = re.match(r'\d{2}_(.+)\.log', log_file.name)
            if match:
                services.add(match.group(1))
        
        return sorted(list(services))
    
    async def get_statistics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        الحصول على إحصائيات السجلات
        Get log statistics
        
        Args:
            hours: عدد الساعات للإحصائيات
            
        Returns:
            Dict[str, Any]: الإحصائيات
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        level_counts = {level: 0 for level in LogLevel}
        service_counts: Dict[str, int] = {}
        total = 0
        
        log_files = await self._get_log_files(start_time, None)
        
        for log_file in log_files:
            entries = await self._read_log_file(log_file)
            
            for entry in entries:
                if entry.timestamp < start_time:
                    continue
                
                level_counts[entry.level] += 1
                service_counts[entry.service] = service_counts.get(entry.service, 0) + 1
                total += 1
        
        return {
            'total_logs': total,
            'level_distribution': {level.value: count for level, count in level_counts.items()},
            'service_distribution': service_counts,
            'period_hours': hours
        }
