"""
مراقب التدريب - Training Monitor
===============================
مراقبة مقاييس التدريب في الوقت الفعلي
Real-time training metrics monitoring
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """مقاييس التدريب - Training metrics"""
    run_id: str
    step: int
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    gpu_utilization: Optional[float] = None
    timestamp: Optional[datetime] = None
    extra_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TrainingRun:
    """جلسة تدريب - Training run"""
    run_id: str
    model_name: str
    dataset_name: str
    start_time: datetime
    worker_id: str
    config: Dict[str, Any]
    status: str = 'running'  # running, completed, failed, paused
    end_time: Optional[datetime] = None
    final_metrics: Optional[Dict[str, Any]] = None


class TrainingMonitor:
    """
    مراقب التدريب
    Training monitor
    
    يدعم تخزين المقاييس في PostgreSQL والتحديثات الفورية عبر WebSocket
    Supports PostgreSQL storage and real-time updates via WebSocket
    """
    
    def __init__(
        self,
        db_url: str,
        websocket_manager: Optional[Any] = None
    ):
        """
        تهيئة مراقب التدريب
        Initialize training monitor
        
        Args:
            db_url: عنوان قاعدة بيانات PostgreSQL
            websocket_manager: مدير WebSocket للتحديثات الفورية
        """
        self.db_url = db_url
        self.websocket_manager = websocket_manager
        self._pool: Optional[Pool] = None
        self._active_runs: Dict[str, TrainingRun] = {}
        self._callbacks: List[Callable[[TrainingMetrics], None]] = []
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """تهيئة الاتصال بقاعدة البيانات - Initialize database connection"""
        try:
            self._pool = await asyncpg.create_pool(self.db_url)
            await self._create_tables()
            logger.info("Training monitor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize training monitor: {e}")
            raise
    
    async def close(self) -> None:
        """إغلاق الاتصال بقاعدة البيانات - Close database connection"""
        if self._pool:
            await self._pool.close()
            logger.info("Training monitor closed")
    
    async def _create_tables(self) -> None:
        """إنشاء جداول قاعدة البيانات - Create database tables"""
        async with self._pool.acquire() as conn:
            # جدول جلسات التدريب
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id VARCHAR(255) PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    dataset_name VARCHAR(255) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    worker_id VARCHAR(255) NOT NULL,
                    config JSONB,
                    status VARCHAR(50) DEFAULT 'running',
                    final_metrics JSONB
                )
            ''')
            
            # جدول مقاييس التدريب
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) REFERENCES training_runs(run_id),
                    step INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL,
                    learning_rate REAL,
                    throughput_samples_per_sec REAL,
                    gpu_utilization REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra_metrics JSONB
                )
            ''')
            
            # فهرس للبحث السريع
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_metrics_run_id 
                ON training_metrics(run_id)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp 
                ON training_metrics(timestamp)
            ''')
    
    async def track_training_run(
        self,
        run_id: str,
        model_name: str,
        dataset_name: str,
        worker_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> TrainingRun:
        """
        بدء متابعة جلسة تدريب جديدة
        Start tracking a new training run
        
        Args:
            run_id: معرف جلسة التدريب
            model_name: اسم النموذج
            dataset_name: اسم مجموعة البيانات
            worker_id: معرف العامل
            config: إعدادات التدريب
            
        Returns:
            TrainingRun: جلسة التدريب
        """
        run = TrainingRun(
            run_id=run_id,
            model_name=model_name,
            dataset_name=dataset_name,
            start_time=datetime.utcnow(),
            worker_id=worker_id,
            config=config or {}
        )
        
        async with self._lock:
            self._active_runs[run_id] = run
        
        # حفظ في قاعدة البيانات
        async with self._pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO training_runs 
                (run_id, model_name, dataset_name, start_time, worker_id, config, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', run_id, model_name, dataset_name, run.start_time, 
                 worker_id, json.dumps(config or {}), 'running')
        
        logger.info(f"Started tracking training run: {run_id}")
        
        # إرسال تحديث عبر WebSocket
        await self._broadcast_update('training_started', {
            'run_id': run_id,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'worker_id': worker_id
        })
        
        return run
    
    async def log_metrics(self, metrics: TrainingMetrics) -> None:
        """
        تسجيل مقاييس التدريب
        Log training metrics
        
        Args:
            metrics: مقاييس التدريب
        """
        # حفظ في قاعدة البيانات
        async with self._pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO training_metrics 
                (run_id, step, epoch, loss, accuracy, learning_rate, 
                 throughput_samples_per_sec, gpu_utilization, timestamp, extra_metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ''', metrics.run_id, metrics.step, metrics.epoch, metrics.loss,
                 metrics.accuracy, metrics.learning_rate,
                 metrics.throughput_samples_per_sec, metrics.gpu_utilization,
                 metrics.timestamp, json.dumps(metrics.extra_metrics or {}))
        
        # استدعاء الدوال المسجلة
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
        
        # إرسال تحديث عبر WebSocket
        await self._broadcast_update('metrics_update', {
            'run_id': metrics.run_id,
            'step': metrics.step,
            'epoch': metrics.epoch,
            'loss': metrics.loss,
            'accuracy': metrics.accuracy,
            'timestamp': metrics.timestamp.isoformat()
        })
    
    async def complete_run(
        self,
        run_id: str,
        status: str = 'completed',
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        إنهاء جلسة تدريب
        Complete a training run
        
        Args:
            run_id: معرف جلسة التدريب
            status: الحالة النهائية (completed, failed)
            final_metrics: المقاييس النهائية
        """
        end_time = datetime.utcnow()
        
        async with self._lock:
            if run_id in self._active_runs:
                run = self._active_runs[run_id]
                run.status = status
                run.end_time = end_time
                run.final_metrics = final_metrics
                del self._active_runs[run_id]
        
        # تحديث قاعدة البيانات
        async with self._pool.acquire() as conn:
            await conn.execute('''
                UPDATE training_runs 
                SET status = $1, end_time = $2, final_metrics = $3
                WHERE run_id = $4
            ''', status, end_time, json.dumps(final_metrics or {}), run_id)
        
        logger.info(f"Training run {run_id} completed with status: {status}")
        
        # إرسال تحديث عبر WebSocket
        await self._broadcast_update('training_completed', {
            'run_id': run_id,
            'status': status,
            'final_metrics': final_metrics
        })
    
    async def get_loss_curve(
        self,
        run_id: str,
        step_interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        الحصول على منحنى الخسارة
        Get loss curve
        
        Args:
            run_id: معرف جلسة التدريب
            step_interval: فاصل الخطوات للعينة
            
        Returns:
            List[Dict[str, Any]]: قائمة نقاط البيانات
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT step, epoch, loss, timestamp
                FROM training_metrics
                WHERE run_id = $1 AND step % $2 = 0
                ORDER BY step ASC
            ''', run_id, step_interval)
        
        return [
            {
                'step': row['step'],
                'epoch': row['epoch'],
                'loss': row['loss'],
                'timestamp': row['timestamp'].isoformat()
            }
            for row in rows
        ]
    
    async def get_accuracy_curve(
        self,
        run_id: str,
        step_interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        الحصول على منحنى الدقة
        Get accuracy curve
        
        Args:
            run_id: معرف جلسة التدريب
            step_interval: فاصل الخطوات للعينة
            
        Returns:
            List[Dict[str, Any]]: قائمة نقاط البيانات
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT step, epoch, accuracy, timestamp
                FROM training_metrics
                WHERE run_id = $1 
                    AND step % $2 = 0
                    AND accuracy IS NOT NULL
                ORDER BY step ASC
            ''', run_id, step_interval)
        
        return [
            {
                'step': row['step'],
                'epoch': row['epoch'],
                'accuracy': row['accuracy'],
                'timestamp': row['timestamp'].isoformat()
            }
            for row in rows
        ]
    
    async def get_throughput(
        self,
        run_id: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        الحصول على الإنتاجية
        Get throughput statistics
        
        Args:
            run_id: معرف جلسة التدريب
            window_size: حجم نافذة المتوسط المتحرك
            
        Returns:
            Dict[str, Any]: إحصائيات الإنتاجية
        """
        async with self._pool.acquire() as conn:
            # المتوسط الكلي
            avg_row = await conn.fetchrow('''
                SELECT AVG(throughput_samples_per_sec) as avg_throughput,
                       MAX(throughput_samples_per_sec) as max_throughput,
                       MIN(throughput_samples_per_sec) as min_throughput
                FROM training_metrics
                WHERE run_id = $1 AND throughput_samples_per_sec IS NOT NULL
            ''', run_id)
            
            # المتوسط المتحرك الأخير
            recent_rows = await conn.fetch('''
                SELECT throughput_samples_per_sec
                FROM training_metrics
                WHERE run_id = $1 AND throughput_samples_per_sec IS NOT NULL
                ORDER BY step DESC
                LIMIT $2
            ''', run_id, window_size)
        
        recent_values = [r['throughput_samples_per_sec'] for r in recent_rows]
        recent_avg = sum(recent_values) / len(recent_values) if recent_values else 0
        
        return {
            'run_id': run_id,
            'average_throughput': avg_row['avg_throughput'] or 0,
            'max_throughput': avg_row['max_throughput'] or 0,
            'min_throughput': avg_row['min_throughput'] or 0,
            'recent_average': recent_avg,
            'window_size': len(recent_values)
        }
    
    async def get_active_runs(self) -> List[TrainingRun]:
        """
        الحصول على الجلسات النشطة
        Get active training runs
        
        Returns:
            List[TrainingRun]: قائمة الجلسات النشطة
        """
        async with self._lock:
            return list(self._active_runs.values())
    
    async def get_run_history(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingRun]:
        """
        الحصول على تاريخ التدريب
        Get training run history
        
        Args:
            limit: الحد الأقصى للنتائج
            offset: الإزاحة
            
        Returns:
            List[TrainingRun]: قائمة جلسات التدريب
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT run_id, model_name, dataset_name, start_time, end_time,
                       worker_id, config, status, final_metrics
                FROM training_runs
                ORDER BY start_time DESC
                LIMIT $1 OFFSET $2
            ''', limit, offset)
        
        return [
            TrainingRun(
                run_id=row['run_id'],
                model_name=row['model_name'],
                dataset_name=row['dataset_name'],
                start_time=row['start_time'],
                worker_id=row['worker_id'],
                config=row['config'] or {},
                status=row['status'],
                end_time=row['end_time'],
                final_metrics=row['final_metrics']
            )
            for row in rows
        ]
    
    def add_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """إضافة دالة استدعاء للمقاييس - Add metrics callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """إزالة دالة الاستدعاء - Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def _broadcast_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """بث التحديثات عبر WebSocket - Broadcast updates via WebSocket"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast({
                    'type': event_type,
                    'data': data
                })
            except Exception as e:
                logger.error(f"Error broadcasting WebSocket update: {e}")
