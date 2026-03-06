"""
Brain Scheduler - مجدول الدماغ

Automatically schedules training jobs based on priorities and resources.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """أولويات المهام"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    IDLE = 5


class JobStatus(Enum):
    """حالات المهام"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduleJob:
    """مهمة مجدولة"""
    job_id: str
    name: str
    layer_name: str
    priority: JobPriority
    config: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_worker: Optional[str] = None
    error_message: Optional[str] = None


class Scheduler:
    """
    مجدول المهام الذكي
    
    يدير:
    - قائمة انتظار المهام
    - توزيع الموارد
    - التدريب في وقت الفراغ
    """
    
    def __init__(
        self,
        check_interval: int = 60,
        max_concurrent: int = 3,
        idle_training: bool = True
    ):
        self.check_interval = check_interval
        self.max_concurrent = max_concurrent
        self.idle_training = idle_training
        
        self.jobs: Dict[str, ScheduleJob] = {}
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_jobs: Dict[str, ScheduleJob] = {}
        self.is_running = False
        self._callbacks: List[Callable] = []
        
    def add_job(
        self,
        name: str,
        layer_name: str,
        priority: JobPriority = JobPriority.MEDIUM,
        config: Dict[str, Any] = None
    ) -> ScheduleJob:
        """إضافة مهمة جديدة"""
        job_id = str(uuid.uuid4())[:12]
        
        job = ScheduleJob(
            job_id=job_id,
            name=name,
            layer_name=layer_name,
            priority=priority,
            config=config or {}
        )
        
        self.jobs[job_id] = job
        
        # Add to priority queue (lower number = higher priority)
        self.queue.put_nowait((priority.value, job_id))
        
        logger.info(f"Added job {job_id}: {name} (priority={priority.name})")
        
        return job
    
    def get_next_job(self) -> Optional[ScheduleJob]:
        """الحصول على المهمة التالية من قائمة الانتظار"""
        try:
            while not self.queue.empty():
                _, job_id = self.queue.get_nowait()
                
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    if job.status == JobStatus.PENDING:
                        return job
        except asyncio.QueueEmpty:
            pass
        
        return None
    
    def start_job(self, job_id: str, worker_id: str) -> bool:
        """بدء تنفيذ مهمة"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if len(self.running_jobs) >= self.max_concurrent:
            logger.warning(f"Cannot start job {job_id}: max concurrent reached")
            return False
        
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.assigned_worker = worker_id
        
        self.running_jobs[job_id] = job
        
        logger.info(f"Started job {job_id} on worker {worker_id}")
        
        # Notify callbacks
        self._notify_callbacks("job_started", job)
        
        return True
    
    def complete_job(self, job_id: str, success: bool = True, error: str = None) -> bool:
        """إكمال مهمة"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error_message = error
        
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
        
        logger.info(f"Completed job {job_id}: {'success' if success else 'failed'}")
        
        # Notify callbacks
        self._notify_callbacks("job_completed", job)
        
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """إلغاء مهمة"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status == JobStatus.RUNNING:
            # Job is running, need to signal worker to stop
            pass
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
        
        logger.info(f"Cancelled job {job_id}")
        
        return True
    
    def check_idle_training(
        self,
        current_cpu: float,
        current_gpu: float,
        active_jobs: int
    ) -> Optional[ScheduleJob]:
        """
        التحقق مما إذا كان يمكن بدء تدريب في وقت الفراغ
        
        Args:
            current_cpu: نسبة استخدام CPU الحالية
            current_gpu: نسبة استخدام GPU الحالية
            active_jobs: عدد المهام النشطة
            
        Returns:
            ScheduleJob | None: مهمة للتشغيل أو None
        """
        if not self.idle_training:
            return None
        
        if active_jobs >= self.max_concurrent:
            return None
        
        # Check if system is idle enough
        if current_cpu > 30 or current_gpu > 20:
            return None
        
        # Look for idle priority jobs
        for job in self.jobs.values():
            if job.status == JobStatus.PENDING and job.priority == JobPriority.IDLE:
                return job
        
        # If no idle jobs, can we run a low priority one?
        if current_cpu < 20 and current_gpu < 10:
            for job in self.jobs.values():
                if job.status == JobStatus.PENDING and job.priority == JobPriority.LOW:
                    return job
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الجدولة"""
        return {
            "is_running": self.is_running,
            "total_jobs": len(self.jobs),
            "pending_jobs": sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED),
            "failed_jobs": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED),
            "max_concurrent": self.max_concurrent,
            "idle_training": self.idle_training,
        }
    
    def list_jobs(
        self,
        status: JobStatus = None,
        limit: int = 50
    ) -> List[ScheduleJob]:
        """قائمة المهام"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by created_at, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    def register_callback(self, callback: Callable):
        """تسجيل دالة استدعاء للأحداث"""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, job: ScheduleJob):
        """إشعار جميع المستمعين"""
        for callback in self._callbacks:
            try:
                callback(event, job)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def run_scheduler_loop(self):
        """حلقة الجدولة الرئيسية"""
        self.is_running = True
        logger.info("Scheduler loop started")
        
        while self.is_running:
            try:
                # Try to schedule pending jobs
                while len(self.running_jobs) < self.max_concurrent:
                    job = self.get_next_job()
                    if not job:
                        break
                    
                    # In real implementation, this would assign to a worker
                    # For now, we just mark as scheduled
                    job.status = JobStatus.SCHEDULED
                    job.scheduled_at = datetime.now()
                    
                    logger.info(f"Scheduled job {job.job_id}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)
        
        logger.info("Scheduler loop stopped")
    
    def stop(self):
        """إيقاف الجدولة"""
        self.is_running = False
