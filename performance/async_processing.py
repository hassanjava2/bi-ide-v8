"""
BI-IDE v8 - Async Task Processing with Celery
Task queues, scheduled jobs, and retry mechanisms
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from uuid import uuid4

from celery import Celery, Task, chain, chord, group
from celery.result import AsyncResult
from celery.schedules import crontab
from celery.signals import task_failure, task_postrun, task_prerun, task_success
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded
import redis

logger = logging.getLogger(__name__)
T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 3
    NORMAL = 6
    LOW = 9


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


@dataclass
class TaskConfig:
    """Task configuration"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 3600  # 1 hour
    task_soft_time_limit: int = 3300  # 55 minutes
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 1000


@dataclass
class RetryPolicy:
    """Retry configuration"""
    max_retries: int = 3
    countdown: int = 60  # seconds
    exponential_backoff: bool = True
    max_countdown: int = 3600  # 1 hour
    retry_exceptions: Tuple[type, ...] = (Exception,)


# Initialize Celery app
celery_app = Celery('bi_ide_v8')


def configure_celery(config: TaskConfig):
    """Configure Celery with settings"""
    celery_app.conf.update(
        broker_url=config.broker_url,
        result_backend=config.result_backend,
        task_serializer=config.task_serializer,
        accept_content=config.accept_content,
        result_serializer=config.result_serializer,
        timezone=config.timezone,
        enable_utc=config.enable_utc,
        task_track_started=config.task_track_started,
        task_time_limit=config.task_time_limit,
        task_soft_time_limit=config.task_soft_time_limit,
        worker_prefetch_multiplier=config.worker_prefetch_multiplier,
        worker_max_tasks_per_child=config.worker_max_tasks_per_child,
        task_routes={
            'ai_tasks.*': {'queue': 'ai'},
            'data_tasks.*': {'queue': 'data'},
            'email_tasks.*': {'queue': 'email'},
            'export_tasks.*': {'queue': 'export'},
            'default': {'queue': 'default'},
        },
        task_default_queue='default',
        task_default_exchange='default',
        task_default_routing_key='default',
    )


# Custom Task Class with enhanced functionality
class BaseTask(Task):
    """Enhanced base task class"""
    
    _execution_count = 0
    _start_time = None
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(
            f"Task {self.name}[{task_id}] failed: {exc}",
            extra={
                'task_id': task_id,
                'task_name': self.name,
                'args': args,
                'kwargs': kwargs,
                'exception': str(exc),
                'traceback': einfo.traceback if einfo else None
            }
        )
        
        # Send alert for critical tasks
        if self.name.startswith('critical.'):
            self._send_alert(task_id, exc)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        execution_time = time.time() - self._start_time if self._start_time else 0
        logger.info(
            f"Task {self.name}[{task_id}] succeeded in {execution_time:.2f}s"
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(
            f"Task {self.name}[{task_id}] retrying: {exc}"
        )
    
    def _send_alert(self, task_id: str, exception: Exception):
        """Send alert for critical task failure"""
        # Implementation for alerting (Slack, PagerDuty, etc.)
        logger.critical(f"ALERT: Critical task {self.name}[{task_id}] failed: {exception}")


# Task Registry
task_registry: Dict[str, Callable] = {}


def register_task(name: str, queue: str = "default", 
                  retry_policy: Optional[RetryPolicy] = None,
                  bind: bool = True):
    """Decorator to register a task"""
    def decorator(func: Callable) -> Callable:
        retry = retry_policy or RetryPolicy()
        
        @celery_app.task(
            name=name,
            queue=queue,
            base=BaseTask,
            bind=bind,
            max_retries=retry.max_retries,
            default_retry_delay=retry.countdown,
            autoretry_for=retry.retry_exceptions,
            retry_backoff=retry.exponential_backoff,
            retry_backoff_max=retry.max_countdown,
        )
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self._start_time = time.time()
            return func(self, *args, **kwargs)
        
        task_registry[name] = wrapper
        return wrapper
    return decorator


# Task Definitions

@register_task('ai_tasks.generate_insights', queue='ai', 
               retry_policy=RetryPolicy(max_retries=3, countdown=30))
def generate_ai_insights(self, data: Dict[str, Any], model: str = "gpt-4") -> Dict[str, Any]:
    """Generate AI insights from data"""
    from wise_ai_engine import WiseAIEngine
    
    try:
        engine = WiseAIEngine()
        insights = engine.analyze(data, model)
        return {
            'status': 'success',
            'insights': insights,
            'model_used': model
        }
    except SoftTimeLimitExceeded:
        logger.error("AI insight generation timed out")
        raise
    except Exception as exc:
        logger.error(f"AI insight generation failed: {exc}")
        raise self.retry(exc=exc)


@register_task('data_tasks.process_large_dataset', queue='data',
               retry_policy=RetryPolicy(max_retries=2, countdown=60))
def process_large_dataset(self, dataset_id: str, operations: List[str]) -> Dict[str, Any]:
    """Process large dataset with chunking"""
    try:
        # Implementation
        return {
            'status': 'success',
            'dataset_id': dataset_id,
            'processed_rows': 1000000
        }
    except Exception as exc:
        raise self.retry(exc=exc)


@register_task('email_tasks.send_notification', queue='email',
               retry_policy=RetryPolicy(max_retries=5, countdown=300))
def send_email_notification(self, to: str, template: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
    """Send email notification"""
    try:
        # Implementation
        return {
            'status': 'sent',
            'to': to,
            'template': template
        }
    except Exception as exc:
        raise self.retry(exc=exc)


@register_task('export_tasks.export_report', queue='export',
               retry_policy=RetryPolicy(max_retries=2, countdown=60))
def export_report(self, report_id: str, format: str = "pdf") -> Dict[str, Any]:
    """Export report to various formats"""
    try:
        # Implementation
        return {
            'status': 'exported',
            'report_id': report_id,
            'format': format,
            'download_url': f"/downloads/{report_id}.{format}"
        }
    except Exception as exc:
        raise self.retry(exc=exc)


@register_task('critical.data_sync', queue='default',
               retry_policy=RetryPolicy(max_retries=10, countdown=10))
def critical_data_sync(self, sync_id: str) -> Dict[str, Any]:
    """Critical data synchronization task"""
    try:
        # Implementation
        return {
            'status': 'synced',
            'sync_id': sync_id
        }
    except Exception as exc:
        raise self.retry(exc=exc)


class AsyncTaskProcessor:
    """Main task processor with advanced features"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self._celery = celery_app
        configure_celery(config)
        self._redis = redis.from_url(config.broker_url)
        self._scheduled_tasks: Dict[str, Any] = {}
    
    # Task Execution
    
    def submit_task(self, task_name: str, *args, **kwargs) -> AsyncResult:
        """Submit a task for execution"""
        task = self._celery.send_task(task_name, args=args, kwargs=kwargs)
        logger.info(f"Task submitted: {task_name}[{task.id}]")
        return task
    
    def submit_task_with_priority(self, task_name: str, priority: TaskPriority,
                                   *args, **kwargs) -> AsyncResult:
        """Submit task with priority"""
        task = self._celery.send_task(
            task_name, 
            args=args, 
            kwargs=kwargs,
            priority=priority.value
        )
        return task
    
    def submit_task_with_delay(self, task_name: str, countdown: int,
                                *args, **kwargs) -> AsyncResult:
        """Submit task with delay"""
        task = self._celery.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            countdown=countdown
        )
        return task
    
    def submit_task_at_time(self, task_name: str, eta: datetime,
                             *args, **kwargs) -> AsyncResult:
        """Schedule task for specific time"""
        task = self._celery.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            eta=eta
        )
        return task
    
    # Workflow Patterns
    
    def chain_tasks(self, *tasks: Dict[str, Any]) -> chain:
        """Chain tasks sequentially"""
        task_signatures = []
        for task_def in tasks:
            sig = self._celery.signature(
                task_def['name'],
                args=task_def.get('args', ()),
                kwargs=task_def.get('kwargs', {}),
                queue=task_def.get('queue', 'default')
            )
            task_signatures.append(sig)
        return chain(*task_signatures)
    
    def parallel_tasks(self, *tasks: Dict[str, Any]) -> group:
        """Run tasks in parallel"""
        task_signatures = []
        for task_def in tasks:
            sig = self._celery.signature(
                task_def['name'],
                args=task_def.get('args', ()),
                kwargs=task_def.get('kwargs', {}),
                queue=task_def.get('queue', 'default')
            )
            task_signatures.append(sig)
        return group(*task_signatures)
    
    def chord_tasks(self, *tasks: Dict[str, Any], 
                    callback: Dict[str, Any]) -> chord:
        """Run tasks in parallel with callback"""
        task_signatures = []
        for task_def in tasks:
            sig = self._celery.signature(
                task_def['name'],
                args=task_def.get('args', ()),
                kwargs=task_def.get('kwargs', {}),
                queue=task_def.get('queue', 'default')
            )
            task_signatures.append(sig)
        
        callback_sig = self._celery.signature(
            callback['name'],
            args=callback.get('args', ()),
            kwargs=callback.get('kwargs', {}),
            queue=callback.get('queue', 'default')
        )
        return chord(group(*task_signatures))(callback_sig)
    
    # Scheduled Jobs
    
    def schedule_periodic_task(self, name: str, task: str, schedule: Union[str, int],
                                args: tuple = (), kwargs: dict = None):
        """Schedule periodic task"""
        if isinstance(schedule, str):
            # Parse cron expression
            parts = schedule.split()
            if len(parts) == 5:
                schedule_obj = crontab(
                    minute=parts[0],
                    hour=parts[1],
                    day_of_month=parts[2],
                    month_of_year=parts[3],
                    day_of_week=parts[4]
                )
            else:
                raise ValueError("Invalid cron expression")
        else:
            # Interval in seconds
            schedule_obj = schedule
        
        self._celery.conf.beat_schedule[name] = {
            'task': task,
            'schedule': schedule_obj,
            'args': args,
            'kwargs': kwargs or {}
        }
        self._scheduled_tasks[name] = {
            'task': task,
            'schedule': schedule
        }
        logger.info(f"Scheduled task: {name} -> {task} ({schedule})")
    
    def remove_scheduled_task(self, name: str):
        """Remove scheduled task"""
        if name in self._celery.conf.beat_schedule:
            del self._celery.conf.beat_schedule[name]
            del self._scheduled_tasks[name]
            logger.info(f"Removed scheduled task: {name}")
    
    # Task Management
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        result = AsyncResult(task_id, app=self._celery)
        if result.ready():
            return result.get()
        return None
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status"""
        result = AsyncResult(task_id, app=self._celery)
        status_map = {
            'PENDING': TaskStatus.PENDING,
            'STARTED': TaskStatus.STARTED,
            'SUCCESS': TaskStatus.SUCCESS,
            'FAILURE': TaskStatus.FAILURE,
            'RETRY': TaskStatus.RETRY,
            'REVOKED': TaskStatus.REVOKED,
        }
        return status_map.get(result.state, TaskStatus.PENDING)
    
    def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """Revoke/cancel a task"""
        self._celery.control.revoke(task_id, terminate=terminate)
        logger.info(f"Task revoked: {task_id}")
        return True
    
    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        result = AsyncResult(task_id, app=self._celery)
        if result.state == 'FAILURE':
            result.retry()
            return True
        return False
    
    # Queue Management
    
    def get_queue_length(self, queue: str = "default") -> int:
        """Get number of tasks in queue"""
        inspect = self._celery.control.inspect()
        active_queues = inspect.active_queues()
        # This is simplified; real implementation would use Redis/RabbitMQ directly
        return 0
    
    def purge_queue(self, queue: str = "default") -> int:
        """Purge all tasks from queue"""
        count = self._celery.control.purge()
        logger.info(f"Purged {count} tasks from queue")
        return count
    
    def get_active_tasks(self) -> Dict[str, Any]:
        """Get currently active tasks"""
        inspect = self._celery.control.inspect()
        return inspect.active() or {}
    
    def get_scheduled_tasks(self) -> Dict[str, Any]:
        """Get scheduled tasks"""
        inspect = self._celery.control.inspect()
        return inspect.scheduled() or {}
    
    # Rate Limiting
    
    def rate_limit_task(self, task_name: str, rate: str):
        """Set rate limit for task (e.g., '10/m')"""
        self._celery.control.rate_limit(task_name, rate)
        logger.info(f"Rate limit set for {task_name}: {rate}")
    
    # Monitoring
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        inspect = self._celery.control.inspect()
        return {
            'active': inspect.active(),
            'scheduled': inspect.scheduled(),
            'reserved': inspect.reserved(),
            'revoked': inspect.revoked(),
            'stats': inspect.stats(),
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check Celery health"""
        try:
            ping = self._celery.control.ping(timeout=5.0)
            return {
                'broker_connected': bool(ping),
                'workers_available': len(ping) > 0 if ping else False
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'broker_connected': False,
                'workers_available': False
            }


# Signal handlers
@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    """Handle task pre-run"""
    logger.debug(f"Task starting: {task.name}[{task_id}]")


@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
    """Handle task post-run"""
    logger.debug(f"Task completed: {task.name}[{task_id}] - {state}")


@task_success.connect
def on_task_success(sender=None, result=None, **kwargs):
    """Handle task success"""
    pass


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure"""
    logger.error(f"Task failed: {task_id} - {exception}")


# Singleton instance
_processor: Optional[AsyncTaskProcessor] = None


def init_processor(config: TaskConfig) -> AsyncTaskProcessor:
    """Initialize global task processor"""
    global _processor
    _processor = AsyncTaskProcessor(config)
    return _processor


def get_processor() -> AsyncTaskProcessor:
    """Get global task processor"""
    if _processor is None:
        raise RuntimeError("Task processor not initialized")
    return _processor


# Convenience functions
def submit_task(task_name: str, *args, **kwargs) -> AsyncResult:
    """Submit task convenience function"""
    return get_processor().submit_task(task_name, *args, **kwargs)


def schedule_task(task_name: str, eta: datetime, *args, **kwargs) -> AsyncResult:
    """Schedule task convenience function"""
    return get_processor().submit_task_at_time(task_name, eta, *args, **kwargs)
