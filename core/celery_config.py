"""
Celery Configuration - إعدادات المهام في الخلفية
"""
import os
from celery import Celery
from celery.signals import task_prerun, task_postrun
from .logging_config import logger

# Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    'bi_ide',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        'core.tasks',
    ]
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """Log task start"""
    logger.info(f"Task started: {task.name}[{task_id}]")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """Log task completion"""
    logger.info(f"Task completed: {task.name}[{task_id}] - Status: {state}")


# Tasks
def training_task(func):
    """Decorator for training tasks"""
    return celery_app.task(
        bind=True,
        max_retries=3,
        default_retry_delay=60
    )(func)
