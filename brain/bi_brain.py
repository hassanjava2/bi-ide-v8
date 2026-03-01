"""
BI Brain - دماغ BI

Main brain orchestrator that coordinates scheduling, evaluation, and idle training.
"""

import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from .scheduler import Scheduler, JobPriority
from .evaluator import ModelEvaluator, EvaluationType
from .config import BrainConfig

logger = logging.getLogger(__name__)


class BIBrain:
    """
    دماغ BI الرئيسي
    
    ينسق بين:
    - الجدولة التلقائية
    - تقييم النماذج
    - التدريب في وقت الفراغ
    """
    
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig.from_env()
        self.scheduler = Scheduler(
            check_interval=self.config.check_interval_seconds,
            max_concurrent=self.config.max_concurrent_jobs,
            idle_training=self.config.idle_training_enabled
        )
        self.evaluator = ModelEvaluator(
            min_improvement_delta=self.config.min_improvement_delta
        )
        self.is_running = False
        self._idle_training_task = None
        
        # Ensure directories exist
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Register scheduler callbacks
        self.scheduler.register_callback(self._on_job_event)
        
        logger.info("BI Brain initialized")
    
    async def start(self):
        """بدء الدماغ"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting BI Brain...")
        
        # Start scheduler
        scheduler_task = asyncio.create_task(self.scheduler.run_scheduler_loop())
        
        # Start idle training monitor (if enabled)
        if self.config.idle_training_enabled:
            self._idle_training_task = asyncio.create_task(self._idle_training_loop())
        
        logger.info("BI Brain is running")
        
        return scheduler_task
    
    def stop(self):
        """إيقاف الدماغ"""
        self.is_running = False
        self.scheduler.stop()
        
        if self._idle_training_task:
            self._idle_training_task.cancel()
        
        logger.info("BI Brain stopped")
    
    async def schedule_training(
        self,
        name: str,
        layer_name: str = "general",
        priority: str = "medium",
        config: Dict[str, Any] = None
    ) -> str:
        """
        جدولة مهمة تدريب
        
        Args:
            name: اسم المهمة
            layer_name: اسم الطبقة
            priority: الأولوية (critical, high, medium, low, idle)
            config: إعدادات إضافية
            
        Returns:
            str: معرف المهمة
        """
        priority_map = {
            "critical": JobPriority.CRITICAL,
            "high": JobPriority.HIGH,
            "medium": JobPriority.MEDIUM,
            "low": JobPriority.LOW,
            "idle": JobPriority.IDLE
        }
        
        job_priority = priority_map.get(priority.lower(), JobPriority.MEDIUM)
        
        job = self.scheduler.add_job(
            name=name,
            layer_name=layer_name,
            priority=job_priority,
            config=config or {}
        )
        
        logger.info(f"Scheduled training job: {job.job_id} ({priority})")
        
        return job.job_id
    
    async def evaluate_model(
        self,
        model_id: str,
        model_name: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        تقييم نموذج والتحقق من جاهزيته للنشر
        
        Args:
            model_id: معرف النموذج
            model_name: اسم النموذج
            metrics: المقاييس
            
        Returns:
            Dict: نتيجة التقييم
        """
        result = await self.evaluator.evaluate(
            model_id=model_id,
            model_name=model_name,
            metrics=metrics,
            evaluation_type=EvaluationType.PRE_DEPLOY
        )
        
        should_deploy = self.evaluator.should_deploy(result.evaluation_id)
        
        return {
            "evaluation_id": result.evaluation_id,
            "model_id": model_id,
            "improvement_percentage": result.improvement_percentage,
            "passed_threshold": result.passed_threshold,
            "should_deploy": should_deploy,
            "metrics": result.metrics
        }
    
    async def _idle_training_loop(self):
        """حلقة مراقبة التدريب في وقت الفراغ"""
        logger.info("Idle training monitor started")
        
        while self.is_running:
            try:
                # In real implementation, this would check actual system metrics
                # For now, we simulate
                current_cpu = await self._get_current_cpu()
                current_gpu = await self._get_current_gpu()
                
                idle_job = self.scheduler.check_idle_training(
                    current_cpu=current_cpu,
                    current_gpu=current_gpu,
                    active_jobs=len(self.scheduler.running_jobs)
                )
                
                if idle_job:
                    logger.info(f"Starting idle training job: {idle_job.job_id}")
                    # In real implementation, this would trigger actual training
                    # For now, we just log
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Idle training loop error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)
        
        logger.info("Idle training monitor stopped")
    
    async def _get_current_cpu(self) -> float:
        """الحصول على استخدام CPU الحالي"""
        # In real implementation, use psutil
        # For now, return a simulated value
        return 25.0  # 25% usage
    
    async def _get_current_gpu(self) -> float:
        """الحصول على استخدام GPU الحالي"""
        # In real implementation, use nvidia-smi
        # For now, return a simulated value
        return 10.0  # 10% usage
    
    def _on_job_event(self, event: str, job):
        """معالجة أحداث المهام"""
        logger.debug(f"Job event: {event} - {job.job_id}")
        
        if event == "job_completed":
            # Could trigger evaluation here
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الدماغ"""
        return {
            "is_running": self.is_running,
            "scheduler": self.scheduler.get_status(),
            "evaluations_count": len(self.evaluator.evaluations),
            "config": {
                "max_concurrent_jobs": self.config.max_concurrent_jobs,
                "idle_training_enabled": self.config.idle_training_enabled,
                "min_improvement_delta": self.config.min_improvement_delta,
            }
        }
    
    def get_jobs(self, status: str = None, limit: int = 50) -> list:
        """الحصول على قائمة المهام"""
        from .scheduler import JobStatus
        
        job_status = None
        if status:
            try:
                job_status = JobStatus(status)
            except ValueError:
                pass
        
        return self.scheduler.list_jobs(status=job_status, limit=limit)
    
    def get_evaluations(self, model_id: str = None, limit: int = 50) -> list:
        """الحصول على قائمة التقييمات"""
        evaluations = self.evaluator.list_evaluations(model_id=model_id, limit=limit)
        return [e.to_dict() for e in evaluations]


# Singleton instance
brain = BIBrain()
