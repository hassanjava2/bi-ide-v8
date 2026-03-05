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
        
        # Hierarchy connection for coordinated decisions
        self._hierarchy = None
        
        logger.info("BI Brain initialized")
    
    @property
    def hierarchy(self):
        """Lazy-load hierarchy to avoid circular imports"""
        if self._hierarchy is None:
            try:
                from hierarchy import ai_hierarchy
                self._hierarchy = ai_hierarchy
            except ImportError:
                logger.warning("Hierarchy not available")
        return self._hierarchy
    
    async def consult_hierarchy(self, question: str) -> Dict[str, Any]:
        """
        استشارة النظام الهرمي لاتخاذ قرار
        
        Args:
            question: السؤال أو الأمر
            
        Returns:
            Dict: نتيجة التشاور
        """
        if not self.hierarchy:
            return {"status": "hierarchy_unavailable", "answer": None}
        
        try:
            result = self.hierarchy.ask(question)
            return {
                "status": "success",
                "answer": result.get("response"),
                "wise_man": result.get("wise_man"),
                "confidence": result.get("confidence", 0.0),
                "source": result.get("response_source")
            }
        except Exception as e:
            logger.error(f"Hierarchy consultation failed: {e}")
            return {"status": "error", "error": str(e)}
    
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
        """الحصول على استخدام CPU الحالي باستخدام psutil"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback: read from /proc/stat on Linux
            try:
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    fields = line.strip().split()
                    idle = int(fields[4])
                    total = sum(int(x) for x in fields[1:])
                    return round((1.0 - idle / total) * 100, 1) if total > 0 else 0.0
            except (FileNotFoundError, Exception):
                return 0.0  # Cannot determine
    
    async def _get_current_gpu(self) -> float:
        """الحصول على استخدام GPU الحالي باستخدام nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # May have multiple GPUs, take the max
                values = [float(v.strip()) for v in result.stdout.strip().split('\n') if v.strip()]
                return max(values) if values else 0.0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        
        # Fallback: try pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return float(util.gpu)
        except Exception:
            return 0.0  # No GPU available
    
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
