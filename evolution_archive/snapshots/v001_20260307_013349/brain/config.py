"""
Brain Configuration - إعدادات الدماغ
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BrainConfig:
    """إعدادات النظام الدماغي"""
    
    # Scheduler settings
    check_interval_seconds: int = 60
    max_concurrent_jobs: int = 3
    idle_training_enabled: bool = True
    idle_cpu_threshold: float = 30.0  # Start training when CPU < 30%
    
    # Evaluation settings
    min_improvement_delta: float = 0.02  # 2% improvement required
    auto_deploy_threshold: float = 0.95  # 95% confidence for auto-deploy
    evaluation_dataset_size: int = 1000
    
    # Resource limits
    max_jobs_per_day: int = 10
    max_training_hours: int = 4
    
    # Paths
    models_dir: str = "data/models"
    checkpoints_dir: str = "data/checkpoints"
    logs_dir: str = "logs/brain"
    
    @classmethod
    def from_env(cls) -> "BrainConfig":
        """Create config from environment variables"""
        return cls(
            check_interval_seconds=int(os.getenv("BRAIN_CHECK_INTERVAL", "60")),
            max_concurrent_jobs=int(os.getenv("BRAIN_MAX_JOBS", "3")),
            idle_training_enabled=os.getenv("BRAIN_IDLE_TRAINING", "true").lower() == "true",
            idle_cpu_threshold=float(os.getenv("BRAIN_IDLE_CPU", "30.0")),
            min_improvement_delta=float(os.getenv("BRAIN_MIN_IMPROVEMENT", "0.02")),
            auto_deploy_threshold=float(os.getenv("BRAIN_AUTO_DEPLOY", "0.95")),
            models_dir=os.getenv("BRAIN_MODELS_DIR", "data/models"),
            checkpoints_dir=os.getenv("BRAIN_CHECKPOINTS_DIR", "data/checkpoints"),
            logs_dir=os.getenv("BRAIN_LOGS_DIR", "logs/brain"),
        )
