"""
BI-IDE v8 Training System - نظام التدريب V8

متوافق مع قوانين RTX5090:
- LoRA Model Only للاستنتاج
- Ollama = تدريب فقط
- Thermal Protection حتى 97°C
"""

from .lora_inference import LoRAInferenceEngine
from .training_pipeline import TrainingPipeline
from .model_manager import ModelManager
from .thermal_guard import ThermalGuard

__all__ = [
    'LoRAInferenceEngine',
    'TrainingPipeline', 
    'ModelManager',
    'ThermalGuard',
]
