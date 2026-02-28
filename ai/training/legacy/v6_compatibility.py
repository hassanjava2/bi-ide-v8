"""
V6 Compatibility Layer - طبقة توافق v6

Provides backward compatibility for code using v6 APIs.
يوفر توافقاً للكود الذي يستخدم واجهات v6.
"""

import warnings
from typing import Any, Optional
import sys
from pathlib import Path

# Add new training module to path
_training_module_path = Path(__file__).parent.parent
if str(_training_module_path) not in sys.path:
    sys.path.insert(0, str(_training_module_path))

# Import new modules with compatibility aliases
try:
    from ai.training.advanced_trainer import AdvancedTrainer
    from ai.training.evaluation_engine import EvaluationEngine
    from ai.training.continuous_trainer import ContinuousTrainer
    from ai.training.model_converter import ModelConverter
    V8_AVAILABLE = True
except ImportError:
    V8_AVAILABLE = False


class FineTuneV6:
    """
    Compatibility wrapper for finetune.py
    غلاف توافق لـ finetune.py
    """
    
    def __init__(self):
        warnings.warn(
            "FineTuneV6 is deprecated. Use AdvancedTrainer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = AdvancedTrainer()
        else:
            raise ImportError("V8 training module not available")
    
    def train(self, *args, **kwargs) -> Any:
        """Delegate to AdvancedTrainer"""
        return self._trainer.train(*args, **kwargs)


class FineTuneChatV6:
    """
    Compatibility wrapper for finetune-chat.py
    غلاف توافق لـ finetune-chat.py
    """
    
    def __init__(self):
        warnings.warn(
            "FineTuneChatV6 is deprecated. Use AdvancedTrainer with mode='chat'.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = AdvancedTrainer(mode='chat')
        else:
            raise ImportError("V8 training module not available")
    
    def train(self, *args, **kwargs) -> Any:
        """Delegate to AdvancedTrainer"""
        return self._trainer.train(*args, **kwargs)


class EvaluateModelV6:
    """
    Compatibility wrapper for evaluate-model.py
    غلاف توافق لـ evaluate-model.py
    """
    
    def __init__(self):
        warnings.warn(
            "EvaluateModelV6 is deprecated. Use EvaluationEngine instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._engine = EvaluationEngine()
        else:
            raise ImportError("V8 training module not available")
    
    def evaluate(self, *args, **kwargs) -> Any:
        """Delegate to EvaluationEngine"""
        return self._engine.evaluate_model(*args, **kwargs)


class ContinuousTrainV6:
    """
    Compatibility wrapper for continuous-train.py
    غلاف توافق لـ continuous-train.py
    """
    
    def __init__(self):
        warnings.warn(
            "ContinuousTrainV6 is deprecated. Use ContinuousTrainer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if V8_AVAILABLE:
            self._trainer = ContinuousTrainer()
        else:
            raise ImportError("V8 training module not available")
    
    def start(self, *args, **kwargs) -> Any:
        """Delegate to ContinuousTrainer"""
        return self._trainer.start(*args, **kwargs)


def migrate_v6_config(config: dict) -> dict:
    """
    Convert v6 config format to v8
    تحويل إعدادات v6 إلى v8
    
    Args:
        config: V6 configuration dict
        
    Returns:
        V8 configuration dict
    """
    mapping = {
        'MODEL_NAME': 'model_name',
        'MAX_LENGTH': 'max_length',
        'BATCH_SIZE': 'batch_size',
        'EPOCHS': 'epochs',
        'LEARNING_RATE': 'learning_rate',
        'LORA_R': 'lora_r',
        'LORA_ALPHA': 'lora_alpha',
        'NUM_WORKERS': 'num_workers',
    }
    
    return {
        mapping.get(k, k): v 
        for k, v in config.items()
    }


# Export compatibility classes
__all__ = [
    'FineTuneV6',
    'FineTuneChatV6', 
    'EvaluateModelV6',
    'ContinuousTrainV6',
    'migrate_v6_config',
    'V8_AVAILABLE',
]
