"""
Brain Module - وحدة الدماغ

MVP Brain system for auto-scheduling, model evaluation, and training.
"""

from .bi_brain import BIBrain, brain
from .scheduler import Scheduler, ScheduleJob
from .evaluator import ModelEvaluator, EvaluationResult
from .config import BrainConfig

# New training pipeline modules (optional — deps may not be installed)
try:
    from .data_preprocessor import DataPreprocessor
except ImportError:
    DataPreprocessor = None

try:
    from .capsule_500 import CAPSULE_REGISTRY, count_capsules, get_categories
except ImportError:
    CAPSULE_REGISTRY = {}
    count_capsules = lambda: 0
    get_categories = lambda: {}

try:
    from .capsule_bridge import CapsuleBridge, bridge
except ImportError:
    CapsuleBridge = None
    bridge = None

__all__ = [
    'BIBrain',
    'brain',
    'Scheduler',
    'ScheduleJob',
    'ModelEvaluator',
    'EvaluationResult',
    'BrainConfig',
    'DataPreprocessor',
    'CAPSULE_REGISTRY',
    'count_capsules',
    'get_categories',
    'CapsuleBridge',
    'bridge',
]

