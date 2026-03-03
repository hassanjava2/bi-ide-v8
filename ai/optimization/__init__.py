"""
Model Optimization - تحسين النماذج
================================
Quantization and optimization for inference speed
"""

try:
    from .benchmark import Benchmark
except ModuleNotFoundError:
    Benchmark = None

try:
    from .quantization import ModelQuantizer, QuantizationConfig
except ModuleNotFoundError:
    ModelQuantizer = None
    QuantizationConfig = None

__all__ = [
    "ModelQuantizer",
    "QuantizationConfig",
    "Benchmark"
]
