"""
BI-IDE v8 AI Module
AI Enhancement Phase 2
"""

__version__ = '2.0.0'

# Tokenizer
from ai.tokenizer.bpe_tokenizer import BPETokenizer, train_bpe_tokenizer
from ai.tokenizer.arabic_processor import ArabicProcessor
from ai.tokenizer.code_tokenizer import CodeTokenizer

# Optimization
from ai.optimization.quantization import ModelQuantizer, QuantizationConfig
from ai.optimization.benchmark import Benchmark

__all__ = [
    # Tokenizer
    'BPETokenizer',
    'train_bpe_tokenizer',
    'ArabicProcessor',
    'CodeTokenizer',
    
    # Optimization
    'ModelQuantizer',
    'QuantizationConfig',
    'Benchmark',
]
