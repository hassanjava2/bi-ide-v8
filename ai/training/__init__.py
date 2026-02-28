"""
AI Training Module - وحدة تدريب الذكاء الاصطناعي

This module provides a comprehensive training pipeline for AI models
with support for PyTorch 2.x and CUDA 12.x.

الميزات الرئيسية:
- Advanced Trainer: Merged trainer supporting multiple modes
- Evaluation Engine: Model evaluation and data validation
- Multi-GPU Trainer: Distributed training with DDP
- Continuous Trainer: Automated continuous training
- Model Converter: Export to multiple formats (ONNX, GGUF)
"""

# Core imports - existing modules
from .data_collection import (
    DataCollector,
    QualityFilter,
    Deduplicator,
    DataSample
)
from .preprocessing import (
    DataPreprocessor,
    PreprocessingConfig,
    TextCleaner,
    DataAugmenter,
    create_preprocessing_pipeline
)
from .auto_evaluation import (
    EvaluationPipeline,
    PerplexityCalculator,
    BenchmarkEvaluator,
    ABTestFramework,
    HumanEvaluationInterface,
    EvaluationResult
)
from .deployment import (
    ModelDeployment,
    ModelRegistry,
    DeploymentConfig,
    ModelVersion,
    create_deployment_pipeline
)

# ═══════════════════════════════════════════════════════════════
# V6 Migration - New Merged Modules
# ═══════════════════════════════════════════════════════════════

# Advanced Trainer - المدرب المتقدم (من v6-scripts: finetune.py, finetune-chat.py, finetune-extended.py)
try:
    from .advanced_trainer import (
        AdvancedTrainer,
        TrainingConfig,
        TrainingResult,
        TrainingMode,
        ModelType,
        create_trainer,
        quick_train,
    )
    ADVANCED_TRAINER_AVAILABLE = True
except ImportError as e:
    ADVANCED_TRAINER_AVAILABLE = False

# Evaluation Engine - محرك التقييم (من v6-scripts: evaluate-model.py, validate-data.py)
try:
    from .evaluation_engine import (
        EvaluationEngine,
        ValidationResult,
        DataQualityReport,
        ModelEvaluationReport,
        EvaluationMetrics,
        ValidationStatus,
        Language,
    )
    EVALUATION_ENGINE_AVAILABLE = True
except ImportError as e:
    EVALUATION_ENGINE_AVAILABLE = False

# Multi-GPU Trainer - مدرب متعدد GPU (جديد لـ PyTorch 2.x DDP)
try:
    from .multi_gpu_trainer import (
        MultiGPUTrainer,
        FaultTolerantTrainer,
        DistributedConfig,
        MultiGPUConfig,
        setup_distributed,
        cleanup_distributed,
        get_gpu_info,
        print_gpu_info,
        is_distributed_available,
        get_rank,
        get_world_size,
        is_main_process,
    )
    MULTI_GPU_TRAINER_AVAILABLE = True
except ImportError as e:
    MULTI_GPU_TRAINER_AVAILABLE = False

# Continuous Trainer - المدرب المستمر (من v6-scripts: continuous-train.py, auto-finetune.py)
try:
    from .continuous_trainer import (
        ContinuousTrainer,
        ContinuousTrainingConfig,
        TrainingSession,
        ModelVersion,
        TrainingState,
        CurriculumType,
        CURRICULA,
    )
    CONTINUOUS_TRAINER_AVAILABLE = True
except ImportError as e:
    CONTINUOUS_TRAINER_AVAILABLE = False

# Model Converter - محول النماذج (من v6-scripts: convert-to-gguf.py, convert-to-onnx.py)
try:
    from .model_converter import (
        ModelConverter,
        ConversionResult,
        ConversionConfig,
        ConversionFormat,
        QuantizationType,
    )
    MODEL_CONVERTER_AVAILABLE = True
except ImportError as e:
    MODEL_CONVERTER_AVAILABLE = False

# Legacy migration tool
try:
    from .legacy.migrate_v6_scripts import (
        V6MigrationTool,
        MigrationStatus,
        ScriptInfo,
        MigrationReport,
    )
    MIGRATION_TOOL_AVAILABLE = True
except ImportError as e:
    MIGRATION_TOOL_AVAILABLE = False

# RTX 4090 Trainer (existing)
try:
    from .rtx4090_trainer import (
        RTX4090Trainer,
        ContinuousTrainer as RTX4090ContinuousTrainer,
        get_rtx4090_trainer
    )
    RTX4090_TRAINER_AVAILABLE = True
except ImportError:
    RTX4090_TRAINER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Convenience Functions - دوال مساعدة
# ═══════════════════════════════════════════════════════════════

def get_trainer(
    mode: str = "advanced",
    distributed: bool = False,
    **kwargs
) -> Any:
    """
    Get appropriate trainer based on requirements
    
    Args:
        mode: Trainer mode (advanced/continuous/rtx4090)
        distributed: Whether to use distributed training
        **kwargs: Additional arguments
        
    Returns:
        Trainer instance
    """
    if distributed and MULTI_GPU_TRAINER_AVAILABLE:
        return MultiGPUTrainer(**kwargs)
    
    if mode == "continuous" and CONTINUOUS_TRAINER_AVAILABLE:
        return ContinuousTrainer(**kwargs)
    
    if mode == "rtx4090" and RTX4090_TRAINER_AVAILABLE:
        return get_rtx4090_trainer()
    
    if ADVANCED_TRAINER_AVAILABLE:
        return AdvancedTrainer(**kwargs)
    
    raise ImportError("No suitable trainer available")


def train_model(
    data: List[Dict[str, str]],
    output_name: str = "finetuned",
    mode: str = "completion",
    epochs: int = 3,
    **kwargs
) -> Any:
    """
    Quick training function
    
    Args:
        data: Training data
        output_name: Output directory name
        mode: Training mode
        epochs: Number of epochs
        **kwargs: Additional arguments
        
    Returns:
        Training result
    """
    if not ADVANCED_TRAINER_AVAILABLE:
        raise ImportError("AdvancedTrainer not available")
    
    from .advanced_trainer import quick_train
    return quick_train(data, output_name, epochs)


def evaluate_model(
    model_path: Union[str, Path],
    validation_data: Optional[List[Dict]] = None
) -> Any:
    """
    Evaluate a model
    
    Args:
        model_path: Path to model
        validation_data: Validation samples
        
    Returns:
        Evaluation report
    """
    if not EVALUATION_ENGINE_AVAILABLE:
        raise ImportError("EvaluationEngine not available")
    
    engine = EvaluationEngine()
    return engine.evaluate_model(model_path, validation_data)


def validate_data(
    data: List[Dict[str, Any]],
    fix: bool = False
) -> Any:
    """
    Validate training data
    
    Args:
        data: Data to validate
        fix: Whether to fix issues
        
    Returns:
        Validation report
    """
    if not EVALUATION_ENGINE_AVAILABLE:
        raise ImportError("EvaluationEngine not available")
    
    engine = EvaluationEngine()
    return engine.validate_data(data, fix=fix)


# ═══════════════════════════════════════════════════════════════
# Version Info
# ═══════════════════════════════════════════════════════════════

__version__ = "2.0.0"
__v6_migrated__ = True


def get_module_info() -> Dict[str, Any]:
    """Get module information"""
    return {
        "version": __version__,
        "v6_migrated": __v6_migrated__,
        "available_modules": {
            "advanced_trainer": ADVANCED_TRAINER_AVAILABLE,
            "evaluation_engine": EVALUATION_ENGINE_AVAILABLE,
            "multi_gpu_trainer": MULTI_GPU_TRAINER_AVAILABLE,
            "continuous_trainer": CONTINUOUS_TRAINER_AVAILABLE,
            "model_converter": MODEL_CONVERTER_AVAILABLE,
            "rtx4090_trainer": RTX4090_TRAINER_AVAILABLE,
            "migration_tool": MIGRATION_TOOL_AVAILABLE,
        },
        "pytorch_version": torch.__version__ if 'torch' in sys.modules else None,
        "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else None,
    }


# ═══════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Core exports
    'DataCollector',
    'QualityFilter',
    'Deduplicator',
    'DataSample',
    'DataPreprocessor',
    'PreprocessingConfig',
    'TextCleaner',
    'DataAugmenter',
    'create_preprocessing_pipeline',
    'EvaluationPipeline',
    'PerplexityCalculator',
    'BenchmarkEvaluator',
    'ABTestFramework',
    'HumanEvaluationInterface',
    'EvaluationResult',
    'ModelDeployment',
    'ModelRegistry',
    'DeploymentConfig',
    'ModelVersion',
    'create_deployment_pipeline',
    
    # V6 Migrated modules
    'AdvancedTrainer',
    'TrainingConfig',
    'TrainingResult',
    'TrainingMode',
    'ModelType',
    'create_trainer',
    'quick_train',
    
    'EvaluationEngine',
    'ValidationResult',
    'DataQualityReport',
    'ModelEvaluationReport',
    'EvaluationMetrics',
    'ValidationStatus',
    'Language',
    
    'MultiGPUTrainer',
    'FaultTolerantTrainer',
    'DistributedConfig',
    'MultiGPUConfig',
    'setup_distributed',
    'cleanup_distributed',
    'get_gpu_info',
    'print_gpu_info',
    'is_distributed_available',
    'get_rank',
    'get_world_size',
    'is_main_process',
    
    'ContinuousTrainer',
    'ContinuousTrainingConfig',
    'TrainingSession',
    'ModelVersion',
    'TrainingState',
    'CurriculumType',
    'CURRICULA',
    
    'ModelConverter',
    'ConversionResult',
    'ConversionConfig',
    'ConversionFormat',
    'QuantizationType',
    
    'V6MigrationTool',
    'MigrationStatus',
    'ScriptInfo',
    'MigrationReport',
    
    # RTX 4090
    'RTX4090Trainer',
    'get_rtx4090_trainer',
    
    # Convenience functions
    'get_trainer',
    'train_model',
    'evaluate_model',
    'validate_data',
    'get_module_info',
    
    # Availability flags
    'ADVANCED_TRAINER_AVAILABLE',
    'EVALUATION_ENGINE_AVAILABLE',
    'MULTI_GPU_TRAINER_AVAILABLE',
    'CONTINUOUS_TRAINER_AVAILABLE',
    'MODEL_CONVERTER_AVAILABLE',
    'RTX4090_TRAINER_AVAILABLE',
    'MIGRATION_TOOL_AVAILABLE',
    
    # Version
    '__version__',
    '__v6_migrated__',
]


# Type imports
from typing import List, Dict, Any, Union
import sys
try:
    import torch
except ImportError:
    pass
