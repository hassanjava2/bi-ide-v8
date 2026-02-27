"""Training Pipeline Module"""

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

# RTX 4090 Trainer
try:
    from .rtx4090_trainer import (
        RTX4090Trainer,
        ContinuousTrainer,
        get_rtx4090_trainer
    )
    RTX4090_TRAINER_AVAILABLE = True
except ImportError:
    RTX4090_TRAINER_AVAILABLE = False

__all__ = [
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
    # RTX 4090
    'RTX4090Trainer',
    'ContinuousTrainer',
    'get_rtx4090_trainer',
    'RTX4090_TRAINER_AVAILABLE',
]
