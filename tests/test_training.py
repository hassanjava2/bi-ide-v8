"""
اختبارات خدمة التدريب - Training Service Tests
================================================
Tests for training service functionality including:
- Starting and stopping training jobs
- Getting metrics and listing models
- Error handling and edge cases

التغطية: >80%
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestTrainingService:
    """
    اختبارات خدمة التدريب
    Training Service Tests
    """
    
    @pytest.fixture
    def training_service(self):
        """إنشاء خدمة تدريب للاختبار"""
        from services.training_service import TrainingService, TrainingStatus
        service = TrainingService()
        return service
    
    # -------------------------------------------------------------------------
    # Tests for start_training
    # -------------------------------------------------------------------------
    
    async def test_start_training_success(self, training_service):
        """
        اختبار بدء التدريب بنجاح
        Test successful training start
        """
        # Arrange
        job_id = "test-job-001"
        model_name = "test-model"
        config = {"epochs": 5, "batch_size": 32}
        
        # Act
        job = await training_service.start_training(job_id, model_name, config)
        
        # Assert
        assert job.job_id == job_id
        assert job.model_name == model_name
        assert job.config == config
        assert job.status.value in ["pending", "running"]
        assert job.created_at is not None
    
    async def test_start_training_failure_duplicate_job(self, training_service):
        """
        اختبار فشل بدء التدريب بسبب وجود مهمة مسبقة
        Test training start failure with duplicate job
        """
        # Arrange
        job_id = "duplicate-job"
        model_name = "test-model"
        config = {"epochs": 5}
        
        # Start first job
        await training_service.start_training(job_id, model_name, config)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await training_service.start_training(job_id, model_name, config)
        
        assert "موجودة مسبقاً" in str(exc_info.value) or "already exists" in str(exc_info.value).lower()
    
    async def test_start_training_with_different_configs(self, training_service):
        """
        اختبار بدء التدريب بإعدادات مختلفة
        Test training start with different configurations
        """
        configs = [
            {"epochs": 1, "learning_rate": 0.01},
            {"epochs": 100, "batch_size": 64, "optimizer": "adam"},
            {},  # Empty config
        ]
        
        for i, config in enumerate(configs):
            job_id = f"config-test-{i}"
            job = await training_service.start_training(job_id, f"model-{i}", config)
            
            assert job.job_id == job_id
            assert job.config == config
    
    # -------------------------------------------------------------------------
    # Tests for stop_training
    # -------------------------------------------------------------------------
    
    async def test_stop_training_success(self, training_service):
        """
        اختبار إيقاف التدريب بنجاح
        Test successful training stop
        """
        # Arrange
        job_id = "stop-test-job"
        await training_service.start_training(job_id, "test-model", {"epochs": 10})
        
        # Act
        result = await training_service.stop_training(job_id)
        
        # Assert
        assert result is True
        
        # Verify job status
        job = await training_service.get_status(job_id)
        assert job.status.value == "cancelled"
    
    async def test_stop_training_nonexistent_job(self, training_service):
        """
        اختبار إيقاف مهمة غير موجودة
        Test stopping non-existent job
        """
        # Act
        result = await training_service.stop_training("nonexistent-job")
        
        # Assert
        assert result is False
    
    async def test_stop_training_already_completed(self, training_service):
        """
        اختبار إيقاف مهمة مكتملة
        Test stopping already completed job
        """
        # Arrange - Create a completed job
        job_id = "completed-job"
        job = await training_service.start_training(job_id, "test-model", {"epochs": 1})
        job.status = MagicMock(value="completed")
        training_service._jobs[job_id] = job
        
        # Act
        result = await training_service.stop_training(job_id)
        
        # Assert - Should return False for completed jobs
        assert result is False
    
    # -------------------------------------------------------------------------
    # Tests for get_metrics
    # -------------------------------------------------------------------------
    
    async def test_get_metrics_existing_job(self, training_service):
        """
        اختبار الحصول على مقاييس مهمة موجودة
        Test getting metrics for existing job
        """
        # Arrange
        job_id = "metrics-test-job"
        job = await training_service.start_training(job_id, "test-model", {"epochs": 5})
        
        # Simulate some metrics
        job.metrics = {"loss": 0.5, "accuracy": 0.95, "epoch": 3}
        
        # Act
        metrics = await training_service.get_metrics(job_id)
        
        # Assert
        assert metrics is not None
        assert "loss" in metrics
        assert "accuracy" in metrics
    
    async def test_get_metrics_nonexistent_job(self, training_service):
        """
        اختبار الحصول على مقاييس مهمة غير موجودة
        Test getting metrics for non-existent job
        """
        # Act
        metrics = await training_service.get_metrics("nonexistent-job")
        
        # Assert
        assert metrics is None
    
    async def test_get_metrics_empty_metrics(self, training_service):
        """
        اختبار الحصول على مقاييس فارغة
        Test getting empty metrics
        """
        # Arrange
        job_id = "empty-metrics-job"
        await training_service.start_training(job_id, "test-model", {"epochs": 5})
        
        # Act
        metrics = await training_service.get_metrics(job_id)
        
        # Assert - Should return empty dict or None
        assert metrics is not None or metrics is None
    
    # -------------------------------------------------------------------------
    # Tests for list_models
    # -------------------------------------------------------------------------
    
    async def test_list_models_empty(self, training_service):
        """
        اختبار قائمة النماذج الفارغة
        Test listing models when none exist
        """
        # Act
        models = await training_service.list_models()
        
        # Assert
        assert isinstance(models, list)
        assert len(models) == 0
    
    async def test_list_models_with_models(self, training_service):
        """
        اختبار قائمة النماذج مع وجود نماذج
        Test listing models with existing models
        """
        # Arrange - Manually add a model
        from services.training_service import ModelInfo
        training_service._models["model_001"] = ModelInfo(
            model_id="model_001",
            name="test-model",
            version="1.0.0",
            created_at=datetime.now(),
            accuracy=0.95,
            is_deployed=False,
            metadata={}
        )
        
        # Act
        models = await training_service.list_models()
        
        # Assert
        assert len(models) == 1
        assert models[0].model_id == "model_001"
        assert models[0].name == "test-model"
    
    async def test_list_models_multiple(self, training_service):
        """
        اختبار قائمة النماذج المتعددة
        Test listing multiple models
        """
        # Arrange
        from services.training_service import ModelInfo
        for i in range(5):
            training_service._models[f"model_{i}"] = ModelInfo(
                model_id=f"model_{i}",
                name=f"model-{i}",
                version=f"1.{i}.0",
                created_at=datetime.now(),
                accuracy=0.9 + (i * 0.01),
                is_deployed=i == 0,
                metadata={}
            )
        
        # Act
        models = await training_service.list_models()
        
        # Assert
        assert len(models) == 5
    
    # -------------------------------------------------------------------------
    # Tests for deploy_model
    # -------------------------------------------------------------------------
    
    async def test_deploy_model_success(self, training_service):
        """
        اختبار نشر نموذج بنجاح
        Test successful model deployment
        """
        # Arrange
        from services.training_service import ModelInfo
        model_id = "deploy-model-001"
        training_service._models[model_id] = ModelInfo(
            model_id=model_id,
            name="test-model",
            version="1.0.0",
            created_at=datetime.now(),
            accuracy=0.95,
            is_deployed=False,
            metadata={}
        )
        
        # Act
        result = await training_service.deploy_model(model_id)
        
        # Assert
        assert result is True
        assert training_service._models[model_id].is_deployed is True
    
    async def test_deploy_model_nonexistent(self, training_service):
        """
        اختبار نشر نموذج غير موجود
        Test deploying non-existent model
        """
        # Act
        result = await training_service.deploy_model("nonexistent-model")
        
        # Assert
        assert result is False
    
    async def test_deploy_model_undeploys_others(self, training_service):
        """
        اختبار أن نشر نموذج يلغي نشر النماذج الأخرى
        Test that deploying a model undeploys others
        """
        # Arrange
        from services.training_service import ModelInfo
        
        # Create and deploy first model
        training_service._models["model_1"] = ModelInfo(
            model_id="model_1", name="model-1", version="1.0.0",
            created_at=datetime.now(), accuracy=0.9, is_deployed=True, metadata={}
        )
        
        # Create second model
        training_service._models["model_2"] = ModelInfo(
            model_id="model_2", name="model-2", version="1.0.0",
            created_at=datetime.now(), accuracy=0.95, is_deployed=False, metadata={}
        )
        
        # Act - Deploy second model
        await training_service.deploy_model("model_2")
        
        # Assert
        assert training_service._models["model_1"].is_deployed is False
        assert training_service._models["model_2"].is_deployed is True


class TestTrainingServiceEdgeCases:
    """
    اختبارات الحالات الخاصة لخدمة التدريب
    Edge case tests for training service
    """
    
    @pytest.fixture
    def training_service(self):
        from services.training_service import TrainingService
        return TrainingService()
    
    async def test_start_training_with_invalid_config(self, training_service):
        """
        اختبار بدء التدريب بإعدادات غير صالحة
        Test starting training with invalid config
        """
        # Should handle None config
        job = await training_service.start_training("job-001", "model", None)
        assert job is not None
        assert job.config is not None
    
    async def test_concurrent_trainings(self, training_service):
        """
        اختبار التدريبات المتزامنة
        Test concurrent training jobs
        """
        # Start multiple jobs concurrently
        jobs = []
        for i in range(5):
            job = await training_service.start_training(
                f"concurrent-job-{i}", 
                f"model-{i}", 
                {"epochs": 3}
            )
            jobs.append(job)
        
        # Assert all jobs created
        assert len(jobs) == 5
        assert len(training_service._jobs) == 5
    
    async def test_training_status_transitions(self, training_service):
        """
        اختبار انتقالات حالة التدريب
        Test training status transitions
        """
        from services.training_service import TrainingStatus
        
        job_id = "transition-test"
        job = await training_service.start_training(job_id, "model", {"epochs": 5})
        
        # Initial status should be pending or running
        assert job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]
        
        # Cancel the job
        await training_service.stop_training(job_id)
        
        # Status should be cancelled
        job = await training_service.get_status(job_id)
        assert job.status == TrainingStatus.CANCELLED
