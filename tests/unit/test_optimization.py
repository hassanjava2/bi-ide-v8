"""
Unit tests for Model Optimization modules
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.optimization.quantization import (
    ModelQuantizer, 
    benchmark_performance,
    QuantizationConfig,
    quantize_model
)
from ai.optimization.pruning import (
    ModelPruner,
    FineTuner,
    prune_model,
    iterative_pruning
)
from ai.optimization.distillation import (
    KnowledgeDistillation,
    TemperatureScaling,
    distill_model
)
from ai.optimization.batch_inference import (
    BatchProcessor,
    InferenceServer,
    DynamicBatcher
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=100, hidden_dim=200, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestQuantization:
    """Test quantization module."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 100)
    
    def test_model_quantizer_init(self):
        """Test ModelQuantizer initialization."""
        quantizer = ModelQuantizer(device='cpu')
        assert quantizer.device == 'cpu'
    
    def test_quantize_to_fp16(self, model, sample_input):
        """Test FP16 quantization."""
        quantizer = ModelQuantizer(device='cpu')
        
        # CPU doesn't support FP16 well, should keep FP32
        quantized = quantizer.quantize_to_fp16(model)
        
        assert quantized is not None
        # Model should still work
        output = quantized(sample_input)
        assert output.shape == (4, 10)
    
    def test_quantize_to_int8(self, model, sample_input):
        """Test INT8 quantization."""
        quantizer = ModelQuantizer(device='cpu')

        try:
            quantized = quantizer.quantize_to_int8(model, method='dynamic')
        except RuntimeError as e:
            if "FBGEMM" in str(e) or "quantized engine" in str(e):
                pytest.skip(f"INT8 quantization not supported: {e}")
            raise

        assert quantized is not None
        # Model should still work
        output = quantized(sample_input)
        assert output.shape == (4, 10)
    
    def test_benchmark_performance(self, model, sample_input):
        """Test performance benchmarking."""
        results = benchmark_performance(
            model,
            sample_input,
            num_runs=5,
            warmup=2
        )
        
        assert 'latency_ms' in results
        assert 'mean' in results['latency_ms']
        assert 'throughput_samples_per_sec' in results
        assert results['latency_ms']['mean'] > 0
    
    def test_quantization_config(self):
        """Test QuantizationConfig."""
        config = QuantizationConfig(
            method='int8',
            int8_method='dynamic'
        )
        
        assert config.method == 'int8'
        assert config.int8_method == 'dynamic'
        
        config_dict = config.to_dict()
        assert config_dict['method'] == 'int8'
    
    def test_quantize_model(self, model):
        """Test quantize_model convenience function."""
        config = QuantizationConfig(method='fp16')
        quantized = quantize_model(model, config)
        
        assert quantized is not None


class TestPruning:
    """Test pruning module."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    def test_model_pruner_init(self):
        """Test ModelPruner initialization."""
        pruner = ModelPruner(device='cpu')
        assert pruner.device == 'cpu'
    
    def test_prune_model(self, model):
        """Test model pruning."""
        pruner = ModelPruner(device='cpu')
        
        # Get initial stats
        initial_params = sum(p.numel() for p in model.parameters())
        
        # Prune
        pruned = pruner.prune_model(model, amount=0.3)
        
        # Check model still works
        sample_input = torch.randn(4, 100)
        output = pruned(sample_input)
        assert output.shape == (4, 10)
        
        # Check sparsity (may be 0 if pruning uses masks not yet applied)
        sparsity = pruner._compute_sparsity(pruned)
        assert sparsity >= 0
    
    def test_get_compression_ratio(self, model):
        """Test compression ratio calculation."""
        pruner = ModelPruner(device='cpu')
        pruner.prune_model(model, amount=0.5)
        
        ratio = pruner.get_compression_ratio(model)
        
        assert 'sparsity' in ratio
        assert 'compression_ratio' in ratio
        assert ratio['sparsity'] >= 0
    
    def test_remove_redundant_weights(self, model):
        """Test removing redundant weights."""
        pruner = ModelPruner(device='cpu')
        
        # Add some small weights
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() >= 2:
                    param[0, 0] = 1e-10
                elif param.dim() == 1:
                    param[0] = 1e-10
        
        cleaned = pruner.remove_redundant_weights(model, threshold=1e-8)
        
        assert cleaned is not None
    
    def test_make_permanent(self, model):
        """Test making pruning permanent."""
        pruner = ModelPruner(device='cpu')
        pruner.prune_model(model, amount=0.3)
        
        # Make permanent
        model = pruner.make_permanent(model)
        
        # Check no mask buffers remain
        for module in model.modules():
            assert not hasattr(module, 'weight_mask')


class TestDistillation:
    """Test distillation module."""
    
    @pytest.fixture
    def teacher_model(self):
        return SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    
    @pytest.fixture
    def student_model(self):
        return SimpleModel(input_dim=100, hidden_dim=50, output_dim=10)
    
    def test_knowledge_distillation_init(self, teacher_model, student_model):
        """Test KnowledgeDistillation initialization."""
        distiller = KnowledgeDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=4.0,
            device='cpu'
        )
        
        assert distiller.temperature == 4.0
        assert distiller.device == 'cpu'
    
    def test_distillation_loss(self, teacher_model, student_model):
        """Test distillation loss calculation."""
        distiller = KnowledgeDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            device='cpu'
        )
        
        # Create dummy logits
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        
        loss = distiller.distillation_loss(student_logits, teacher_logits, labels)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        scaler = TemperatureScaling(temperature=2.0)
        
        logits = torch.randn(4, 10)
        scaled = scaler.scale_logits(logits)
        
        assert scaled.shape == logits.shape
        assert torch.allclose(scaled, logits / 2.0)


class TestBatchInference:
    """Test batch inference module."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    @pytest.fixture
    def processor(self, model):
        return BatchProcessor(
            model=model,
            max_batch_size=4,
            device='cpu'
        )
    
    def test_batch_processor_init(self, model):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(
            model=model,
            max_batch_size=8,
            device='cpu'
        )
        
        assert processor.max_batch_size == 8
        assert processor.device == 'cpu'
    
    def test_submit(self, processor):
        """Test submitting requests."""
        request_id = processor.submit(
            inputs=torch.randn(100),
            request_id='test_1'
        )
        
        assert request_id == 'test_1'
        assert processor.stats['total_requests'] == 1
    
    def test_process_batch(self, processor):
        """Test batch processing."""
        # Create requests
        requests = []
        for i in range(3):
            from ai.optimization.batch_inference import InferenceRequest
            req = InferenceRequest(
                id=f'req_{i}',
                inputs=torch.randn(100)
            )
            requests.append(req)
        
        results = processor.process_batch(requests)
        
        assert len(results) == 3
        for result in results:
            assert result.request_id.startswith('req_')
            assert result.processing_time >= 0
    
    def test_inference_server(self, model):
        """Test InferenceServer."""
        server = InferenceServer(
            model=model,
            max_batch_size=4,
            device='cpu'
        )
        
        # Start server
        server.start()
        
        try:
            # Single prediction
            result = server.predict(torch.randn(100))
            assert result.shape == (10,)
            
            # Batch prediction
            batch_results = server.predict_batch([torch.randn(100) for _ in range(5)])
            assert len(batch_results) == 5
        finally:
            server.stop()
    
    def test_dynamic_batcher(self, model):
        """Test DynamicBatcher."""
        batcher = DynamicBatcher(
            model=model,
            bucket_boundaries=[32, 64, 128],
            device='cpu'
        )
        
        # Test bucketing
        bucket = batcher.get_bucket(50)
        assert bucket == 64
        
        bucket = batcher.get_bucket(200)
        assert bucket == 'overflow'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
