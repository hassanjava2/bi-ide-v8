"""
Multi-GPU Trainer - مدرب متعدد GPU
Updated for PyTorch 2.x with DDP

Features / المميزات:
  • DistributedDataParallel (DDP)
  • DataParallel fallback
  • Multi-node support
  • Gradient synchronization
  • Load balancing
  • Fault tolerance
  • Automatic mixed precision (AMP)
  • Gradient accumulation across GPUs

PyTorch 2.x + CUDA 12.x Compatible
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl"
) -> bool:
    """
    إعداد التوزيع
    Setup distributed training
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend (nccl, gloo, mpi)
        
    Returns:
        True if successful
    """
    try:
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
            dist.init_process_group(backend, rank=rank, world_size=world_size)
        return True
    except Exception as e:
        logger.error(f"Failed to setup distributed: {e}")
        return False


def cleanup_distributed():
    """تنظيف التوزيع - Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class DistributedConfig:
    """
    إعدادات التدريب الموزع - Distributed training configuration
    
    Attributes:
        world_size: عدد العمليات الكلي
        rank: رتبة العملية الحالية
        local_rank: الرتبة المحلية في العقدة
        backend: Backend للتواصل (nccl/gloo)
        init_method: طريقة التهيئة (env/tcp/file)
        find_unused_parameters: البحث عن parameters غير المستخدمة
        gradient_as_bucket_view: تحسين استخدام الذاكرة
        bucket_cap_mb: حجم bucket للـ gradients
    """
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25
    
    @property
    def is_main_process(self) -> bool:
        """هل هذه العملية الرئيسية؟"""
        return self.rank == 0
    
    @property
    def is_distributed(self) -> bool:
        """هل التدريب موزع؟"""
        return self.world_size > 1


@dataclass
class MultiGPUConfig:
    """إعدادات متعدد GPU - Multi-GPU configuration"""
    use_ddp: bool = True
    use_amp: bool = True
    sync_batchnorm: bool = True
    gradient_sync_freq: int = 1
    
    # إعدادات Fault Tolerance
    checkpoint_freq: int = 500
    max_retries: int = 3
    
    # Load balancing
    balance_data: bool = True
    drop_last: bool = False


class MultiGPUTrainer:
    """
    مدرب متعدد GPU - Multi-GPU Trainer
    
    يدعم:
    - DistributedDataParallel (DDP) للأداء الأمثل
    - DataParallel كخيار احتياطي
    - تدريب متعدد العقد (Multi-node)
    - التسامح مع الأخطاء (Fault Tolerance)
    """
    
    def __init__(
        self,
        dist_config: Optional[DistributedConfig] = None,
        gpu_config: Optional[MultiGPUConfig] = None,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize Multi-GPU trainer
        
        Args:
            dist_config: Distributed configuration
            gpu_config: Multi-GPU configuration
            base_dir: Base project directory
        """
        self.dist_config = dist_config or self._auto_detect_config()
        self.gpu_config = gpu_config or MultiGPUConfig()
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self._setup_complete = False
        self._training_step = 0
        
        self._setup_logging()
        
        logger.info("=" * 60)
        logger.info("Multi-GPU Trainer - مدرب متعدد GPU")
        logger.info("=" * 60)
        logger.info(f"   World size: {self.dist_config.world_size}")
        logger.info(f"   Rank: {self.dist_config.rank}")
        logger.info(f"   Local rank: {self.dist_config.local_rank}")
        logger.info(f"   Backend: {self.dist_config.backend}")
        logger.info(f"   DDP: {self.gpu_config.use_ddp}")
        logger.info(f"   AMP: {self.gpu_config.use_amp}")
    
    def _auto_detect_config(self) -> DistributedConfig:
        """اكتشاف الإعدادات تلقائياً - Auto-detect configuration"""
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        return DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend=backend
        )
    
    def _setup_logging(self):
        """إعداد التسجيل - Setup logging"""
        if not self.dist_config.is_main_process:
            logging.disable(logging.INFO)
    
    def setup(self) -> bool:
        """
        إعداد بيئة التدريب
        Setup training environment
        
        Returns:
            True if successful
        """
        if self._setup_complete:
            return True
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.dist_config.local_rank)
            logger.info(f"GPU {self.dist_config.local_rank}: {torch.cuda.get_device_name()}")
        
        if self.dist_config.is_distributed and self.gpu_config.use_ddp:
            if not setup_distributed(
                self.dist_config.rank,
                self.dist_config.world_size,
                self.dist_config.backend
            ):
                logger.error("Failed to setup distributed training")
                return False
            logger.info("Distributed training initialized")
        
        if self.gpu_config.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP enabled")
        
        self._setup_complete = True
        return True
    
    def prepare_model(
        self,
        model: nn.Module,
        device_ids: Optional[List[int]] = None
    ) -> nn.Module:
        """
        تحضير النموذج للتدريب متعدد GPU
        Prepare model for multi-GPU training
        """
        if not self._setup_complete:
            self.setup()
        
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.dist_config.local_rank}")
            model = model.to(device)
        else:
            device = torch.device("cpu")
        
        if self.gpu_config.sync_batchnorm and torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("Synchronized BatchNorm")
        
        if self.dist_config.is_distributed and self.gpu_config.use_ddp:
            model = DDP(
                model,
                device_ids=[self.dist_config.local_rank],
                output_device=self.dist_config.local_rank,
                find_unused_parameters=self.dist_config.find_unused_parameters,
                gradient_as_bucket_view=self.dist_config.gradient_as_bucket_view,
                bucket_cap_mb=self.dist_config.bucket_cap_mb
            )
            logger.info("Model wrapped with DDP")
        elif torch.cuda.device_count() > 1 and not self.dist_config.is_distributed:
            device_ids = device_ids or list(range(torch.cuda.device_count()))
            model = DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel")
        
        self.model = model
        return model
    
    def prepare_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        **kwargs
    ) -> DataLoader:
        """
        تحضير DataLoader
        Prepare DataLoader with distributed sampling
        """
        sampler = None
        
        if self.dist_config.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist_config.world_size,
                rank=self.dist_config.rank,
                shuffle=shuffle,
                drop_last=self.gpu_config.drop_last
            )
            shuffle = False
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn,
            drop_last=self.gpu_config.drop_last,
            **kwargs
        )
        
        return loader
    
    def backward_pass(
        self,
        loss: torch.Tensor,
        accumulate_gradients: bool = False
    ):
        """
        خطوة الـ backward مع AMP
        Backward pass with AMP support
        """
        if self.scaler is not None and self.gpu_config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if self.dist_config.is_distributed and not accumulate_gradients:
            self._sync_gradients()
    
    def _sync_gradients(self):
        """مزامنة التدرجات عبر GPUs - Sync gradients across GPUs"""
        if self.model is None:
            return
        
        if isinstance(self.model, DDP):
            return
        
        world_size = self.dist_config.world_size
        if world_size > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        خطوة الـ optimizer مع AMP
        Optimizer step with AMP
        """
        if self.scaler is not None and self.gpu_config.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        self._training_step += 1
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict] = None
    ):
        """
        حفظ checkpoint
        Save checkpoint (main process only)
        """
        if not self.dist_config.is_main_process:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self._training_step,
            'model_state_dict': self._get_model_state_dict(),
            'dist_config': {
                'world_size': self.dist_config.world_size,
                'rank': self.dist_config.rank
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def _get_model_state_dict(self) -> Dict:
        """الحصول على state dict - Get model state dict"""
        if self.model is None:
            return {}
        
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        elif isinstance(self.model, DataParallel):
            return self.model.module.state_dict()
        
        return self.model.state_dict()
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None
    ) -> Dict:
        """
        تحميل checkpoint
        Load checkpoint
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return {}
        
        if map_location is None:
            map_location = f"cuda:{self.dist_config.local_rank}" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=map_location)
        
        model_state = checkpoint.get('model_state_dict', {})
        
        if self.model is not None:
            if isinstance(self.model, (DDP, DataParallel)):
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self._training_step = checkpoint.get('step', 0)
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def barrier(self):
        """حاجز تزامن - Synchronization barrier"""
        if self.dist_config.is_distributed:
            dist.barrier()
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        جمع البيانات من جميع GPUs
        Gather tensors from all GPUs
        """
        if not self.dist_config.is_distributed:
            return [tensor]
        
        world_size = self.dist_config.world_size
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return gathered
    
    def reduce_dict(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        تقليل قاموس من جميع GPUs
        Reduce dict across all GPUs
        """
        if not self.dist_config.is_distributed:
            return data
        
        reduced = {}
        for key, value in data.items():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced[key] = value / self.dist_config.world_size
        
        return reduced
    
    def cleanup(self):
        """تنظيف الموارد - Cleanup resources"""
        cleanup_distributed()
        logger.info("Distributed training cleaned up")


class FaultTolerantTrainer(MultiGPUTrainer):
    """
    مدرب متسامح مع الأخطاء
    Fault-tolerant trainer with automatic recovery
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failure_count = 0
        self._last_checkpoint = None
    
    def train_with_recovery(
        self,
        train_func: Callable,
        checkpoint_dir: Path,
        max_retries: Optional[int] = None
    ) -> Any:
        """
        تدريب مع استرداد تلقائي
        Train with automatic recovery
        """
        max_retries = max_retries or self.gpu_config.max_retries
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Training attempt {attempt + 1}/{max_retries}")
                result = train_func()
                logger.info("Training completed successfully")
                return result
                
            except Exception as e:
                self._failure_count += 1
                logger.error(f"Training failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    checkpoint_path = checkpoint_dir / f"emergency_checkpoint_{attempt + 1}.pt"
                    try:
                        self.save_checkpoint(checkpoint_path, epoch=0)
                        self._last_checkpoint = checkpoint_path
                        logger.info("Emergency checkpoint saved")
                    except Exception as save_error:
                        logger.error(f"Failed to save checkpoint: {save_error}")
                    
                    wait_time = min(2 ** attempt, 60)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    raise
        
        return None


def get_gpu_info() -> Dict[str, Any]:
    """
    الحصول على معلومات GPUs
    Get GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor
            })
    
    return info


def print_gpu_info():
    """طباعة معلومات GPUs - Print GPU information"""
    info = get_gpu_info()
    
    print("=" * 60)
    print("GPU Information - معلومات GPUs")
    print("=" * 60)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"Device Count: {info['device_count']}")
    
    for device in info['devices']:
        print(f"\n  GPU {device['id']}: {device['name']}")
        print(f"    Memory: {device['total_memory_gb']:.1f} GB")
        print(f"    Compute Capability: {device['major']}.{device['minor']}")
    print("=" * 60)


def is_distributed_available() -> bool:
    """التحقق من توفر التدريب الموزع"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """الحصول على رتبة العملية الحالية"""
    if is_distributed_available():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """الحصول على عدد العمليات الكلي"""
    if is_distributed_available():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """هل هذه العملية الرئيسية؟"""
    return get_rank() == 0


if __name__ == "__main__":
    print_gpu_info()
    print("\nMultiGPUTrainer ready!")
