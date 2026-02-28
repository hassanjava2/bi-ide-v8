"""
⚡ Resource Manager — إدارة موارد الحاسبة الفعلية
التحكم بنسبة استخدام CPU و GPU للتدريب

يدعم:
- تحديد نسبة CPU (10%-100%) — عدد threads + عمليات متوازية
- تحديد نسبة GPU (10%-100%) — حجم المودل + batch size + VRAM
- مراقبة حية للاستهلاك الفعلي
- مودل قابل للتوسع (5M - 200M+ parameters)
"""

import os
import sys
import time
import threading
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ================================================================
#  Hardware Detection
# ================================================================

def detect_system_resources() -> Dict[str, Any]:
    """Detect all available hardware resources."""
    info = {
        "cpu_cores_physical": multiprocessing.cpu_count(),
        "cpu_cores_logical": os.cpu_count() or 1,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "gpu_available": False,
        "gpu_name": "None",
        "gpu_vram_total_gb": 0.0,
        "gpu_vram_used_gb": 0.0,
        "gpu_vram_free_gb": 0.0,
        "gpu_temp_c": 0,
        "gpu_utilization": 0,
        "cuda_version": "N/A",
        "pytorch_version": "N/A",
    }

    # RAM
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    
    # GPU
    if TORCH_AVAILABLE:
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_total_gb"] = round(props.total_mem / (1024**3), 1) if hasattr(props, 'total_mem') else round(props.total_memory / (1024**3), 1)
            info["gpu_vram_used_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            info["gpu_vram_free_gb"] = round(info["gpu_vram_total_gb"] - info["gpu_vram_used_gb"], 2)
            info["cuda_version"] = torch.version.cuda or "N/A"

    # GPU temp and utilization via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                info["gpu_utilization"] = int(parts[0].strip())
                info["gpu_temp_c"] = int(parts[1].strip())
                info["gpu_vram_used_gb"] = round(float(parts[2].strip()) / 1024, 2)
                info["gpu_vram_total_gb"] = round(float(parts[3].strip()) / 1024, 2)
                info["gpu_vram_free_gb"] = round(info["gpu_vram_total_gb"] - info["gpu_vram_used_gb"], 2)
    except Exception:
        pass

    # CPU utilization
    if PSUTIL_AVAILABLE:
        info["cpu_utilization"] = psutil.cpu_percent(interval=0.3)
    
    return info


# ================================================================
#  Scalable Model — حجم يتناسب مع الموارد
# ================================================================

class ScalableTransformer(nn.Module):
    """Transformer model that scales based on resource allocation.
    
    Scale presets:
    - tiny:   ~2M params  (d=128, layers=2, heads=4)   — 10% GPU
    - small:  ~8M params  (d=256, layers=4, heads=8)   — 30% GPU
    - medium: ~35M params (d=512, layers=8, heads=8)   — 50% GPU
    - large:  ~85M params (d=768, layers=12, heads=12) — 80% GPU
    - xlarge: ~200M params(d=1024, layers=24, heads=16) — 100% GPU
    """

    PRESETS = {
        "tiny":   {"d_model": 128,  "nhead": 4,  "num_layers": 2,  "ff_dim": 512},
        "small":  {"d_model": 256,  "nhead": 8,  "num_layers": 4,  "ff_dim": 1024},
        "medium": {"d_model": 512,  "nhead": 8,  "num_layers": 8,  "ff_dim": 2048},
        "large":  {"d_model": 768,  "nhead": 12, "num_layers": 12, "ff_dim": 3072},
        "xlarge": {"d_model": 1024, "nhead": 16, "num_layers": 24, "ff_dim": 4096},
    }

    def __init__(self, preset: str = "medium", vocab_size: int = 32000, seq_len: int = 256):
        super().__init__()
        cfg = self.PRESETS.get(preset, self.PRESETS["medium"])
        self.preset = preset
        self.d_model = cfg["d_model"]
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, cfg["d_model"])
        self.pos_encoding = nn.Embedding(seq_len, cfg["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["ff_dim"],
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg["num_layers"])
        self.layer_norm = nn.LayerNorm(cfg["d_model"])
        self.fc_out = nn.Linear(cfg["d_model"], vocab_size)

        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.fc_out(x)
        return x


class SyntheticTrainingDataset(Dataset):
    """Generates training data that exercises compute.
    Uses varied sequence patterns to ensure the model actually learns.
    """

    def __init__(self, size: int = 10000, seq_len: int = 256, vocab_size: int = 32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate patterned sequences the model can learn
        # Pattern: shifted repeating sequences with noise
        import random
        base_pattern_len = random.randint(4, 32)
        base = torch.randint(1, self.vocab_size, (base_pattern_len,))
        
        # Repeat pattern to fill sequence
        repeats = (self.seq_len + base_pattern_len) // base_pattern_len + 1
        full = base.repeat(repeats)[:self.seq_len + 1]
        
        # Add some noise (10% of tokens)
        noise_mask = torch.rand(self.seq_len + 1) < 0.1
        noise = torch.randint(1, self.vocab_size, (self.seq_len + 1,))
        full = torch.where(noise_mask, noise, full)

        input_ids = full[:self.seq_len]
        target_ids = full[1:self.seq_len + 1]
        return input_ids, target_ids


# ================================================================
#  Resource Manager — التحكم الرئيسي
# ================================================================

@dataclass
class ResourceConfig:
    """Configuration for resource utilization."""
    cpu_percent: int = 80       # 10-100
    gpu_percent: int = 80       # 10-100
    ram_limit_percent: int = 80 # 10-100

    def validate(self):
        self.cpu_percent = max(10, min(100, self.cpu_percent))
        self.gpu_percent = max(10, min(100, self.gpu_percent))
        self.ram_limit_percent = max(10, min(100, self.ram_limit_percent))


class ResourceManager:
    """Manages real hardware resource utilization for training."""

    def __init__(self):
        self.config = ResourceConfig()
        self.system_info = detect_system_resources()
        self.training_active = False
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = False
        self.lock = threading.Lock()

        # Training state
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.dataset = None
        self.dataloader = None

        # Live metrics
        self.metrics = {
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "samples_processed": 0,
            "throughput_samples_sec": 0.0,
            "gpu_utilization": 0,
            "gpu_temp_c": 0,
            "gpu_vram_used_gb": 0.0,
            "gpu_vram_total_gb": 0.0,
            "cpu_utilization": 0.0,
            "ram_used_gb": 0.0,
            "model_preset": "medium",
            "model_params": 0,
            "batch_size": 0,
            "dataset_size": 0,
            "training_active": False,
            "start_time": None,
            "elapsed_seconds": 0,
            "estimated_remaining": "",
        }

        # CPU stress threads (for max CPU utilization)
        self._cpu_stress_threads: List[threading.Thread] = []
        self._cpu_stress_active = False

    def get_model_preset(self) -> str:
        """Select model size based on GPU allocation."""
        gpu_pct = self.config.gpu_percent
        if gpu_pct >= 90:
            return "xlarge"
        elif gpu_pct >= 70:
            return "large"
        elif gpu_pct >= 50:
            return "medium"
        elif gpu_pct >= 25:
            return "small"
        else:
            return "tiny"

    def get_batch_size(self) -> int:
        """Calculate batch size from GPU allocation and VRAM."""
        vram_gb = self.system_info.get("gpu_vram_total_gb", 0)
        gpu_pct = self.config.gpu_percent
        preset = self.get_model_preset()

        if not self.system_info.get("gpu_available", False):
            # CPU-only: smaller batches
            return max(4, int(16 * gpu_pct / 100))

        # Estimate VRAM needed per sample based on model size
        vram_budget_gb = vram_gb * (gpu_pct / 100) * 0.85  # 85% of allocated VRAM

        batch_map = {
            "tiny":   min(256, max(8, int(vram_budget_gb * 64))),
            "small":  min(128, max(8, int(vram_budget_gb * 32))),
            "medium": min(64,  max(4, int(vram_budget_gb * 12))),
            "large":  min(32,  max(2, int(vram_budget_gb * 6))),
            "xlarge": min(16,  max(1, int(vram_budget_gb * 2))),
        }
        return batch_map.get(preset, 16)

    def get_dataset_size(self) -> int:
        """Calculate dataset size based on resource allocation."""
        base = 5000
        return int(base * (self.config.gpu_percent / 100) * 2) + 1000

    def get_num_workers(self) -> int:
        """DataLoader workers based on CPU allocation."""
        cores = self.system_info.get("cpu_cores_physical", 1)
        return max(0, min(int(cores * self.config.cpu_percent / 100) - 1, 8))

    def configure(self, cpu_percent: int = None, gpu_percent: int = None,
                  ram_limit_percent: int = None) -> Dict[str, Any]:
        """Update resource configuration. Can be called while training."""
        with self.lock:
            if cpu_percent is not None:
                self.config.cpu_percent = cpu_percent
            if gpu_percent is not None:
                self.config.gpu_percent = gpu_percent
            if ram_limit_percent is not None:
                self.config.ram_limit_percent = ram_limit_percent
            self.config.validate()

            # Apply CPU thread limit immediately
            if TORCH_AVAILABLE:
                cores = self.system_info.get("cpu_cores_logical", 1)
                threads = max(1, int(cores * self.config.cpu_percent / 100))
                torch.set_num_threads(threads)
                os.environ["OMP_NUM_THREADS"] = str(threads)
                os.environ["MKL_NUM_THREADS"] = str(threads)

        preset = self.get_model_preset()
        batch_size = self.get_batch_size()

        return {
            "status": "configured",
            "cpu_percent": self.config.cpu_percent,
            "gpu_percent": self.config.gpu_percent,
            "ram_limit_percent": self.config.ram_limit_percent,
            "model_preset": preset,
            "model_params_estimate": {
                "tiny": "~2M", "small": "~8M", "medium": "~35M",
                "large": "~85M", "xlarge": "~200M"
            }.get(preset, "?"),
            "batch_size": batch_size,
            "cpu_threads": torch.get_num_threads() if TORCH_AVAILABLE else 1,
            "dataloader_workers": self.get_num_workers(),
        }

    def get_live_status(self) -> Dict[str, Any]:
        """Get real-time resource utilization."""
        # Refresh system info
        self.system_info = detect_system_resources()

        status = {
            **self.metrics,
            "config": {
                "cpu_percent": self.config.cpu_percent,
                "gpu_percent": self.config.gpu_percent,
                "ram_limit_percent": self.config.ram_limit_percent,
            },
            "system": self.system_info,
            "training_active": self.training_active,
        }

        if self.training_active and self.metrics.get("start_time"):
            elapsed = time.time() - self.metrics["start_time"]
            status["elapsed_seconds"] = int(elapsed)
            total_epochs = self.metrics.get("total_epochs", 1)
            current_epoch = self.metrics.get("epoch", 0)
            if current_epoch > 0:
                secs_per_epoch = elapsed / current_epoch
                remaining = secs_per_epoch * (total_epochs - current_epoch)
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                status["estimated_remaining"] = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        return status

    # ────────────────────────────────────────────────
    #  Training Engine
    # ────────────────────────────────────────────────

    def start_training(self, epochs: int = 100) -> Dict[str, Any]:
        """Start real intensive training that uses hardware resources."""
        if self.training_active:
            return {"status": "already_running", "message": "Training in progress"}

        if not TORCH_AVAILABLE:
            return {"status": "error", "message": "PyTorch not available"}

        self.stop_flag = False
        preset = self.get_model_preset()
        batch_size = self.get_batch_size()
        dataset_size = self.get_dataset_size()
        num_workers = self.get_num_workers()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply CPU thread limits
        cores = self.system_info.get("cpu_cores_logical", 1)
        threads = max(1, int(cores * self.config.cpu_percent / 100))
        torch.set_num_threads(threads)

        # Build model
        self.model = ScalableTransformer(preset=preset).to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Build dataset
        seq_len = 256 if preset in ("xlarge", "large") else 128
        self.dataset = SyntheticTrainingDataset(
            size=dataset_size, seq_len=seq_len, vocab_size=32000
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        # Update initial metrics
        self.metrics.update({
            "model_preset": preset,
            "model_params": self.model.param_count,
            "batch_size": batch_size,
            "dataset_size": dataset_size,
            "total_epochs": epochs,
            "epoch": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "training_active": True,
            "start_time": time.time(),
        })

        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(device, epochs, scheduler),
            daemon=True,
        )
        self.training_active = True
        self.training_thread.start()

        # Start CPU stress if CPU% is high
        if self.config.cpu_percent >= 70:
            self._start_cpu_stress()

        return {
            "status": "started",
            "device": str(device),
            "model_preset": preset,
            "model_params": f"{self.model.param_count / 1e6:.1f}M",
            "batch_size": batch_size,
            "dataset_size": dataset_size,
            "epochs": epochs,
            "cpu_threads": threads,
            "dataloader_workers": num_workers,
        }

    def _training_loop(self, device, epochs, scheduler):
        """Real training loop with intensive computation."""
        criterion = nn.CrossEntropyLoss()
        scaler = None
        use_amp = device.type == "cuda" and hasattr(torch.cuda.amp, "GradScaler")

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Gradient accumulation for extra compute
        grad_accum_steps = max(1, self.config.gpu_percent // 25)

        try:
            for epoch in range(1, epochs + 1):
                if self.stop_flag:
                    break

                epoch_start = time.time()
                self.model.train()
                total_loss = 0.0
                correct = 0
                total_tokens = 0
                samples_this_epoch = 0

                for batch_idx, (input_ids, target_ids) in enumerate(self.dataloader):
                    if self.stop_flag:
                        break

                    input_ids = input_ids.to(device, non_blocking=True)
                    target_ids = target_ids.to(device, non_blocking=True)

                    # Mixed precision forward pass for GPU efficiency
                    if use_amp and scaler:
                        with torch.cuda.amp.autocast():
                            output = self.model(input_ids)
                            loss = criterion(
                                output.reshape(-1, output.size(-1)),
                                target_ids.reshape(-1)
                            )
                            loss = loss / grad_accum_steps

                        scaler.scale(loss).backward()

                        if (batch_idx + 1) % grad_accum_steps == 0:
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    else:
                        output = self.model(input_ids)
                        loss = criterion(
                            output.reshape(-1, output.size(-1)),
                            target_ids.reshape(-1)
                        )
                        loss = loss / grad_accum_steps
                        loss.backward()

                        if (batch_idx + 1) % grad_accum_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    # Track metrics
                    total_loss += loss.item() * grad_accum_steps
                    _, predicted = output.max(-1)
                    correct += predicted.eq(target_ids).sum().item()
                    total_tokens += target_ids.numel()
                    samples_this_epoch += input_ids.size(0)

                scheduler.step()
                epoch_time = time.time() - epoch_start

                # Calculate metrics
                avg_loss = total_loss / max(1, len(self.dataloader))
                accuracy = 100.0 * correct / max(1, total_tokens)
                throughput = samples_this_epoch / max(0.01, epoch_time)

                # Update GPU metrics
                gpu_util = 0
                gpu_temp = 0
                gpu_vram_used = 0.0
                gpu_vram_total = 0.0
                if device.type == "cuda":
                    gpu_vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    try:
                        import subprocess
                        r = subprocess.run(
                            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=3
                        )
                        if r.returncode == 0:
                            parts = r.stdout.strip().split(",")
                            gpu_util = int(parts[0].strip())
                            gpu_temp = int(parts[1].strip())
                    except Exception:
                        pass

                cpu_util = psutil.cpu_percent(interval=0) if PSUTIL_AVAILABLE else 0
                ram_used = psutil.virtual_memory().used / (1024**3) if PSUTIL_AVAILABLE else 0

                # Update shared metrics
                with self.lock:
                    self.metrics.update({
                        "epoch": epoch,
                        "loss": round(avg_loss, 6),
                        "accuracy": round(accuracy, 2),
                        "samples_processed": self.metrics.get("samples_processed", 0) + samples_this_epoch,
                        "throughput_samples_sec": round(throughput, 1),
                        "gpu_utilization": gpu_util,
                        "gpu_temp_c": gpu_temp,
                        "gpu_vram_used_gb": round(gpu_vram_used, 2),
                        "gpu_vram_total_gb": round(gpu_vram_total, 2),
                        "cpu_utilization": cpu_util,
                        "ram_used_gb": round(ram_used, 2),
                    })

                # Log progress
                if epoch % 5 == 0 or epoch == 1:
                    print(f"📊 Epoch {epoch}/{epochs} | "
                          f"Loss={avg_loss:.4f} | Acc={accuracy:.1f}% | "
                          f"{throughput:.0f} samples/s | "
                          f"GPU={gpu_util}% {gpu_temp}°C | "
                          f"VRAM={gpu_vram_used:.1f}/{gpu_vram_total:.1f}GB | "
                          f"CPU={cpu_util:.0f}%")

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_active = False
            self.metrics["training_active"] = False
            self._stop_cpu_stress()
            # Free GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("✅ Training finished")

    def stop_training(self) -> Dict[str, Any]:
        """Stop training gracefully."""
        if not self.training_active:
            return {"status": "not_running"}
        
        self.stop_flag = True
        self._stop_cpu_stress()
        
        return {
            "status": "stopping",
            "epoch": self.metrics.get("epoch", 0),
            "total_epochs": self.metrics.get("total_epochs", 0),
        }

    # ────────────────────────────────────────────────
    #  CPU Stress — for maximum CPU utilization
    # ────────────────────────────────────────────────

    def _start_cpu_stress(self):
        """Start CPU-intensive background threads to maximize utilization."""
        if self._cpu_stress_active:
            return

        self._cpu_stress_active = True
        cores = self.system_info.get("cpu_cores_physical", 1)
        # Use a fraction of cores for stress (rest used by training DataLoader)
        stress_cores = max(1, int(cores * (self.config.cpu_percent / 100) * 0.5))

        for i in range(stress_cores):
            t = threading.Thread(target=self._cpu_stress_worker, daemon=True)
            self._cpu_stress_threads.append(t)
            t.start()

    def _cpu_stress_worker(self):
        """CPU-intensive computation for stress testing."""
        import math
        while self._cpu_stress_active and self.training_active:
            # Heavy math operations
            x = 1.0
            for _ in range(100000):
                x = math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1)
            # Small sleep to allow control
            time.sleep(0.001)

    def _stop_cpu_stress(self):
        """Stop all CPU stress threads."""
        self._cpu_stress_active = False
        self._cpu_stress_threads.clear()


# ================================================================
#  Singleton instance
# ================================================================

resource_manager = ResourceManager()
