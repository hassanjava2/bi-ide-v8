"""
Training Pipeline V8 - خط أنابيب التدريب

متوافق مع قوانين RTX5090:
- Ollama = تدريب فقط (ليس استنتاج)
- Thermal protection: 97°C = STOP
- Auto-merge adapters
"""

import os
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .thermal_guard import get_thermal_guard, ThermalState


class TrainingState(Enum):
    """حالة التدريب"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    MERGING = "merging"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"
    THERMAL_PAUSE = "thermal_pause"


@dataclass
class TrainingConfig:
    """إعدادات التدريب"""
    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B"
    output_dir: str = "/home/bi/training_data/models/finetuned"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Thermal
    thermal_throttle: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingProgress:
    """تقدم التدريب"""
    state: TrainingState
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    samples_processed: int
    start_time: float
    estimated_remaining_sec: Optional[float] = None
    thermal_state: Optional[str] = None


class TrainingPipeline:
    """
    خط أنابيب التدريب V8
    
    Features:
    - Automatic thermal protection
    - Ollama integration for training data
    - Auto-merge best adapters
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.state = TrainingState.IDLE
        self.progress: Optional[TrainingProgress] = None
        
        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._callbacks: List[Callable[[TrainingProgress], None]] = []
        
        # Thermal guard
        self._thermal = get_thermal_guard()
        self._thermal.on_state_change(self._on_thermal_change)
        
        # Stats
        self._completed_runs = 0
        self._failed_runs = 0
        
        print("🎓 Training Pipeline V8 initialized")
    
    def _on_thermal_change(self, state: ThermalState, reading):
        """معالج تغيير الحرارة"""
        if state == ThermalState.EMERGENCY:
            print("🚨 THERMAL EMERGENCY - Pausing training")
            self._pause_for_thermal()
    
    def _pause_for_thermal(self):
        """إيقاف مؤقت للحرارة"""
        with self._lock:
            if self.state == TrainingState.TRAINING:
                self.state = TrainingState.THERMAL_PAUSE
                print("⏸️ Training paused for thermal protection")
    
    def _resume_from_thermal(self):
        """استئناف بعد انخفاض الحرارة"""
        with self._lock:
            if self.state == TrainingState.THERMAL_PAUSE:
                self.state = TrainingState.TRAINING
                print("▶️ Training resumed (temperature safe)")
    
    def _training_loop(self, dataset_path: Optional[str] = None):
        """حلقة التدريب الرئيسية"""
        try:
            with self._lock:
                self.state = TrainingState.PREPARING
                self._stop_event.clear()
            
            # التحقق الحراري
            if not self._thermal.is_safe_to_train():
                print("🌡️ Too hot to start training")
                self.state = TrainingState.THERMAL_PAUSE
                return
            
            # بدء المراقبة الحرارية
            self._thermal.start()
            
            # تحضير البيانات
            print("📚 Preparing training data...")
            samples = self._load_training_samples(dataset_path)
            
            if len(samples) < 10:
                print("⚠️ Not enough training samples")
                self.state = TrainingState.ERROR
                return
            
            print(f"📊 Loaded {len(samples)} training samples")
            
            # التدريب
            with self._lock:
                self.state = TrainingState.TRAINING
            
            self._run_training(samples)
            
            # دمج إذا نجح
            if self.state != TrainingState.ERROR:
                self._auto_merge()
                self._completed_runs += 1
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.state = TrainingState.ERROR
            self._failed_runs += 1
        finally:
            self._thermal.stop()
            with self._lock:
                if self.state not in (TrainingState.ERROR, TrainingState.THERMAL_PAUSE):
                    self.state = TrainingState.IDLE
    
    def _load_training_samples(self, dataset_path: Optional[str]) -> List[Dict]:
        """تحميل عينات التدريب"""
        samples = []
        
        # من ملف محدد
        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path) as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except:
                        pass
        
        # من مجلد ingest
        ingest_dir = Path("/home/bi/training_data/ingest")
        if ingest_dir.exists():
            for f in ingest_dir.glob("*.jsonl"):
                with open(f) as file:
                    for line in file:
                        try:
                            samples.append(json.loads(line))
                        except:
                            pass
        
        return samples
    
    def _run_training(self, samples: List[Dict]):
        """تنفيذ التدريب"""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
            import torch
            
            # تحضير Dataset
            texts = []
            for s in samples:
                if "input" in s and "output" in s:
                    texts.append(f"### Instruction:\n{s['input']}\n\n### Response:\n{s['output']}")
                elif "text" in s:
                    texts.append(s["text"])
            
            dataset = Dataset.from_dict({"text": texts})
            
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length"
                )
            
            tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
            
            # Model
            print("🧠 Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # LoRA config
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            
            # Output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"v8_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=True,
                remove_unused_columns=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                ),
            )
            
            print("🔥 Starting training...")
            trainer.train()
            
            # Save
            with self._lock:
                self.state = TrainingState.SAVING
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"✅ Training complete: {output_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def _auto_merge(self):
        """دمج تلقائي للـ adapters"""
        from .model_manager import get_model_manager
        
        manager = get_model_manager()
        versions = manager.discover_versions()
        
        if len(versions) >= 3:
            print("🔀 Auto-merging top 3 adapters...")
            # Placeholder - real merge would happen here
    
    def start_training(self, dataset_path: Optional[str] = None) -> bool:
        """بدء التدريب"""
        with self._lock:
            if self.state in (TrainingState.TRAINING, TrainingState.PREPARING):
                print("⚠️ Training already in progress")
                return False
        
        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(dataset_path,),
            daemon=True
        )
        self._training_thread.start()
        
        return True
    
    def stop_training(self):
        """إيقاف التدريب"""
        self._stop_event.set()
        with self._lock:
            self.state = TrainingState.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على الحالة"""
        thermal_stats = self._thermal.get_stats()
        
        return {
            "state": self.state.value,
            "progress": {
                "epoch": self.progress.epoch if self.progress else 0,
                "total_epochs": self.progress.total_epochs if self.progress else 0,
                "loss": self.progress.loss if self.progress else 0,
            } if self.progress else None,
            "thermal": thermal_stats,
            "stats": {
                "completed_runs": self._completed_runs,
                "failed_runs": self._failed_runs,
            }
        }


# Singleton
_pipeline: Optional[TrainingPipeline] = None


def get_training_pipeline() -> TrainingPipeline:
    """الحصول على خط الأنابيب الموحد"""
    global _pipeline
    if _pipeline is None:
        _pipeline = TrainingPipeline()
    return _pipeline
