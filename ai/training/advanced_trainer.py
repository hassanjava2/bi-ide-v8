"""
Advanced Trainer - المدرب المتقدم
Merged from: finetune.py, finetune-chat.py, finetune-extended.py

Features / المميزات:
  • Multiple model types support
  • Chat and completion modes
  • Extended training options
  • LoRA support with configurable rank
  • Checkpoint resume capability
  • Mixed precision (FP16/BF16)
  • Gradient accumulation
  • Data augmentation

PyTorch 2.x + CUDA 12.x Compatible
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """وضع التدريب - Training mode"""
    COMPLETION = "completion"  # تدريب إكمال النص
    CHAT = "chat"              # تدريب المحادثة
    EXTENDED = "extended"      # تدريب مكثف


class ModelType(Enum):
    """نوع النموذج - Model type"""
    CAUSAL_LM = "causal_lm"    # نموذج اللغة السببي
    SEQ2SEQ = "seq2seq"        # تسلسل إلى تسلسل


@dataclass
class TrainingConfig:
    """
    إعدادات التدريب - Training configuration
    
    Attributes:
        model_name: اسم النموذج من HuggingFace
        max_length: الطول الأقصى للتسلسل
        batch_size: حجم الدفعة
        epochs: عدد epochs
        learning_rate: معدل التعلم
        lora_r: رتبة LoRA
        lora_alpha: معامل LoRA
        gradient_accumulation_steps: خطوات تراكم التدرج
        warmup_ratio: نسبة الاحماء
        fp16: استخدام FP16
        bf16: استخدام BF16 (أفضل لـ A100/RTX 4090)
        save_steps: حفظ كل N خطوة
        eval_steps: تقييم كل N خطوة
        num_workers: عدد عمال تحميل البيانات
        cpu_threads: عدد خيوط المعالج
        data_augmentation: توسيع البيانات
        max_steps: الحد الأقصى للخطوات (اختياري)
        pulse_max_minutes: وقت التدريب الأقصى (لـ Hostinger)
    """
    model_name: str = "Qwen/Qwen2-0.5B"
    max_length: int = 512
    batch_size: int = 4
    epochs: float = 3.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 500
    eval_steps: int = 500
    num_workers: int = 4
    cpu_threads: int = 8
    data_augmentation: bool = False
    max_steps: int = 0
    pulse_max_minutes: int = 0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    
    # Extended training settings
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    def __post_init__(self):
        """Validate configuration"""
        if self.fp16 and self.bf16:
            logger.warning("Both fp16 and bf16 enabled, defaulting to bf16")
            self.fp16 = False


@dataclass
class TrainingResult:
    """نتيجة التدريب - Training result"""
    success: bool
    output_dir: Path
    final_loss: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    train_samples: int = 0
    eval_samples: int = 0
    training_time_seconds: float = 0.0
    error_message: Optional[str] = None


class AdvancedTrainer:
    """
    المدرب المتقدم - Advanced Trainer
    
    يدعم تدريب النماذج بعدة أوضاع:
    - Completion: تدريب على إكمال النص
    - Chat: تدريب على المحادثات بصيغة ChatML
    - Extended: تدريب مكثف مع توسيع البيانات
    
    Supports training in multiple modes:
    - Completion: Text completion training
    - Chat: ChatML format conversation training
    - Extended: Intensive training with data augmentation
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        mode: TrainingMode = TrainingMode.COMPLETION,
        output_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            mode: Training mode (completion/chat/extended)
            output_dir: Output directory for model
            base_dir: Base project directory
        """
        self.config = config or TrainingConfig()
        self.mode = mode
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        
        if output_dir is None:
            mode_suffix = {
                TrainingMode.COMPLETION: "finetuned",
                TrainingMode.CHAT: "finetuned-chat",
                TrainingMode.EXTENDED: "finetuned-extended"
            }
            self.output_dir = self.base_dir / "models" / mode_suffix[mode]
        else:
            self.output_dir = output_dir
        
        self.training_data: List[Dict[str, str]] = []
        self.model = None
        self.tokenizer = None
        self._trainer = None
        
        # إعداد الخيوط
        torch.set_num_threads(self.config.cpu_threads)
        torch.set_num_interop_threads(self.config.cpu_threads)
        
        logger.info(f"🚀 AdvancedTrainer initialized")
        logger.info(f"   Mode: {mode.value}")
        logger.info(f"   Model: {self.config.model_name}")
        logger.info(f"   Output: {self.output_dir}")
    
    def check_dependencies(self) -> bool:
        """
        فحص المكتبات المطلوبة
        Check required dependencies
        
        Returns:
            True if all dependencies available
        """
        required = ['torch', 'transformers', 'datasets', 'peft', 'accelerate']
        missing = []
        
        for lib in required:
            try:
                __import__(lib)
            except ImportError:
                missing.append(lib)
        
        if missing:
            logger.error(f"❌ Missing libraries: {', '.join(missing)}")
            logger.error(f"   Install: pip install {' '.join(missing)}")
            return False
        
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   VRAM: {vram:.1f} GB")
        else:
            logger.info("⚠️  CPU only mode")
        
        return True
    
    def load_training_data(
        self,
        data_source: Optional[Union[Path, List[Dict]]] = None
    ) -> List[Dict[str, str]]:
        """
        تحميل بيانات التدريب
        Load training data
        
        Args:
            data_source: Path to data file or list of examples
            
        Returns:
            List of training examples
        """
        all_data = []
        
        # تحميل من ملف
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_data = data
                elif isinstance(data, dict) and 'samples' in data:
                    all_data = data['samples']
        
        # تحميل من قائمة
        elif isinstance(data_source, list):
            all_data = data_source
        
        # تحميل تلقائي من المجلدات المعروفة
        else:
            # قاعدة المعرفة
            kb_path = self.base_dir / "data" / "knowledge" / "rag-knowledge-base.json"
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                for doc in docs:
                    text = doc.get('text', '')
                    answer = doc.get('answer', '')
                    if text and answer:
                        all_data.append({
                            "instruction": text,
                            "output": answer
                        })
            
            # ملفات التدريب
            training_dir = self.base_dir / "training" / "output"
            if training_dir.exists():
                for json_file in training_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                    except:
                        pass
        
        # توسيع البيانات في الوضع الممتد
        if self.mode == TrainingMode.EXTENDED or self.config.data_augmentation:
            all_data = self._augment_data(all_data)
        
        self.training_data = all_data
        logger.info(f"📊 Loaded {len(all_data)} training samples")
        return all_data
    
    def _augment_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        توسيع البيانات
        Augment training data
        
        Args:
            data: Original data
            
        Returns:
            Augmented data
        """
        augmented = []
        
        for item in data:
            inp = item.get('instruction', '')
            out = item.get('output', '')
            
            # النسخة الأصلية
            augmented.append(item)
            
            # نسخة معكوسة
            if len(out) > 100 and self.mode == TrainingMode.EXTENDED:
                augmented.append({
                    "instruction": f"لخّص المفهوم التالي:\n{out[:300]}",
                    "output": inp
                })
            
            # أسئلة لماذا
            if len(out) > 100:
                augmented.append({
                    "instruction": f"لماذا نستخدم هذا الأسلوب:\n{out[:200]}",
                    "output": f"نستخدم هذا الأسلوب لأنه:\n{inp}\n\nالتفاصيل:\n{out[:300]}"
                })
            
            # أمثلة للكود
            if 'function' in out.lower() or 'def ' in out.lower():
                augmented.append({
                    "instruction": f"اشرح هذا الكود بالتفصيل:\n{out[:300]}",
                    "output": f"شرح الكود:\n\nهذا الكود يقوم بـ: {inp}\n\nالتفاصيل:\n{out[:400]}"
                })
        
        # حد أقصى للتوسيع
        if len(augmented) > len(data) * 2:
            augmented = random.sample(augmented, len(data) * 2)
        
        logger.info(f"   🔄 Augmented: {len(data)} → {len(augmented)}")
        return augmented
    
    def _format_for_training(
        self,
        data: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        تنسيق البيانات للتدريب
        Format data for training
        
        Args:
            data: Raw data
            
        Returns:
            Formatted data
        """
        formatted = []
        
        for item in data:
            inp = item.get('instruction', item.get('input', item.get('question', '')))
            out = item.get('output', item.get('answer', item.get('response', '')))
            
            if not inp or not out:
                continue
            
            if self.mode == TrainingMode.CHAT:
                # ChatML format
                formatted.append({
                    "messages": [
                        {"role": "user", "content": inp[:self.config.max_length]},
                        {"role": "assistant", "content": out[:self.config.max_length * 2]}
                    ]
                })
            else:
                # Completion format
                text = f"### سؤال:\n{inp}\n\n### جواب:\n{out}"
                formatted.append({"text": text})
                
                # صيغة إضافية للوضع الممتد
                if self.mode == TrainingMode.EXTENDED and random.random() < 0.3:
                    text2 = f"[INST] {inp} [/INST]\n{out}"
                    formatted.append({"text": text2})
        
        if self.mode == TrainingMode.EXTENDED:
            random.shuffle(formatted)
        
        return formatted
    
    def _setup_model(self):
        """
        إعداد النموذج والـ tokenizer
        Setup model and tokenizer
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️  Device: {device}")
        
        # تحميل tokenizer
        logger.info(f"🤖 Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # تحميل النموذج
        dtype = torch.float16 if device == "cuda" else torch.float32
        if self.config.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # إعداد LoRA
        logger.info("🔧 Setting up LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")
    
    def train(
        self,
        data: Optional[List[Dict[str, str]]] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> TrainingResult:
        """
        بدء التدريب
        Start training
        
        Args:
            data: Training data (loads from files if None)
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training result
        """
        start_time = time.time()
        
        try:
            from transformers import (
                TrainingArguments, 
                Trainer, 
                DataCollatorForLanguageModeling,
                TrainerCallback
            )
            from datasets import Dataset
            
            # فحص المتطلبات
            if not self.check_dependencies():
                return TrainingResult(
                    success=False,
                    output_dir=self.output_dir,
                    error_message="Missing dependencies"
                )
            
            # تحميل البيانات
            if data is None:
                data = self.load_training_data()
            else:
                self.training_data = data
            
            if len(data) < 10:
                return TrainingResult(
                    success=False,
                    output_dir=self.output_dir,
                    error_message="Insufficient training data (< 10 samples)"
                )
            
            # تنسيق البيانات
            formatted = self._format_for_training(data)
            
            # إنشاء Dataset
            dataset = Dataset.from_list(formatted)
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_ds = split['train']
            eval_ds = split['test']
            
            logger.info(f"   Train: {len(train_ds)} | Eval: {len(eval_ds)}")
            
            # إعداد النموذج
            self._setup_model()
            
            # Tokenization
            logger.info("🔤 Tokenizing...")
            
            def tokenize(examples):
                if self.mode == TrainingMode.CHAT and "messages" in examples:
                    # معالجة ChatML
                    texts = []
                    for messages in examples["messages"]:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        texts.append(text)
                    return self.tokenizer(
                        texts,
                        truncation=True,
                        max_length=self.config.max_length,
                        padding="max_length"
                    )
                else:
                    return self.tokenizer(
                        examples["text"],
                        truncation=True,
                        max_length=self.config.max_length,
                        padding="max_length"
                    )
            
            train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
            eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)
            
            # إعداد مجلد الإخراج
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # إعدادات التدريب
            use_max_steps = self.config.max_steps > 0
            
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config.epochs if not use_max_steps else 1,
                max_steps=self.config.max_steps if use_max_steps else -1,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type=self.config.lr_scheduler_type,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                fp16=self.config.fp16 and torch.cuda.is_available(),
                bf16=self.config.bf16 and torch.cuda.is_available(),
                logging_steps=10,
                eval_strategy="steps" if use_max_steps else "epoch",
                eval_steps=self.config.eval_steps if use_max_steps else None,
                save_strategy="steps" if use_max_steps else "epoch",
                save_steps=self.config.save_steps if use_max_steps else None,
                save_total_limit=3,
                load_best_model_at_end=True,
                report_to="none",
                dataloader_num_workers=self.config.num_workers,
                dataloader_pin_memory=False,
            )
            
            # Callback للتحكم بالوقت (Hostinger pulse mode)
            class TimeLimitCallback(TrainerCallback):
                def __init__(self, max_minutes):
                    self.max_minutes = max_minutes
                    self.start = None
                
                def on_train_begin(self, args, state, control, **kwargs):
                    self.start = time.time()
                    return control
                
                def on_step_end(self, args, state, control, **kwargs):
                    if self.max_minutes <= 0:
                        return control
                    elapsed_min = (time.time() - self.start) / 60
                    if elapsed_min >= self.max_minutes:
                        control.should_training_stop = True
                        control.should_save = True
                    return control
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Trainer
            callbacks = []
            if self.config.pulse_max_minutes > 0:
                callbacks.append(TimeLimitCallback(self.config.pulse_max_minutes))
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_tok,
                eval_dataset=eval_tok,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            # بدء التدريب
            logger.info(f"\n🚀 Starting training...")
            logger.info(f"   Epochs: {self.config.epochs}")
            logger.info(f"   Batch: {self.config.batch_size} × {self.config.gradient_accumulation_steps}")
            logger.info(f"   LoRA: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            logger.info("=" * 50)
            
            if resume_from_checkpoint:
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer.train()
            
            # حفظ النموذج
            logger.info("\n💾 Saving model...")
            self.model.save_pretrained(self.output_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.output_dir)
            
            training_time = time.time() - start_time
            
            logger.info(f"\n✅ Training complete!")
            logger.info(f"   📁 {self.output_dir}")
            logger.info(f"   ⏱️  Time: {training_time/60:.1f} minutes")
            
            return TrainingResult(
                success=True,
                output_dir=self.output_dir,
                train_samples=len(train_ds),
                eval_samples=len(eval_ds),
                training_time_seconds=training_time
            )
            
        except Exception as e:
            logger.exception("Training failed")
            return TrainingResult(
                success=False,
                output_dir=self.output_dir,
                error_message=str(e),
                training_time_seconds=time.time() - start_time
            )
    
    def train_chat(
        self,
        chat_data: Optional[List[List[Dict]]] = None,
        **kwargs
    ) -> TrainingResult:
        """
        تدريب المحادثة (اختصار)
        Train chat model (convenience method)
        
        Args:
            chat_data: Chat data in ChatML format
            **kwargs: Additional arguments for train()
            
        Returns:
            Training result
        """
        self.mode = TrainingMode.CHAT
        self.config.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.config.max_length = 1024
        self.config.batch_size = 2
        self.config.gradient_accumulation_steps = 8
        
        if chat_data:
            # تحويل ChatML إلى التنسيق المتوقع
            formatted = []
            for conversation in chat_data:
                if len(conversation) >= 2:
                    formatted.append({
                        "instruction": conversation[0].get("content", ""),
                        "output": conversation[1].get("content", "")
                    })
            return self.train(data=formatted, **kwargs)
        
        return self.train(**kwargs)
    
    def train_extended(
        self,
        data: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> TrainingResult:
        """
        تدريب مكثف (اختصار)
        Extended training (convenience method)
        
        Args:
            data: Training data
            **kwargs: Additional arguments for train()
            
        Returns:
            Training result
        """
        self.mode = TrainingMode.EXTENDED
        self.config.epochs = 15
        self.config.lora_r = 32
        self.config.lora_alpha = 64
        self.config.target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.config.data_augmentation = True
        self.config.learning_rate = 1e-4
        
        return self.train(data=data, **kwargs)


# دوال مساعدة - Helper functions
def create_trainer(
    mode: str = "completion",
    model_name: Optional[str] = None,
    **config_kwargs
) -> AdvancedTrainer:
    """
    إنشاء مدرب جديد
    Create a new trainer
    
    Args:
        mode: Training mode (completion/chat/extended)
        model_name: Model name
        **config_kwargs: Additional config options
        
    Returns:
        AdvancedTrainer instance
    """
    mode_enum = TrainingMode(mode)
    config = TrainingConfig(**config_kwargs)
    if model_name:
        config.model_name = model_name
    
    return AdvancedTrainer(config=config, mode=mode_enum)


def quick_train(
    data: List[Dict[str, str]],
    output_name: str = "quick_finetuned",
    epochs: int = 3
) -> TrainingResult:
    """
    تدريب سريع
    Quick training shortcut
    
    Args:
        data: Training data
        output_name: Output directory name
        epochs: Number of epochs
        
    Returns:
        Training result
    """
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "models" / output_name
    
    config = TrainingConfig(epochs=epochs, batch_size=4)
    trainer = AdvancedTrainer(config=config, output_dir=output_dir)
    
    return trainer.train(data=data)


if __name__ == "__main__":
    # اختبار بسيط
    print("=" * 50)
    print("Advanced Trainer - Test")
    print("=" * 50)
    
    trainer = AdvancedTrainer()
    trainer.check_dependencies()
    
    print("\n✅ AdvancedTrainer ready!")
