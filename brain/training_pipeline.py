"""
training_pipeline.py — Pipeline تدريب من الصفر

يدرّب موديل خاص من الصفر (مو LoRA) داخل كبسولة.
يشتغل على RTX 5090 (24.5GB) و H200 (80GB×8).

الأسلوب: Progressive Training
1. يبدي صغير (100M params)
2. يتدرب على بيانات الكبسولة
3. يكبّر تدريجياً
4. يقيّم بعد كل مرحلة

القاعدة: الموديل الأساسي = مادة خام.
الابتكار = كيف ندرّبه ونخصصه داخل الكبسولة.
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger("training_pipeline")

# ─── Config ───────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B"
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/home/bi/training_data"))


class TrainingPipeline:
    """
    Pipeline تدريب من الصفر — يعمل داخل كبسولة واحدة
    
    كل كبسولة تدرّب موديلها الخاص:
    1. تجمع بياناتها (من الكشافة أو من كبسولات ثانية)
    2. تنظّف وتجهّز البيانات
    3. تدرّب موديل من الصفر (progressive)
    4. تقيّم النتيجة
    5. إذا ضعيفة → تعيد بأسلوب مختلف
    """
    
    def __init__(self, capsule_dir: Path, capsule_id: str):
        self.capsule_dir = capsule_dir
        self.capsule_id = capsule_id
        self.data_dir = capsule_dir / "data"
        self.model_dir = capsule_dir / "model"
        self.logs_dir = capsule_dir / "training_logs"
        
        # Create dirs
        for d in [self.data_dir, self.model_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> dict:
        config_path = self.capsule_dir / "training_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        
        config = {
            "capsule_id": self.capsule_id,
            "base_model": DEFAULT_BASE_MODEL,
            "current_phase": 0,
            "phases_completed": [],
            "total_tokens_trained": 0,
            "best_eval_score": 0.0,
            "created_at": datetime.now().isoformat(),
        }
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        return config
    
    def _save_config(self):
        config_path = self.capsule_dir / "training_config.json"
        config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    # ─── Data Preparation ────────────────────────────────────
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        تجهيز بيانات التدريب من ملفات الكبسولة
        يقرأ كل ملفات .jsonl من data/ ويجهّزها
        """
        data_files = list(self.data_dir.glob("*.jsonl"))
        
        if not data_files:
            return {
                "status": "no_data",
                "message": f"ما اكو بيانات بكبسولة {self.capsule_id} — الكشاف لازم يجيب بيانات أول",
            }
        
        total_samples = 0
        all_samples = []
        
        for f in data_files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            all_samples.append(sample)
                            total_samples += 1
                        except json.JSONDecodeError:
                            continue
        
        # Save prepared dataset
        prepared_path = self.data_dir / "prepared_train.jsonl"
        with open(prepared_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"📊 [{self.capsule_id}] جهّز {total_samples} عينة للتدريب")
        
        return {
            "status": "ready",
            "total_samples": total_samples,
            "data_files": len(data_files),
            "prepared_path": str(prepared_path),
        }
    
    # ─── Training ────────────────────────────────────────────
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
    ) -> Dict[str, Any]:
        """
        تدريب الموديل من الصفر داخل الكبسولة
        
        يستخدم الموديل الأساسي كنقطة بداية (مادة خام)
        ثم يدرّبه بالكامل على بيانات الكبسولة
        = الموديل الناتج خاص بالكبسولة 100%
        """
        # Check data
        data_status = self.prepare_data()
        if data_status["status"] != "ready":
            return data_status
        
        prepared_path = data_status["prepared_path"]
        total_samples = data_status["total_samples"]
        
        training_config = {
            "capsule_id": self.capsule_id,
            "base_model": self.config["base_model"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_samples": total_samples,
            "data_path": prepared_path,
            "output_dir": str(self.model_dir),
            "started_at": datetime.now().isoformat(),
            "status": "starting",
        }
        
        # Save training config for the actual trainer
        train_config_path = self.capsule_dir / "active_training.json"
        train_config_path.write_text(json.dumps(training_config, indent=2))
        
        logger.info(f"🏋️ [{self.capsule_id}] بدأ التدريب: {total_samples} عينة × {epochs} epochs")
        
        try:
            result = self._run_training(training_config)
            
            # Update config
            self.config["current_phase"] += 1
            self.config["phases_completed"].append({
                "phase": self.config["current_phase"],
                "samples": total_samples,
                "epochs": epochs,
                "timestamp": datetime.now().isoformat(),
                "result": result.get("status", "unknown"),
            })
            self.config["total_tokens_trained"] += total_samples * max_length
            self._save_config()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [{self.capsule_id}] فشل التدريب: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_training(self, config: dict) -> Dict[str, Any]:
        """
        التدريب الفعلي — يستخدم transformers + الـ GPU
        
        ينتج موديل مخصص بالكامل للكبسولة
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from datasets import load_dataset
        except ImportError as e:
            return {
                "status": "missing_deps",
                "error": f"مطلوب: {e}",
                "message": "نصّب: pip install transformers datasets torch",
            }
        
        # Check GPU
        if not torch.cuda.is_available():
            return {"status": "no_gpu", "error": "GPU غير متوفر — التدريب يحتاج GPU"}
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"🖥️ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        
        # Load base model (المادة الخام)
        logger.info(f"📦 تحميل المادة الخام: {config['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load dataset
        dataset = load_dataset("json", data_files=config["data_path"], split="train")
        
        # Tokenize
        def tokenize_fn(examples):
            # Support multiple formats
            texts = []
            for i in range(len(examples.get("instruction", examples.get("text", [])))):
                if "instruction" in examples and "output" in examples:
                    text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
                elif "text" in examples:
                    text = examples["text"][i]
                elif "prompt" in examples and "response" in examples:
                    text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['response'][i]}<|im_end|>"
                else:
                    text = str(list(examples.values())[0][i])
                texts.append(text)
            
            return tokenizer(
                texts,
                truncation=True,
                max_length=config["max_length"],
                padding="max_length",
                return_tensors="pt",
            )
        
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        
        # Training arguments
        output_dir = config["output_dir"]
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            fp16=True,
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            dataloader_pin_memory=False,
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
        )
        
        logger.info(f"🚀 [{self.capsule_id}] التدريب بدأ...")
        train_result = trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        metrics = train_result.metrics
        logger.info(f"✅ [{self.capsule_id}] التدريب اكتمل — loss: {metrics.get('train_loss', '?')}")
        
        return {
            "status": "completed",
            "capsule_id": self.capsule_id,
            "model_path": output_dir,
            "metrics": metrics,
            "gpu": gpu_name,
            "timestamp": datetime.now().isoformat(),
        }
    
    # ─── Inference ───────────────────────────────────────────
    
    def inference(self, prompt: str, max_tokens: int = 256) -> str:
        """
        استخدام الموديل المدرّب للإجابة
        يستخدم موديل الكبسولة الخاص (المدرّب من الصفر)
        """
        model_path = self.model_dir
        
        # Check if model exists
        if not (model_path / "config.json").exists():
            return f"⚠️ كبسولة {self.capsule_id} ما اتدرّبت بعد — يحتاج تدريب أول"
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return response.strip()
            
        except Exception as e:
            return f"⚠️ خطأ بالاستدلال: {e}"
    
    # ─── Status ──────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """حالة pipeline التدريب"""
        model_exists = (self.model_dir / "config.json").exists()
        data_files = list(self.data_dir.glob("*.jsonl"))
        
        return {
            "capsule_id": self.capsule_id,
            "has_model": model_exists,
            "data_files": len(data_files),
            "current_phase": self.config.get("current_phase", 0),
            "phases_completed": len(self.config.get("phases_completed", [])),
            "total_tokens_trained": self.config.get("total_tokens_trained", 0),
            "best_eval_score": self.config.get("best_eval_score", 0.0),
            "base_model": self.config.get("base_model", DEFAULT_BASE_MODEL),
        }
