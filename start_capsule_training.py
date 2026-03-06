#!/usr/bin/env python3
"""
start_capsule_training.py — Capsule Training Launcher

يشتغل على أي جهاز:
- RTX 5090: تدريب GPU كامل
- Hostinger: تجهيز بيانات + تدريب CPU (أبطأ بس يساعد)
- Windows: حسب الموارد

يسوي:
1. ينشئ كبسولة تجريبية
2. يحمّل البيانات الموجودة
3. يبدي التدريب
4. يسجّل كل شي بـ logs

الاستخدام:
  python3 start_capsule_training.py
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup project path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("capsule_launcher")


def detect_environment():
    """اكتشاف البيئة — GPU? CPU? أي جهاز؟"""
    env = {
        "hostname": os.uname().nodename,
        "has_gpu": False,
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    
    try:
        import torch
        env["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            env["has_gpu"] = True
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    except ImportError:
        env["pytorch"] = "NOT INSTALLED"
    
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = "NOT INSTALLED"
    
    return env


def find_training_data():
    """البحث عن بيانات التدريب"""
    search_paths = [
        Path("/home/bi/training_data/downloads"),
        Path("/home/bi/training_data"),
        Path("/opt/bi-iq-app/shared_data"),
        Path.home() / "training_data",
        Path("/data/training"),
    ]
    
    all_files = []
    for path in search_paths:
        if path.exists():
            jsonl_files = list(path.glob("**/*.jsonl"))
            all_files.extend(jsonl_files)
            logger.info(f"📂 {path}: {len(jsonl_files)} ملفات JSONL")
    
    return all_files


def prepare_capsule_data(data_files, capsule_data_dir, max_samples=10000):
    """تجهيز بيانات للكبسولة — يأخذ أفضل العينات"""
    capsule_data_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    output_path = capsule_data_dir / "train_data.jsonl"
    
    with open(output_path, "w") as out:
        for f in data_files:
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            # Accept multiple formats
                            has_data = (
                                ("instruction" in sample and "output" in sample)
                                or ("input_text" in sample and "output_text" in sample)
                                or ("prompt" in sample and "response" in sample)
                                or "text" in sample
                            )
                            if has_data:
                                out.write(line + "\n")
                                total += 1
                                if total >= max_samples:
                                    break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"⚠️ خطأ بقراءة {f.name}: {e}")
            
            if total >= max_samples:
                break
    
    logger.info(f"📊 جهّز {total} عينة → {output_path}")
    return total, output_path


def run_training(capsule_dir, data_path, env):
    """تشغيل التدريب — GPU أو CPU حسب المتاح"""
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
        logger.error(f"❌ مكتبة ناقصة: {e}")
        logger.error("نصّب: pip install transformers datasets accelerate torch")
        return {"status": "missing_deps", "error": str(e)}
    
    model_name = "Qwen/Qwen2.5-1.5B"
    output_dir = str(capsule_dir / "model")
    
    # Adjust settings based on hardware
    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if env["has_gpu"]:
        batch_size = 1
        gradient_accum = 16
        fp16 = True
        device_map = "auto"
        max_len = 256  # Reduce to fit in 24GB
        logger.info(f"🚀 GPU training: {env['gpu_name']} ({env['gpu_memory_gb']}GB)")
    else:
        batch_size = 1
        max_len = 256
        gradient_accum = 16
        fp16 = False
        device_map = "cpu"
        logger.info(f"🐢 CPU training: {env['cpu_count']} cores (أبطأ بس يساعد)")
    
    # Load model (المادة الخام)
    logger.info(f"📦 تحميل المادة الخام: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Load dataset
    logger.info(f"📂 تحميل البيانات: {data_path}")
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info(f"📊 {len(dataset)} عينة تدريب")
    
    # Tokenize
    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            if "instruction" in examples and "output" in examples:
                text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            elif "input_text" in examples and "output_text" in examples:
                text = f"<|im_start|>user\n{examples['input_text'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output_text'][i]}<|im_end|>"
            elif "prompt" in examples and "response" in examples:
                text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['response'][i]}<|im_end|>"
            elif "text" in examples:
                text = examples["text"][i]
            else:
                text = str(list(examples.values())[0][i])
            texts.append(text)
        
        return tokenizer(texts, truncation=True, max_length=max_len, padding="max_length")
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Add labels for causal LM (labels = input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized = tokenized.map(add_labels, batched=True)
    
    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=5e-5,
        fp16=fp16,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
        max_steps=1000,  # Limit first run to 1000 steps
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )
    
    logger.info("🏋️ التدريب بدأ...")
    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time
    
    # Save
    trainer.save_model(output_dir)
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass  # Model already saved by trainer
    
    metrics = result.metrics
    logger.info(f"✅ التدريب اكتمل — {elapsed/60:.1f} دقيقة — loss: {metrics.get('train_loss', '?')}")
    
    return {
        "status": "completed",
        "elapsed_minutes": round(elapsed / 60, 1),
        "metrics": metrics,
        "model_path": output_dir,
    }


def main():
    logger.info("=" * 60)
    logger.info("🧠 Capsule Training Launcher — بسم الله")
    logger.info("=" * 60)
    
    # 1. Detect environment
    env = detect_environment()
    logger.info(f"🖥️  {env['hostname']}: GPU={'✅ ' + env['gpu_name'] if env['has_gpu'] else '❌ CPU only'}")
    logger.info(f"📦 PyTorch={env.get('pytorch', '?')} Transformers={env.get('transformers', '?')}")
    
    if env.get("transformers") == "NOT INSTALLED":
        logger.error("❌ transformers مو منصّبة — نصّب: pip install transformers datasets accelerate")
        sys.exit(1)
    
    # 2. Find training data
    data_files = find_training_data()
    if not data_files:
        logger.error("❌ ما لكيت بيانات تدريب JSONL!")
        sys.exit(1)
    logger.info(f"📊 لكيت {len(data_files)} ملفات JSONL")
    
    # 3. Create capsule directory
    capsule_id = f"capsule_{env['hostname']}_{datetime.now().strftime('%Y%m%d')}"
    capsule_dir = PROJECT_ROOT / "capsules" / capsule_id
    capsule_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Prepare data
    max_samples = 5000 if env["has_gpu"] else 1000  # Less for CPU
    total, data_path = prepare_capsule_data(data_files, capsule_dir / "data", max_samples)
    
    if total == 0:
        logger.error("❌ ما لكيت عينات صالحة!")
        sys.exit(1)
    
    # 5. Save environment info
    env_path = capsule_dir / "environment.json"
    env_path.write_text(json.dumps({**env, "training_samples": total, "started_at": datetime.now().isoformat()}, indent=2))
    
    # 6. Train!
    result = run_training(capsule_dir, data_path, env)
    
    # 7. Save result
    result_path = capsule_dir / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    
    logger.info("=" * 60)
    if result.get("status") == "completed":
        logger.info(f"🎉 التدريب نجح! الموديل محفوظ بـ: {result['model_path']}")
        logger.info(f"⏱️  {result['elapsed_minutes']} دقيقة")
    else:
        logger.info(f"❌ التدريب فشل: {result}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
