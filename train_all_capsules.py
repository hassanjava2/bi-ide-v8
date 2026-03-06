#!/usr/bin/env python3
"""
train_all_capsules.py — يدرّب كل كبسولة بالتسلسل على بياناتها الخاصة

كل كبسولة تتدرب بشكل مستقل على بياناتها المتخصصة فقط.
"""

import sys
import os
import json
import time
import glob
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"multi_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("multi_trainer")


def train_capsule(capsule_dir: Path, capsule_id: str):
    """Train one capsule on its data"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset
    
    data_dir = capsule_dir / "data"
    model_dir = capsule_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    # Find all JSONL files for this capsule
    data_files = list(data_dir.glob("*.jsonl"))
    if not data_files:
        logger.warning(f"⚠️ [{capsule_id}] No data files!")
        return {"status": "no_data"}
    
    # Count samples
    total_lines = sum(1 for f in data_files for _ in open(f))
    if total_lines < 10:
        logger.warning(f"⚠️ [{capsule_id}] Only {total_lines} samples — skipping!")
        return {"status": "insufficient_data", "samples": total_lines}
    
    logger.info(f"🎯 [{capsule_id}] Training on {total_lines} samples from {len(data_files)} files")
    
    # Detect hardware
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    has_gpu = torch.cuda.is_available()
    
    # Adjust steps based on data size
    max_steps = min(total_lines * 2, 2000)  # 2 epochs or 2000 steps max
    
    if has_gpu:
        batch_size = 2
        gradient_accum = 4
        use_bf16 = True
        device_map = "auto"
        max_len = 256
    else:
        batch_size = 1
        gradient_accum = 4
        use_bf16 = False
        device_map = "cpu"
        max_len = 128
    
    # Load base model (raw material)
    base_model = "Qwen/Qwen2.5-1.5B"
    logger.info(f"📦 Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    # NOTE: gradient_checkpointing removed — slows training 50x
    
    # Load data
    dataset = load_dataset("json", data_files=[str(f) for f in data_files], split="train")
    logger.info(f"📊 Loaded {len(dataset)} samples")
    
    # Tokenize - handle multiple formats
    def tokenize_fn(examples):
        texts = []
        keys = list(examples.keys())
        for i in range(len(examples[keys[0]])):
            if "input_text" in examples and "output_text" in examples:
                text = f"<|im_start|>user\n{examples['input_text'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output_text'][i]}<|im_end|>"
            elif "instruction" in examples and "output" in examples:
                text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            elif "text" in examples:
                text = examples["text"][i]
            else:
                text = str(list(examples.values())[0][i])
            texts.append(text)
        return tokenizer(texts, truncation=True, max_length=max_len, padding="max_length")
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Add labels
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    tokenized = tokenized.map(add_labels, batched=True)
    
    # Training
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=5e-5,
        bf16=use_bf16,
        fp16=False,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
        max_steps=max_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )
    
    logger.info(f"🏋️ [{capsule_id}] Training {max_steps} steps...")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    
    # Save
    trainer.save_model(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except:
        pass
    
    metrics = result.metrics
    logger.info(f"✅ [{capsule_id}] Done! {elapsed/60:.1f} min — loss: {metrics.get('train_loss', '?')}")
    
    # Save result
    result_data = {
        "capsule_id": capsule_id,
        "status": "completed",
        "elapsed_minutes": round(elapsed / 60, 1),
        "training_samples": total_lines,
        "max_steps": max_steps,
        "metrics": metrics,
    }
    (capsule_dir / "training_result.json").write_text(json.dumps(result_data, indent=2, default=str))
    
    # Free GPU memory
    del model, trainer
    if has_gpu:
        torch.cuda.empty_cache()
    
    return result_data


def main():
    logger.info("=" * 60)
    logger.info("🏭 Multi-Capsule Trainer — بسم الله")
    logger.info("=" * 60)
    
    capsules_dir = PROJECT_ROOT / "capsules"
    
    # Find capsules with data
    trainable = []
    for capsule_dir in sorted(capsules_dir.iterdir()):
        if not capsule_dir.is_dir():
            continue
        capsule_id = capsule_dir.name
        data_dir = capsule_dir / "data"
        if not data_dir.exists():
            continue
        data_files = list(data_dir.glob("*.jsonl"))
        total = sum(1 for f in data_files for _ in open(f))
        if total >= 50:
            trainable.append((capsule_id, capsule_dir, total))
            logger.info(f"  ✅ {capsule_id}: {total} samples")
        else:
            logger.info(f"  ⏭️ {capsule_id}: {total} samples (skipping — too few)")
    
    if not trainable:
        logger.error("❌ No capsules with enough data!")
        sys.exit(1)
    
    logger.info(f"\n🎯 Training {len(trainable)} capsules sequentially\n")
    
    results = []
    for capsule_id, capsule_dir, sample_count in trainable:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧠 Training: {capsule_id} ({sample_count} samples)")
        logger.info(f"{'='*50}")
        
        try:
            result = train_capsule(capsule_dir, capsule_id)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ [{capsule_id}] Failed: {e}")
            results.append({"capsule_id": capsule_id, "status": "failed", "error": str(e)})
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 Training Summary:")
    for r in results:
        status = "✅" if r.get("status") == "completed" else "❌"
        loss = r.get("metrics", {}).get("train_loss", "?")
        mins = r.get("elapsed_minutes", "?")
        logger.info(f"  {status} {r.get('capsule_id')}: loss={loss} time={mins}min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
