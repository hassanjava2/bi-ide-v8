#!/usr/bin/env python3
"""
training_worker.py — عامل التدريب

يشتغل على أي حاسبة بعيدة:
  1. يستقبل بيانات كبسولة (JSONL + config)
  2. يدرب LoRA على الموديل الأساسي
  3. يحفظ الـ adapter
  4. يبلّغ القيادة إنه خلص

الاستخدام:
  python3 training_worker.py --capsule-dir /tmp/capsule_python --config config.yaml
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(message)s",
)
logger = logging.getLogger("worker")


def train_capsule(capsule_dir: Path, base_model: str, config: dict) -> dict:
    """تدريب كبسولة واحدة — نفس منطق brain_daemon._train_capsule"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    data_dir = capsule_dir / "data"
    model_dir = capsule_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    capsule_id = capsule_dir.name

    # === دمج JSONL ===
    merged = data_dir / "merged_train.jsonl"
    count = 0
    with open(merged, "w") as out:
        for f in data_dir.glob("*.jsonl"):
            if f.name == "merged_train.jsonl":
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        sample = {
                            "input_text": d.get("input_text", d.get("instruction", "")),
                            "output_text": d.get("output_text", d.get("output", "")),
                        }
                        if sample["input_text"] and sample["output_text"]:
                            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            count += 1
                    except Exception:
                        pass

    min_samples = config.get("min_samples", 20)
    if count < min_samples:
        result = {"capsule": capsule_id, "status": "skip", "reason": f"only {count} samples (need {min_samples})"}
        logger.info(f"⏭️  {capsule_id}: skipped — {count} samples")
        return result

    has_gpu = torch.cuda.is_available()
    max_steps = min(count * config.get("max_steps_per_sample_ratio", 2), config.get("max_steps_cap", 2000))

    logger.info(f"🏋️ Training {capsule_id}: {count} samples, {max_steps} steps, GPU={'✅' if has_gpu else '❌'}")

    # === الموديل ===
    model_source = base_model
    if (model_dir / "config.json").exists():
        model_source = str(model_dir)
        logger.info(f"  → Continuing from existing model")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if has_gpu else torch.float32,
        device_map="auto" if has_gpu else "cpu",
        trust_remote_code=True,
    )
    if has_gpu:
        model.gradient_checkpointing_enable()

    # === البيانات ===
    dataset = load_dataset("json", data_files=str(merged), split="train")

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples["input_text"])):
            inp = str(examples["input_text"][i])
            out = str(examples["output_text"][i])
            text = "<|im_start|>user\n" + inp + "<|im_end|>\n<|im_start|>assistant\n" + out + "<|im_end|>"
            texts.append(text)
        max_len = config.get("max_seq_length", 256)
        result = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length")
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # === التدريب ===
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("gradient_accumulation", 16) if has_gpu else 4

    args = TrainingArguments(
        output_dir=str(model_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.get("learning_rate", 5e-5),
        bf16=has_gpu and config.get("bf16", True),
        fp16=False,
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        logging_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    loss = result.metrics.get("train_loss", None)

    # === حفظ ===
    trainer.save_model(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except Exception:
        pass

    logger.info(f"✅ {capsule_id}: {elapsed/60:.1f}min, loss={loss}")

    info = {
        "capsule": capsule_id,
        "status": "completed",
        "minutes": round(elapsed / 60, 1),
        "samples": count,
        "steps": max_steps,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "worker": os.uname().nodename,
    }

    # حفظ النتيجة
    (capsule_dir / "result.json").write_text(json.dumps(info, indent=2, default=str))

    # تنظيف الذاكرة
    del model, trainer, tokenized, dataset
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()

    return info


def main():
    parser = argparse.ArgumentParser(description="BI-IDE Training Worker")
    parser.add_argument("--capsule-dir", required=True, help="Path to capsule directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B", help="Base model name")
    parser.add_argument("--config", default=None, help="Training config YAML")
    args = parser.parse_args()

    capsule_dir = Path(args.capsule_dir)
    if not capsule_dir.exists():
        logger.error(f"❌ Capsule dir not found: {capsule_dir}")
        sys.exit(1)

    # تحميل الإعدادات
    config = {}
    if args.config and Path(args.config).exists():
        import yaml
        config = yaml.safe_load(Path(args.config).read_text()).get("training", {})

    result = train_capsule(capsule_dir, args.base_model, config)

    # كتابة النتيجة لـ stdout (يقرأها الـ dispatcher)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
