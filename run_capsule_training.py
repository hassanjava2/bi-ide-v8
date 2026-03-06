#!/usr/bin/env python3
"""
run_capsule_training.py — Train capsules sequentially on RTX 5090
Uses PROVEN settings from successful 3.28s/step run.
"""
import sys, os, json, time, logging, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOG_PATH = Path("/tmp/capsule_sequential_train.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH)), logging.StreamHandler()])
logger = logging.getLogger("train")

CAPSULES_TO_TRAIN = ["code_python", "code_web", "knowledge_arabic"]
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ROOT = Path("/home/bi/bi-ide-v8/capsules")


def train_one(capsule_id):
    data_path = ROOT / capsule_id / "data" / "merged_train.jsonl"
    model_dir = ROOT / capsule_id / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.info(f"SKIP {capsule_id} — no merged_train.jsonl")
        return None

    samples = sum(1 for _ in open(data_path))
    if samples < 50:
        logger.info(f"SKIP {capsule_id} — only {samples} samples")
        return None

    max_steps = min(samples * 2, 2000)
    logger.info(f"{'='*50}")
    logger.info(f"CAPSULE: {capsule_id} | {samples} samples | {max_steps} steps")
    logger.info(f"{'='*50}")

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()

    # Load + tokenize
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} samples")

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples["input_text"])):
            inp = examples["input_text"][i]
            out = examples["output_text"][i]
            text = "<|im_start|>user\n" + inp + "<|im_end|>\n<|im_start|>assistant\n" + out + "<|im_end|>"
            texts.append(text)
        result = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # Training — PROVEN settings (3.28s/step on RTX 5090)
    args = TrainingArguments(
        output_dir=str(model_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        bf16=True,
        fp16=False,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
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

    logger.info(f"START training {capsule_id}...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    loss = result.metrics.get("train_loss", "?")

    trainer.save_model(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except:
        pass

    logger.info(f"DONE {capsule_id}: {elapsed/60:.1f} min, loss={loss}")

    info = {
        "capsule": capsule_id,
        "status": "completed",
        "minutes": round(elapsed / 60, 1),
        "samples": samples,
        "steps": max_steps,
        "loss": loss,
    }
    (ROOT / capsule_id / "result.json").write_text(
        json.dumps(info, indent=2, default=str))

    del model, trainer
    torch.cuda.empty_cache()
    logger.info(f"GPU freed — next capsule")
    return info


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SEQUENTIAL CAPSULE TRAINER — RTX 5090")
    logger.info("=" * 60)

    results = []
    for cid in CAPSULES_TO_TRAIN:
        try:
            r = train_one(cid)
            if r:
                results.append(r)
        except Exception as e:
            logger.error(f"FAILED {cid}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"capsule": cid, "status": "failed", "error": str(e)})

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    for r in results:
        s = "OK" if r.get("status") == "completed" else "FAIL"
        logger.info(f"  {s} {r.get('capsule')}: loss={r.get('loss','?')} time={r.get('minutes','?')}min")
    logger.info("=" * 60)
