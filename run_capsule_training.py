#!/usr/bin/env python3
"""
run_capsule_training.py — Train ALL capsules sequentially on RTX 5090

Auto-discovers capsules with data in capsules/*/data/*.jsonl
Uses PROVEN settings: batch=1, gradient_accum=16, bf16, gradient_checkpointing
"""
import sys, os, json, time, logging, torch, gc
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOG_PATH = Path("/tmp/capsule_training_all.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s",
    handlers=[logging.FileHandler(str(LOG_PATH)), logging.StreamHandler()])
logger = logging.getLogger("train")

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ROOT = Path("/home/bi/bi-ide-v8/capsules")


def normalize_data(capsule_dir):
    """Merge all JSONL files into one normalized merged_train.jsonl"""
    data_dir = capsule_dir / "data"
    if not data_dir.exists():
        return 0

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
                    except:
                        pass
    return count


def train_one(capsule_id, capsule_dir):
    """Train a single capsule"""
    data_path = capsule_dir / "data" / "merged_train.jsonl"
    model_dir = capsule_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        return None

    samples = sum(1 for _ in open(data_path))
    if samples < 30:
        logger.info(f"SKIP {capsule_id} — {samples} samples (need 30+)")
        return None

    max_steps = min(samples * 2, 2000)
    logger.info(f"{'='*50}")
    logger.info(f"CAPSULE: {capsule_id} | {samples} samples | {max_steps} steps")
    logger.info(f"{'='*50}")

    has_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if has_gpu else torch.float32,
        device_map="auto" if has_gpu else "cpu",
        trust_remote_code=True,
    )
    if has_gpu:
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} samples")

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples["input_text"])):
            inp = examples["input_text"][i]
            out = examples["output_text"][i]
            text = "<|im_start|>user\n" + str(inp) + "<|im_end|>\n<|im_start|>assistant\n" + str(out) + "<|im_end|>"
            texts.append(text)
        result = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=str(model_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16 if has_gpu else 4,
        learning_rate=5e-5,
        bf16=has_gpu,
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

    logger.info(f"START {capsule_id}")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    loss = result.metrics.get("train_loss", "?")

    trainer.save_model(str(model_dir))
    try:
        tokenizer.save_pretrained(str(model_dir))
    except:
        pass

    logger.info(f"DONE {capsule_id}: {elapsed/60:.1f}min loss={loss}")

    info = {
        "capsule": capsule_id, "status": "completed",
        "minutes": round(elapsed / 60, 1), "samples": samples,
        "steps": max_steps, "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    (capsule_dir / "result.json").write_text(json.dumps(info, indent=2, default=str))

    # Free GPU memory completely
    del model, trainer, tokenized, dataset
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()
    logger.info(f"GPU freed")
    return info


def main():
    logger.info("=" * 60)
    logger.info("MULTI-CAPSULE TRAINER — ALL CAPSULES")
    logger.info("=" * 60)

    # Auto-discover capsules with data
    trainable = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        count = normalize_data(d)
        if count >= 30:
            trainable.append((cid, d, count))
            logger.info(f"  ✅ {cid}: {count} samples")
        elif count > 0:
            logger.info(f"  ⏭️ {cid}: {count} samples (too few)")

    if not trainable:
        logger.error("No capsules with data! Run generate_capsule_data.py first.")
        sys.exit(1)

    logger.info(f"\n🎯 Training {len(trainable)} capsules\n")

    results = []
    for cid, cdir, cnt in trainable:
        try:
            r = train_one(cid, cdir)
            if r:
                results.append(r)
        except Exception as e:
            logger.error(f"FAILED {cid}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"capsule": cid, "status": "failed", "error": str(e)})

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    total_time = 0
    for r in results:
        s = "✅" if r.get("status") == "completed" else "❌"
        logger.info(f"  {s} {r.get('capsule')}: loss={r.get('loss','?')} time={r.get('minutes','?')}min")
        total_time += r.get("minutes", 0) or 0
    logger.info(f"\nTotal training time: {total_time:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
