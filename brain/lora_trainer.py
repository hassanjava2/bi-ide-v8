#!/usr/bin/env python3
"""
lora_trainer.py — تدريب LoRA/QLoRA للكبسولات
Full LoRA/QLoRA training pipeline with curriculum learning

الاستخدام:
  # QLoRA على RTX 5090 (محلي)
  python lora_trainer.py --data training_data/ --model Qwen/Qwen2.5-7B-Instruct --qlora

  # Full LoRA على Vast.ai 4x B200
  python lora_trainer.py --data training_data/ --model meta-llama/Llama-3-70B-Instruct --lora --multi-gpu

  # تدريب كبسولة محددة
  python lora_trainer.py --data training_data/ --capsule hacking --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_seq_length": 4096,

    # LoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # QLoRA
    "use_qlora": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",

    # Training
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",
    "max_grad_norm": 1.0,

    # Curriculum Learning
    "curriculum_enabled": True,
    "curriculum_tiers": 13,

    # Saving
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 5,

    # Output
    "output_dir": "./training_output",
    "logging_steps": 10,
}


# ═══════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════

def load_training_data(data_path: str, capsule_filter: Optional[str] = None,
                       curriculum_tier: Optional[int] = None):
    """Load unified JSONL training data with optional filters"""
    samples = []
    data_path = Path(data_path)

    files = list(data_path.glob("*.jsonl")) if data_path.is_dir() else [data_path]

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Filter by capsule
                if capsule_filter:
                    cap = item.get("capsule", "")
                    if not cap.startswith(capsule_filter):
                        continue

                # Filter by curriculum tier
                if curriculum_tier is not None:
                    if item.get("difficulty", 0) > curriculum_tier:
                        continue

                samples.append(item)

    print(f"📚 Loaded {len(samples):,} training samples")
    if capsule_filter:
        print(f"   Filtered by capsule: {capsule_filter}")
    if curriculum_tier is not None:
        print(f"   Filtered by tier: <= {curriculum_tier}")

    return samples


def format_for_training(sample: dict, tokenizer=None) -> str:
    """Format a sample into chat template"""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Fallback: ChatML format
    return (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


# ═══════════════════════════════════════════════
# Training setup
# ═══════════════════════════════════════════════

def setup_model_and_tokenizer(config: dict):
    """Load model with LoRA/QLoRA configuration"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    model_name = config["model_name"]
    print(f"🧠 Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA config
    if config.get("use_qlora"):
        print("   Using QLoRA (4-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("   Using full precision LoRA")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def create_dataset(samples: list, tokenizer, max_length: int):
    """Create HuggingFace Dataset from samples"""
    from torch.utils.data import Dataset

    class TrainingDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            text = format_for_training(self.samples[idx], self.tokenizer)
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

    return TrainingDataset(samples, tokenizer, max_length)


# ═══════════════════════════════════════════════
# Curriculum training loop
# ═══════════════════════════════════════════════

def train_with_curriculum(model, tokenizer, all_samples: list, config: dict):
    """Train with curriculum learning — easy → hard"""
    from transformers import TrainingArguments, Trainer
    import torch

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.get("curriculum_enabled"):
        # Train everything at once
        print("\n🔄 Training all data at once (no curriculum)")
        dataset = create_dataset(all_samples, tokenizer, config["max_seq_length"])
        _run_training(model, tokenizer, dataset, config, str(output_dir / "all"))
        return

    # Group by difficulty
    tiers = {}
    for s in all_samples:
        d = s.get("difficulty", 5)
        tiers.setdefault(d, []).append(s)

    print(f"\n📚 Curriculum Learning — {len(tiers)} tiers:")
    for t in sorted(tiers):
        print(f"   Tier {t:2d}: {len(tiers[t]):>8,} samples")

    # Train tier by tier
    for tier in sorted(tiers.keys()):
        tier_samples = tiers[tier]
        if not tier_samples:
            continue

        print(f"\n{'='*50}")
        print(f"🎓 Tier {tier}: {len(tier_samples):,} samples")
        print(f"{'='*50}")

        dataset = create_dataset(tier_samples, tokenizer, config["max_seq_length"])
        tier_dir = str(output_dir / f"tier_{tier:02d}")
        _run_training(model, tokenizer, dataset, config, tier_dir)

    # Final merged checkpoint
    print(f"\n✅ Curriculum training complete!")
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"📁 Final model saved to: {output_dir / 'final'}")


def _run_training(model, tokenizer, dataset, config, output_dir):
    """Run training with given dataset"""
    from transformers import TrainingArguments, Trainer
    import torch

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler"],
        max_grad_norm=config["max_grad_norm"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_steps=config["logging_steps"],
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.get("use_qlora") else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)


# ═══════════════════════════════════════════════
# Export to GGUF
# ═══════════════════════════════════════════════

def export_gguf(model_dir: str, output_path: str, quant: str = "q4_k_m"):
    """Export trained LoRA model to GGUF for Ollama deployment"""
    import subprocess

    print(f"\n📦 Exporting to GGUF ({quant})...")
    print(f"   Model: {model_dir}")
    print(f"   Output: {output_path}")

    # Merge LoRA
    merge_cmd = [
        "python3", "-m", "peft.merge_and_unload",
        "--model_name_or_path", model_dir,
        "--output_dir", f"{model_dir}_merged",
    ]

    # Convert to GGUF
    convert_cmd = [
        "python3", "llama.cpp/convert_hf_to_gguf.py",
        f"{model_dir}_merged",
        "--outfile", output_path,
        "--outtype", quant,
    ]

    try:
        subprocess.run(merge_cmd, check=True)
        subprocess.run(convert_cmd, check=True)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ GGUF exported: {output_path} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"❌ GGUF export failed: {e}")
        print("   Install llama.cpp and try again")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BI-IDE LoRA/QLoRA Trainer")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL dir or file)")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"], help="Base model name/path")
    parser.add_argument("--output", default="./training_output", help="Output directory")
    parser.add_argument("--capsule", default=None, help="Filter by capsule (e.g. 'hacking', 'software')")
    parser.add_argument("--tier", type=int, default=None, help="Max curriculum tier")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit)")
    parser.add_argument("--lora", action="store_true", help="Use full LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--export-gguf", default=None, help="Export to GGUF path after training")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU training")
    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "model_name": args.model,
        "output_dir": args.output,
        "use_qlora": args.qlora or (not args.lora),
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "curriculum_enabled": not args.no_curriculum,
    })

    print("🏛️ BI-IDE LoRA/QLoRA Trainer")
    print(f"   Model: {config['model_name']}")
    print(f"   Mode: {'QLoRA' if config['use_qlora'] else 'LoRA'}")
    print(f"   LoRA rank: {config['lora_r']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch: {config['batch_size']} × {config['gradient_accumulation']} = {config['batch_size'] * config['gradient_accumulation']}")
    print(f"   LR: {config['learning_rate']}")
    print(f"   Curriculum: {'ON' if config['curriculum_enabled'] else 'OFF'}")

    # Load data
    samples = load_training_data(args.data, args.capsule, args.tier)
    if not samples:
        print("❌ No training data found!")
        sys.exit(1)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Train
    train_with_curriculum(model, tokenizer, samples, config)

    # Export GGUF
    if args.export_gguf:
        final_dir = os.path.join(config["output_dir"], "final")
        export_gguf(final_dir, args.export_gguf)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
