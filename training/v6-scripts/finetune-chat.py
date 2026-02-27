#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE – تدريب نموذج المحادثة (Qwen2.5-3B-Instruct + LoRA)
يحمّل بيانات محادثة بصيغة ChatML ويدرّب النموذج للرد بجودة عالية.

المتطلبات:
    pip install torch transformers datasets peft accelerate
    python training/prepare-chat-data.py   # أولاً

الاستخدام:
    python training/finetune-chat.py
"""

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training" / "output"
CHAT_DATA_FILE = TRAINING_DIR / "chat_training_data.json"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned-chat"

# Qwen2.5-3B-Instruct – جودة محادثة عالية
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 1024
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
LORA_R = 16
LORA_ALPHA = 32
NUM_WORKERS = 6
CPU_THREADS = 8
GRADIENT_ACCUMULATION_STEPS = 8


def check_dependencies():
    missing = []
    for lib in ["torch", "transformers", "datasets", "peft", "accelerate"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"❌ مكتبات ناقصة: {', '.join(missing)}")
        print(f"   ثبّتها بـ: pip install {' '.join(missing)}")
        sys.exit(1)
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU فقط'}")


def load_chat_data():
    if not CHAT_DATA_FILE.exists():
        print(f"❌ ملف البيانات غير موجود: {CHAT_DATA_FILE}")
        print("   شغّل أولاً: python training/prepare-chat-data.py")
        sys.exit(1)
    with open(CHAT_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("❌ توقع قائمة محادثات في chat_training_data.json")
        sys.exit(1)
    return data


def train():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ الجهاز: {device}")
    print(f"   CPU threads: {CPU_THREADS} | Data workers: {NUM_WORKERS}")

    conversations = load_chat_data()
    print(f"\n📊 محادثات محمّلة: {len(conversations)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_conversation(conv):
        """تحويل محادثة إلى نص وتقطيع. Labels = نفس الـ input_ids مع -100 للـ padding."""
        if not conv or not isinstance(conv, list):
            return None
        text = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        out = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors=None,
        )
        ids = out["input_ids"]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        out["labels"] = [id if id != pad_id else -100 for id in ids]
        return out

    rows = []
    for conv in conversations:
        row = tokenize_conversation(conv)
        if row is not None:
            rows.append(row)

    if not rows:
        print("❌ لا توجد عينات صالحة بعد التقطيع.")
        sys.exit(1)

    dataset = Dataset.from_list(rows)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    print(f"   Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    print(f"\n🤖 تحميل النموذج: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=(device == "cuda"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        warmup_ratio=0.1,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print(f"\n🚀 بدء تدريب المحادثة ({EPOCHS} epochs, batch={BATCH_SIZE})...")
    print("=" * 50)
    trainer.train()

    print("\n💾 حفظ النموذج...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ تم التدريب بنجاح! النموذج: {OUTPUT_DIR}")
    print("   الخطوة التالية: python training/convert-to-gguf.py")


if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Bi IDE – تدريب نموذج المحادثة (Qwen2.5-3B-Instruct)")
    print("=" * 50)
    check_dependencies()
    print()
    train()
