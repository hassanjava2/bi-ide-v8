"""
Bi IDE – Fine-tuning Script
تدريب نموذج AI على بياناتك الخاصة

المتطلبات:
    pip install torch transformers datasets peft accelerate

الاستخدام:
    python training/finetune.py

RTX 3060 (6GB VRAM) → LoRA على نموذج صغير
"""

import json
import os
import sys
from pathlib import Path

# ══════════════════════════════════════
# الإعدادات
# ══════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training" / "output"
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge" / "rag-knowledge-base.json"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned"

# نموذج صغير - إعدادات محسّنة للسيرفر (8 cores, 31GB RAM)
# ملاحظة: نخليها قابلة للتعديل من البيئة حتى نقدر نطبّق خطة نبضات Hostinger (تقليل الضغط).
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2-0.5B")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))          # أكبر = أسرع
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))          # تحميل بيانات بالتوازي
CPU_THREADS = int(os.environ.get("CPU_THREADS", "8"))          # عدد خيوط المعالج

# خطة نبضات: إيقاف تدريجي بعد مدة محددة (دقائق) لحفظ الأداء على Hostinger
PULSE_MAX_MINUTES = int(os.environ.get("PULSE_MAX_MINUTES", "0"))

# تدريب عملاق: حد أقصى للخطوات (يُستخدم لتمديد التدريب بدل الاعتماد على epochs فقط)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))

def check_dependencies():
    """فحص المكتبات المطلوبة"""
    missing = []
    for lib in ['torch', 'transformers', 'datasets', 'peft', 'accelerate']:
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
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f} GB")

def load_training_data():
    """تحميل كل بيانات التدريب"""
    all_data = []
    
    # 1. قاعدة المعرفة
    if KNOWLEDGE_FILE.exists():
        print(f"📚 تحميل قاعدة المعرفة...")
        with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        for doc in docs:
            text = doc.get('text', '')
            answer = doc.get('answer', '')
            if text and answer:
                all_data.append({
                    "instruction": text,
                    "output": answer[:MAX_LENGTH * 2]
                })
        print(f"   ✅ {len(docs)} مستند")
    
    # 2. ملفات التدريب
    if TRAINING_DIR.exists():
        for f in sorted(TRAINING_DIR.glob("*.json")):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    content = json.load(fh)
                if isinstance(content, list):
                    count = 0
                    for item in content:
                        inp = item.get('instruction', item.get('input', item.get('question', '')))
                        out = item.get('output', item.get('answer', item.get('response', '')))
                        if inp and out:
                            all_data.append({
                                "instruction": str(inp)[:MAX_LENGTH],
                                "output": str(out)[:MAX_LENGTH * 2]
                            })
                            count += 1
                    if count > 0:
                        print(f"   📄 {f.name}: {count} عينة")
            except Exception as e:
                print(f"   ⚠️ {f.name}: {e}")
    
    print(f"\n📊 المجموع: {len(all_data)} عينة تدريب")
    return all_data

def format_for_training(data):
    """تحويل البيانات لصيغة التدريب"""
    formatted = []
    for item in data:
        text = f"### سؤال:\n{item['instruction']}\n\n### جواب:\n{item['output']}"
        formatted.append({"text": text})
    return formatted

def train():
    """التدريب الفعلي"""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from transformers import TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    # استغلال عدد خيوط محدد (مهم لتخفيف الضغط)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ الجهاز: {device}")
    print(f"   CPU threads: {CPU_THREADS} | Data workers: {NUM_WORKERS}")
    
    # 1. تحميل البيانات
    print("\n📦 تحميل البيانات...")
    raw_data = load_training_data()
    if len(raw_data) < 10:
        print("❌ بيانات قليلة جداً (أقل من 10). أضف المزيد.")
        return
    
    formatted = format_for_training(raw_data)
    dataset = Dataset.from_list(formatted)
    
    # تقسيم train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    # 2. تحميل النموذج
    print(f"\n🤖 تحميل النموذج: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 3. إعداد LoRA
    print("🔧 إعداد LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")
    
    # 4. Tokenize
    print("🔤 Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_tokenized = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # 5. تدريب
    output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    class StopAfterTimeCallback(TrainerCallback):
        def __init__(self, max_minutes):
            self.max_minutes = max_minutes
            self.start = None

        def on_train_begin(self, args, state, control, **kwargs):
            import time
            self.start = time.time()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if not self.max_minutes or self.max_minutes <= 0:
                return control
            import time
            elapsed_min = (time.time() - (self.start or time.time())) / 60
            if elapsed_min >= self.max_minutes:
                # إيقاف تدريجي + طلب حفظ checkpoint
                control.should_training_stop = True
                control.should_save = True
            return control
    
    # إذا MAX_STEPS مفعّل، نفضّل استراتيجية save/eval بالخطوات
    use_max_steps = isinstance(MAX_STEPS, int) and MAX_STEPS > 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS if not use_max_steps else 1,
        max_steps=MAX_STEPS if use_max_steps else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        fp16=device == "cuda",
        logging_steps=10,
        eval_strategy="steps" if use_max_steps else "epoch",
        eval_steps=EVAL_STEPS if use_max_steps else None,
        save_strategy="steps" if use_max_steps else "epoch",
        save_steps=SAVE_STEPS if use_max_steps else None,
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
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[StopAfterTimeCallback(PULSE_MAX_MINUTES)] if PULSE_MAX_MINUTES > 0 else None,
    )
    
    print(f"\n🚀 بدء التدريب ({EPOCHS} epochs, batch={BATCH_SIZE})...")
    print("=" * 50)
    
    trainer.train()
    
    # 6. حفظ
    print("\n💾 حفظ النموذج...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ تم التدريب بنجاح!")
    print(f"   📁 النموذج: {output_dir}")
    print(f"   📊 عينات: {len(train_dataset)}")
    print(f"   🔄 Epochs: {EPOCHS}")
    print(f"\n💡 الخطوة التالية: تحويل لـ ONNX لاستخدامه في Bi IDE")

if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Bi IDE – Fine-tuning")
    print("=" * 50)
    
    check_dependencies()
    
    print()
    train()
