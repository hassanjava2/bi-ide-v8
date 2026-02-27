"""
Bi IDE – تدريب مكثف (10 ساعات)
Extended Fine-tuning – يستغل كل بياناتك لأقصى حد

المتطلبات: نفس finetune.py
الاستخدام: python training/finetune-extended.py

RTX 3060 × 10 ساعات = تدريب عميق جداً
"""

import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training" / "output"
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge" / "rag-knowledge-base.json"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned-extended"

# ══════════════════════════════════════
# الإعدادات – محسّنة لـ 10 ساعات
# ══════════════════════════════════════
MODEL_NAME = "Qwen/Qwen2-0.5B"
MAX_LENGTH = 768          # أطول من السابق
BATCH_SIZE = 2            # أصغر = ذاكرة أقل = تعلم أدق
GRAD_ACCUM = 8            # تعويض الـ batch الصغير
EPOCHS = 15               # أكثر بكثير
LEARNING_RATE = 1e-4      # أبطأ = تعلم أدق
WARMUP_RATIO = 0.05
LORA_R = 32               # أكبر = قدرة تعلم أعلى
LORA_ALPHA = 64
SAVE_EVERY_HOURS = 2      # حفظ كل ساعتين

def load_all_data():
    """تحميل وتوسيع كل البيانات"""
    all_data = []
    
    # 1. قاعدة المعرفة
    if KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        for doc in docs:
            text = doc.get('text', '')
            answer = doc.get('answer', '')
            if text and answer:
                # النسخة الأصلية
                all_data.append({"instruction": text, "output": answer})
                # نسخة معكوسة (الجواب كسياق والسؤال كمخرج)
                all_data.append({"instruction": f"لخّص المفهوم التالي:\n{answer[:300]}", "output": text})
        print(f"  📚 قاعدة معرفة: {len(docs)} → {len(all_data)} (مع توسيع)")
    
    # 2. كل ملفات التدريب
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
                            all_data.append({"instruction": str(inp)[:MAX_LENGTH], "output": str(out)[:MAX_LENGTH]})
                            count += 1
                    if count > 0:
                        print(f"  📄 {f.name}: {count}")
            except:
                pass
    
    # 3. توليد بيانات إضافية (Data Augmentation)
    augmented = augment_data(all_data)
    all_data.extend(augmented)
    
    print(f"\n📊 المجموع الكلي: {len(all_data)} عينة")
    return all_data

def augment_data(data):
    """توسيع البيانات – توليد عينات إضافية"""
    augmented = []
    print("\n🔄 توسيع البيانات (Data Augmentation)...")
    
    for item in data:
        inp = item['instruction']
        out = item['output']
        
        # 1. سؤال "لماذا" 
        if len(out) > 100:
            augmented.append({
                "instruction": f"لماذا نستخدم هذا الأسلوب:\n{out[:200]}",
                "output": f"نستخدم هذا الأسلوب لأنه:\n{inp}\n\nالتفاصيل:\n{out[:300]}"
            })
        
        # 2. سؤال "ما هي البدائل"
        if 'function' in out.lower() or 'def ' in out.lower() or 'class' in out.lower():
            augmented.append({
                "instruction": f"ما هي أفضل الممارسات لـ: {inp[:100]}",
                "output": f"أفضل الممارسات:\n1. {out[:200]}\n2. تأكد من معالجة الأخطاء\n3. أضف توثيقاً واضحاً\n4. اكتب اختبارات"
            })
        
        # 3. سؤال "اشرح الكود"
        if '```' in out or 'function' in out or 'const ' in out:
            augmented.append({
                "instruction": f"اشرح هذا الكود بالتفصيل:\n{out[:300]}",
                "output": f"شرح الكود:\n\nهذا الكود يقوم بـ: {inp}\n\nالتفاصيل:\n{out[:400]}"
            })
    
    # حد أقصى للتوسيع
    if len(augmented) > len(data):
        augmented = random.sample(augmented, len(data))
    
    print(f"  ✅ {len(augmented)} عينة مُولّدة")
    return augmented

def format_conversations(data):
    """تحويل لصيغة محادثة"""
    formatted = []
    for item in data:
        # صيغة 1: سؤال وجواب
        formatted.append({"text": f"### سؤال:\n{item['instruction']}\n\n### جواب:\n{item['output']}"})
        
        # صيغة 2: تعليمة ومخرج (لبعض العينات)
        if random.random() < 0.3:
            formatted.append({"text": f"[INST] {item['instruction']} [/INST]\n{item['output']}"})
    
    random.shuffle(formatted)
    return formatted

def train():
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n🖥️  الجهاز: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {vram:.1f} GB")
    
    start_time = time.time()
    end_time = start_time + (10 * 3600)  # 10 ساعات
    print(f"   ⏰ البدء: {datetime.now().strftime('%H:%M')}")
    print(f"   ⏰ النهاية المتوقعة: {(datetime.now() + timedelta(hours=10)).strftime('%H:%M')}")
    
    # 1. البيانات
    print("\n📦 تحميل وتوسيع البيانات...")
    raw_data = load_all_data()
    formatted = format_conversations(raw_data)
    dataset = Dataset.from_list(formatted)
    
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split['train']
    eval_ds = split['test']
    print(f"   Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    
    # 2. النموذج – بدون device_map="auto" لضمان حفظ الأوزان بشكل صحيح
    print(f"\n🤖 تحميل: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    # نقل النموذج للـ GPU يدوياً
    if device == "cuda":
        model = model.to(device)
    
    # 3. LoRA أكبر
    print("🔧 إعداد LoRA (extended)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")
    
    # 4. Tokenize
    print("🔤 Tokenizing...")
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
    
    # 5. Callback لحفظ الأوزان بشكل صحيح مع كل checkpoint
    class SavePeftModelCallback(TrainerCallback):
        """يضمن حفظ أوزان LoRA مع كل checkpoint"""
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(checkpoint_dir):
                # حفظ أوزان LoRA بصيغة safetensors
                kwargs['model'].save_pretrained(checkpoint_dir, safe_serialization=True)
                print(f"\n   💾 تم حفظ أوزان LoRA في: checkpoint-{state.global_step}")
            return control
    
    # 6. تدريب
    output = str(OUTPUT_DIR)
    os.makedirs(output, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        fp16=device == "cuda",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback()],
    )
    
    print(f"\n{'='*60}")
    print(f"🚀 تدريب مكثف – {EPOCHS} epochs")
    print(f"   Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"   LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"   Max length: {MAX_LENGTH}")
    print(f"   LR: {LEARNING_RATE} (cosine)")
    print(f"{'='*60}\n")
    
    # البحث عن checkpoint سابق صالح للاستكمال
    last_checkpoint = None
    if os.path.isdir(output):
        checkpoints = []
        for d in os.listdir(output):
            cp_path = os.path.join(output, d)
            if d.startswith("checkpoint-") and os.path.isdir(cp_path):
                # تحقق أن الـ checkpoint يحتوي أوزان فعلية
                has_weights = any(
                    f.endswith('.safetensors') or f.endswith('.bin')
                    for f in os.listdir(cp_path)
                )
                if has_weights:
                    checkpoints.append(cp_path)
                else:
                    print(f"   ⚠️ {d} بدون أوزان - يتم تجاوزه")
        
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"\n🔄 استكمال التدريب من: {os.path.basename(last_checkpoint)}")
            print(f"   (الخطوات المتبقية ستُكمَل تلقائياً)")
    
    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if any(
            d.startswith("checkpoint-") 
            for d in os.listdir(output) 
            if os.path.isdir(os.path.join(output, d))
        ):
            print("\n⚠️  وُجدت checkpoints قديمة بدون أوزان - يتم البدء من جديد")
            print("   (هالمرة الأوزان راح تنحفظ بشكل صحيح مع كل checkpoint)")
        trainer.train()
    
    # 7. حفظ النهائي
    elapsed = (time.time() - start_time) / 3600
    print(f"\n💾 حفظ النموذج النهائي... (بعد {elapsed:.1f} ساعة)")
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    
    # تحقق من الحفظ
    saved_files = [f for f in os.listdir(output) if f.endswith('.safetensors')]
    if saved_files:
        print(f"   ✅ تم الحفظ بنجاح: {saved_files}")
    else:
        # حاول الحفظ بصيغة bin كاحتياط
        print("   ⚠️ safetensors لم تنحفظ، نحاول bin...")
        model.save_pretrained(output, safe_serialization=False)
        saved_bin = [f for f in os.listdir(output) if f.endswith('.bin')]
        if saved_bin:
            print(f"   ✅ تم الحفظ: {saved_bin}")
        else:
            print("   ❌ فشل الحفظ! حاول يدوياً")
    
    # 8. تقرير
    print(f"\n{'='*60}")
    print(f"✅ تم التدريب المكثف!")
    print(f"   📁 {output}")
    print(f"   📊 عينات: {len(train_ds)}")
    print(f"   🔄 Epochs: {EPOCHS}")
    print(f"   ⏱️  المدة: {elapsed:.1f} ساعة")
    print(f"   🧠 LoRA: r={LORA_R}, target=7 layers")
    print(f"{'='*60}")
    print(f"\n💡 بعد الانتهاء شغّل: python training/convert-to-onnx.py")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Bi IDE – تدريب مكثف (Extended Fine-tuning)")
    print("   مصمم لـ 10 ساعات متواصلة")
    print("=" * 60)
    
    # فحص المتطلبات
    for lib in ['torch', 'transformers', 'datasets', 'peft', 'accelerate']:
        try:
            __import__(lib)
        except ImportError:
            print(f"❌ ناقص: {lib}")
            sys.exit(1)
    
    import torch
    print(f"\n✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    train()
