"""
Bi IDE – تدريب تلقائي بالخلفية
Auto Fine-tuning – يشتغل تلقائياً عندما الكمبيوتر فاضي

يجمع كل البيانات المتعلمة من:
- LearningCore (auto-training-data.json)
- ErrorLearner (error-training-data.json)
- ProjectLearner (training/output/auto_learned_training.json)
- كل ملفات training/output/*.json

الاستخدام:
  python training/auto-finetune.py              # تدريب فوري
  python training/auto-finetune.py --watch      # مراقبة + تدريب عند idle
  python training/auto-finetune.py --check      # فحص البيانات المتاحة فقط
"""

import json
import os
import sys
import time
import glob
import psutil
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training" / "output"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = MODELS_DIR / "finetuned"
MERGED_DIR = MODELS_DIR / "merged"
ONNX_DIR = MODELS_DIR / "bi-ai-onnx"
STATE_FILE = BASE_DIR / "data" / "learning" / "auto-finetune-state.json"

# إعدادات التدريب
MODEL_NAME = "Qwen/Qwen2-0.5B"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 5              # أقل من finetune-extended (تدريب سريع)
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
MIN_DATA_COUNT = 200     # أقل عدد بيانات للبدء
MIN_IDLE_CPU = 40        # أقصى CPU% للبدء
MIN_HOURS_BETWEEN = 12   # أقل فترة بين تدريبين

def load_state():
    """تحميل حالة التدريب السابقة"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_training": None, "total_trainings": 0, "total_samples": 0}

def save_state(state):
    """حفظ حالة التدريب"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def collect_all_data():
    """جمع كل بيانات التدريب من كل المصادر"""
    all_data = []
    sources = {}
    
    # 1. ملفات training/output/*.json
    if TRAINING_DIR.exists():
        for json_file in TRAINING_DIR.glob("*.json"):
            if json_file.name == "training_report.json":
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    valid = [d for d in data if d.get('instruction') and d.get('output')]
                    all_data.extend(valid)
                    sources[json_file.name] = len(valid)
            except:
                pass
    
    # 2. قاعدة المعرفة
    kb_files = ['auto-training-data.json', 'error-training-data.json', 'auto-learning-data.json']
    for kb_file in kb_files:
        kb_path = KNOWLEDGE_DIR / kb_file
        if kb_path.exists():
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    valid = [d for d in data if d.get('instruction') and d.get('output')]
                    all_data.extend(valid)
                    sources[kb_file] = len(valid)
            except:
                pass
    
    # 3. RAG knowledge base
    rag_file = KNOWLEDGE_DIR / "rag-knowledge-base.json"
    if rag_file.exists():
        try:
            with open(rag_file, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            for doc in docs:
                text = doc.get('text', '')
                answer = doc.get('answer', '')
                if text and answer:
                    all_data.append({
                        'instruction': text[:200],
                        'output': answer[:500]
                    })
            sources['rag-knowledge-base.json'] = len(docs)
        except:
            pass
    
    # إزالة التكرار
    seen = set()
    unique_data = []
    for item in all_data:
        key = f"{item.get('instruction', '')[:50]}|{item.get('output', '')[:50]}"
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    return unique_data, sources

def check_conditions(state):
    """فحص شروط التدريب"""
    issues = []
    
    # فحص CPU
    cpu = psutil.cpu_percent(interval=2)
    if cpu > MIN_IDLE_CPU:
        issues.append(f"CPU عالي: {cpu}% (الحد: {MIN_IDLE_CPU}%)")
    
    # فحص آخر تدريب
    if state.get('last_training'):
        last = datetime.fromisoformat(state['last_training'])
        hours_since = (datetime.now() - last).total_seconds() / 3600
        if hours_since < MIN_HOURS_BETWEEN:
            issues.append(f"آخر تدريب قبل {hours_since:.1f} ساعة (الحد: {MIN_HOURS_BETWEEN})")
    
    # فحص البيانات
    data, sources = collect_all_data()
    if len(data) < MIN_DATA_COUNT:
        issues.append(f"بيانات قليلة: {len(data)} (الحد: {MIN_DATA_COUNT})")
    
    return issues, data, sources

def format_data_for_training(data):
    """تحويل البيانات لصيغة التدريب"""
    formatted = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        formatted.append({"text": text})
    
    return formatted

def train(data):
    """تنفيذ التدريب"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"❌ مكتبات مفقودة: {e}")
        print("   pip install torch transformers peft datasets")
        return False
    
    print(f"\n📦 تحميل النموذج: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # تحضير البيانات
    formatted = format_data_for_training(data)
    dataset = Dataset.from_list(formatted)
    
    def tokenize(example):
        result = tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = dataset.map(tokenize, remove_columns=["text"])
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # إعدادات التدريب
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    
    print(f"\n🚀 بدء التدريب: {len(split['train'])} عينة تدريب, {len(split['test'])} عينة اختبار")
    trainer.train()
    
    # حفظ
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✅ تم حفظ النموذج في: {OUTPUT_DIR}")
    
    return True

def merge_and_convert():
    """دمج LoRA وتحويل لـ ONNX"""
    print("\n🔗 دمج LoRA مع النموذج الأساسي...")
    
    try:
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
        model = model.merge_and_unload()
        
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"✅ تم الدمج في: {MERGED_DIR}")
        
        # تحويل ONNX
        print("\n📦 تحويل لـ ONNX...")
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            ort_model = ORTModelForCausalLM.from_pretrained(str(MERGED_DIR), export=True)
            ONNX_DIR.mkdir(parents=True, exist_ok=True)
            ort_model.save_pretrained(str(ONNX_DIR))
            tokenizer.save_pretrained(str(ONNX_DIR))
            print(f"✅ تم التحويل لـ ONNX في: {ONNX_DIR}")
        except Exception as e:
            print(f"⚠️ فشل تحويل ONNX: {e}")
            print("   يمكنك تحويله يدوياً: python training/convert-to-onnx.py")
        
        return True
    except Exception as e:
        print(f"❌ فشل الدمج: {e}")
        return False

def main():
    args = sys.argv[1:]
    state = load_state()
    
    # فحص فقط
    if '--check' in args:
        data, sources = collect_all_data()
        print(f"\n📊 بيانات التدريب المتاحة: {len(data)} عينة")
        print("\nالمصادر:")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  📄 {source}: {count}")
        print(f"\nآخر تدريب: {state.get('last_training', 'لم يتم بعد')}")
        print(f"إجمالي التدريبات: {state.get('total_trainings', 0)}")
        return
    
    # وضع المراقبة
    if '--watch' in args:
        print("👁️ وضع المراقبة - يراقب ويتدرب عند idle...")
        while True:
            issues, data, sources = check_conditions(state)
            if issues:
                print(f"⏳ [{datetime.now().strftime('%H:%M')}] لا يمكن التدريب:")
                for issue in issues:
                    print(f"   - {issue}")
                time.sleep(300)  # فحص كل 5 دقائق
                continue
            
            print(f"\n✅ الشروط متحققة! بدء التدريب ({len(data)} عينة)...")
            success = train(data)
            if success:
                merge_and_convert()
                state['last_training'] = datetime.now().isoformat()
                state['total_trainings'] = state.get('total_trainings', 0) + 1
                state['total_samples'] = len(data)
                save_state(state)
            
            time.sleep(3600)  # انتظار ساعة بعد التدريب
        return
    
    # تدريب فوري
    print("🚀 تدريب فوري...")
    data, sources = collect_all_data()
    print(f"📊 بيانات: {len(data)} عينة من {len(sources)} مصدر")
    
    if len(data) < 10:
        print("❌ بيانات قليلة جداً! أضف مواضيع للتعلم أولاً.")
        return
    
    for source, count in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"  📄 {source}: {count}")
    
    success = train(data)
    if success:
        merge_and_convert()
        state['last_training'] = datetime.now().isoformat()
        state['total_trainings'] = state.get('total_trainings', 0) + 1
        state['total_samples'] = len(data)
        save_state(state)
        print("\n🎉 التدريب اكتمل بنجاح!")

if __name__ == "__main__":
    main()
