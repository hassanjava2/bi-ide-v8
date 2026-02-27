"""
Bi IDE – تحويل النموذج المُدرَّب لـ ONNX
الخطوة 1: دمج LoRA مع النموذج الأصلي
الخطوة 2: تحويل لـ ONNX

الاستخدام:
    python training/convert-to-onnx.py
"""

import os
import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
FINETUNED_DIR = BASE_DIR / "models" / "finetuned"
FINETUNED_EXT_DIR = BASE_DIR / "models" / "finetuned-extended"
MERGED_DIR = BASE_DIR / "models" / "merged"
ONNX_OUTPUT = BASE_DIR / "models" / "bi-ai-onnx"

def convert():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # البحث عن أحدث نموذج مُدرَّب (الممتد أولاً)
    global FINETUNED_DIR
    if FINETUNED_EXT_DIR.exists() and any(FINETUNED_EXT_DIR.glob("*.safetensors")):
        FINETUNED_DIR = FINETUNED_EXT_DIR
        print(f"📂 استخدام النموذج الممتد: {FINETUNED_DIR}")
    elif FINETUNED_DIR.exists() and (
        any(FINETUNED_DIR.glob("*.safetensors")) or any(FINETUNED_DIR.glob("*.bin"))
    ):
        print(f"📂 استخدام النموذج الأساسي: {FINETUNED_DIR}")
    else:
        print(f"❌ لا يوجد نموذج مُدرَّب!")
        print(f"   تحقق من: {FINETUNED_DIR}")
        print(f"   أو: {FINETUNED_EXT_DIR}")
        sys.exit(1)

    # ══════════════════════════════════════
    # الخطوة 1: دمج LoRA
    # ══════════════════════════════════════
    print("🔧 الخطوة 1: دمج LoRA مع النموذج الأصلي...")
    
    # تحميل النموذج الأصلي
    config_path = FINETUNED_DIR / "adapter_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B")
    else:
        base_model_name = "Qwen/Qwen2-0.5B"
    
    print(f"   Base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # دمج LoRA
    model = PeftModel.from_pretrained(base_model, str(FINETUNED_DIR))
    model = model.merge_and_unload()
    print("   ✅ LoRA merged")
    
    # حفظ النموذج المدمج
    merged = str(MERGED_DIR)
    os.makedirs(merged, exist_ok=True)
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    print(f"   ✅ Saved to {merged}")
    
    # ══════════════════════════════════════
    # الخطوة 2: تحويل ONNX
    # ══════════════════════════════════════
    print("\n🔄 الخطوة 2: تحويل لـ ONNX...")
    
    output = str(ONNX_OUTPUT)
    os.makedirs(output, exist_ok=True)
    
    try:
        # محاولة optimum
        from optimum.onnxruntime import ORTModelForCausalLM
        
        ort_model = ORTModelForCausalLM.from_pretrained(merged, export=True)
        ort_model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print("   ✅ ONNX exported (optimum)")
        
    except Exception as e1:
        print(f"   ⚠️ optimum فشل: {e1}")
        print("   🔄 جاري المحاولة بـ CLI...")
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "optimum.exporters.onnx",
             "--model", merged,
             "--task", "text-generation",
             output],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            tokenizer.save_pretrained(output)
            print("   ✅ ONNX exported (CLI)")
        else:
            print(f"   ⚠️ CLI فشل: {result.stderr[:200]}")
            print("\n💡 النموذج المدمج جاهز في:", merged)
            print("   يمكنك تحويله يدوياً لاحقاً بـ:")
            print(f"   optimum-cli export onnx --model {merged} --task text-generation {output}")
            
            # على الأقل انسخ المدمج كنتيجة
            for f in Path(merged).glob("*"):
                shutil.copy2(f, output)
            tokenizer.save_pretrained(output)
            print(f"\n✅ النموذج المدمج (بدون ONNX) محفوظ في: {output}")
            return
    
    # النتيجة + إصدار (model versioning)
    total = sum(f.stat().st_size for f in Path(output).rglob('*') if f.is_file())
    print(f"\n✅ تم التحويل بنجاح!")
    print(f"   📁 {output}")
    print(f"   💾 {total / 1024 / 1024:.1f} MB")

    registry_path = BASE_DIR / "models" / "model-registry.json"
    try:
        import json
        from datetime import datetime
        registry = {'versions': [], 'current': None}
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        version = len(registry.get('versions', [])) + 1
        version_dir = ONNX_OUTPUT / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        for f in Path(output).rglob('*'):
            if f.is_file():
                rel = f.relative_to(Path(output))
                dest = version_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        registry.setdefault('versions', []).append({
            'version': version,
            'path': str(version_dir),
            'timestamp': datetime.now().isoformat(),
            'size_mb': round(total / 1024 / 1024, 1)
        })
        registry['current'] = version
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        print(f"   📌 Version v{version} saved; registry updated.")
    except Exception as e:
        print(f"   ⚠️ Versioning: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("🔄 Bi IDE – ONNX Conversion")
    print("=" * 50)
    convert()
