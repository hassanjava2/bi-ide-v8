#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE – دمج LoRA وتحويل نموذج المحادثة إلى GGUF (Q4_K_M)
1. يدمج أوزان LoRA مع النموذج الأصلي
2. يحوّل إلى GGUF باستخدام convert_hf_to_gguf.py من llama.cpp
3. يكمّم إلى Q4_K_M إن وُجد تنفيذي quantize

الاستخدام:
    python training/convert-to-gguf.py
    python training/convert-to-gguf.py --merged-dir models/finetuned-chat-merged --out models/bi-chat-gguf
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FINETUNED_CHAT = BASE_DIR / "models" / "finetuned-chat"
MERGED_DIR = BASE_DIR / "models" / "finetuned-chat-merged"
OUTPUT_DIR = BASE_DIR / "models" / "bi-chat-gguf"
CONVERT_SCRIPT = BASE_DIR / "training" / "convert_hf_to_gguf.py"
GGUF_SCRIPT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"


def merge_lora():
    """دمج أوزان LoRA مع النموذج الأصلي وحفظ النموذج المدمج."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    if not FINETUNED_CHAT.exists() or not (FINETUNED_CHAT / "adapter_config.json").exists():
        print(f"❌ مجلد LoRA غير موجود: {FINETUNED_CHAT}")
        print("   شغّل أولاً: python training/finetune-chat.py")
        sys.exit(1)

    print("🔄 تحميل النموذج الأساسي + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(FINETUNED_CHAT))
    print("🔄 دمج LoRA مع النموذج الأساسي...")
    model = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"💾 حفظ النموذج المدمج في {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print("✅ تم الدمج والحفظ.")
    return MERGED_DIR


def ensure_convert_script():
    if CONVERT_SCRIPT.exists():
        return str(CONVERT_SCRIPT)
    print("📥 تنزيل convert_hf_to_gguf.py من llama.cpp...")
    try:
        urllib.request.urlretrieve(GGUF_SCRIPT_URL, CONVERT_SCRIPT)
        return str(CONVERT_SCRIPT)
    except Exception as e:
        print(f"❌ فشل التنزيل: {e}")
        print("   حمّل يدوياً من: https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py")
        sys.exit(1)


def run_convert_to_gguf(merged_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = out_dir / "bi-chat-3b-f16.gguf"
    script = ensure_convert_script()
    cmd = [sys.executable, script, str(merged_dir), "--outfile", str(gguf_path), "--outtype", "f16"]
    print(f"🔄 تشغيل تحويل GGUF: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(BASE_DIR))
    if r.returncode != 0:
        print("❌ فشل تحويل GGUF. تأكد من تثبيت gguf: pip install gguf")
        sys.exit(1)
    print(f"✅ تم إنشاء: {gguf_path}")
    return gguf_path


def run_quantize(gguf_f16_path, out_dir):
    out_dir = Path(out_dir)
    q4_path = out_dir / "bi-chat-3b-q4.gguf"
    quantize_bin = shutil.which("quantize") or shutil.which("llama-quantize")
    if not quantize_bin:
        print("⚠️ تنفيذي quantize غير موجود في PATH.")
        print("   لإنشاء Q4_K_M: ثبّت llama.cpp ثم شغّل:")
        print(f"   quantize {gguf_f16_path} {q4_path} Q4_K_M")
        return None
    cmd = [quantize_bin, str(gguf_f16_path), str(q4_path), "Q4_K_M"]
    print(f"🔄 تشغيل التكميم: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("❌ فشل التكميم.")
        return None
    print(f"✅ تم إنشاء: {q4_path}")
    return q4_path


def main():
    parser = argparse.ArgumentParser(description="دمج LoRA وتحويل إلى GGUF")
    parser.add_argument("--skip-merge", action="store_true", help="تخطي الدمج (استخدام مجلد merged موجود)")
    parser.add_argument("--merged-dir", type=str, default=str(MERGED_DIR))
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    print("=" * 50)
    print("Bi IDE – تحويل نموذج المحادثة إلى GGUF")
    print("=" * 50)

    if args.skip_merge and Path(args.merged_dir).exists():
        merged_dir = Path(args.merged_dir)
        print(f"استخدام مجلد مدمج موجود: {merged_dir}")
    else:
        merged_dir = merge_lora()

    gguf_f16 = run_convert_to_gguf(merged_dir, args.out)
    run_quantize(gguf_f16, args.out)
    print("\n✅ انتهى. انسخ ملف bi-chat-3b-q4.gguf (أو f16) إلى models/ واستخدمه في Bi IDE.")


if __name__ == "__main__":
    main()
