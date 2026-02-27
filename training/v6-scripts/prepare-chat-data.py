#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - تحضير بيانات المحادثة (ChatML)
يحوّل بيانات instruction/output إلى صيغة محادثة متعددة الأدوار لتدريب نموذج المحادثة.

الاستخدام:
    python training/prepare-chat-data.py
    python training/prepare-chat-data.py --max-length 1024 --out training/output/chat_training_data.json
"""

import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training" / "output"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
OUTPUT_FILE = TRAINING_DIR / "chat_training_data.json"
SKIP_FILES = {
    "all_training_data.json", "validated_training_data.json",
    "validation_report.json", "training_report.json", "cleanup_summary.json",
    "chat_training_data.json"
}
MAX_CONTENT_LENGTH = 2048  # مناسب لـ Qwen2.5-3B


def get_instruction(item):
    return (
        item.get("instruction")
        or item.get("input")
        or item.get("question")
        or item.get("prompt")
        or ""
    )


def get_output(item):
    return (
        item.get("output")
        or item.get("answer")
        or item.get("response")
        or item.get("completion")
        or ""
    )


def load_rag_knowledge():
    """تحميل قاعدة المعرفة RAG"""
    out = []
    rag_file = KNOWLEDGE_DIR / "rag-knowledge-base.json"
    if not rag_file.exists():
        return out
    try:
        with open(rag_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for doc in docs:
            text = doc.get("text", "")
            answer = doc.get("answer", "")
            if text and answer:
                out.append({"instruction": text[:MAX_CONTENT_LENGTH], "output": answer[:MAX_CONTENT_LENGTH * 2]})
    except Exception as e:
        print(f"   ⚠️ rag-knowledge-base.json: {e}")
    return out


def load_training_jsons(max_length):
    """تحميل كل ملفات التدريب من training/output"""
    out = []
    if not TRAINING_DIR.exists():
        return out
    for path in sorted(TRAINING_DIR.glob("*.json")):
        if path.name in SKIP_FILES:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"   ⚠️ {path.name}: {e}")
            continue
        if isinstance(data, list):
            for item in data:
                inp = get_instruction(item)
                out_val = get_output(item)
                if inp and out_val:
                    out.append({
                        "instruction": str(inp)[:max_length],
                        "output": str(out_val)[:max_length * 2],
                    })
        elif isinstance(data, dict):
            if "samples" in data:
                for item in data["samples"]:
                    inp = get_instruction(item)
                    out_val = get_output(item)
                    if inp and out_val:
                        out.append({"instruction": str(inp)[:max_length], "output": str(out_val)[:max_length * 2]})
            elif "examples" in data:
                for item in data["examples"]:
                    inp = get_instruction(item)
                    out_val = get_output(item)
                    if inp and out_val:
                        out.append({"instruction": str(inp)[:max_length], "output": str(out_val)[:max_length * 2]})
    return out


def to_chatml_conversation(instruction, output):
    """تحويل زوج سؤال/جواب إلى محادثة ChatML (قائمة رسائل)."""
    return [
        {"role": "user", "content": instruction.strip()},
        {"role": "assistant", "content": output.strip()},
    ]


def main():
    parser = argparse.ArgumentParser(description="تحضير بيانات المحادثة ChatML")
    parser.add_argument("--max-length", type=int, default=1024, help="أقصى طول للمحتوى")
    parser.add_argument("--out", type=str, default=str(OUTPUT_FILE), help="ملف الإخراج")
    parser.add_argument("--no-followup", action="store_true", help="بدون إضافة جمل متابعة")
    args = parser.parse_args()
    out_path = Path(args.out)
    max_length = args.max_length

    print("=" * 50)
    print("Bi IDE – تحضير بيانات المحادثة (ChatML)")
    print("=" * 50)

    conversations = []

    # 1. قاعدة المعرفة
    print("\n📚 تحميل قاعدة المعرفة...")
    rag = load_rag_knowledge()
    for item in rag:
        conversations.append(to_chatml_conversation(item["instruction"], item["output"]))
    print(f"   ✅ {len(rag)} محادثة من RAG")

    # 2. ملفات التدريب
    print("\n📄 تحميل training/output/*.json...")
    raw = load_training_jsons(max_length)
    for item in raw:
        conversations.append(to_chatml_conversation(item["instruction"], item["output"]))
    print(f"   ✅ {len(raw)} محادثة من ملفات التدريب")

    # إزالة تكرارات تقريبية (نفس أول رسالة user)
    seen = set()
    unique = []
    for conv in conversations:
        key = conv[0]["content"][:200] if conv else ""
        if key not in seen:
            seen.add(key)
            unique.append(conv)
    if len(unique) < len(conversations):
        print(f"   📌 إزالة تكرارات: {len(conversations) - len(unique)}")

    conversations = unique
    print(f"\n📊 المجموع: {len(conversations)} محادثة (ChatML)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"\n✅ تم الحفظ: {out_path}")
    print("   الخطوة التالية: python training/finetune-chat.py")


if __name__ == "__main__":
    main()
