#!/usr/bin/env python3
"""
Knowledge Scout — كشّاف المعرفة
Non-stop continuous downloader that fills /data with ALL human knowledge.
Runs 24/7 on RTX 5090 — never stops, never sleeps.

Categories:
  - Wikipedia (Arabic + English + other languages)
  - Scientific Papers (arXiv)
  - Programming (Stack Overflow, GitHub, documentation)
  - Books (OpenTextbook, Project Gutenberg)
  - Engineering & Science datasets
  - Arabic language & culture
  - Survival & Civilization rebuilding knowledge

Usage:
  nohup python3 knowledge_scout.py > /tmp/scout.log 2>&1 &
"""

import sys
import os
import json
import time
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# ─── Paths ────────────────────────────────────────────────────
DATA_ROOT = Path("/data")           # Second 4TB NVMe
SCOUT_STATE = DATA_ROOT / ".scout_state.json"

# ─── All Knowledge Sources ───────────────────────────────────
SOURCES = [
    # ════════ WIKIPEDIA ════════
    {
        "name": "wikipedia_ar",
        "category": "wikipedia",
        "type": "huggingface",
        "path": "wikimedia/wikipedia",
        "subset": "20231101.ar",
        "split": "train",
        "output": "wikipedia/ar.jsonl",
        "batch": 2000,
        "priority": 1,
        "desc": "Wikipedia Arabic — أساس المعرفة العربية",
    },
    {
        "name": "wikipedia_en",
        "category": "wikipedia",
        "type": "huggingface",
        "path": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "output": "wikipedia/en.jsonl",
        "batch": 5000,
        "priority": 2,
        "desc": "Wikipedia English — 6M+ articles",
    },
    {
        "name": "wikipedia_simple",
        "category": "wikipedia",
        "type": "huggingface",
        "path": "wikimedia/wikipedia",
        "subset": "20231101.simple",
        "split": "train",
        "output": "wikipedia/simple.jsonl",
        "batch": 2000,
        "priority": 3,
        "desc": "Simple English Wikipedia — easy to learn from",
    },
    # ════════ PROGRAMMING ════════
    {
        "name": "code_alpaca",
        "category": "programming",
        "type": "huggingface",
        "path": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "output": "datasets/code_alpaca.jsonl",
        "batch": 5000,
        "priority": 1,
        "desc": "Code instructions — 20K coding tasks",
    },
    {
        "name": "python_code",
        "category": "programming",
        "type": "huggingface",
        "path": "Nan-Do/code-search-net-python",
        "split": "train",
        "output": "datasets/python_code.jsonl",
        "batch": 5000,
        "priority": 2,
        "desc": "Python code search — functions + docs",
    },
    {
        "name": "stackoverflow",
        "category": "programming",
        "type": "huggingface",
        "path": "koutch/stackoverflow_python",
        "split": "train",
        "output": "datasets/stackoverflow_python.jsonl",
        "batch": 5000,
        "priority": 3,
        "desc": "Stack Overflow Python Q&A",
    },
    {
        "name": "openhermes",
        "category": "programming",
        "type": "huggingface",
        "path": "teknium/OpenHermes-2.5",
        "split": "train",
        "output": "datasets/openhermes.jsonl",
        "batch": 5000,
        "priority": 4,
        "desc": "OpenHermes — 1M high-quality instruction pairs",
    },
    # ════════ ARABIC ════════
    {
        "name": "arabic_alpaca",
        "category": "arabic",
        "type": "huggingface",
        "path": "FreedomIntelligence/alpaca-gpt4-arabic",
        "split": "train",
        "output": "datasets/arabic_alpaca.jsonl",
        "batch": 5000,
        "priority": 1,
        "desc": "Arabic instruction-following data",
    },
    {
        "name": "arabic_poems",
        "category": "arabic",
        "type": "huggingface",
        "path": "arbml/CIDAR",
        "split": "train",
        "output": "datasets/arabic_cidar.jsonl",
        "batch": 5000,
        "priority": 2,
        "desc": "Arabic cultural & instructional data",
    },
    # ════════ SCIENCE ════════
    {
        "name": "sciq",
        "category": "science",
        "type": "huggingface",
        "path": "allenai/sciq",
        "split": "train",
        "output": "datasets/sciq.jsonl",
        "batch": 5000,
        "priority": 1,
        "desc": "Science exam questions",
    },
    {
        "name": "math",
        "category": "science",
        "type": "huggingface",
        "path": "lighteval/MATH",
        "split": "train",
        "output": "datasets/math.jsonl",
        "batch": 5000,
        "priority": 2,
        "desc": "Mathematics problems + solutions",
    },
    # ════════ SURVIVAL & CIVILIZATION ════════
    {
        "name": "medical_qa",
        "category": "survival",
        "type": "huggingface",
        "path": "medmcqa/medmcqa",
        "split": "train",
        "output": "datasets/medical_qa.jsonl",
        "batch": 5000,
        "priority": 1,
        "desc": "Medical knowledge Q&A — essential for survival",
    },
    {
        "name": "wikibooks",
        "category": "survival",
        "type": "huggingface",
        "path": "wikimedia/wikisource",
        "subset": "20231201.ar",
        "split": "train",
        "output": "books/arabic_wikisource.jsonl",
        "batch": 2000,
        "priority": 2,
        "desc": "Arabic WikiSource — books & references",
    },
    # ════════ GENERAL KNOWLEDGE ════════
    {
        "name": "dolly",
        "category": "general",
        "type": "huggingface",
        "path": "databricks/databricks-dolly-15k",
        "split": "train",
        "output": "datasets/dolly.jsonl",
        "batch": 5000,
        "priority": 1,
        "desc": "Dolly — 15K diverse instruction-following",
    },
    {
        "name": "oasst",
        "category": "general",
        "type": "huggingface",
        "path": "OpenAssistant/oasst1",
        "split": "train",
        "output": "datasets/oasst.jsonl",
        "batch": 5000,
        "priority": 2,
        "desc": "OpenAssistant — human-generated conversations",
    },
    {
        "name": "slimorca",
        "category": "general",
        "type": "huggingface",
        "path": "Open-Orca/SlimOrca",
        "split": "train",
        "output": "datasets/slimorca.jsonl",
        "batch": 5000,
        "priority": 3,
        "desc": "SlimOrca — 500K high-quality GPT-4 conversations",
    },
]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state() -> dict:
    if SCOUT_STATE.exists():
        with SCOUT_STATE.open() as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with SCOUT_STATE.open("w") as f:
        json.dump(state, f, indent=2)


def convert_item(item: dict, source: dict) -> Optional[dict]:
    """Convert any dataset item to training format."""
    cat = source.get("category", "")
    name = source["name"]

    # Wikipedia
    if cat == "wikipedia" or "wiki" in name:
        text = item.get("text", "")
        title = item.get("title", "")
        if len(text) < 50:
            return None
        return {
            "input": f"{'ما هو' if 'ar' in name else 'What is'} {title}?",
            "output": text[:4000],
            "source": name,
            "category": cat,
        }

    # Code / Programming
    if cat == "programming":
        # Try multiple field names
        inp = (item.get("instruction") or item.get("prompt") or
               item.get("question") or item.get("func_documentation_string") or
               item.get("title") or "")
        out = (item.get("output") or item.get("completion") or
               item.get("answer") or item.get("func_code_string") or
               item.get("body") or "")
        # OpenHermes format
        convos = item.get("conversations", [])
        if convos and not inp:
            for c in convos:
                if c.get("from") == "human":
                    inp = c.get("value", "")
                elif c.get("from") == "gpt":
                    out = c.get("value", "")
        if inp and out:
            return {"input": inp[:2000], "output": out[:4000], "source": name, "category": cat}

    # Arabic
    if cat == "arabic":
        inp = item.get("instruction", "") or item.get("input", "")
        out = item.get("output", "")
        if inp and out:
            return {"input": inp[:2000], "output": out[:4000], "source": name, "category": cat}

    # Science / Math
    if cat == "science":
        q = item.get("question", "")
        a = item.get("correct_answer", "") or item.get("solution", "") or item.get("answer", "")
        support = item.get("support", "")
        if q and a:
            full_answer = f"{a}\n{support}" if support else a
            return {"input": q[:2000], "output": full_answer[:4000], "source": name, "category": cat}

    # Medical / Survival
    if cat == "survival":
        q = item.get("question", "") or item.get("title", "")
        text = item.get("text", "") or item.get("exp", "")
        if "opa" in item:  # medmcqa format
            options = [item.get(f"op{c}", "") for c in "abcd" if item.get(f"op{c}")]
            ans_idx = item.get("cop", 0)
            ans = options[ans_idx] if ans_idx < len(options) else ""
            exp = item.get("exp", "")
            out = f"{ans}\n{exp}" if exp else ans
            if q and out:
                return {"input": q[:2000], "output": out[:4000], "source": name, "category": cat}
        if q and text:
            return {"input": q[:2000], "output": text[:4000], "source": name, "category": cat}

    # General (Dolly, OASST, SlimOrca)
    if cat == "general":
        # Dolly format
        inp = item.get("instruction", "") or item.get("question", "")
        ctx = item.get("context", "")
        out = item.get("response", "") or item.get("output", "")
        # OASST format
        if not inp and item.get("text"):
            return {"input": item["text"][:2000], "output": "", "source": name, "category": cat}
        # SlimOrca format
        convos = item.get("conversations", [])
        if convos and not inp:
            for c in convos:
                role = c.get("from", "") or c.get("role", "")
                val = c.get("value", "") or c.get("content", "")
                if role in ("human", "user"):
                    inp = val
                elif role in ("gpt", "assistant"):
                    out = val
        if inp and out:
            full_inp = f"{inp}\n\nContext: {ctx}" if ctx else inp
            return {"input": full_inp[:2000], "output": out[:4000], "source": name, "category": cat}

    return None


def download_source(source: dict, state: dict) -> int:
    """Download one batch from a source. Returns count of new items."""
    from datasets import load_dataset

    name = source["name"]
    key = f"idx_{name}"
    start_idx = state.get(key, 0)
    batch = source.get("batch", 2000)

    output_path = DATA_ROOT / source["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        kwargs = {"path": source["path"], "split": source["split"], "streaming": True}
        if "subset" in source:
            kwargs["name"] = source["subset"]

        ds = load_dataset(**kwargs, trust_remote_code=True)
        count = 0

        with output_path.open("a", encoding="utf-8") as f:
            for i, item in enumerate(ds.skip(start_idx)):
                if i >= batch:
                    break
                sample = convert_item(item, source)
                if sample:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1

        state[key] = start_idx + batch
        save_state(state)

        size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0
        log(f"   ✅ {name}: +{count} samples (total idx: {start_idx + batch}, file: {size_mb:.1f}MB)")
        return count

    except Exception as e:
        log(f"   ⚠️ {name}: {e}")
        state[f"error_{name}"] = str(e)
        save_state(state)
        return 0


def check_internet() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def get_disk_usage() -> dict:
    import shutil
    total, used, free = shutil.disk_usage(str(DATA_ROOT))
    return {
        "total_gb": round(total / 1e9, 1),
        "used_gb": round(used / 1e9, 1),
        "free_gb": round(free / 1e9, 1),
        "percent": round(used / total * 100, 1),
    }


def main():
    log("=" * 70)
    log("🔭 KNOWLEDGE SCOUT — كشّاف المعرفة v1.0")
    log(f"   Target: {DATA_ROOT} (Second 4TB NVMe)")
    log(f"   Sources: {len(SOURCES)} datasets")
    log(f"   Mode: NON-STOP CONTINUOUS — never sleeps")
    disk = get_disk_usage()
    log(f"   Disk: {disk['used_gb']}GB used / {disk['free_gb']}GB free ({disk['percent']}%)")
    log("=" * 70)

    state = load_state()
    state["started_at"] = datetime.now().isoformat()
    state["total_samples"] = state.get("total_samples", 0)
    save_state(state)

    round_num = 0
    while True:
        round_num += 1
        disk = get_disk_usage()
        log(f"\n{'='*50}")
        log(f"🔄 Round {round_num} — Disk: {disk['used_gb']}GB / {disk['total_gb']}GB ({disk['percent']}%)")

        if disk["free_gb"] < 50:
            log("⚠️ Less than 50GB free — pausing downloads for 10 minutes")
            time.sleep(600)
            continue

        if not check_internet():
            log("📴 No internet — waiting 30 seconds...")
            time.sleep(30)
            continue

        round_total = 0
        for source in sorted(SOURCES, key=lambda s: (s.get("priority", 99))):
            log(f"\n🔍 [{source['category']}] {source['desc']}")
            try:
                count = download_source(source, state)
                round_total += count
                state["total_samples"] = state.get("total_samples", 0) + count
                save_state(state)
            except Exception as e:
                log(f"   ❌ Fatal error: {e}")
                traceback.print_exc()

            # No sleep between sources — NON-STOP mode
            # Just a tiny yield to prevent CPU starvation
            time.sleep(0.1)

        log(f"\n📊 Round {round_num} done: +{round_total} samples | Total: {state.get('total_samples', 0)}")
        disk = get_disk_usage()
        log(f"💾 Disk: {disk['used_gb']}GB used, {disk['free_gb']}GB free")

        # No sleep between rounds! Continuous download.
        # Just 1 second to let system breathe
        time.sleep(1)


if __name__ == "__main__":
    main()
