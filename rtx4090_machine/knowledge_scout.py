#!/usr/bin/env python3
"""
Knowledge Scout v2.0 — كشّاف المعرفة (TURBO)
Downloads ALL human knowledge into /data — FAST.

Changes from v1:
  - Batch size: 5K → 50K+ per source
  - Parallel downloads: 3 threads simultaneously
  - Fixed broken dataset URLs
  - Skip permanently failed sources
  - Non-streaming mode for datasets under 5GB (much faster)

Usage:
  nohup python3 knowledge_scout.py > /tmp/scout.log 2>&1 &
"""

import sys
import os
import json
import time
import hashlib
import traceback
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────────
DATA_ROOT = Path("/data")           # Second 4TB NVMe
SCOUT_STATE = DATA_ROOT / ".scout_state.json"
_state_lock = threading.Lock()

# ─── All Knowledge Sources — VERIFIED WORKING ─────────────────
SOURCES = [
    # ════════ WIKIPEDIA — Foundation of Knowledge ════════
    {"name": "wikipedia_ar", "category": "wikipedia", "type": "huggingface", "path": "wikimedia/wikipedia", "subset": "20231101.ar", "split": "train", "output": "wikipedia/ar.jsonl", "batch": 50000, "priority": 1, "desc": "Wikipedia Arabic — 1.2M articles"},
    {"name": "wikipedia_en", "category": "wikipedia", "type": "huggingface", "path": "wikimedia/wikipedia", "subset": "20231101.en", "split": "train", "output": "wikipedia/en.jsonl", "batch": 100000, "priority": 1, "desc": "Wikipedia English — 6M+ articles"},
    {"name": "wikipedia_simple", "category": "wikipedia", "type": "huggingface", "path": "wikimedia/wikipedia", "subset": "20231101.simple", "split": "train", "output": "wikipedia/simple.jsonl", "batch": 50000, "priority": 3, "desc": "Simple English Wikipedia"},

    # ════════ PROGRAMMING — ALL LANGUAGES ════════
    {"name": "code_alpaca", "category": "programming", "type": "huggingface", "path": "sahil2801/CodeAlpaca-20k", "split": "train", "output": "datasets/code_alpaca.jsonl", "batch": 20000, "priority": 1, "stream": False, "desc": "Code instructions — 20K"},
    {"name": "python_instructions", "category": "programming", "type": "huggingface", "path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "output": "datasets/python_instructions.jsonl", "batch": 18000, "priority": 1, "stream": False, "desc": "Python instructions — 18K"},
    {"name": "code_feedback", "category": "programming", "type": "huggingface", "path": "m-a-p/CodeFeedback-Filtered-Instruction", "split": "train", "output": "datasets/code_feedback.jsonl", "batch": 50000, "priority": 2, "desc": "Code feedback — instruction + response"},
    {"name": "code_exercises", "category": "programming", "type": "huggingface", "path": "jinaai/code_exercises", "split": "train", "output": "datasets/code_exercises.jsonl", "batch": 50000, "priority": 2, "desc": "Code exercises — practice problems"},
    {"name": "openhermes", "category": "programming", "type": "huggingface", "path": "teknium/OpenHermes-2.5", "split": "train", "output": "datasets/openhermes.jsonl", "batch": 100000, "priority": 1, "desc": "OpenHermes — 1M instruction pairs"},

    # ════════ MATHEMATICS ════════
    {"name": "gsm8k", "category": "science", "type": "huggingface", "path": "openai/gsm8k", "subset": "main", "split": "train", "output": "datasets/gsm8k.jsonl", "batch": 10000, "priority": 1, "stream": False, "desc": "GSM8K — grade school math"},
    {"name": "metamath", "category": "science", "type": "huggingface", "path": "meta-math/MetaMathQA", "split": "train", "output": "datasets/metamath.jsonl", "batch": 100000, "priority": 1, "desc": "MetaMathQA — 400K math questions"},
    {"name": "math_camel", "category": "science", "type": "huggingface", "path": "camel-ai/math", "split": "train", "output": "datasets/math_camel.jsonl", "batch": 50000, "priority": 1, "desc": "CAMEL Math — math dialogues"},

    # ════════ PHYSICS ════════
    {"name": "physics_camel", "category": "science", "type": "huggingface", "path": "camel-ai/physics", "split": "train", "output": "datasets/physics_camel.jsonl", "batch": 50000, "priority": 1, "desc": "CAMEL Physics — physics dialogues"},
    {"name": "sciq", "category": "science", "type": "huggingface", "path": "allenai/sciq", "split": "train", "output": "datasets/sciq.jsonl", "batch": 15000, "priority": 1, "stream": False, "desc": "Science exam questions"},

    # ════════ ARABIC ════════
    {"name": "arabic_alpaca", "category": "arabic", "type": "huggingface", "path": "FreedomIntelligence/alpaca-gpt4-arabic", "split": "train", "output": "datasets/arabic_alpaca.jsonl", "batch": 50000, "priority": 1, "desc": "Arabic instruction-following"},
    {"name": "arabic_cidar", "category": "arabic", "type": "huggingface", "path": "arbml/CIDAR", "split": "train", "output": "datasets/arabic_cidar.jsonl", "batch": 50000, "priority": 2, "desc": "Arabic cultural & instructional data"},
    {"name": "arabic_poems", "category": "arabic", "type": "huggingface", "path": "arbml/Arabic_Poems_1M", "split": "train", "output": "datasets/arabic_poems.jsonl", "batch": 100000, "priority": 2, "desc": "Arabic poetry — 1M poems"},

    # ════════ SURVIVAL & CIVILIZATION REBUILDING ════════
    {"name": "medical_flashcards", "category": "survival", "type": "huggingface", "path": "medalpaca/medical_meadow_medical_flashcards", "split": "train", "output": "datasets/medical_flashcards.jsonl", "batch": 50000, "priority": 1, "stream": False, "desc": "Medical flashcards"},
    {"name": "medical_dialog", "category": "survival", "type": "huggingface", "path": "ruslanmv/ai-medical-chatbot", "split": "train", "output": "datasets/medical_dialog.jsonl", "batch": 100000, "priority": 2, "desc": "Medical dialogues"},
    {"name": "wikihow", "category": "survival", "type": "huggingface", "path": "b-mc2/wikihow_lists", "split": "train", "output": "datasets/wikihow.jsonl", "batch": 50000, "priority": 1, "desc": "WikiHow — practical instructions"},

    # ════════ GENERAL KNOWLEDGE & CONVERSATIONS ════════
    {"name": "dolly", "category": "general", "type": "huggingface", "path": "databricks/databricks-dolly-15k", "split": "train", "output": "datasets/dolly.jsonl", "batch": 15000, "priority": 1, "stream": False, "desc": "Dolly — 15K instructions"},
    {"name": "oasst", "category": "general", "type": "huggingface", "path": "OpenAssistant/oasst2", "split": "train", "output": "datasets/oasst.jsonl", "batch": 50000, "priority": 1, "desc": "OpenAssistant 2 — human conversations"},
    {"name": "slimorca", "category": "general", "type": "huggingface", "path": "Open-Orca/SlimOrca-Dedup", "split": "train", "output": "datasets/slimorca.jsonl", "batch": 100000, "priority": 1, "desc": "SlimOrca — 500K GPT-4 conversations"},
    {"name": "self_instruct", "category": "general", "type": "huggingface", "path": "yizhongw/self_instruct", "split": "train", "output": "datasets/self_instruct.jsonl", "batch": 52000, "priority": 2, "stream": False, "desc": "Self-Instruct — 52K instructions"},
    {"name": "ultrachat", "category": "general", "type": "huggingface", "path": "stingning/ultrachat", "split": "train", "output": "datasets/ultrachat.jsonl", "batch": 100000, "priority": 2, "desc": "UltraChat — 1.5M conversations"},

    # ════════ MASSIVE DATASETS — FILL THE DISK ════════
    {"name": "the_pile_subset", "category": "general", "type": "huggingface", "path": "monology/pile-uncopyrighted", "split": "train", "output": "datasets/pile_uncopyrighted.jsonl", "batch": 200000, "priority": 2, "desc": "The Pile (uncopyrighted) — books, papers, code"},
    {"name": "red_pajama", "category": "general", "type": "huggingface", "path": "togethercomputer/RedPajama-Data-1T-Sample", "split": "train", "output": "datasets/red_pajama.jsonl", "batch": 200000, "priority": 2, "desc": "RedPajama — 1T token sample"},
    {"name": "starcoderdata", "category": "programming", "type": "huggingface", "path": "bigcode/starcoderdata", "subset": "python", "split": "train", "output": "datasets/starcoder_python.jsonl", "batch": 200000, "priority": 2, "desc": "StarCoder Python — massive code dataset"},
]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state() -> dict:
    with _state_lock:
        if SCOUT_STATE.exists():
            with SCOUT_STATE.open() as f:
                return json.load(f)
    return {}


def save_state(state: dict):
    with _state_lock:
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
            "output": text[:8000],  # Longer context for better learning
            "source": name,
            "category": cat,
        }

    # Code / Programming
    if cat == "programming":
        inp = (item.get("instruction") or item.get("prompt") or
               item.get("question") or item.get("func_documentation_string") or
               item.get("title") or item.get("content") or "")
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
        # Pile / RedPajama format — plain text
        if not inp and not out:
            text = item.get("text", "")
            if text and len(text) > 100:
                return {"input": "Continue this code:", "output": text[:8000], "source": name, "category": cat}
        if inp and out:
            return {"input": inp[:4000], "output": out[:8000], "source": name, "category": cat}

    # Arabic
    if cat == "arabic":
        inp = item.get("instruction", "") or item.get("input", "") or item.get("text", "")
        out = item.get("output", "") or item.get("poem", "")
        if inp and out:
            return {"input": inp[:4000], "output": out[:8000], "source": name, "category": cat}
        # Poetry — single text field
        if "poem" in name and (item.get("poem_text") or item.get("text")):
            text = item.get("poem_text", "") or item.get("text", "")
            return {"input": "اكتب قصيدة", "output": text[:8000], "source": name, "category": "arabic"}

    # Science / Math
    if cat == "science":
        q = item.get("question", "") or item.get("query", "")
        a = item.get("correct_answer", "") or item.get("solution", "") or item.get("answer", "") or item.get("response", "")
        support = item.get("support", "")
        # CAMEL format
        if not q:
            msg1 = item.get("message_1", "")
            msg2 = item.get("message_2", "")
            if msg1 and msg2:
                q, a = msg1, msg2
        if q and a:
            full_answer = f"{a}\n{support}" if support else a
            return {"input": q[:4000], "output": full_answer[:8000], "source": name, "category": cat}

    # Medical / Survival
    if cat == "survival":
        q = item.get("question", "") or item.get("title", "") or item.get("Patient", "") or item.get("input", "")
        text = item.get("text", "") or item.get("exp", "") or item.get("Doctor", "") or item.get("output", "")
        if "opa" in item:  # medmcqa format
            options = [item.get(f"op{c}", "") for c in "abcd" if item.get(f"op{c}")]
            ans_idx = item.get("cop", 0)
            ans = options[ans_idx] if ans_idx < len(options) else ""
            exp = item.get("exp", "")
            out = f"{ans}\n{exp}" if exp else ans
            if q and out:
                return {"input": q[:4000], "output": out[:8000], "source": name, "category": cat}
        if q and text:
            return {"input": q[:4000], "output": text[:8000], "source": name, "category": cat}

    # General (Dolly, OASST, SlimOrca, Pile, RedPajama)
    if cat == "general":
        inp = item.get("instruction", "") or item.get("question", "")
        ctx = item.get("context", "")
        out = item.get("response", "") or item.get("output", "")
        # OASST format
        if not inp and item.get("text"):
            text = item["text"]
            if len(text) > 100:
                return {"input": text[:4000], "output": "", "source": name, "category": cat}
        # SlimOrca / conversation format
        convos = item.get("conversations", [])
        if convos and not inp:
            for c in convos:
                role = c.get("from", "") or c.get("role", "")
                val = c.get("value", "") or c.get("content", "")
                if role in ("human", "user"):
                    inp = val
                elif role in ("gpt", "assistant"):
                    out = val
        # Pile / RedPajama — plain text
        if not inp and not out:
            text = item.get("text", "")
            if text and len(text) > 100:
                return {"input": "Explain:", "output": text[:8000], "source": name, "category": cat}
        if inp and out:
            full_inp = f"{inp}\n\nContext: {ctx}" if ctx else inp
            return {"input": full_inp[:4000], "output": out[:8000], "source": name, "category": cat}

    return None


def download_source(source: dict, state: dict) -> int:
    """Download one batch from a source. Returns count of new items."""
    from datasets import load_dataset

    name = source["name"]
    key = f"idx_{name}"
    err_key = f"error_{name}"
    start_idx = state.get(key, 0)
    batch = source.get("batch", 50000)

    output_path = DATA_ROOT / source["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        use_streaming = source.get("stream", True)
        kwargs = {"path": source["path"], "split": source["split"], "streaming": use_streaming}
        if "subset" in source:
            kwargs["name"] = source["subset"]

        ds = load_dataset(**kwargs)
        count = 0

        with output_path.open("a", encoding="utf-8") as f:
            if use_streaming:
                for i, item in enumerate(ds.skip(start_idx)):
                    if i >= batch:
                        break
                    sample = convert_item(item, source)
                    if sample:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
                    # Progress every 10K
                    if (i + 1) % 10000 == 0:
                        log(f"      {name}: {i+1}/{batch} processed...")
            else:
                # Non-streaming: much faster for small datasets
                end_idx = min(start_idx + batch, len(ds))
                for i in range(start_idx, end_idx):
                    item = ds[i]
                    sample = convert_item(item, source)
                    if sample:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1

        with _state_lock:
            state[key] = start_idx + batch
            if err_key in state:
                del state[err_key]  # Clear old errors on success
            save_state(state)

        size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0
        log(f"   ✅ {name}: +{count} samples (total idx: {start_idx + batch}, file: {size_mb:.1f}MB)")
        return count

    except Exception as e:
        err_msg = str(e)
        log(f"   ⚠️ {name}: {err_msg[:100]}")
        with _state_lock:
            state[err_key] = err_msg[:200]
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
    log("🔭 KNOWLEDGE SCOUT v2.0 — كشّاف المعرفة (TURBO)")
    log(f"   Target: {DATA_ROOT} (Second 4TB NVMe)")
    log(f"   Sources: {len(SOURCES)} datasets")
    log(f"   Mode: TURBO — parallel downloads, 50K+ batches")
    log(f"   Max parallel: 3 threads")
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

        # Sort by priority, skip permanently failed
        active_sources = []
        for s in sorted(SOURCES, key=lambda x: x.get("priority", 99)):
            err_key = f"error_{s['name']}"
            if err_key in state:
                err = state[err_key]
                if "doesn't exist" in err or "cannot be accessed" in err:
                    continue  # Skip permanently broken datasets
            active_sources.append(s)

        round_total = 0

        # Download in parallel — 3 threads
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="scout") as executor:
            futures = {}
            for source in active_sources:
                log(f"   📥 Queuing: [{source['category']}] {source['desc']}")
                future = executor.submit(download_source, source, state)
                futures[future] = source

            for future in as_completed(futures):
                source = futures[future]
                try:
                    count = future.result()
                    round_total += count
                    with _state_lock:
                        state["total_samples"] = state.get("total_samples", 0) + count
                        save_state(state)
                except Exception as e:
                    log(f"   ❌ Fatal error in {source['name']}: {e}")
                    traceback.print_exc()

        log(f"\n📊 Round {round_num} done: +{round_total} samples | Total: {state.get('total_samples', 0)}")
        disk = get_disk_usage()
        log(f"💾 Disk: {disk['used_gb']}GB used, {disk['free_gb']}GB free")

        # Tiny pause then continue
        time.sleep(2)


if __name__ == "__main__":
    main()
