#!/usr/bin/env python3
"""
Bulk Data Downloader — ملء 4TB بالمعرفة البشرية
Downloads MASSIVE datasets directly (no streaming).
Uses huggingface_hub for maximum speed + resume support.

Target: Fill /data (4TB NVMe) with ALL human knowledge.

Estimated sizes:
  - Wikipedia ALL languages:  ~80GB
  - RedPajama-1T sample:      ~5GB
  - The Pile (uncopyrighted): ~200GB
  - StarCoder (all langs):    ~250GB
  - FineWeb-Edu:              ~500GB
  - OSCAR multilingual:       ~300GB+
  - C4 (Colossal Cleaned):    ~800GB
  - ArXiv papers:             ~100GB
  - Books/Gutenberg:           ~50GB
  Total potential:           ~2.3TB

Usage:
  nohup python3 bulk_downloader.py > /tmp/bulk_download.log 2>&1 &
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_ROOT = Path("/data")
STATE_FILE = DATA_ROOT / ".bulk_state.json"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_disk_free_gb():
    import shutil
    _, _, free = shutil.disk_usage(str(DATA_ROOT))
    return round(free / 1e9, 1)


def load_state():
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_state(state):
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Download Methods
# ═══════════════════════════════════════════════════════════════

def hf_download(repo_id: str, output_dir: str, subset: str = None, file_pattern: str = None):
    """Download a full HuggingFace dataset using Python API (fastest method with resume)."""
    from huggingface_hub import snapshot_download
    
    dest = DATA_ROOT / output_dir
    dest.mkdir(parents=True, exist_ok=True)
    
    log(f"   ⬇️  snapshot_download {repo_id} → {dest}")
    
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(dest),
    }
    if file_pattern:
        kwargs["allow_patterns"] = [file_pattern]
    
    snapshot_download(**kwargs)
    
    # Get size
    size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    log(f"   ✅ {repo_id}: {size_gb:.1f}GB downloaded")
    return size_gb


def hf_stream_to_jsonl(repo_id: str, output_file: str, subset: str = None, 
                         max_samples: int = 0, split: str = "train"):
    """Download dataset via Python datasets library and save as JSONL."""
    from datasets import load_dataset
    
    dest = DATA_ROOT / output_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if already large enough
    if dest.exists() and dest.stat().st_size > 1e9:  # > 1GB
        log(f"   ⏭️  {repo_id}: Already {dest.stat().st_size/1e9:.1f}GB — skipping")
        return dest.stat().st_size / 1e9
    
    kwargs = {"path": repo_id, "split": split, "streaming": True}
    if subset:
        kwargs["name"] = subset
    
    log(f"   ⬇️  Streaming {repo_id} → {dest}")
    ds = load_dataset(**kwargs)
    
    count = 0
    with dest.open("a", encoding="utf-8") as f:
        for item in ds:
            # Universal text extraction
            text = ""
            if "text" in item:
                text = item["text"]
            elif "content" in item:
                text = item["content"]
            elif "conversations" in item:
                parts = []
                for c in item["conversations"]:
                    val = c.get("value", "") or c.get("content", "")
                    parts.append(val)
                text = "\n".join(parts)
            else:
                # Try all text-like fields
                for key in ["instruction", "input", "output", "response", "answer", "question"]:
                    if key in item and item[key]:
                        text += item[key] + "\n"
            
            if text and len(text) > 50:
                sample = {"text": text[:16000], "source": repo_id}
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
            
            if max_samples > 0 and count >= max_samples:
                break
            
            if count % 100000 == 0 and count > 0:
                size_gb = dest.stat().st_size / 1e9
                log(f"      {repo_id}: {count:,} samples ({size_gb:.1f}GB)...")
    
    size_gb = dest.stat().st_size / 1e9
    log(f"   ✅ {repo_id}: {count:,} samples ({size_gb:.1f}GB)")
    return size_gb


def wget_download(url: str, output_dir: str, filename: str = None):
    """Download a file via wget with resume support."""
    dest = DATA_ROOT / output_dir
    dest.mkdir(parents=True, exist_ok=True)
    
    cmd = ["wget", "-c", "--no-check-certificate", "-q", "--show-progress",
           "-P", str(dest), url]
    if filename:
        cmd.extend(["-O", str(dest / filename)])
    
    log(f"   ⬇️  wget {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    
    if result.returncode != 0:
        raise RuntimeError(f"wget failed: {result.stderr[:200]}")


# ═══════════════════════════════════════════════════════════════
# Dataset Definitions — ordered by priority + size
# ═══════════════════════════════════════════════════════════════

DATASETS = [
    # ──── Priority 1: Core Knowledge (100-200GB) ────
    {
        "name": "wikipedia_all",
        "desc": "📚 Wikipedia — ALL major languages (AR, EN, DE, FR, ES, ZH, JA, RU)",
        "size_est": "~80GB",
        "method": "hf_cli",
        "tasks": [
            {"repo": "wikimedia/wikipedia", "subset": "20231101.ar", "dir": "wikipedia/ar"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.en", "dir": "wikipedia/en"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.de", "dir": "wikipedia/de"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.fr", "dir": "wikipedia/fr"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.es", "dir": "wikipedia/es"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.zh", "dir": "wikipedia/zh"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.ja", "dir": "wikipedia/ja"},
            {"repo": "wikimedia/wikipedia", "subset": "20231101.ru", "dir": "wikipedia/ru"},
        ]
    },
    {
        "name": "fineweb_edu",
        "desc": "🎓 FineWeb-Edu — High-quality educational web content",
        "size_est": "~500GB",
        "method": "stream",
        "repo": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "output": "fineweb_edu/sample.jsonl",
        "max_samples": 10000000,  # 10M samples
    },
    {
        "name": "red_pajama",
        "desc": "🔴 RedPajama — 1T token sample (diverse web text)",
        "size_est": "~5GB",
        "method": "hf_cli",
        "repo": "togethercomputer/RedPajama-Data-1T-Sample",
        "dir": "red_pajama",
    },
    
    # ──── Priority 2: Code (250GB+) ────
    {
        "name": "starcoder_python",
        "desc": "⭐ StarCoder — Python code (massive)",
        "size_est": "~50GB",
        "method": "stream",
        "repo": "bigcode/starcoderdata",
        "subset": "python",
        "output": "code/starcoder_python.jsonl",
        "max_samples": 5000000,
    },
    {
        "name": "starcoder_javascript",
        "desc": "⭐ StarCoder — JavaScript code",
        "size_est": "~30GB",
        "method": "stream",
        "repo": "bigcode/starcoderdata",
        "subset": "javascript",
        "output": "code/starcoder_javascript.jsonl",
        "max_samples": 3000000,
    },
    {
        "name": "starcoder_java",
        "desc": "⭐ StarCoder — Java code",
        "size_est": "~25GB",
        "method": "stream",
        "repo": "bigcode/starcoderdata",
        "subset": "java",
        "output": "code/starcoder_java.jsonl",
        "max_samples": 2000000,
    },
    {
        "name": "starcoder_cpp",
        "desc": "⭐ StarCoder — C/C++ code",
        "size_est": "~20GB",
        "method": "stream",
        "repo": "bigcode/starcoderdata",
        "subset": "c",
        "output": "code/starcoder_cpp.jsonl",
        "max_samples": 2000000,
    },
    
    # ──── Priority 3: Science & Papers (100GB) ────
    {
        "name": "arxiv_papers",
        "desc": "📄 ArXiv — Scientific papers (CS, Physics, Math)",
        "size_est": "~100GB",
        "method": "stream",
        "repo": "togethercomputer/RedPajama-Data-1T-Sample",
        "output": "arxiv/papers.jsonl",
        "max_samples": 2000000,
    },
    
    # ──── Priority 4: Multilingual (300GB+) ────
    {
        "name": "oscar_arabic",
        "desc": "🌍 OSCAR — Arabic web corpus (huge)",
        "size_est": "~20GB",
        "method": "stream",
        "repo": "oscar-corpus/OSCAR-2301",
        "subset": "ar",
        "output": "oscar/arabic.jsonl",
        "max_samples": 5000000,
    },
    
    # ──── Priority 5: Books (50GB) ────
    {
        "name": "pile_books",
        "desc": "📖 Pile of Law + Books3 subset",
        "size_est": "~50GB",
        "method": "stream",
        "repo": "monology/pile-uncopyrighted",
        "output": "books/pile_uncopyrighted.jsonl",
        "max_samples": 5000000,
    },
    
    # ──── Priority 6: Conversations & Instructions (50GB) ────
    {
        "name": "ultrachat_200k",
        "desc": "💬 UltraChat 200K — high-quality conversations",
        "size_est": "~5GB",
        "method": "hf_cli",
        "repo": "HuggingFaceH4/ultrachat_200k",
        "dir": "conversations/ultrachat_200k",
    },
    {
        "name": "open_orca",
        "desc": "🐋 OpenOrca — 4M GPT-4 augmented conversations",
        "size_est": "~15GB",
        "method": "hf_cli",
        "repo": "Open-Orca/OpenOrca",
        "dir": "conversations/open_orca",
    },
    {
        "name": "capybara",
        "desc": "🦫 Capybara — Multi-turn conversations",
        "size_est": "~1GB",
        "method": "hf_cli",
        "repo": "LDJnr/Capybara",
        "dir": "conversations/capybara",
    },
]


def process_dataset(ds_config: dict, state: dict):
    """Process a single dataset configuration."""
    name = ds_config["name"]
    
    if name in state.get("completed", []):
        log(f"⏭️  {name}: Already completed — skipping")
        return 0
    
    # Check disk space
    free_gb = get_disk_free_gb()
    if free_gb < 100:
        log(f"⚠️ Only {free_gb}GB free — stopping downloads")
        return -1
    
    log(f"\n{'━'*60}")
    log(f"📥 {ds_config['desc']}")
    log(f"   Estimated size: {ds_config.get('size_est', 'unknown')}")
    log(f"   Disk free: {free_gb}GB")
    
    try:
        method = ds_config["method"]
        total_gb = 0
        
        if method == "hf_cli":
            if "tasks" in ds_config:
                # Multiple sub-downloads (like Wikipedia languages)
                for task in ds_config["tasks"]:
                    try:
                        gb = hf_download(task["repo"], task["dir"])
                        total_gb += gb
                    except Exception as e:
                        log(f"   ⚠️ Sub-task {task.get('dir', task['repo'])} failed: {e}")
            else:
                total_gb = hf_download(ds_config["repo"], ds_config["dir"])
        
        elif method == "stream":
            total_gb = hf_stream_to_jsonl(
                ds_config["repo"],
                ds_config["output"],
                subset=ds_config.get("subset"),
                max_samples=ds_config.get("max_samples", 0),
            )
        
        elif method == "wget":
            wget_download(ds_config["url"], ds_config["dir"], ds_config.get("filename"))
            total_gb = sum(
                f.stat().st_size for f in (DATA_ROOT / ds_config["dir"]).rglob("*") if f.is_file()
            ) / 1e9
        
        # Mark completed
        state.setdefault("completed", []).append(name)
        state[f"size_{name}"] = f"{total_gb:.1f}GB"
        save_state(state)
        
        log(f"✅ {name}: COMPLETE ({total_gb:.1f}GB)")
        return total_gb
        
    except Exception as e:
        log(f"❌ {name}: FAILED — {e}")
        traceback.print_exc()
        state.setdefault("failed", {})[name] = str(e)[:200]
        save_state(state)
        return 0


def main():
    log("=" * 70)
    log("🚀 BULK DOWNLOADER — ملء 4 تيرابايت بالمعرفة")
    log(f"   Target: {DATA_ROOT}")
    log(f"   Datasets: {len(DATASETS)}")
    free_gb = get_disk_free_gb()
    log(f"   Disk free: {free_gb}GB")
    log(f"   Goal: Fill as much as possible!")
    log("=" * 70)
    
    state = load_state()
    state["started_at"] = datetime.now().isoformat()
    save_state(state)
    
    total_downloaded = 0
    
    for ds in DATASETS:
        result = process_dataset(ds, state)
        if result < 0:
            log("⚠️ Disk space critical — stopping")
            break
        total_downloaded += result
        
        free_gb = get_disk_free_gb()
        log(f"\n💾 Progress: {total_downloaded:.1f}GB downloaded | {free_gb}GB remaining")
    
    log(f"\n{'='*70}")
    log(f"🏁 BULK DOWNLOAD COMPLETE")
    log(f"   Total downloaded: {total_downloaded:.1f}GB")
    log(f"   Disk remaining: {get_disk_free_gb()}GB")
    log(f"   Completed: {len(state.get('completed', []))} datasets")
    log(f"   Failed: {len(state.get('failed', {}))} datasets")
    log(f"{'='*70}")
    
    # After bulk download, loop continuous on remaining
    log("\n🔄 Switching to continuous mode — downloading more data forever...")
    while True:
        for ds in DATASETS:
            if ds["name"] not in state.get("completed", []):
                process_dataset(ds, state)
        time.sleep(60)


if __name__ == "__main__":
    main()
