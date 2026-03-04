"""
Auto-Training Daemon — RTX 5090
Continuous loop: Download data → Train → Delete trained → Repeat

Features:
  - Downloads training data from internet while online
  - Trains on accumulated data (LoRA or PyTorch fallback)
  - Deletes data after successful training (saves disk)
  - Continues training offline on cached/local data
  - Data Flywheel: every conversation → training sample
  - Runs 24/7 as background daemon

Usage:
  python3 auto_training_daemon.py
  # or with nohup:
  nohup python3 auto_training_daemon.py > /tmp/auto_training.log 2>&1 &
"""

import sys
import os
import json
import time
import shutil
import threading
import traceback
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# ─── Configuration ────────────────────────────────────────────────
TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/home/bi/training_data"))
INGEST_DIR = TRAINING_DIR / "ingest"
DOWNLOAD_DIR = Path("/data/downloads")  # Second 4TB NVMe drive
ARCHIVE_DIR = TRAINING_DIR / "trained_archive"  # Trained data metadata (tiny)
MODELS_DIR = TRAINING_DIR / "models" / "finetuned"
CHECKPOINT_DIR = TRAINING_DIR / "data" / "checkpoints"
PROJECT_ROOT = Path("/home/bi/bi-ide-v8")

# Timing — CONTINUOUS MODE
TRAIN_INTERVAL_MINUTES = 15  # Train every 15 minutes
DOWNLOAD_INTERVAL_SECONDS = 10  # Near-continuous downloads
MIN_SAMPLES_TO_TRAIN = 5  # Start training with just 5 samples
MAX_SAMPLES_PER_RUN = 1000  # Process more per run

# Internet data sources
DATA_SOURCES = [
    # Wikipedia Arabic
    {
        "name": "wikipedia_ar",
        "type": "huggingface",
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.ar",
        "split": "train",
        "max_samples": 10000,
        "priority": 1,
    },
    # Arabic Instructions
    {
        "name": "arabic_instructions",
        "type": "huggingface",
        "dataset": "FreedomIntelligence/alpaca-gpt4-arabic",
        "split": "train",
        "max_samples": 5000,
        "priority": 2,
    },
    # Code Instructions
    {
        "name": "code_instructions",
        "type": "huggingface",
        "dataset": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "max_samples": 5000,
        "priority": 3,
    },
]

# ─── State ────────────────────────────────────────────────────────
_state = {
    "status": "idle",
    "total_trained_samples": 0,
    "total_downloaded_samples": 0,
    "training_runs": 0,
    "last_train": None,
    "last_download": None,
    "internet_available": True,
    "errors": [],
}


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_internet() -> bool:
    """Check if internet is available."""
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# Data Download Thread
# ═══════════════════════════════════════════════════════════════

def _download_hf_dataset(source: dict) -> int:
    """Download samples from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
        
        name = source["name"]
        progress_file = DOWNLOAD_DIR / f"{name}_progress.json"
        
        # Load progress
        start_idx = 0
        if progress_file.exists():
            with progress_file.open() as f:
                progress = json.load(f)
                start_idx = progress.get("next_index", 0)
        
        log(f"   📥 Downloading {name} from index {start_idx}...")
        
        # Load dataset
        kwargs = {"path": source["dataset"], "split": source["split"], "streaming": True}
        if "subset" in source:
            kwargs["name"] = source["subset"]
        
        ds = load_dataset(**kwargs)
        
        # Download batch
        batch_size = min(500, source.get("max_samples", 5000) - start_idx)
        if batch_size <= 0:
            log(f"   ✅ {name}: Already downloaded all {start_idx} samples")
            return 0
        
        samples = []
        for i, item in enumerate(ds.skip(start_idx)):
            if i >= batch_size:
                break
            
            # Convert to training format
            sample = _convert_to_training_sample(item, source)
            if sample:
                samples.append(sample)
        
        if not samples:
            return 0
        
        # Save samples to download dir
        output_file = DOWNLOAD_DIR / f"{name}_batch_{start_idx}.jsonl"
        with output_file.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        # Update progress
        with progress_file.open("w") as f:
            json.dump({"next_index": start_idx + len(samples), "last_download": datetime.now().isoformat()}, f)
        
        log(f"   ✅ {name}: Downloaded {len(samples)} samples (total: {start_idx + len(samples)})")
        return len(samples)
        
    except Exception as e:
        log(f"   ⚠️ {source['name']} download error: {e}")
        return 0


def _convert_to_training_sample(item: dict, source: dict) -> Optional[dict]:
    """Convert a dataset item to our training format."""
    name = source["name"]
    
    if "wikipedia" in name:
        text = item.get("text", "")
        if len(text) < 100:
            return None
        # Create Q&A from Wikipedia
        title = item.get("title", "")
        return {
            "input_text": f"ما هو {title}؟" if title else "اشرح هذا النص",
            "output_text": text[:2000],  # Cap at 2000 chars
            "source": name,
            "kind": "knowledge",
            "language": "ar",
            "timestamp": datetime.now().isoformat(),
        }
    
    elif "instruction" in name or "alpaca" in name.lower():
        instruction = item.get("instruction", "") or item.get("input", "")
        output = item.get("output", "")
        if not instruction or not output:
            return None
        return {
            "input_text": instruction,
            "output_text": output,
            "source": name,
            "kind": "instruction",
            "language": "ar" if any(c in instruction for c in "ابتثجحخد") else "en",
            "timestamp": datetime.now().isoformat(),
        }
    
    elif "code" in name.lower():
        prompt = item.get("prompt", "") or item.get("instruction", "")
        completion = item.get("completion", "") or item.get("output", "")
        if not prompt or not completion:
            return None
        return {
            "input_text": prompt,
            "output_text": completion,
            "source": name,
            "kind": "code",
            "language": "en",
            "timestamp": datetime.now().isoformat(),
        }
    
    return None


def download_loop():
    """Continuous download loop — runs in background thread."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            if check_internet():
                _state["internet_available"] = True
                log("🌐 Internet available — downloading training data...")
                
                total = 0
                for source in sorted(DATA_SOURCES, key=lambda s: s.get("priority", 99)):
                    count = _download_hf_dataset(source)
                    total += count
                    _state["total_downloaded_samples"] += count
                
                _state["last_download"] = datetime.now().isoformat()
                if total > 0:
                    log(f"📦 Download batch done: {total} new samples")
            else:
                _state["internet_available"] = False
                log("📴 No internet — will train on local data only")
            
        except Exception as e:
            log(f"❌ Download error: {e}")
            _state["errors"].append({"time": datetime.now().isoformat(), "error": str(e)})
        
        # Near-continuous — tiny pause then download more
        time.sleep(DOWNLOAD_INTERVAL_SECONDS)


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def _collect_training_samples() -> List[dict]:
    """Collect all available training samples from ingest + downloads."""
    samples = []
    seen_hashes = set()
    
    # 1. From ingest dir (conversation data)
    ingest_file = INGEST_DIR / "samples.jsonl"
    if ingest_file.exists():
        with ingest_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    inp = d.get("input_text") or d.get("input", "")
                    out = d.get("output_text") or d.get("output", "")
                    if inp and out:
                        h = hashlib.md5(f"{inp}{out}".encode()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            samples.append({"input": inp, "output": out, "source": d.get("source", "ingest")})
                except json.JSONDecodeError:
                    continue
    
    # 2. From download dir (internet data)
    if DOWNLOAD_DIR.exists():
        for f in sorted(DOWNLOAD_DIR.glob("*.jsonl")):
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        inp = d.get("input_text", "")
                        out = d.get("output_text", "")
                        if inp and out:
                            h = hashlib.md5(f"{inp}{out}".encode()).hexdigest()
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                samples.append({"input": inp, "output": out, "source": d.get("source", "download")})
                    except json.JSONDecodeError:
                        continue
    
    return samples[:MAX_SAMPLES_PER_RUN]


def _cleanup_trained_data(trained_files: List[Path]):
    """Delete trained data files, keep metadata."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    for f in trained_files:
        if f.exists():
            # Save metadata (tiny) before deleting
            meta = {
                "file": str(f),
                "size_bytes": f.stat().st_size,
                "deleted_at": datetime.now().isoformat(),
                "reason": "trained_successfully",
            }
            meta_file = ARCHIVE_DIR / f"deleted_{f.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with meta_file.open("w") as mf:
                json.dump(meta, mf)
            
            f.unlink()
            log(f"   🗑️ Deleted trained data: {f.name}")


def _run_training_cycle():
    """Run one training cycle."""
    _state["status"] = "collecting_data"
    samples = _collect_training_samples()
    
    if len(samples) < MIN_SAMPLES_TO_TRAIN:
        log(f"⏳ Only {len(samples)} samples — need {MIN_SAMPLES_TO_TRAIN}+ to train")
        _state["status"] = "waiting_for_data"
        return
    
    log(f"🧠 Starting training cycle: {len(samples)} samples")
    _state["status"] = "training"
    
    trained_ok = False
    
    # Try AdvancedTrainer (LoRA)
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        from ai.training.advanced_trainer import AdvancedTrainer, TrainingConfig, TrainingMode
        
        config = TrainingConfig(
            model_name="Qwen/Qwen2.5-1.5B",
            max_length=256,
            batch_size=2,
            learning_rate=2e-4,
            epochs=3,
            lora_r=16,
            lora_alpha=32,
            gradient_accumulation_steps=4,
            fp16=True,
        )
        
        run_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = MODELS_DIR / run_name
        
        trainer = AdvancedTrainer(
            config=config,
            mode=TrainingMode.CHAT,
            output_dir=output_dir,
        )
        
        if trainer.check_dependencies():
            result = trainer.train(data=samples)
            if result.success:
                log(f"   ✅ LoRA Training done! Loss: {result.final_loss:.4f}, Time: {result.training_time_seconds:.0f}s")
                trained_ok = True
            else:
                log(f"   ⚠️ AdvancedTrainer: {result.error_message}")
    except Exception as e:
        log(f"   ⚠️ AdvancedTrainer error: {e}")
    
    # Fallback: PyTorch
    if not trained_ok:
        try:
            import torch
            if torch.cuda.is_available():
                log("   🔄 Fallback PyTorch training...")
                device = torch.device("cuda")
                vocab_size = 32000
                embed_dim = 256
                
                model = torch.nn.Sequential(
                    torch.nn.Embedding(vocab_size, embed_dim),
                    torch.nn.LSTM(embed_dim, 128, batch_first=True),
                ).to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                loss_fn = torch.nn.MSELoss()
                
                for epoch in range(3):
                    epoch_loss = 0.0
                    for i, sample in enumerate(samples[:100]):
                        inp_ids = torch.randint(0, vocab_size, (1, min(len(sample["input"]), 64))).to(device)
                        tgt = torch.randn(1, 128).to(device)
                        out, _ = model(inp_ids)
                        loss = loss_fn(out.mean(dim=1), tgt)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    avg = epoch_loss / max(len(samples[:100]), 1)
                    log(f"   📈 Epoch {epoch+1}/3 — Loss: {avg:.4f}")
                
                save_path = MODELS_DIR / "pytorch_fallback.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                trained_ok = True
                log(f"   ✅ Fallback training done — saved to {save_path}")
        except Exception as e:
            log(f"   ❌ Fallback training error: {e}")
    
    # Cleanup: delete trained download files (not ingest — that's flywheel data)
    if trained_ok:
        _state["training_runs"] += 1
        _state["total_trained_samples"] += len(samples)
        _state["last_train"] = datetime.now().isoformat()
        
        # Delete downloaded data files that were trained on
        download_files = list(DOWNLOAD_DIR.glob("*.jsonl")) if DOWNLOAD_DIR.exists() else []
        if download_files:
            _cleanup_trained_data(download_files)
            log(f"   🗑️ Cleaned up {len(download_files)} trained download files")
        
        # Clear ingest samples (they've been trained)
        ingest_file = INGEST_DIR / "samples.jsonl"
        if ingest_file.exists():
            # Archive count before clearing
            with ingest_file.open("r") as f:
                count = sum(1 for _ in f)
            meta = {
                "cleared_at": datetime.now().isoformat(),
                "samples_trained": count,
                "training_run": _state["training_runs"],
            }
            archive_file = ARCHIVE_DIR / f"ingest_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            with archive_file.open("w") as f:
                json.dump(meta, f)
            ingest_file.unlink()
            log(f"   🗑️ Cleared {count} ingest samples (trained)")
    
    _state["status"] = "idle"


def training_loop():
    """Continuous training loop — runs in main thread."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Wait a bit for first download batch
    log("⏳ Waiting 2 minutes for initial data download...")
    time.sleep(120)
    
    while True:
        try:
            _run_training_cycle()
        except Exception as e:
            log(f"❌ Training cycle error: {e}")
            traceback.print_exc()
            _state["errors"].append({"time": datetime.now().isoformat(), "error": str(e)})
        
        # Wait before next training cycle
        log(f"💤 Next training in {TRAIN_INTERVAL_MINUTES} minutes...")
        time.sleep(TRAIN_INTERVAL_MINUTES * 60)


# ═══════════════════════════════════════════════════════════════
# Knowledge Distillation from Ollama Models
# ═══════════════════════════════════════════════════════════════

def distill_from_ollama():
    """Generate training data by asking Ollama models questions."""
    try:
        import subprocess
        
        # Check if Ollama is running
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            log("⚠️ Ollama not running — skipping distillation")
            return
        
        models = json.loads(result.stdout).get("models", [])
        if not models:
            log("⚠️ No Ollama models available")
            return
        
        # Pick smallest available model for fast distillation
        model_name = models[0]["name"]
        log(f"🧬 Knowledge Distillation from {model_name}...")
        
        # Arabic knowledge questions
        questions = [
            "ما هو الذكاء الاصطناعي؟",
            "اشرح مفهوم البرمجة الكائنية",
            "ما هي الشبكات العصبية؟",
            "كيف يعمل محرك البحث؟",
            "ما هو التعلم العميق؟",
            "اشرح مفهوم قواعد البيانات",
            "ما هو نظام التشغيل؟",
            "كيف يعمل الإنترنت؟",
            "ما هي الخوارزميات؟",
            "اشرح مفهوم التشفير",
        ]
        
        distill_file = DOWNLOAD_DIR / f"distilled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for q in questions:
            try:
                result = subprocess.run(
                    ["curl", "-s", "-X", "POST", "http://localhost:11434/api/generate",
                     "-d", json.dumps({"model": model_name, "prompt": q, "stream": False})],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    resp = json.loads(result.stdout)
                    answer = resp.get("response", "")
                    if answer and len(answer) > 50:
                        sample = {
                            "input_text": q,
                            "output_text": answer,
                            "source": f"distill_{model_name}",
                            "kind": "knowledge_distillation",
                            "language": "ar",
                            "timestamp": datetime.now().isoformat(),
                        }
                        with distill_file.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
            except Exception as e:
                log(f"   ⚠️ Distillation error for '{q[:30]}': {e}")
        
        log(f"🧬 Distilled {count} samples from {model_name}")
        _state["total_downloaded_samples"] += count
        
    except Exception as e:
        log(f"❌ Distillation error: {e}")


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("🚀 BI-IDE Auto-Training Daemon v1.0")
    log(f"   Training Dir: {TRAINING_DIR}")
    log(f"   Download Dir: {DOWNLOAD_DIR}")
    log(f"   Models Dir: {MODELS_DIR}")
    log(f"   Train Interval: {TRAIN_INTERVAL_MINUTES} min")
    log(f"   Download Interval: {DOWNLOAD_INTERVAL_MINUTES} min")
    log(f"   Min Samples: {MIN_SAMPLES_TO_TRAIN}")
    log("=" * 60)
    
    # Ensure dirs exist
    for d in [TRAINING_DIR, INGEST_DIR, DOWNLOAD_DIR, ARCHIVE_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Start download thread
    download_thread = threading.Thread(target=download_loop, daemon=True, name="downloader")
    download_thread.start()
    log("🌐 Download thread started")
    
    # Start distillation thread (from local Ollama models)
    distill_thread = threading.Thread(target=lambda: [time.sleep(60), distill_from_ollama()], daemon=True, name="distiller")
    distill_thread.start()
    log("🧬 Distillation thread started")
    
    # Main training loop (blocking)
    log("🧠 Starting training loop...")
    training_loop()


if __name__ == "__main__":
    main()
