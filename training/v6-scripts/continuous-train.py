#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - Continuous training pipeline.
- Collect new data (smart-learn / auto-learning)
- Train every 12h or when 200+ new samples
- Convert to ONNX with versioning
- Optional webhook/email report

Usage:
    python training/continuous-train.py           # one-shot
    python training/continuous-train.py --watch  # loop every 12h
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training"
OUTPUT_DIR = TRAINING_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "model-registry.json"
MIN_SAMPLES_TO_TRAIN = 200
INTERVAL_SEC = int(os.environ.get('TRAIN_INTERVAL', 3600))  # default 1 hour, override with env
INTENSIVE_MODE = os.environ.get('INTENSIVE_MODE', 'false').lower() == 'true'
TRAIN_CHAT_MODEL = os.environ.get('TRAIN_CHAT_MODEL', 'false').lower() == 'true'


def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'versions': [], 'last_training': None, 'last_sample_count': 0}


def save_registry(reg):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def count_training_samples():
    total = 0
    all_path = OUTPUT_DIR / "all_training_data.json"
    if all_path.exists():
        try:
            with open(all_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total = len(data) if isinstance(data, list) else len(data.get('samples', data.get('examples', [])))
        except Exception:
            pass
    validated = OUTPUT_DIR / "validated_training_data.json"
    if validated.exists():
        try:
            with open(validated, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total = max(total, len(data))
        except Exception:
            pass
    for p in OUTPUT_DIR.glob("*.json"):
        if p.name in ('all_training_data.json', 'validated_training_data.json', 'validation_report.json', 'training_report.json'):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            n = len(d) if isinstance(d, list) else len(d.get('samples', d.get('examples', [])))
            total += n
        except Exception:
            pass
    return total


CURRICULUM_CYCLE = ["js", "python", "web", "laravel", "security", "ai", "survival", "factory"]

def get_next_curriculum(reg):
    """اختيار المنهج التالي بالتناوب"""
    last = reg.get('last_curriculum_index', -1)
    next_idx = (last + 1) % len(CURRICULUM_CYCLE)
    return next_idx, CURRICULUM_CYCLE[next_idx]

def run_smart_learn(curriculum="all"):
    print(f"Running smart-learn (curriculum: {curriculum})...")
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "smart-learn.py"), "--curriculum", curriculum],
                      cwd=str(BASE_DIR), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("smart-learn stderr:", r.stderr[:500] if r.stderr else "none")
    else:
        print("smart-learn completed successfully.")
    return r.returncode == 0


def run_validate():
    print("Validating data...")
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "validate-data.py"), "--fix"],
                      cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def run_finetune():
    print("Running finetune...")
    sys.stdout.flush()
    # stdout يطبع مباشرة عشان نشوف تقدم التدريب بالـ logs
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "finetune.py")],
                      cwd=str(BASE_DIR), stderr=subprocess.PIPE, text=True, timeout=3600 * 4)
    if r.returncode != 0:
        print("finetune stderr:", r.stderr[:800] if r.stderr else "none")
    else:
        print("Finetune completed successfully.")
    return r.returncode == 0


def run_convert_to_onnx():
    print("Converting to ONNX...")
    sys.stdout.flush()
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "convert-to-onnx.py")],
                      cwd=str(BASE_DIR), stderr=subprocess.PIPE, text=True, timeout=600)
    if r.returncode != 0:
        print("convert-to-onnx stderr:", r.stderr[:500] if r.stderr else "none")
    else:
        print("ONNX conversion completed.")
    return r.returncode == 0


def run_prepare_chat_data():
    print("Preparing chat training data (ChatML)...")
    sys.stdout.flush()
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "prepare-chat-data.py")],
                      cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("prepare-chat-data stderr:", r.stderr[:500] if r.stderr else "none")
    return r.returncode == 0


def run_finetune_chat():
    print("Running finetune-chat (Qwen2.5-3B-Instruct)...")
    sys.stdout.flush()
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "finetune-chat.py")],
                      cwd=str(BASE_DIR), stderr=subprocess.PIPE, text=True, timeout=3600 * 24)
    if r.returncode != 0:
        print("finetune-chat stderr:", r.stderr[:800] if r.stderr else "none")
    else:
        print("Finetune-chat completed successfully.")
    return r.returncode == 0


def run_convert_to_gguf():
    print("Converting chat model to GGUF...")
    sys.stdout.flush()
    r = subprocess.run([sys.executable, str(TRAINING_DIR / "convert-to-gguf.py")],
                      cwd=str(BASE_DIR), stderr=subprocess.PIPE, text=True, timeout=3600)
    if r.returncode != 0:
        print("convert-to-gguf stderr:", r.stderr[:500] if r.stderr else "none")
    else:
        print("GGUF conversion completed.")
    return r.returncode == 0


def one_cycle():
    reg = load_registry()
    
    cur_name = 'all'
    if INTENSIVE_MODE:
        cur_idx, cur_name = get_next_curriculum(reg)
        print(f"=== Intensive Mode: Curriculum [{cur_idx+1}/{len(CURRICULUM_CYCLE)}] = {cur_name} ===")
        run_smart_learn(cur_name)
        reg['last_curriculum_index'] = cur_idx
    else:
        run_smart_learn()
    
    n = count_training_samples()
    print(f"Training samples count: {n} (min to train: {MIN_SAMPLES_TO_TRAIN})")
    sys.stdout.flush()
    
    if n < MIN_SAMPLES_TO_TRAIN:
        print("Not enough samples; skipping training.")
        return False
    if not run_validate():
        print("Validation failed or produced no output.")
    if not run_finetune():
        print("Finetune failed.")
        return False
    if not run_convert_to_onnx():
        print("ONNX conversion failed.")
        return False
    if TRAIN_CHAT_MODEL:
        if not run_prepare_chat_data():
            print("Prepare chat data failed; skipping chat training.")
        elif not run_finetune_chat():
            print("Finetune-chat failed.")
        else:
            run_convert_to_gguf()
    reg['last_training'] = datetime.now().isoformat()
    reg['last_sample_count'] = n
    if 'versions' not in reg:
        reg['versions'] = []
    version = len(reg['versions']) + 1
    reg['versions'].append({
        'version': version,
        'timestamp': reg['last_training'],
        'sample_count': n,
        'curriculum': cur_name if INTENSIVE_MODE else 'all'
    })
    save_registry(reg)
    print(f"=== Cycle done. Version: {version} | Samples: {n} ===")
    return True


def main():
    watch = '--watch' in sys.argv
    if watch:
        interval_h = INTERVAL_SEC / 3600
        mode = "INTENSIVE" if INTENSIVE_MODE else "Normal"
        print(f"Watch mode ({mode}): cycle every {interval_h}h. Ctrl+C to stop.")
        print(f"Curricula: {', '.join(CURRICULUM_CYCLE)}")
        sys.stdout.flush()
        cycle_num = 0
        while True:
            cycle_num += 1
            print(f"\n{'='*60}")
            print(f"Cycle {cycle_num} started at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"{'='*60}")
            sys.stdout.flush()
            try:
                one_cycle()
            except Exception as e:
                print(f"Cycle {cycle_num} error: {e}")
            print(f"Cycle {cycle_num} finished. Sleeping {interval_h}h...")
            sys.stdout.flush()
            time.sleep(INTERVAL_SEC)
    else:
        one_cycle()


if __name__ == "__main__":
    main()
