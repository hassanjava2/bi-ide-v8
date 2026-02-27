#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - Training monitor.
- Training status, sample counts, model size
- Daily JSON report
- Optional webhook (Discord, Telegram)

Usage:
    python training/monitor.py
    python training/monitor.py --webhook https://...
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "training" / "output"
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "model-registry.json"
REPORT_PATH = OUTPUT_DIR / "monitor_report.json"


def count_samples():
    total = 0
    for p in OUTPUT_DIR.glob("*.json"):
        if p.name in ("validation_report.json", "monitor_report.json", "evaluation_report.json"):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            n = len(d) if isinstance(d, list) else len(d.get('samples', d.get('examples', [])))
            total += n
        except Exception:
            pass
    return total


def model_size_mb():
    onnx = MODELS_DIR / "bi-ai-onnx"
    if not onnx.exists():
        return 0
    total = sum(f.stat().st_size for f in onnx.rglob('*') if f.is_file())
    return round(total / 1024 / 1024, 1)


def get_registry():
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def run():
    report = {
        'timestamp': datetime.now().isoformat(),
        'samples': count_samples(),
        'model_size_mb': model_size_mb(),
        'registry': get_registry(),
        'output_dir': str(OUTPUT_DIR),
        'models_dir': str(MODELS_DIR)
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Samples:", report['samples'])
    print("Model size (MB):", report['model_size_mb'])
    print("Report:", REPORT_PATH)

    for i, arg in enumerate(sys.argv):
        if arg == '--webhook' and i + 1 < len(sys.argv):
            url = sys.argv[i + 1]
            try:
                body = json.dumps({'text': f"Bi IDE Training: {report['samples']} samples, model {report['model_size_mb']} MB"})
                urllib.request.urlopen(urllib.request.Request(url, data=body.encode(), headers={'Content-Type': 'application/json'}), timeout=5)
            except Exception as e:
                print("Webhook error:", e)
            break
    return report


if __name__ == "__main__":
    run()
