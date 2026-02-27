#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - تدريب محدود بوقت (مثلاً 16 ساعة)
تشغّله على السيرفر وتطفّي حاسوبك — السيرفر يكمل حتى انتهاء المدة.

الاستخدام:
    python training/run-16h.py              # 16 ساعة افتراضي
    python training/run-16h.py --hours 12
    python training/run-16h.py --hours 24

على السيرفر (يبقى يشتغل بعد إغلاق SSH):
    nohup python3 training/run-16h.py --hours 16 > logs/16h.log 2>&1 &
    tail -f logs/16h.log
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training"


def main():
    parser = argparse.ArgumentParser(description="تدريب لمدة محددة (ساعات)")
    parser.add_argument("--hours", type=float, default=16, help="عدد الساعات (افتراضي 16)")
    args = parser.parse_args()
    hours = max(0.5, min(168, args.hours))  # بين 30 دقيقة و 7 أيام
    end_time = datetime.now() + timedelta(hours=hours)
    script = TRAINING_DIR / "continuous-train.py"
    if not script.exists():
        print("continuous-train.py not found")
        sys.exit(1)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cycle = 0
    print("=" * 60)
    print(f"Bi IDE – تدريب محدود بوقت: {hours} ساعة")
    print(f"ينتهي عند: {end_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    sys.stdout.flush()
    while datetime.now() < end_time:
        cycle += 1
        remaining = (end_time - datetime.now()).total_seconds() / 3600
        print(f"\n[{datetime.now().strftime('%H:%M')}] دورة {cycle} (متبقي ~{remaining:.1f}h)")
        sys.stdout.flush()
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            env=env,
            timeout=int(remaining * 3600) + 3600,
        )
        if r.returncode != 0:
            print(f"دورة {cycle} خرجت برمز {r.returncode}")
        if datetime.now() >= end_time:
            break
        # استراحة قصيرة بين الدورات
        time.sleep(60)
    print("\n" + "=" * 60)
    print(f"انتهت المدة. عدد الدورات: {cycle}")
    print("آخر نموذج في: models/bi-ai-onnx/ و models/model-registry.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
