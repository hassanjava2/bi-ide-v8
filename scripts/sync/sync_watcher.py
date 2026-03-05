#!/usr/bin/env python3
"""
sync_watcher.py — يراقب مجلد التدريب ويرفع تلقائياً للـ VPS

يُشغّل على RTX 5090 كخدمة (systemd) أو بالطرفية.
كل ما ينتج LoRA model جديد أو بيانات تدريب جديدة → يرفع للـ VPS فوراً.

Usage: python3 sync_watcher.py
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────
WATCH_DIRS = [
    "/home/bi/training_data/models/finetuned",
    "/home/bi/training_data/data",
    "/home/bi/training_data/ingest",
]

VPS_HOST = "root@76.13.154.123"
VPS_SYNC_DIR = "/opt/bi-iq-app/shared_data"

CHECK_INTERVAL = 60  # seconds between checks
SYNC_SCRIPT = str(Path(__file__).parent / "sync_rtx_to_vps.sh")

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/bi-sync-watcher.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("sync_watcher")

# ─── State tracking ──────────────────────────────────────────
STATE_FILE = Path("/tmp/bi-sync-state.json")


def get_dir_hash(dir_path: str) -> str:
    """حساب hash سريع لمحتويات المجلد (أسماء الملفات + أحجام + تواريخ)"""
    h = hashlib.md5()
    try:
        for root, dirs, files in os.walk(dir_path):
            for f in sorted(files):
                filepath = os.path.join(root, f)
                try:
                    stat = os.stat(filepath)
                    h.update(f"{filepath}:{stat.st_size}:{int(stat.st_mtime)}".encode())
                except OSError:
                    pass
    except Exception:
        pass
    return h.hexdigest()


def load_state() -> dict:
    """تحميل حالة السنكرونايز السابقة"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    """حفظ حالة السنكرونايز"""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_sync(mode: str = "--full"):
    """تشغيل سكربت النسخ المتماثل"""
    logger.info(f"🔄 بدء النسخ المتماثل ({mode})...")

    try:
        result = subprocess.run(
            ["bash", SYNC_SCRIPT, mode],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        if result.returncode == 0:
            logger.info("✅ النسخ المتماثل اكتمل بنجاح")
        else:
            logger.error(f"❌ فشل النسخ: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("⏱️ انتهت مهلة النسخ (10 دقائق)")
    except FileNotFoundError:
        logger.error(f"❌ سكربت النسخ غير موجود: {SYNC_SCRIPT}")
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")


def check_for_changes() -> bool:
    """فحص إذا في تغييرات بالمجلدات المراقبة"""
    state = load_state()
    changed = False

    for dir_path in WATCH_DIRS:
        if not os.path.exists(dir_path):
            continue

        current_hash = get_dir_hash(dir_path)
        previous_hash = state.get(dir_path, "")

        if current_hash != previous_hash:
            logger.info(f"📝 تغييرات مكتشفة في: {dir_path}")
            state[dir_path] = current_hash
            changed = True

    if changed:
        save_state(state)

    return changed


def main():
    logger.info("=" * 50)
    logger.info("🔍 بدء مراقب النسخ المتماثل")
    logger.info(f"   المجلدات: {len(WATCH_DIRS)}")
    logger.info(f"   VPS: {VPS_HOST}")
    logger.info(f"   فترة الفحص: {CHECK_INTERVAL}s")
    logger.info("=" * 50)

    # Initial sync
    logger.info("📋 نسخ أولي...")
    run_sync("--full")

    # Watch loop
    sync_count = 0
    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            if check_for_changes():
                sync_count += 1
                logger.info(f"🔄 تغييرات مكتشفة — نسخ #{sync_count}")
                run_sync("--full")
            else:
                logger.debug("لا تغييرات")

        except KeyboardInterrupt:
            logger.info("⛔ إيقاف المراقب")
            break
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
