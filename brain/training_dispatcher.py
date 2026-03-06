#!/usr/bin/env python3
"""
training_dispatcher.py — المرسل 🚀

يوزع الكبسولات على الحاسبات المتوفرة للتدريب:
  1. يقرأ سجل الحاسبات (training_fleet.yaml)
  2. يشوف أي كبسولة تحتاج تدريب
  3. يحزم البيانات ويرسلها SSH/rsync
  4. يشغل training_worker.py عن بعد
  5. يسحب الـ adapter لما يكمل
  6. يحمّله محلياً

الاستخدام:
  python3 training_dispatcher.py                    # تدريب كل الكبسولات الجاهزة
  python3 training_dispatcher.py --capsule python   # كبسولة واحدة
  python3 training_dispatcher.py --dry-run          # عرض بدون تنفيذ
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dispatcher] %(message)s",
)
logger = logging.getLogger("dispatcher")

# المسارات
ROOT = Path(__file__).parent.parent
CAPSULES_DIR = ROOT / "capsules"
FLEET_CONFIG = ROOT / "config" / "training_fleet.yaml"
WORKER_SCRIPT = ROOT / "brain" / "training_worker.py"
MIN_SAMPLES = 20
TRAINING_LOCK = ".training_in_progress"  # ملف قفل أثناء التدريب
import shutil


class WorkerSlot:
    """مكان فارغ على حاسبة بعيدة"""

    def __init__(self, worker_config: dict):
        self.name = worker_config["name"]
        self.host = worker_config["host"]
        self.user = worker_config.get("user", "bi")
        self.vram_gb = worker_config.get("vram_gb", 24)
        self.max_concurrent = worker_config.get("max_concurrent", 1)
        self.role = worker_config.get("role", "training")
        self.workspace = worker_config.get("workspace", "~/bi-ide-v8")
        self.venv = worker_config.get("venv", "python3")
        self.ssh_key = worker_config.get("ssh_key", "~/.ssh/id_rsa")
        self.active_jobs = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.active_jobs < self.max_concurrent

    def acquire(self) -> bool:
        with self._lock:
            if self.active_jobs < self.max_concurrent:
                self.active_jobs += 1
                return True
            return False

    def release(self):
        with self._lock:
            self.active_jobs = max(0, self.active_jobs - 1)


class TrainingDispatcher:
    """المرسل — يوزع الكبسولات على الحاسبات"""

    def __init__(self, fleet_path: Path = FLEET_CONFIG):
        self.fleet_config = yaml.safe_load(fleet_path.read_text())
        self.workers = [WorkerSlot(w) for w in self.fleet_config.get("workers", [])]
        self.base_model = self.fleet_config.get("base_model", "Qwen/Qwen2.5-1.5B")
        self.training_config = self.fleet_config.get("training", {})
        self.results = []
        self._active_inboxes = {}  # capsule_id → inbox_path

        logger.info(f"📡 Fleet: {len(self.workers)} workers")
        for w in self.workers:
            logger.info(f"   {w.name}: {w.host} ({w.vram_gb}GB, max {w.max_concurrent} concurrent)")

    def find_capsules_to_train(self, specific: str = None) -> list:
        """يلكي الكبسولات الي تحتاج تدريب"""
        capsules = []
        if not CAPSULES_DIR.exists():
            logger.warning(f"No capsules dir: {CAPSULES_DIR}")
            return capsules

        for cap_dir in sorted(CAPSULES_DIR.iterdir()):
            if not cap_dir.is_dir():
                continue
            cap_id = cap_dir.name
            if specific and cap_id != specific:
                continue

            data_dir = cap_dir / "data"
            if not data_dir.exists():
                continue

            # عد العينات
            count = 0
            for f in data_dir.glob("*.jsonl"):
                if f.name == "merged_train.jsonl":
                    continue
                count += sum(1 for line in open(f) if line.strip())

            if count >= MIN_SAMPLES:
                capsules.append({"id": cap_id, "dir": cap_dir, "samples": count})

        logger.info(f"🎯 Found {len(capsules)} capsules ready for training")
        return capsules

    def _find_worker(self) -> WorkerSlot | None:
        """يلكي حاسبة فاضية"""
        for worker in self.workers:
            if worker.available:
                return worker
        return None

    def _ssh_cmd(self, worker: WorkerSlot, cmd: str, timeout: int = 600) -> tuple:
        """ينفذ أمر SSH"""
        ssh = [
            "ssh", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            f"{worker.user}@{worker.host}",
            cmd
        ]
        try:
            result = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"

    def _rsync_to(self, worker: WorkerSlot, src: Path, dest: str) -> bool:
        """يرسل ملفات rsync"""
        cmd = [
            "rsync", "-az", "--delete",
            str(src) + "/",
            f"{worker.user}@{worker.host}:{dest}/"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except Exception:
            return False

    def _rsync_from(self, worker: WorkerSlot, src: str, dest: Path) -> bool:
        """يسحب ملفات rsync"""
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync", "-az",
            f"{worker.user}@{worker.host}:{src}/",
            str(dest) + "/"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except Exception:
            return False

    # ══════════════════════════════════════════════
    # صندوق الوارد — حماية البيانات أثناء التدريب
    # ══════════════════════════════════════════════

    def _activate_inbox(self, cap_dir: Path, cap_id: str):
        """تفعيل صندوق الوارد — الكشافة تكتب هنا أثناء التدريب"""
        inbox = cap_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        # ملف قفل يخبر الكشافة تكتب بـ inbox/ بدال data/
        (cap_dir / TRAINING_LOCK).write_text(json.dumps({
            "started": datetime.now().isoformat(),
            "capsule": cap_id,
        }))
        self._active_inboxes[cap_id] = inbox
        logger.info(f"   📥 Inbox activated for {cap_id}")

    def _merge_inbox(self, cap_dir: Path, cap_id: str):
        """دمج صندوق الوارد — البيانات الجديدة تنتقل لـ data/"""
        inbox = cap_dir / "inbox"
        data_dir = cap_dir / "data"
        lock_file = cap_dir / TRAINING_LOCK

        if not inbox.exists():
            return 0

        # نقل كل JSONL من inbox → data
        moved = 0
        for f in inbox.glob("*.jsonl"):
            dest = data_dir / f"inbox_{f.name}"
            shutil.move(str(f), str(dest))
            moved += 1

        # حذف القفل
        if lock_file.exists():
            lock_file.unlink()

        # حذف المجلد الفارغ
        if inbox.exists() and not any(inbox.iterdir()):
            inbox.rmdir()

        self._active_inboxes.pop(cap_id, None)

        if moved > 0:
            logger.info(f"   📬 Merged {moved} inbox files → data/ for {cap_id}")
        return moved

    @staticmethod
    def is_training(cap_dir: Path) -> bool:
        """هل الكبسولة قيد التدريب؟ (يستخدمها الكشافة)"""
        return (cap_dir / TRAINING_LOCK).exists()

    @staticmethod
    def get_write_dir(cap_dir: Path) -> Path:
        """وين الكشافة تكتب؟ inbox/ لو قيد التدريب، data/ لو لا"""
        if (cap_dir / TRAINING_LOCK).exists():
            inbox = cap_dir / "inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            return inbox
        return cap_dir / "data"

    # ══════════════════════════════════════════════
    # إرسال كبسولة للتدريب
    # ══════════════════════════════════════════════

    def dispatch_capsule(self, capsule: dict, worker: WorkerSlot) -> dict:
        """يرسل كبسولة لحاسبة بعيدة ويدربها (مع حماية inbox)"""
        cap_id = capsule["id"]
        cap_dir = capsule["dir"]

        logger.info(f"🛫 {cap_id} → {worker.name} ({capsule['samples']} samples)")

        if not worker.acquire():
            return {"capsule": cap_id, "status": "error", "reason": "worker busy"}

        # تفعيل صندوق الوارد قبل الإرسال
        self._activate_inbox(cap_dir, cap_id)

        try:
            t0 = time.time()

            # === هل الحاسبة محلية؟ ===
            is_local = worker.host in ("localhost", "127.0.0.1", "192.168.1.164")

            if is_local:
                result = self._train_local(capsule, worker)
            else:
                result = self._train_remote(capsule, worker)

            elapsed = time.time() - t0
            result["minutes"] = round(elapsed / 60, 1)

            # دمج صندوق الوارد بعد انتهاء التدريب
            inbox_merged = self._merge_inbox(cap_dir, cap_id)
            result["inbox_merged"] = inbox_merged

            if result.get("status") == "completed":
                logger.info(f"   ✅ {cap_id}: done in {elapsed/60:.1f}min on {worker.name}")

            return result

        except Exception as e:
            logger.error(f"   ❌ {cap_id}: {e}")
            # حتى لو فشل — ندمج الـ inbox
            self._merge_inbox(cap_dir, cap_id)
            return {"capsule": cap_id, "status": "error", "reason": str(e)}
        finally:
            worker.release()

    def _train_remote(self, capsule: dict, worker: WorkerSlot) -> dict:
        """تدريب على حاسبة بعيدة عبر SSH"""
        cap_id = capsule["id"]
        cap_dir = capsule["dir"]

        remote_dir = f"/tmp/bi-capsule-{cap_id}"
        remote_data = f"{remote_dir}/data"
        remote_model = f"{remote_dir}/model"

        # إنشاء المجلدات
        self._ssh_cmd(worker, f"mkdir -p {remote_data} {remote_model}")

        # rsync البيانات
        logger.info(f"   📦 Syncing data to {worker.name}...")
        if not self._rsync_to(worker, cap_dir / "data", remote_data):
            return {"capsule": cap_id, "status": "error", "reason": "rsync data failed"}

        # rsync الموديل الموجود (لو فيه)
        if (cap_dir / "model" / "config.json").exists():
            self._rsync_to(worker, cap_dir / "model", remote_model)

        # rsync سكربت العامل
        self._ssh_cmd(worker, f"mkdir -p {remote_dir}/brain")
        subprocess.run([
            "scp", str(WORKER_SCRIPT),
            f"{worker.user}@{worker.host}:{remote_dir}/brain/training_worker.py"
        ], capture_output=True, timeout=30)

        # تشغيل التدريب
        logger.info(f"   🏋️ Training on {worker.name}...")
        train_cmd = (
            f"cd {remote_dir} && "
            f"{worker.venv} brain/training_worker.py "
            f"--capsule-dir {remote_dir} "
            f"--base-model {self.base_model}"
        )
        code, stdout, stderr = self._ssh_cmd(worker, train_cmd, timeout=3600)

        if code != 0:
            logger.error(f"   ❌ Training failed: {stderr[:200]}")
            return {"capsule": cap_id, "status": "error", "reason": stderr[:200]}

        # سحب الـ adapter
        logger.info(f"   🛬 Pulling adapter from {worker.name}...")
        if not self._rsync_from(worker, remote_model, cap_dir / "model"):
            return {"capsule": cap_id, "status": "error", "reason": "rsync adapter failed"}

        # سحب result.json
        self._rsync_from(worker, f"{remote_dir}/result.json", cap_dir)

        # تنظيف
        self._ssh_cmd(worker, f"rm -rf {remote_dir}")

        return {"capsule": cap_id, "status": "completed", "worker": worker.name}

    def _train_local(self, capsule: dict, worker: WorkerSlot) -> dict:
        """تدريب محلي — نفس الحاسبة"""
        cap_id = capsule["id"]
        cap_dir = capsule["dir"]

        train_cmd = (
            f"cd {ROOT} && "
            f"{worker.venv} -m brain.training_worker "
            f"--capsule-dir {cap_dir} "
            f"--base-model {self.base_model}"
        )
        code, stdout, stderr = self._ssh_cmd(worker, train_cmd, timeout=3600)

        if code != 0:
            return {"capsule": cap_id, "status": "error", "reason": stderr[:200]}

        # parse result
        try:
            result = json.loads(stdout.strip().split("\n")[-1])
            return result
        except Exception:
            return {"capsule": cap_id, "status": "completed", "worker": worker.name}

    def dispatch_all(self, capsules: list, dry_run: bool = False) -> list:
        """يوزع كل الكبسولات بالتوازي"""
        if dry_run:
            logger.info("🔍 DRY RUN — عرض بدون تنفيذ:")
            for cap in capsules:
                worker = self._find_worker()
                w_name = worker.name if worker else "⏳ waiting"
                logger.info(f"  {cap['id']} ({cap['samples']} samples) → {w_name}")
            return []

        logger.info(f"🚀 Dispatching {len(capsules)} capsules to {len(self.workers)} workers")

        max_threads = sum(w.max_concurrent for w in self.workers)
        results = []

        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            futures = {}

            for capsule in capsules:
                # انتظر حاسبة فاضية
                while True:
                    worker = self._find_worker()
                    if worker:
                        break
                    time.sleep(5)

                future = pool.submit(self.dispatch_capsule, capsule, worker)
                futures[future] = capsule

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                cap = futures[future]
                status = result.get("status", "unknown")
                if status == "completed":
                    logger.info(f"✅ {cap['id']} completed")
                else:
                    logger.warning(f"⚠️ {cap['id']}: {status}")

        # ملخص
        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") == "error")
        skipped = sum(1 for r in results if r.get("status") == "skip")

        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Results: {completed} ✅ | {failed} ❌ | {skipped} ⏭️")
        logger.info(f"{'='*50}")

        return results


def main():
    parser = argparse.ArgumentParser(description="BI-IDE Training Dispatcher")
    parser.add_argument("--capsule", default=None, help="Train specific capsule")
    parser.add_argument("--fleet", default=str(FLEET_CONFIG), help="Fleet config path")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    fleet_path = Path(args.fleet)
    if not fleet_path.exists():
        logger.error(f"❌ Fleet config not found: {fleet_path}")
        sys.exit(1)

    dispatcher = TrainingDispatcher(fleet_path)
    capsules = dispatcher.find_capsules_to_train(specific=args.capsule)

    if not capsules:
        logger.info("No capsules need training")
        return

    results = dispatcher.dispatch_all(capsules, dry_run=args.dry_run)

    # حفظ النتائج
    if results:
        report = ROOT / "logs" / f"dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"📄 Report saved: {report}")


if __name__ == "__main__":
    main()
