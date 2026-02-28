"""
🔄 Sync Manager — مزامنة البيانات والنماذج

المسؤوليات:
- مزامنة checkpoints من كل العقد → RTX 5090 (العقدة الرئيسية)
- مزامنة بيانات التدريب من Hostinger → RTX 5090
- نقل التدريبات السابقة إلى RTX 5090
- نسخ احتياطي يومي على هارد 16TB خارجي
- مراقبة حالة المزامنة

يعمل كـ background task يتحقق كل 60 ثانية.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ──────────── Config ────────────

SYNC_INTERVAL_SEC = int(os.getenv("SYNC_INTERVAL_SEC", "60"))
PRIMARY_NODE_URL = os.getenv("PRIMARY_NODE_URL", "")  # e.g. http://192.168.1.100:8010
ORCHESTRATOR_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "")
BACKUP_DRIVE_PATH = os.getenv("BACKUP_DRIVE_PATH", "/mnt/backup")  # 16TB external
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SYNC_STATE_FILE = DATA_DIR / "sync" / "sync_state.json"


# ──────────── Sync State ────────────

class SyncState:
    """Tracks what has been synced."""

    def __init__(self):
        self.synced_files: Dict[str, str] = {}  # path -> checksum
        self.last_sync: Optional[str] = None
        self.last_backup: Optional[str] = None
        self.sync_errors: List[Dict[str, Any]] = []
        self.stats = {
            "total_synced": 0,
            "total_bytes_synced": 0,
            "total_backups": 0,
            "failed_syncs": 0,
        }
        self._load()

    def _load(self):
        SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if SYNC_STATE_FILE.exists():
            try:
                data = json.loads(SYNC_STATE_FILE.read_text())
                self.synced_files = data.get("synced_files", {})
                self.last_sync = data.get("last_sync")
                self.last_backup = data.get("last_backup")
                self.stats = data.get("stats", self.stats)
            except Exception:
                pass

    def save(self):
        try:
            SYNC_STATE_FILE.write_text(json.dumps({
                "synced_files": self.synced_files,
                "last_sync": self.last_sync,
                "last_backup": self.last_backup,
                "stats": self.stats,
                "sync_errors": self.sync_errors[-20:],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def needs_sync(self, filepath: str, checksum: str) -> bool:
        return self.synced_files.get(filepath) != checksum

    def mark_synced(self, filepath: str, checksum: str, size: int):
        self.synced_files[filepath] = checksum
        self.last_sync = datetime.now(timezone.utc).isoformat()
        self.stats["total_synced"] += 1
        self.stats["total_bytes_synced"] += size

    def log_error(self, error: str):
        self.sync_errors.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "error": error,
        })
        self.stats["failed_syncs"] += 1
        if len(self.sync_errors) > 50:
            self.sync_errors = self.sync_errors[-50:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_sync": self.last_sync,
            "last_backup": self.last_backup,
            "stats": self.stats,
            "synced_files_count": len(self.synced_files),
            "recent_errors": self.sync_errors[-5:],
        }


# ──────────── File Helpers ────────────

def file_checksum(path: Path) -> str:
    """SHA256 checksum of a file (first 16 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def find_syncable_files(base_dirs: List[Path],
                        extensions: set = None) -> List[Path]:
    """Find files that should be synced."""
    if extensions is None:
        extensions = {
            ".pt", ".pth", ".safetensors", ".onnx",  # Models
            ".json", ".jsonl",  # Training data
            ".bin", ".npz", ".npy",  # Arrays
            ".db", ".sqlite",  # Databases
            ".log",  # Training logs
        }

    files = []
    for base in base_dirs:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix in extensions:
                files.append(f)
    return files


# ──────────── Sync to Primary (RTX 5090) ────────────

class SyncManager:
    """
    Manages syncing training data and checkpoints.
    
    Modes:
    1. Server → Primary: Hostinger pushes to RTX 5090
    2. Worker → Server → Primary: Workers upload to server, server relays to RTX 5090
    3. Backup: Primary backs up to 16TB external drive
    """

    def __init__(self):
        self.state = SyncState()
        self.is_running = False
        self._primary_reachable = False

    async def start(self):
        """Start the sync background loop."""
        self.is_running = True
        print("🔄 Sync Manager started")
        
        while self.is_running:
            try:
                await self._sync_cycle()
            except Exception as e:
                self.state.log_error(f"Sync cycle error: {e}")
                print(f"⚠️ Sync error: {e}")
            
            self.state.save()
            await asyncio.sleep(SYNC_INTERVAL_SEC)

    def stop(self):
        self.is_running = False
        self.state.save()
        print("🔄 Sync Manager stopped")

    async def _sync_cycle(self):
        """One sync cycle: check files, sync what's needed."""
        # 1. Check if primary is reachable
        self._primary_reachable = await self._check_primary()

        # 2. Collect files that need syncing
        sync_dirs = [
            DATA_DIR / "orchestrator" / "artifacts",
            DATA_DIR / "learning",
            DATA_DIR / "knowledge",
            Path("models"),
            Path("checkpoints"),
            Path("learning_data"),
        ]

        files = find_syncable_files(sync_dirs)
        pending = []

        for f in files:
            checksum = file_checksum(f)
            if self.state.needs_sync(str(f), checksum):
                pending.append((f, checksum))

        if not pending:
            return

        print(f"🔄 {len(pending)} files to sync")

        # 3. Sync to primary if reachable
        if self._primary_reachable and PRIMARY_NODE_URL:
            for filepath, checksum in pending:
                success = await self._push_to_primary(filepath, checksum)
                if success:
                    self.state.mark_synced(str(filepath), checksum, filepath.stat().st_size)

        # 4. Daily backup check
        await self._check_daily_backup()

    async def _check_primary(self) -> bool:
        """Check if primary node (RTX 5090) is reachable."""
        if not PRIMARY_NODE_URL:
            return False
        try:
            resp = requests.get(
                f"{PRIMARY_NODE_URL}/api/v1/orchestrator/health",
                headers={"X-Orchestrator-Token": ORCHESTRATOR_TOKEN},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def _push_to_primary(self, filepath: Path, checksum: str) -> bool:
        """Push a file to the primary node via HTTP upload."""
        if not PRIMARY_NODE_URL:
            return False

        try:
            with open(filepath, "rb") as f:
                resp = requests.post(
                    f"{PRIMARY_NODE_URL}/api/v1/training-data/sync-file",
                    files={"file": (filepath.name, f)},
                    data={
                        "relative_path": str(filepath),
                        "checksum": checksum,
                        "source": "hostinger",
                    },
                    headers={"X-Orchestrator-Token": ORCHESTRATOR_TOKEN},
                    timeout=120,
                )
                if resp.status_code == 200:
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"  📤 Synced: {filepath.name} ({size_mb:.1f}MB)")
                    return True
                else:
                    self.state.log_error(
                        f"Push failed for {filepath.name}: HTTP {resp.status_code}"
                    )
                    return False
        except Exception as e:
            self.state.log_error(f"Push failed for {filepath.name}: {e}")
            return False

    async def _check_daily_backup(self):
        """Check if daily backup is needed (runs on RTX 5090 only)."""
        backup_path = Path(BACKUP_DRIVE_PATH)
        if not backup_path.exists():
            return  # Not the RTX 5090 machine or drive not mounted

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_backup and self.state.last_backup.startswith(today):
            return  # Already backed up today

        print(f"💾 Starting daily backup to {backup_path}")
        try:
            await self._run_backup(backup_path)
            self.state.last_backup = datetime.now(timezone.utc).isoformat()
            self.state.stats["total_backups"] += 1
            print("💾 Daily backup complete ✅")
        except Exception as e:
            self.state.log_error(f"Backup failed: {e}")
            print(f"❌ Backup failed: {e}")

    async def _run_backup(self, backup_path: Path):
        """Run daily incremental backup."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = backup_path / "bi-ide-backup" / today

        # Source directories to back up
        sources = [
            DATA_DIR,
            Path("models"),
            Path("checkpoints"),
            Path("learning_data"),
            Path("hierarchy"),
            Path("ai"),
        ]

        for src in sources:
            if not src.exists():
                continue
            dest = target / src.name
            dest.mkdir(parents=True, exist_ok=True)

            # Use rsync if available, else shutil
            try:
                proc = await asyncio.create_subprocess_exec(
                    "rsync", "-a", "--delete", str(src) + "/", str(dest) + "/",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"rsync failed: {stderr.decode()}")
            except FileNotFoundError:
                # rsync not available, use shutil
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest, dirs_exist_ok=True)

        # Cleanup old backups (keep 14 daily + 8 weekly + 6 monthly)
        await self._cleanup_old_backups(backup_path / "bi-ide-backup")

    async def _cleanup_old_backups(self, backup_root: Path):
        """Rotation: keep 14 daily + 8 weekly + 6 monthly backups."""
        if not backup_root.exists():
            return

        all_backups = sorted(
            [d for d in backup_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )

        keep = set()

        # Keep last 14 daily
        for b in all_backups[:14]:
            keep.add(b.name)

        # Keep last 8 weekly (Sundays)
        weekly_count = 0
        for b in all_backups:
            try:
                dt = datetime.strptime(b.name, "%Y-%m-%d")
                if dt.weekday() == 6:  # Sunday
                    keep.add(b.name)
                    weekly_count += 1
                    if weekly_count >= 8:
                        break
            except ValueError:
                continue

        # Keep last 6 monthly (1st of month)
        monthly_count = 0
        for b in all_backups:
            if b.name.endswith("-01"):
                keep.add(b.name)
                monthly_count += 1
                if monthly_count >= 6:
                    break

        # Remove old backups
        for b in all_backups:
            if b.name not in keep:
                try:
                    shutil.rmtree(b)
                    print(f"  🗑️ Removed old backup: {b.name}")
                except Exception:
                    pass

    def get_status(self) -> Dict[str, Any]:
        """Get sync status."""
        return {
            **self.state.to_dict(),
            "primary_reachable": self._primary_reachable,
            "primary_url": PRIMARY_NODE_URL or "not configured",
            "backup_drive": BACKUP_DRIVE_PATH,
            "is_running": self.is_running,
        }


# ──────────── Singleton ────────────

sync_manager = SyncManager()
