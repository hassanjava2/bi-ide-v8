#!/usr/bin/env python3
"""
emergency_downloader.py — تنزيل بيانات الطوارئ 🚨📥

قبل ما ينقطع النت — ننزل كل شي:
  1. Wikipedia (عربي + إنجليزي)
  2. Llama 3 70B GGUF
  3. Mistral 7B GGUF
  4. Stack Overflow Data Dump
  5. ArXiv Papers
  6. CVE Database
  7. OpenStreetMap Iraq
  8. Project Gutenberg (كتب)

يشتغل بالخلفية + يكمل من وينما وقف (resume)
"""

import json
import os
import hashlib
import logging
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("emergency_dl")

PROJECT_ROOT = Path(__file__).parent.parent
DOWNLOAD_ROOT = PROJECT_ROOT / "emergency_data"
PROGRESS_FILE = DOWNLOAD_ROOT / ".download_progress.json"


@dataclass
class DownloadItem:
    """عنصر تنزيل"""
    name: str
    url: str
    filename: str
    size_gb: float
    priority: int          # 0 = أعلى
    category: str
    description: str
    completed: bool = False
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_mbps: float = 0
    started_at: str = ""
    finished_at: str = ""


# ═══════════════════════════════════════════════════════════
# قائمة التنزيلات
# ═══════════════════════════════════════════════════════════

DOWNLOADS: List[Dict] = [
    # 🔴 P0 — إلزامي
    {
        "name": "Mistral 7B GGUF (Q4)",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "models/mistral-7b-Q4.gguf",
        "size_gb": 4.4, "priority": 0, "category": "models",
        "description": "نموذج خفيف يشتغل على أي جهاز",
    },
    {
        "name": "Wikipedia Arabic (Dump)",
        "url": "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2",
        "filename": "knowledge/ar_wikipedia.xml.bz2",
        "size_gb": 2.5, "priority": 0, "category": "knowledge",
        "description": "ويكيبيديا العربية كاملة",
    },
    {
        "name": "Wikipedia English (Dump)",
        "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        "filename": "knowledge/en_wikipedia.xml.bz2",
        "size_gb": 22, "priority": 0, "category": "knowledge",
        "description": "ويكيبيديا الإنجليزية كاملة",
    },
    # 🟡 P1 — مهم
    {
        "name": "Llama 3 8B GGUF (Q4)",
        "url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "filename": "models/llama3-8b-Q4.gguf",
        "size_gb": 4.9, "priority": 1, "category": "models",
        "description": "Llama 3 — ذكي جداً للحجم",
    },
    {
        "name": "CVE Database (NVD)",
        "url": "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-recent.json.gz",
        "filename": "security/cve_recent.json.gz",
        "size_gb": 0.05, "priority": 1, "category": "security",
        "description": "ثغرات أمنية حديثة",
    },
    # 🟢 P2 — مفيد
    {
        "name": "Project Gutenberg (Books)",
        "url": "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
        "filename": "books/gutenberg_catalog.csv",
        "size_gb": 0.01, "priority": 2, "category": "books",
        "description": "كتالوج كتب مجانية",
    },
]


class EmergencyDownloader:
    """
    منزّل بيانات الطوارئ

    - تنزيل متوازي (3 threads)
    - يكمل من وينما وقف (resume)
    - يسجّل التقدم
    - ترتيب بالأولوية
    """

    def __init__(self, download_dir: Path = None, max_parallel: int = 3):
        self.download_dir = download_dir or DOWNLOAD_ROOT
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.items: List[DownloadItem] = []
        self.active_downloads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._load_progress()
        self._init_items()

    def _init_items(self):
        """تهيئة قائمة التنزيلات"""
        for dl in DOWNLOADS:
            # تجاهل المكتملة
            existing = next((i for i in self.items if i.name == dl["name"]), None)
            if existing and existing.completed:
                continue

            filepath = self.download_dir / dl["filename"]
            item = DownloadItem(
                name=dl["name"],
                url=dl["url"],
                filename=dl["filename"],
                size_gb=dl["size_gb"],
                priority=dl["priority"],
                category=dl["category"],
                description=dl["description"],
                downloaded_bytes=filepath.stat().st_size if filepath.exists() else 0,
                total_bytes=int(dl["size_gb"] * 1024**3),
            )

            if not existing:
                self.items.append(item)

        # ترتيب بالأولوية
        self.items.sort(key=lambda x: (x.completed, x.priority))

    def _load_progress(self):
        """تحميل حالة التقدم"""
        if PROGRESS_FILE.exists():
            try:
                data = json.loads(PROGRESS_FILE.read_text())
                for d in data:
                    self.items.append(DownloadItem(**d))
            except Exception:
                pass

    def _save_progress(self):
        """حفظ حالة التقدم"""
        try:
            data = []
            for item in self.items:
                data.append({
                    "name": item.name, "url": item.url,
                    "filename": item.filename, "size_gb": item.size_gb,
                    "priority": item.priority, "category": item.category,
                    "description": item.description, "completed": item.completed,
                    "downloaded_bytes": item.downloaded_bytes,
                    "total_bytes": item.total_bytes,
                    "started_at": item.started_at, "finished_at": item.finished_at,
                })
            PROGRESS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Save progress error: {e}")

    def _download_file(self, item: DownloadItem):
        """تنزيل ملف واحد (مع resume)"""
        filepath = self.download_dir / item.filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        item.started_at = datetime.now().isoformat()
        start_time = time.time()

        try:
            # فحص حجم الملف على السيرفر
            req = Request(item.url, method="HEAD")
            req.add_header("User-Agent", "BI-IDE-EmergencyDownloader/1.0")
            try:
                response = urlopen(req, timeout=15)
                total = int(response.headers.get("Content-Length", 0))
                if total > 0:
                    item.total_bytes = total
            except Exception:
                pass

            # Resume support
            existing_size = filepath.stat().st_size if filepath.exists() else 0
            if existing_size > 0 and existing_size < item.total_bytes:
                logger.info(f"▶️ Resuming {item.name} from {existing_size / (1024**3):.1f} GB")

            # wget for resume support (more reliable)
            cmd = [
                "wget", "-c",            # continue/resume
                "--progress=dot:giga",
                "-O", str(filepath),
                "--user-agent=BI-IDE-EmergencyDownloader/1.0",
                "--timeout=30",
                "--tries=5",
                item.url,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*12)

            if filepath.exists():
                item.downloaded_bytes = filepath.stat().st_size
                item.completed = True
                item.finished_at = datetime.now().isoformat()
                elapsed = time.time() - start_time
                if elapsed > 0:
                    item.speed_mbps = round(item.downloaded_bytes / (1024**2) / elapsed, 1)
                logger.info(f"✅ {item.name}: {item.downloaded_bytes / (1024**3):.1f} GB in {elapsed:.0f}s")

        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ {item.name}: timeout (will resume next run)")
        except FileNotFoundError:
            # wget ما موجود — نستخدم curl
            try:
                cmd = [
                    "curl", "-L", "-C", "-",
                    "-o", str(filepath),
                    "-A", "BI-IDE-EmergencyDownloader/1.0",
                    "--connect-timeout", "30",
                    "--retry", "5",
                    item.url,
                ]
                subprocess.run(cmd, capture_output=True, timeout=3600*12)
                if filepath.exists():
                    item.downloaded_bytes = filepath.stat().st_size
                    item.completed = True
                    item.finished_at = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"❌ {item.name}: {e}")
        except Exception as e:
            logger.error(f"❌ {item.name}: {e}")

        with self._lock:
            self._save_progress()

    def start_downloads(self, priority: int = None):
        """بدء التنزيلات"""
        pending = [i for i in self.items if not i.completed]
        if priority is not None:
            pending = [i for i in pending if i.priority <= priority]

        if not pending:
            print("✅ كل البيانات منزلة!")
            return

        print(f"📥 Emergency Download — {len(pending)} files pending\n")

        for item in pending:
            pct = (item.downloaded_bytes / max(item.total_bytes, 1)) * 100
            status = f"▶️ {pct:.0f}%" if item.downloaded_bytes > 0 else "⏳"
            print(f"  {status} [{item.category}] {item.name} ({item.size_gb:.1f} GB) — P{item.priority}")

        print(f"\nStarting {min(self.max_parallel, len(pending))} parallel downloads...\n")

        threads = []
        for item in pending[:self.max_parallel]:
            t = threading.Thread(target=self._download_file, args=(item,), daemon=True)
            t.name = item.name
            threads.append(t)
            self.active_downloads[item.name] = t
            t.start()
            logger.info(f"📥 Started: {item.name}")

        # انتظر كل الأولوية الأعلى
        for t in threads:
            t.join()

        # تابع الباقي
        remaining = [i for i in pending[self.max_parallel:] if not i.completed]
        for item in remaining:
            self._download_file(item)

    def get_status(self) -> Dict:
        """حالة التنزيلات"""
        total_gb = sum(i.size_gb for i in self.items)
        done_gb = sum(i.size_gb for i in self.items if i.completed)
        return {
            "total_items": len(self.items),
            "completed": sum(1 for i in self.items if i.completed),
            "pending": sum(1 for i in self.items if not i.completed),
            "total_gb": round(total_gb, 1),
            "downloaded_gb": round(done_gb, 1),
            "progress_pct": round(done_gb / max(total_gb, 0.1) * 100, 1),
            "items": [
                {
                    "name": i.name, "category": i.category,
                    "size_gb": i.size_gb, "priority": i.priority,
                    "completed": i.completed,
                    "pct": round(i.downloaded_bytes / max(i.total_bytes, 1) * 100, 1),
                }
                for i in self.items
            ],
        }

    def quick_status(self) -> str:
        """حالة سريعة"""
        s = self.get_status()
        lines = [f"📥 Emergency Data: {s['downloaded_gb']:.1f}/{s['total_gb']:.1f} GB ({s['progress_pct']:.0f}%)\n"]
        for i in s["items"]:
            icon = "✅" if i["completed"] else f"⏳ {i['pct']:.0f}%"
            lines.append(f"  {icon} P{i['priority']} [{i['category']}] {i['name']} ({i['size_gb']:.1f}GB)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

downloader = EmergencyDownloader()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BI-IDE Emergency Data Downloader")
    parser.add_argument("--start", action="store_true", help="Start downloading P0")
    parser.add_argument("--all", action="store_true", help="Download all priorities")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    if args.status or (not args.start and not args.all):
        print(downloader.quick_status())
    elif args.start:
        print("🚨 Emergency Download — P0 (Critical) only\n")
        downloader.start_downloads(priority=0)
    elif args.all:
        print("🚨 Emergency Download — ALL priorities\n")
        downloader.start_downloads()
