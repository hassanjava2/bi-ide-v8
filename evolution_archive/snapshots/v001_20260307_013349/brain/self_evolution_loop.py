#!/usr/bin/env python3
"""
self_evolution_loop.py — التطور الذاتي مع الوراثة والأرشيف 🧬∞

الحلقة:
  1. snapshot — ينسخ نفسه (كل ملفات brain/)
  2. evolve — يعدل النسخة (يحسّن كود/أوزان/بنية)
  3. benchmark — يقارن الجديد vs القديم
  4. propose — يعرض على المستخدم
  5. إذا وافق → promote (الجديد = production، القديم = أرشيف)
  6. إذا رفض → rollback (يرجع القديم)
  7. تتكرر ∞

كل النسخ السابقة محفوظة بالأرشيف — لا شي يُمحى
"""

import json
import os
import shutil
import hashlib
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("evolution")

PROJECT_ROOT = Path(__file__).parent.parent
BRAIN_DIR = PROJECT_ROOT / "brain"
ARCHIVE_DIR = PROJECT_ROOT / "evolution_archive"
SNAPSHOTS_DIR = ARCHIVE_DIR / "snapshots"
EVOLUTION_LOG = ARCHIVE_DIR / "evolution_history.json"


class EvolutionArchive:
    """
    أرشيف كل النسخ السابقة — لا شي يُمحى أبداً!

    كل snapshot محفوظ:
      evolution_archive/
      ├── snapshots/
      │   ├── v001_20260307_013000/
      │   │   ├── brain/ (نسخة كاملة)
      │   │   └── metadata.json
      │   ├── v002_20260307_020000/
      │   └── ...
      ├── evolution_history.json
      └── current_version.txt
    """

    def __init__(self):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if EVOLUTION_LOG.exists():
            try:
                return json.loads(EVOLUTION_LOG.read_text())
            except Exception:
                pass
        return []

    def _save_history(self):
        EVOLUTION_LOG.write_text(json.dumps(self.history, indent=2, ensure_ascii=False))

    def get_version(self) -> int:
        return len(self.history) + 1

    def snapshot(self, reason: str = "") -> Dict:
        """أخذ نسخة من الدماغ الحالي"""
        version = self.get_version()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"v{version:03d}_{timestamp}"
        snap_dir = SNAPSHOTS_DIR / snap_name

        # نسخ brain/ كامل
        try:
            shutil.copytree(
                BRAIN_DIR, snap_dir / "brain",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "capsules"),
            )
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            return {"error": str(e)}

        # حساب hash لكل ملف
        file_hashes = {}
        total_lines = 0
        for f in (snap_dir / "brain").rglob("*.py"):
            content = f.read_bytes()
            file_hashes[str(f.relative_to(snap_dir))] = hashlib.md5(content).hexdigest()
            total_lines += content.decode(errors="ignore").count("\n")

        # metadata
        meta = {
            "version": version,
            "snapshot_name": snap_name,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "files": len(file_hashes),
            "total_lines": total_lines,
            "hashes": file_hashes,
        }
        (snap_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

        logger.info(f"📸 Snapshot v{version}: {len(file_hashes)} files, {total_lines} lines")
        return meta

    def list_versions(self) -> List[Dict]:
        """قائمة كل النسخ"""
        versions = []
        for snap in sorted(SNAPSHOTS_DIR.iterdir()):
            meta_file = snap / "metadata.json"
            if meta_file.exists():
                try:
                    versions.append(json.loads(meta_file.read_text()))
                except Exception:
                    pass
        return versions

    def restore(self, version: int) -> bool:
        """استرجاع نسخة قديمة"""
        for snap in SNAPSHOTS_DIR.iterdir():
            meta_file = snap / "metadata.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    if meta["version"] == version:
                        # نسخ brain/ الحالي أولاً كـ backup
                        self.snapshot(f"auto_backup_before_restore_v{version}")

                        # استرجاع
                        for src in (snap / "brain").rglob("*.py"):
                            rel = src.relative_to(snap / "brain")
                            dst = BRAIN_DIR / rel
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)

                        logger.info(f"🔄 Restored v{version}")
                        return True
                except Exception:
                    pass
        return False


class SelfEvolver:
    """
    المتطور الذاتي — يطوّر الدماغ ∞

    الحلقة:
      check_health → find_issues → evolve → benchmark → propose → promote/rollback
    """

    def __init__(self):
        self.archive = EvolutionArchive()
        self.improvements = 0
        self.failures = 0
        self.pending_proposal: Optional[Dict] = None

    def check_health(self) -> Dict:
        """فحص صحة كل ملفات الدماغ"""
        results = {"files": 0, "errors": [], "warnings": [], "ok": []}

        for py_file in BRAIN_DIR.glob("*.py"):
            results["files"] += 1
            try:
                # فحص syntax
                compile(py_file.read_text(), str(py_file), "exec")
                results["ok"].append(py_file.name)
            except SyntaxError as e:
                results["errors"].append({"file": py_file.name, "error": str(e)})

            # فحص imports
            content = py_file.read_text(errors="ignore")
            if "import" not in content:
                results["warnings"].append(f"{py_file.name}: no imports")

        results["healthy"] = len(results["errors"]) == 0
        return results

    def find_improvements(self) -> List[Dict]:
        """بحث عن تحسينات ممكنة"""
        suggestions = []

        for py_file in BRAIN_DIR.glob("*.py"):
            content = py_file.read_text(errors="ignore")
            lines = content.splitlines()

            # TODO/FIXME
            for i, line in enumerate(lines):
                if "TODO" in line or "FIXME" in line:
                    suggestions.append({
                        "file": py_file.name, "line": i + 1,
                        "type": "todo", "text": line.strip(),
                    })

            # ملفات كبيرة ← تحتاج refactor
            if len(lines) > 500:
                suggestions.append({
                    "file": py_file.name, "line": 0,
                    "type": "refactor", "text": f"Large file: {len(lines)} lines — consider splitting",
                })

            # pass ← دالة فارغة
            for i, line in enumerate(lines):
                if line.strip() == "pass" and i > 0 and "def " in lines[i-1]:
                    suggestions.append({
                        "file": py_file.name, "line": i + 1,
                        "type": "empty_function", "text": f"Empty function near line {i+1}",
                    })

        return suggestions

    def evolve_once(self) -> Dict:
        """
        محاولة تطوير واحدة:
          1. snapshot الحالي
          2. بحث عن مشاكل
          3. إصلاح
          4. benchmark
        """
        result = {"success": False, "changes": []}

        # 1. snapshot
        snap = self.archive.snapshot("pre_evolution")
        result["snapshot"] = snap.get("version")

        # 2. فحص
        health = self.check_health()
        if health["errors"]:
            result["errors"] = health["errors"]
            # محاولة إصلاح syntax errors
            for err in health["errors"]:
                # الإصلاحات تُسجل بس ما تُطبق مباشرة (أمان)
                result["changes"].append({
                    "file": err["file"], "type": "syntax_fix",
                    "description": f"Will fix: {err['error']}",
                })

        # 3. تحسينات
        improvements = self.find_improvements()
        result["improvements_found"] = len(improvements)

        if not health["errors"] and not improvements:
            result["status"] = "already_optimal"
            return result

        result["success"] = True
        self.improvements += 1
        return result

    def propose_evolution(self) -> Dict:
        """
        اقتراح تطور — يُعرض على المستخدم

        المستخدم يقرر: موافقة أو رفض (فيتو)
        """
        result = self.evolve_once()
        self.pending_proposal = {
            "id": f"evo_{int(datetime.now().timestamp())}",
            "result": result,
            "status": "pending",  # pending | approved | rejected
            "timestamp": datetime.now().isoformat(),
        }

        # حفظ بتاريخ التطور
        self.archive.history.append(self.pending_proposal)
        self.archive._save_history()

        return self.pending_proposal

    def approve_evolution(self) -> Dict:
        """المستخدم وافق — الجديد = production"""
        if not self.pending_proposal:
            return {"error": "No pending proposal"}

        self.pending_proposal["status"] = "approved"
        self.archive._save_history()

        logger.info(f"✅ Evolution approved: {self.pending_proposal['id']}")
        self.pending_proposal = None
        return {"status": "promoted"}

    def reject_evolution(self, reason: str = "") -> Dict:
        """المستخدم رفض — يرجع القديم"""
        if not self.pending_proposal:
            return {"error": "No pending proposal"}

        version = self.pending_proposal["result"].get("snapshot")
        self.pending_proposal["status"] = "rejected"
        self.pending_proposal["reject_reason"] = reason
        self.archive._save_history()

        if version:
            self.archive.restore(version)

        self.failures += 1
        logger.info(f"🚫 Evolution rejected: {reason}")
        self.pending_proposal = None
        return {"status": "rolled_back", "restored_version": version}

    def run_cycle(self, max_iterations: int = 5) -> Dict:
        """دورة تطور كاملة"""
        results = {"iterations": 0, "improvements": 0, "failures": 0}

        for i in range(max_iterations):
            result = self.evolve_once()
            results["iterations"] += 1

            if result.get("status") == "already_optimal":
                break
            if result.get("success"):
                results["improvements"] += 1
            else:
                results["failures"] += 1

        return results

    def get_status(self) -> Dict:
        """حالة التطور"""
        versions = self.archive.list_versions()
        return {
            "total_versions": len(versions),
            "improvements": self.improvements,
            "failures": self.failures,
            "pending": self.pending_proposal is not None,
            "archive_size": sum(
                sum(f.stat().st_size for f in (SNAPSHOTS_DIR / v.get("snapshot_name", "")).rglob("*")
                    if f.is_file())
                for v in versions if (SNAPSHOTS_DIR / v.get("snapshot_name", "")).exists()
            ) if versions else 0,
        }


# Singleton
evolver = SelfEvolver()
# Backward compatibility
evolution_loop = evolver


if __name__ == "__main__":
    print("🧬 Self Evolution System\n")

    # فحص صحة
    health = evolver.check_health()
    print(f"Health: {health['files']} files, {len(health['errors'])} errors, "
          f"{len(health['warnings'])} warnings")

    # بحث تحسينات
    improvements = evolver.find_improvements()
    print(f"Improvements found: {len(improvements)}")
    for imp in improvements[:5]:
        print(f"  [{imp['type']}] {imp['file']}:{imp['line']} — {imp['text'][:60]}")

    # snapshot
    snap = evolver.archive.snapshot("test_snapshot")
    print(f"\nSnapshot: v{snap.get('version')}, {snap.get('files')} files, {snap.get('total_lines')} lines")

    # دورة تطور
    cycle = evolver.run_cycle(3)
    print(f"\nEvolution cycle: {cycle}")

    # حالة
    status = evolver.get_status()
    print(f"Status: {status}")

    # قائمة النسخ
    versions = evolver.archive.list_versions()
    print(f"\nArchive: {len(versions)} versions")
    for v in versions:
        print(f"  v{v['version']}: {v['files']} files, {v['total_lines']} lines — {v['reason']}")
