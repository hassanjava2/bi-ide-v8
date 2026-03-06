#!/usr/bin/env python3
"""
ide_controller.py — الكبسولات تتحكم بالـ IDE 🎮

يعطي الكبسولات القدرة على:
  1. قراءة ملفات المشروع
  2. كتابة/تعديل ملفات
  3. تشغيل أوامر (بالـ sandbox)
  4. بناء المشروع (compile/build)
  5. اختبار (pytest, npm test)
  6. عمل git commit + push

يستخدمه:
  - auto_programmer.py → لما يحتاج يصلح شي
  - project_orchestrator.py → لما ينفذ مشروع
  - brain_daemon.py → لما يتطور ذاتياً
  - self_evolution_loop.py → لما يحسّن نفسه
"""

import os
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("ide_controller")

PROJECT_ROOT = Path(__file__).parent.parent
BACKUP_DIR = PROJECT_ROOT / ".brain_backups"

# حدود أمان
MAX_FILE_SIZE = 100_000       # 100KB لكل ملف
MAX_COMMAND_TIMEOUT = 60      # 60 ثانية
PROTECTED_PATHS = {
    ".git",
    "node_modules",
    ".brain_backups",
}


class IDEController:
    """الكبسولات تتحكم بالـ IDE — قراءة، كتابة، تشغيل"""

    def __init__(self, project_root: Path = None):
        self.root = project_root or PROJECT_ROOT
        self.actions_log: List[Dict] = []

    # ═══════════════════════════════════════════
    # القراءة
    # ═══════════════════════════════════════════

    def read_file(self, relative_path: str) -> Dict:
        """قراءة ملف من المشروع"""
        fp = (self.root / relative_path).resolve()
        if not self._is_safe(fp):
            return {"error": "unsafe path"}
        if not fp.exists():
            return {"error": "not found"}
        try:
            content = fp.read_text(errors="replace")
            return {"path": relative_path, "content": content,
                    "lines": len(content.splitlines())}
        except Exception as e:
            return {"error": str(e)}

    def list_files(self, relative_dir: str = "", extensions: List[str] = None) -> List[str]:
        """قائمة الملفات"""
        dp = (self.root / relative_dir).resolve()
        if not self._is_safe(dp):
            return []
        files = []
        try:
            for f in dp.rglob("*"):
                if f.is_file() and self._is_safe(f):
                    if extensions and f.suffix.lstrip(".") not in extensions:
                        continue
                    files.append(str(f.relative_to(self.root)))
        except:
            pass
        return files[:200]  # حد أقصى

    def search_in_files(self, query: str, extensions: List[str] = None) -> List[Dict]:
        """بحث بالمشروع"""
        results = []
        for f in self.list_files("", extensions):
            try:
                content = (self.root / f).read_text(errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if query.lower() in line.lower():
                        results.append({"file": f, "line": i, "content": line.strip()})
                        if len(results) >= 50:
                            return results
            except:
                pass
        return results

    # ═══════════════════════════════════════════
    # الكتابة
    # ═══════════════════════════════════════════

    def write_file(self, relative_path: str, content: str, backup: bool = True) -> Dict:
        """كتابة ملف (مع نسخة احتياطية)"""
        fp = (self.root / relative_path).resolve()
        if not self._is_safe(fp):
            return {"error": "unsafe path"}
        if len(content) > MAX_FILE_SIZE:
            return {"error": "file too large"}

        # نسخة احتياطية
        backup_path = None
        if backup and fp.exists():
            backup_path = self._backup(fp)

        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        self._log("write", relative_path, backup_path)

        return {"status": "ok", "path": relative_path,
                "backup": str(backup_path) if backup_path else None}

    def edit_file(self, relative_path: str, old_text: str, new_text: str) -> Dict:
        """تعديل جزء من ملف (find & replace)"""
        fp = (self.root / relative_path).resolve()
        if not self._is_safe(fp) or not fp.exists():
            return {"error": "file not found or unsafe"}

        content = fp.read_text(errors="replace")
        if old_text not in content:
            return {"error": "target text not found"}

        backup_path = self._backup(fp)
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        self._log("edit", relative_path, backup_path)

        return {"status": "ok", "path": relative_path}

    def create_file(self, relative_path: str, content: str) -> Dict:
        """إنشاء ملف جديد"""
        fp = (self.root / relative_path).resolve()
        if fp.exists():
            return {"error": "file already exists"}
        return self.write_file(relative_path, content, backup=False)

    def delete_file(self, relative_path: str) -> Dict:
        """حذف ملف (مع نسخة احتياطية)"""
        fp = (self.root / relative_path).resolve()
        if not self._is_safe(fp) or not fp.exists():
            return {"error": "not found or unsafe"}

        backup_path = self._backup(fp)
        fp.unlink()
        self._log("delete", relative_path, backup_path)

        return {"status": "ok", "backup": str(backup_path)}

    # ═══════════════════════════════════════════
    # تشغيل الأوامر
    # ═══════════════════════════════════════════

    def run_command(self, command: str, cwd: str = None,
                     timeout: int = None) -> Dict:
        """تشغيل أمر (sandbox — مو root)"""
        timeout = min(timeout or MAX_COMMAND_TIMEOUT, MAX_COMMAND_TIMEOUT)

        # أوامر محظورة
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:", "sudo rm"]
        for d in dangerous:
            if d in command:
                return {"error": f"dangerous command blocked: {d}"}

        work_dir = (self.root / cwd) if cwd else self.root

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                timeout=timeout, cwd=str(work_dir),
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
            output = result.stdout.decode(errors="replace")
            error = result.stderr.decode(errors="replace")

            self._log("command", command, None)

            return {
                "status": "ok" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout": output[-5000:],  # آخر 5000 حرف
                "stderr": error[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    def run_tests(self, test_type: str = "auto") -> Dict:
        """تشغيل الاختبارات"""
        if test_type == "python" or (test_type == "auto" and
                (self.root / "pytest.ini").exists()):
            return self.run_command("python3 -m pytest -x --tb=short 2>&1 | tail -30")
        elif test_type == "node" or (test_type == "auto" and
                (self.root / "package.json").exists()):
            return self.run_command("npm test 2>&1 | tail -30")
        else:
            return self.run_command("python3 -m py_compile brain/*.py 2>&1")

    def build_project(self, target: str = "auto") -> Dict:
        """بناء المشروع"""
        if target == "tauri" or (target == "auto" and
                (self.root / "apps/desktop-tauri").exists()):
            return self.run_command(
                "cd apps/desktop-tauri && npm run build 2>&1 | tail -20",
                timeout=120)
        return {"error": "unknown build target"}

    # ═══════════════════════════════════════════
    # Git
    # ═══════════════════════════════════════════

    def git_status(self) -> Dict:
        """حالة git"""
        return self.run_command("git status --short 2>&1 | head -20")

    def git_commit(self, message: str, files: List[str] = None) -> Dict:
        """git add + commit"""
        if files:
            add_cmd = "git add " + " ".join(f'"{f}"' for f in files)
        else:
            add_cmd = "git add -A"
        result = self.run_command(f'{add_cmd} && git commit -m "{message}" 2>&1')
        self._log("git_commit", message, None)
        return result

    def git_push(self) -> Dict:
        """git push"""
        return self.run_command("git push origin main 2>&1")

    # ═══════════════════════════════════════════
    # أمان
    # ═══════════════════════════════════════════

    def _is_safe(self, fp: Path) -> bool:
        """هل المسار آمن؟"""
        try:
            fp.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        for p in PROTECTED_PATHS:
            if p in str(fp):
                return False
        return True

    def _backup(self, fp: Path) -> Path:
        """نسخة احتياطية"""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel = fp.relative_to(self.root)
        backup = BACKUP_DIR / f"{rel.name}.{ts}.bak"
        shutil.copy2(fp, backup)
        return backup

    def _log(self, action: str, target: str, backup: Path):
        """تسجيل"""
        self.actions_log.append({
            "action": action, "target": target,
            "backup": str(backup) if backup else None,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.actions_log) > 200:
            self.actions_log = self.actions_log[-100:]

    def get_log(self) -> List[Dict]:
        """سجل العمليات"""
        return self.actions_log


# Singleton
ide_controller = IDEController()
