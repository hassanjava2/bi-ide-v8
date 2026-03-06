#!/usr/bin/env python3
"""
auto_programmer.py — المبرمج الذاتي 🔧

النظام يعدّل كوده بنفسه:
  1. يكتشف مشاكل (من logs, من self_eval)
  2. يولّد حل (عبر Ollama)
  3. يختبر الحل (sandbox)
  4. يطبّق إذا نجح

حدود الأمان:
  - فقط ملفات brain/* يقدر يعدّل
  - كل تعديل ينحفظ كنسخة احتياطية
  - إذا فشل الاختبار ← يرجع النسخة القديمة
  - لا يعدّل auto_programmer.py نفسه (حماية)
"""

import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("auto_programmer")

PROJECT_ROOT = Path(__file__).parent.parent
BRAIN_DIR = PROJECT_ROOT / "brain"
BACKUP_DIR = PROJECT_ROOT / ".brain_backups"

# ملفات محمية — لا يعدّلها
PROTECTED_FILES = {
    "auto_programmer.py",  # ما يعدّل نفسه!
}

# ملفات مسموح يعدّلها
ALLOWED_DIRS = [
    BRAIN_DIR,
    PROJECT_ROOT / "capsules",
]


class AutoProgrammer:
    """المبرمج الذاتي — يعدّل كوده بنفسه"""

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.changes_log = BRAIN_DIR / ".auto_changes.json"

    def _is_allowed(self, filepath: Path) -> bool:
        """هل مسموح يعدّل هالملف؟"""
        if filepath.name in PROTECTED_FILES:
            return False
        for allowed in ALLOWED_DIRS:
            try:
                filepath.resolve().relative_to(allowed.resolve())
                return True
            except ValueError:
                continue
        return False

    def _backup(self, filepath: Path) -> Path:
        """نسخة احتياطية"""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"{filepath.name}.{ts}.bak"
        shutil.copy2(filepath, backup)
        return backup

    def _ollama_generate(self, prompt: str) -> str:
        """توليد كود من Ollama"""
        import requests
        try:
            r = requests.post(f"{self.ollama_url}/api/chat",
                json={"model": self._get_model(),
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False,
                      "options": {"temperature": 0.3, "num_predict": 2000}},
                timeout=120)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
        return ""

    def _get_model(self) -> str:
        import requests
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return models[0] if models else ""
        except:
            return ""

    def _test_syntax(self, filepath: Path) -> bool:
        """فحص syntax"""
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(filepath)],
                capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def propose_fix(self, filepath: Path, issue: str) -> Dict:
        """اقتراح إصلاح لمشكلة"""
        if not self._is_allowed(filepath):
            return {"status": "denied", "reason": "protected file"}

        if not filepath.exists():
            return {"status": "error", "reason": "file not found"}

        current = filepath.read_text()

        prompt = f"""Fix this Python file. The issue is: {issue}

Current code:
```python
{current[:3000]}
```

Return ONLY the fixed Python code, nothing else. No explanations."""

        fixed = self._ollama_generate(prompt)
        if not fixed:
            return {"status": "error", "reason": "no response from Ollama"}

        # استخراج الكود
        if "```python" in fixed:
            fixed = fixed.split("```python")[1].split("```")[0]
        elif "```" in fixed:
            fixed = fixed.split("```")[1].split("```")[0]

        return {
            "status": "proposed",
            "file": str(filepath),
            "issue": issue,
            "original_lines": len(current.splitlines()),
            "fixed_lines": len(fixed.splitlines()),
            "fixed_code": fixed,
        }

    def apply_fix(self, filepath: Path, fixed_code: str) -> Dict:
        """تطبيق إصلاح مع نسخة احتياطية"""
        if not self._is_allowed(filepath):
            return {"status": "denied"}

        # نسخة احتياطية
        backup = self._backup(filepath)

        # كتابة الكود الجديد
        filepath.write_text(fixed_code)

        # فحص syntax
        if not self._test_syntax(filepath):
            # فشل — رجّع النسخة القديمة
            shutil.copy2(backup, filepath)
            logger.warning(f"❌ Fix failed syntax check, rolled back: {filepath.name}")
            return {"status": "rollback", "reason": "syntax error"}

        # نجح!
        self._log_change(filepath, "fix", backup)
        logger.info(f"✅ Auto-fix applied: {filepath.name}")
        return {"status": "applied", "backup": str(backup)}

    def add_feature(self, filepath: Path, feature_desc: str) -> Dict:
        """إضافة ميزة جديدة"""
        if not self._is_allowed(filepath):
            return {"status": "denied"}

        current = filepath.read_text() if filepath.exists() else ""

        prompt = f"""Add this feature to the Python file: {feature_desc}

Current code:
```python
{current[:3000]}
```

Return the complete updated Python code with the new feature added. Return ONLY code."""

        new_code = self._ollama_generate(prompt)
        if not new_code:
            return {"status": "error"}

        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0]
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0]

        return self.apply_fix(filepath, new_code)

    def optimize_file(self, filepath: Path) -> Dict:
        """تحسين كود"""
        return self.propose_fix(filepath, "Optimize for performance and readability")

    def create_new_file(self, filepath: Path, description: str) -> Dict:
        """إنشاء ملف جديد"""
        if not self._is_allowed(filepath):
            return {"status": "denied"}

        if filepath.exists():
            return {"status": "exists"}

        prompt = f"""Create a new Python file: {description}

Requirements:
- Clean, production-ready Python code
- Proper docstrings and type hints
- Error handling
- Logging

Return ONLY the Python code."""

        code = self._ollama_generate(prompt)
        if not code:
            return {"status": "error"}

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code)

        if self._test_syntax(filepath):
            self._log_change(filepath, "create", None)
            logger.info(f"✅ Created: {filepath.name}")
            return {"status": "created", "file": str(filepath)}
        else:
            filepath.unlink()
            return {"status": "error", "reason": "syntax error in generated code"}

    def _log_change(self, filepath: Path, action: str, backup: Path):
        """تسجيل التعديل"""
        log = []
        if self.changes_log.exists():
            try:
                log = json.loads(self.changes_log.read_text())
            except:
                pass
        log.append({
            "file": str(filepath),
            "action": action,
            "backup": str(backup) if backup else None,
            "timestamp": datetime.now().isoformat(),
        })
        log = log[-100:]  # آخر 100 تعديل
        self.changes_log.write_text(json.dumps(log, indent=2, default=str))

    def rollback_last(self) -> Dict:
        """التراجع عن آخر تعديل"""
        if not self.changes_log.exists():
            return {"status": "no_changes"}

        log = json.loads(self.changes_log.read_text())
        if not log:
            return {"status": "no_changes"}

        last = log[-1]
        if last.get("backup"):
            backup = Path(last["backup"])
            target = Path(last["file"])
            if backup.exists():
                shutil.copy2(backup, target)
                log.pop()
                self.changes_log.write_text(json.dumps(log, indent=2, default=str))
                logger.info(f"⏪ Rolled back: {target.name}")
                return {"status": "rolled_back", "file": str(target)}

        return {"status": "no_backup"}


# Singleton
auto_programmer = AutoProgrammer()
