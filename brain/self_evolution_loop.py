#!/usr/bin/env python3
"""
self_evolution_loop.py — اللولب التطوري 🧬∞

من القوانين (قسم 8):
  "يأخذ نسخة من نفسه → يبرمجها → يطورها → يجربها"
  "يستمر يطور نسخة أحدث"

الدورة:
  1. يفحص الـ logs والأخطاء
  2. يحدد المشاكل والتحسينات
  3. auto_programmer يولّد الحل
  4. ide_controller يطبق
  5. يختبر — إذا نجح → git commit
  6. إذا فشل → rollback
  7. يكرر ∞
"""

import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("evolution")

PROJECT_ROOT = Path(__file__).parent.parent
EVOLUTION_LOG = PROJECT_ROOT / ".evolution_history.json"


class SelfEvolutionLoop:
    """اللولب التطوري — النظام يطور نفسه"""

    def __init__(self):
        self.improvements = 0
        self.failures = 0

    def check_health(self) -> Dict:
        """
        الخطوة 1: فحص صحة النظام
        يقرأ logs ويبحث عن أخطاء
        """
        issues = []

        # فحص logs
        log_files = [
            Path("/tmp/brain_daemon.log"),
            Path("/tmp/train_20.log"),
        ]
        for lf in log_files:
            if lf.exists():
                try:
                    lines = lf.read_text().splitlines()[-50:]
                    for line in lines:
                        ll = line.lower()
                        if "error" in ll or "exception" in ll or "traceback" in ll:
                            issues.append({
                                "type": "log_error",
                                "source": lf.name,
                                "line": line.strip()[:200],
                            })
                except:
                    pass

        # فحص syntax لكل ملفات brain
        brain_dir = PROJECT_ROOT / "brain"
        for py in brain_dir.glob("*.py"):
            try:
                compile(py.read_text(), str(py), "exec")
            except SyntaxError as e:
                issues.append({
                    "type": "syntax_error",
                    "file": str(py.relative_to(PROJECT_ROOT)),
                    "error": str(e),
                })

        # فحص ملفات الكبسولات
        capsules_dir = PROJECT_ROOT / "capsules"
        if capsules_dir.exists():
            for d in capsules_dir.iterdir():
                if d.is_dir():
                    meta = d / "meta.json"
                    if not meta.exists():
                        issues.append({
                            "type": "missing_meta",
                            "capsule": d.name,
                        })

        return {
            "issues": issues[:20],
            "total_issues": len(issues),
            "healthy": len(issues) == 0,
        }

    def suggest_improvements(self) -> List[Dict]:
        """
        الخطوة 2: اقتراح تحسينات
        """
        suggestions = []

        # اقتراح: ملفات بدون docstring
        brain_dir = PROJECT_ROOT / "brain"
        for py in brain_dir.glob("*.py"):
            try:
                content = py.read_text()
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    suggestions.append({
                        "type": "missing_docstring",
                        "file": str(py.relative_to(PROJECT_ROOT)),
                        "action": "add module docstring",
                    })
            except:
                pass

        # اقتراح: imports غير مستخدمة
        # اقتراح: functions طويلة جداً
        for py in brain_dir.glob("*.py"):
            try:
                lines = py.read_text().splitlines()
                if len(lines) > 400:
                    suggestions.append({
                        "type": "large_file",
                        "file": str(py.relative_to(PROJECT_ROOT)),
                        "lines": len(lines),
                        "action": "consider splitting",
                    })
            except:
                pass

        return suggestions[:10]

    def evolve_once(self) -> Dict:
        """
        الخطوة 3: محاولة تطوير واحدة
        - يفحص → يكتشف مشكلة → يصلح → يختبر → يطبق
        """
        from brain.ide_controller import IDEController
        from brain.auto_programmer import AutoProgrammer

        controller = IDEController()
        programmer = AutoProgrammer()

        # 1. فحص
        health = self.check_health()
        if health["healthy"]:
            logger.info("✅ System healthy — no evolution needed")
            return {"status": "healthy", "evolved": False}

        # 2. أول مشكلة
        issue = health["issues"][0]
        logger.info(f"🔧 Found issue: {issue['type']} — {issue}")

        # 3. إصلاح حسب النوع
        result = {"status": "attempted", "issue": issue, "evolved": False}

        if issue["type"] == "syntax_error":
            fp = PROJECT_ROOT / issue["file"]
            fix = programmer.propose_fix(fp, issue["error"])
            if fix.get("status") == "proposed":
                apply = programmer.apply_fix(fp, fix["fixed_code"])
                if apply.get("status") == "applied":
                    # 4. اختبار
                    test = controller.run_command(
                        f"python3 -m py_compile {issue['file']}")
                    if test.get("returncode") == 0:
                        # 5. نجح — commit
                        controller.git_commit(
                            f"auto-fix: {issue['file']} syntax error",
                            [issue["file"]])
                        self.improvements += 1
                        result["evolved"] = True
                        result["status"] = "fixed"
                        logger.info(f"✅ Auto-fixed: {issue['file']}")
                    else:
                        programmer.rollback_last()
                        self.failures += 1
                        result["status"] = "rollback"

        elif issue["type"] == "missing_meta":
            # إنشاء meta.json للكبسولة
            capsule_dir = PROJECT_ROOT / "capsules" / issue["capsule"]
            meta = {
                "id": issue["capsule"],
                "specialty": issue["capsule"].replace("_", " "),
                "parents": [],
                "layer": 0,
                "created": datetime.now().isoformat(),
                "auto_created": True,
            }
            meta_path = capsule_dir / "meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            self.improvements += 1
            result["evolved"] = True
            result["status"] = "meta_created"

        # 6. تسجيل
        self._log_evolution(result)

        return result

    def run_cycle(self, max_fixes: int = 5) -> Dict:
        """
        دورة تطور كاملة — يحاول max_fixes إصلاحات
        """
        logger.info(f"\n🧬 Evolution cycle — max {max_fixes} fixes")
        results = []

        for i in range(max_fixes):
            try:
                r = self.evolve_once()
                results.append(r)
                if r.get("status") == "healthy":
                    break
            except Exception as e:
                logger.error(f"Evolution error: {e}")
                traceback.print_exc()
                break

        evolved = sum(1 for r in results if r.get("evolved"))
        logger.info(f"🧬 Cycle done: {evolved} improvements, "
                    f"{self.failures} rollbacks")

        return {
            "attempts": len(results),
            "evolved": evolved,
            "total_improvements": self.improvements,
            "total_failures": self.failures,
        }

    def _log_evolution(self, result: Dict):
        """تسجيل التطور"""
        log = []
        if EVOLUTION_LOG.exists():
            try:
                log = json.loads(EVOLUTION_LOG.read_text())
            except:
                pass

        result["timestamp"] = datetime.now().isoformat()
        log.append(result)
        log = log[-200:]

        EVOLUTION_LOG.write_text(
            json.dumps(log, indent=2, ensure_ascii=False, default=str))


# Singleton
evolution_loop = SelfEvolutionLoop()
