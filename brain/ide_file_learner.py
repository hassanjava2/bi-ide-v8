#!/usr/bin/env python3
"""
ide_file_learner.py — التعلم من كل ملف بالـ IDE 📚

Data Flywheel — كل ملف تفتحه = بيانات تدريب:
  1. يحلل الكود أوتوماتيكياً
  2. يكتشف الأخطاء ← {خطأ, إصلاح}
  3. يكتشف الأنماط ← {كود, شرح}
  4. يفرز حسب اللغة ← الكبسولة المناسبة
  5. يحفظ كلشي بالذاكرة الأبدية (PostgreSQL)
"""

import json
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("file_learner")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory

# ═══════════════════════════════════════════════════════════
# Language Detection + Capsule Mapping
# ═══════════════════════════════════════════════════════════

LANG_MAP = {
    ".py": ("python", "code_python"),
    ".ts": ("typescript", "code_typescript"),
    ".tsx": ("typescript", "code_typescript"),
    ".js": ("javascript", "code_typescript"),
    ".jsx": ("javascript", "code_typescript"),
    ".rs": ("rust", "code_rust"),
    ".sql": ("sql", "code_sql"),
    ".css": ("css", "code_css"),
    ".scss": ("scss", "code_css"),
    ".html": ("html", "code_typescript"),
    ".c": ("c", "code_python"),
    ".cpp": ("cpp", "code_python"),
    ".java": ("java", "code_python"),
    ".go": ("go", "code_python"),
    ".swift": ("swift", "code_python"),
    ".kt": ("kotlin", "code_python"),
    ".rb": ("ruby", "code_python"),
    ".php": ("php", "code_python"),
    ".sh": ("shell", "devops"),
    ".bash": ("shell", "devops"),
    ".yaml": ("yaml", "devops"),
    ".yml": ("yaml", "devops"),
    ".json": ("json", "devops"),
    ".toml": ("toml", "devops"),
    ".dockerfile": ("docker", "devops"),
    ".md": ("markdown", "knowledge_arabic"),
    ".txt": ("text", "knowledge_arabic"),
}

# ═══════════════════════════════════════════════════════════
# Error Patterns — أنماط الأخطاء الشائعة
# ═══════════════════════════════════════════════════════════

ERROR_PATTERNS = {
    "python": [
        (r"except\s*:", "syntax", "Bare except catches all exceptions", "Use except Exception as e:"),
        (r"eval\s*\(", "security", "eval() is dangerous — code injection risk", "Use ast.literal_eval() or json.loads()"),
        (r"import\s+\*", "syntax", "Wildcard import pollutes namespace", "Import specific names"),
        (r"password\s*=\s*['\"]", "security", "Hardcoded password", "Use environment variables"),
        (r"\.format\(.*input", "security", "Format string with user input — injection risk", "Use parameterized queries"),
        (r"os\.system\(", "security", "os.system() is unsafe", "Use subprocess.run()"),
        (r"print\(.*password", "security", "Printing password to console", "Remove password logging"),
    ],
    "typescript": [
        (r"any\b", "syntax", "Using 'any' type defeats TypeScript's purpose", "Use specific types"),
        (r"eval\s*\(", "security", "eval() is dangerous", "Use JSON.parse() or safer alternatives"),
        (r"innerHTML\s*=", "security", "innerHTML can lead to XSS", "Use textContent or sanitize input"),
        (r"console\.log\(", "syntax", "Console.log left in code", "Remove or use proper logging"),
        (r"password.*=.*['\"]", "security", "Hardcoded password", "Use environment variables"),
    ],
    "rust": [
        (r"unsafe\s*\{", "security", "Unsafe block — memory safety risk", "Minimize unsafe usage"),
        (r"unwrap\(\)", "syntax", "unwrap() can panic", "Use ? operator or match"),
        (r"clone\(\)", "syntax", "Excessive cloning — performance issue", "Consider borrowing"),
    ],
    "sql": [
        (r"SELECT\s+\*", "syntax", "SELECT * is inefficient", "Select specific columns"),
        (r"DROP\s+TABLE", "security", "DROP TABLE is destructive", "Use with extreme caution"),
        (r"--.*password", "security", "Password in SQL comment", "Remove sensitive data from comments"),
    ],
}

# ═══════════════════════════════════════════════════════════
# Best Practice Patterns — أنماط صحيحة للتعلم
# ═══════════════════════════════════════════════════════════

GOOD_PATTERNS = {
    "python": [
        (r"def\s+\w+\(.*\)\s*->", "Type hints in function signature"),
        (r'"""[\s\S]*?"""', "Docstring documentation"),
        (r"with\s+open\(", "Context manager for file handling"),
        (r"logging\.\w+\(", "Proper logging instead of print"),
        (r"try:.*except\s+\w+", "Specific exception handling"),
        (r"@dataclass", "Using dataclass for data structures"),
        (r"from pathlib import Path", "Using pathlib for paths"),
    ],
    "typescript": [
        (r"interface\s+\w+", "Interface definition"),
        (r"type\s+\w+\s*=", "Type alias"),
        (r"async\s+function", "Async function"),
        (r"const\s+\w+:\s*\w+", "Typed constant"),
        (r"useEffect\(", "React hook usage"),
        (r"useMemo\(", "Performance optimization with useMemo"),
    ],
}


class IDEFileLearner:
    """
    التعلم من كل ملف — Data Flywheel

    كل ملف يُفتح بالـ IDE:
      1. يُحلل بحثاً عن أخطاء + أنماط صحيحة
      2. يُولد عينات تدريب
      3. يُحفظ بالذاكرة الأبدية
    """

    def __init__(self):
        self.seen_hashes = set()
        self._load_seen()
        logger.info("📚 IDE File Learner initialized")

    def _load_seen(self):
        """تحميل هاشات الملفات المفحوصة"""
        hash_file = PROJECT_ROOT / "brain" / "capsules" / ".learner_seen.json"
        if hash_file.exists():
            try:
                self.seen_hashes = set(json.loads(hash_file.read_text()))
            except Exception:
                pass

    def _save_seen(self):
        """حفظ هاشات الملفات"""
        hash_file = PROJECT_ROOT / "brain" / "capsules" / ".learner_seen.json"
        try:
            hash_file.write_text(json.dumps(list(self.seen_hashes)[-10000:]))
        except Exception:
            pass

    def learn_from_file(self, file_path: str, content: str = None) -> Dict:
        """
        تعلم من ملف واحد

        Returns:
            {
                "file": "...",
                "language": "python",
                "errors": [{"type": "...", "desc": "...", "fix": "..."}],
                "patterns": [{"desc": "..."}],
                "samples_generated": 5,
            }
        """
        path = Path(file_path)

        # قراءة المحتوى
        if content is None:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return {"file": file_path, "error": "Cannot read file"}

        # تجاهل ملفات صغيرة جداً أو كبيرة جداً
        if len(content) < 50 or len(content) > 500000:
            return {"file": file_path, "skipped": True}

        # هاش للتجنب التكرار
        file_hash = hashlib.md5(content.encode()).hexdigest()
        if file_hash in self.seen_hashes:
            return {"file": file_path, "already_seen": True}
        self.seen_hashes.add(file_hash)

        # كشف اللغة
        ext = path.suffix.lower()
        if ext not in LANG_MAP:
            return {"file": file_path, "unsupported": True}

        language, capsule_id = LANG_MAP[ext]
        lines = content.split("\n")

        # === تحليل الأخطاء ===
        errors = self._find_errors(content, language)

        # === تحليل الأنماط الصحيحة ===
        patterns = self._find_good_patterns(content, language)

        # === توليد عينات تدريب ===
        samples = self._generate_samples(file_path, content, language, capsule_id, errors, patterns)

        # === حفظ بالذاكرة الأبدية ===
        memory.save_file_seen(
            file_path=file_path,
            language=language,
            capsule_id=capsule_id,
            lines_count=len(lines),
            errors_found=len(errors),
            patterns_found=len(patterns),
            file_hash=file_hash,
        )

        for err in errors:
            memory.save_error(
                file_path=file_path,
                error_type=err["type"],
                error_text=err["desc"],
                fix_text=err.get("fix"),
                language=language,
                capsule_id=capsule_id,
            )

        self._save_seen()

        result = {
            "file": file_path,
            "language": language,
            "capsule": capsule_id,
            "lines": len(lines),
            "errors": errors,
            "patterns": patterns,
            "samples_generated": len(samples),
        }

        if errors or patterns:
            logger.info(f"📚 {path.name}: {len(errors)} errors, {len(patterns)} patterns, {len(samples)} samples → {capsule_id}")

        return result

    def learn_from_edit(self, file_path: str, before: str, after: str) -> Dict:
        """
        تعلم من تعديل — before/after

        كل تعديل يسويه المستخدم = عينة تدريب
        """
        if before == after:
            return {"skipped": True}

        ext = Path(file_path).suffix.lower()
        if ext not in LANG_MAP:
            return {"unsupported": True}

        language, capsule_id = LANG_MAP[ext]

        # توليد عينة: "صلح هالكود" → الكود المصلح
        sample = {
            "input_text": f"Review and improve this {language} code:\n```{language}\n{before[:2000]}\n```",
            "output_text": f"Improved code:\n```{language}\n{after[:2000]}\n```",
        }

        # حفظ
        self._save_sample(sample, capsule_id)

        memory.save_knowledge(
            topic=f"Code edit: {Path(file_path).name}",
            content=f"Changed {language} code in {file_path}",
            source="ide_edit",
            capsule_id=capsule_id,
        )

        return {"file": file_path, "language": language, "edit_learned": True}

    def _find_errors(self, content: str, language: str) -> List[Dict]:
        """اكتشاف الأخطاء"""
        errors = []
        patterns = ERROR_PATTERNS.get(language, [])

        for regex, err_type, desc, fix in patterns:
            matches = re.findall(regex, content, re.IGNORECASE)
            if matches:
                errors.append({
                    "type": err_type,
                    "desc": desc,
                    "fix": fix,
                    "count": len(matches),
                })

        return errors

    def _find_good_patterns(self, content: str, language: str) -> List[Dict]:
        """اكتشاف الأنماط الصحيحة"""
        found = []
        patterns = GOOD_PATTERNS.get(language, [])

        for regex, desc in patterns:
            if re.search(regex, content):
                found.append({"desc": desc})

        return found

    def _generate_samples(self, file_path: str, content: str,
                         language: str, capsule_id: str,
                         errors: List[Dict], patterns: List[Dict]) -> List[Dict]:
        """توليد عينات تدريب"""
        samples = []

        # عينة من الأخطاء
        for err in errors[:5]:
            sample = {
                "input_text": f"Find the {err['type']} issue in this {language} code and how to fix it:\n{err['desc']}",
                "output_text": f"**Issue**: {err['desc']}\n**Fix**: {err['fix']}\n**Type**: {err['type']}",
            }
            samples.append(sample)

        # عينة من الكود الجيد
        if len(content) > 100:
            chunk = content[:3000]
            sample = {
                "input_text": f"Explain this {language} code:\n```{language}\n{chunk}\n```",
                "output_text": (
                    f"This is a {language} file ({Path(file_path).name}).\n"
                    f"Good patterns found: {', '.join(p['desc'] for p in patterns[:5])}\n"
                    f"Issues found: {len(errors)}\n"
                ),
            }
            samples.append(sample)

        # حفظ العينات
        for s in samples:
            self._save_sample(s, capsule_id)

        return samples

    def _save_sample(self, sample: Dict, capsule_id: str):
        """حفظ عينة تدريب"""
        data_dir = CAPSULES_ROOT / capsule_id / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_file = data_dir / f"ide_learn_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

learner = IDEFileLearner()


def learn(file_path: str, content: str = None) -> Dict:
    """نقطة الدخول الرئيسية"""
    return learner.learn_from_file(file_path, content)


def learn_edit(file_path: str, before: str, after: str) -> Dict:
    """تعلم من تعديل"""
    return learner.learn_from_edit(file_path, before, after)


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("📚 IDE File Learner — Test\n")

    # تجربة على ملف Python
    test_code = '''
import os

def get_user(name):
    password = "admin123"
    result = eval(input("Enter query: "))
    try:
        data = open("file.txt").read()
    except:
        pass
    return os.system(f"echo {name}")
'''

    result = learner.learn_from_file("/tmp/test_bad.py", test_code)
    print(f"File: {result.get('file')}")
    print(f"Language: {result.get('language')}")
    print(f"Errors: {len(result.get('errors', []))}")
    for err in result.get("errors", []):
        print(f"  ❌ [{err['type']}] {err['desc']} → {err['fix']}")
    print(f"Patterns: {len(result.get('patterns', []))}")
    print(f"Samples: {result.get('samples_generated', 0)}")

    # تجربة على كود جيد
    good_code = '''
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User model with type hints."""
    name: str
    email: str

def get_user(user_id: int) -> User:
    """Fetch user by ID."""
    try:
        with open(Path("users.json")) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Users file not found: {e}")
        raise
    return User(**data)
'''

    result2 = learner.learn_from_file("/tmp/test_good.py", good_code)
    print(f"\nGood file:")
    print(f"Errors: {len(result2.get('errors', []))}")
    print(f"Patterns: {len(result2.get('patterns', []))}")
    for pat in result2.get("patterns", []):
        print(f"  ✅ {pat['desc']}")
    print(f"Samples: {result2.get('samples_generated', 0)}")
