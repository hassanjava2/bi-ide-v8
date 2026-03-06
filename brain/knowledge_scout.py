#!/usr/bin/env python3
"""
knowledge_scout.py — الكشّافة

مصدران للبيانات:
1. كشافة خارجية (Ollama): تولّد Q&A من موديلات محلية
2. كشافة داخلية (FileScanner): تفحص ملفات الجهاز الحقيقية
   - كود Python, TypeScript, Rust, SQL, CSS
   - إعدادات النظام والتطبيقات
   - كود المشروع نفسه
   - أي ملف قابل للتعلم

البيانات الحقيقية > البيانات المولّدة (بلا هلوسة)
الكشافة ما تتعب — تشتغل 24/7
"""

import json
import os
import time
import random
import logging
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger("scout")

CAPSULES_ROOT = Path(__file__).parent.parent / "capsules"

# ═══════════════════════════════════════════════════════════
# مواضيع التوليد لكل كبسولة — تتوسع لحالها
# ═══════════════════════════════════════════════════════════

SCOUT_TOPICS = {
    # ─── البرمجة ──────────────────────────────────
    "code_python": {
        "prompts": [
            "Write a Python function that {action} using {lib}",
            "Create a Python class for {concept} with error handling",
            "Explain this Python concept in detail: {concept}",
            "Write efficient Python code to solve: {problem}",
            "Show best practices for {topic} in Python",
        ],
        "vars": {
            "action": ["sorts data", "parses JSON", "connects to API", "handles files",
                       "manages database", "processes images", "sends emails", "caches results",
                       "validates input", "generates reports", "scrapes web pages",
                       "manages async tasks", "creates CLI tools", "handles websockets"],
            "lib": ["asyncio", "requests", "sqlalchemy", "pandas", "pydantic",
                    "fastapi", "pytest", "pathlib", "logging", "typing"],
            "concept": ["decorators", "generators", "context managers", "metaclasses",
                       "descriptors", "coroutines", "type hints", "dataclasses"],
            "problem": ["find duplicates", "merge sorted lists", "parse log files",
                       "rate limiting", "connection pooling", "batch processing"],
            "topic": ["error handling", "testing", "documentation", "performance",
                     "security", "deployment", "logging", "configuration"],
        },
    },
    "code_typescript": {
        "prompts": [
            "Create a React component for {component} with TypeScript",
            "Write a custom hook for {purpose}",
            "Implement {pattern} in TypeScript",
            "Build a Next.js {feature} with proper types",
        ],
        "vars": {
            "component": ["data table", "form wizard", "file uploader", "search bar",
                         "notification center", "chat widget", "dashboard card"],
            "purpose": ["data fetching", "authentication", "form validation",
                       "infinite scroll", "dark mode", "local storage"],
            "pattern": ["observer pattern", "factory pattern", "state machine",
                       "dependency injection", "middleware chain"],
            "feature": ["API route", "middleware", "server component", "streaming page"],
        },
    },

    # ─── ERP ──────────────────────────────────────
    "erp_accounting": {
        "prompts": [
            "اشرح {topic} في المحاسبة مع أمثلة",
            "كيف أسجل {operation} محاسبياً؟",
            "ما الفرق بين {a} و {b} في المحاسبة؟",
        ],
        "vars": {
            "topic": ["القيد المزدوج", "الإهلاك", "التسويات الجردية", "التكاليف المعيارية",
                     "المحاسبة الحكومية", "محاسبة الشركات", "التحليل المالي"],
            "operation": ["شراء أصل ثابت", "دفع رواتب", "بيع بالأجل", "مرتجع مشتريات",
                        "خصم نقدي", "أقساط قرض", "توزيع أرباح"],
            "a": ["FIFO", "المصروف", "الأصل الثابت", "الربح التشغيلي"],
            "b": ["LIFO", "الخسارة", "الأصل المتداول", "صافي الربح"],
        },
    },
    "erp_sales": {
        "prompts": [
            "اشرح عملية {process} في نظام المبيعات",
            "كيف أحسب {metric} في المبيعات؟",
        ],
        "vars": {
            "process": ["أمر البيع", "الفاتورة الضريبية", "المرتجع", "التحصيل"],
            "metric": ["هامش الربح", "نسبة التحويل", "معدل الاحتفاظ بالعميل"],
        },
    },

    # ─── الأمن ──────────────────────────────────
    "security": {
        "prompts": [
            "Explain {attack} and how to defend against it",
            "Implement secure {feature} in a web application",
            "Best practices for {area} security",
        ],
        "vars": {
            "attack": ["SQL injection", "XSS", "CSRF", "SSRF", "directory traversal",
                      "buffer overflow", "privilege escalation", "session hijacking"],
            "feature": ["authentication", "file upload", "API access", "password storage",
                       "session management", "data encryption"],
            "area": ["API", "database", "container", "network", "cloud", "mobile"],
        },
    },

    # ─── المحادثة ─────────────────────────────────
    "conversation_ar": {
        "prompts": [
            "أجب على السؤال التالي بطريقة مفيدة: {question}",
            "اشرح {topic} بأسلوب بسيط ومفهوم",
        ],
        "vars": {
            "question": ["كيف أتعلم البرمجة؟", "ما هو الذكاء الاصطناعي؟",
                        "كيف أبدأ مشروعي الخاص؟", "ما أفضل لغة برمجة؟"],
            "topic": ["التعلم العميق", "الحوسبة السحابية", "إنترنت الأشياء",
                     "البلوك تشين", "الأمن السيبراني"],
        },
    },
    "iraqi_dialect": {
        "prompts": [
            "رد باللهجة العراقية على: {question}",
            "اشرح {topic} بلهجة عراقية",
        ],
        "vars": {
            "question": ["شلون أتعلم برمجة؟", "شنو أحسن لابتوب؟", "شگد ياخذ وقت؟"],
            "topic": ["الذكاء الاصطناعي", "التطبيقات", "الأمن"],
        },
    },
}


class KnowledgeScout:
    """الكشّافة — تجيب بيانات جديدة بدون توقف"""

    def __init__(self, capsules_dir: Optional[Path] = None,
                 ollama_url: str = "http://127.0.0.1:11434"):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        self.ollama_url = ollama_url
        self.total_generated = 0
        self.cycle_count = 0

    def _ollama_generate(self, model: str, prompt: str, timeout: int = 90) -> str:
        """توليد نص من Ollama"""
        import requests
        try:
            r = requests.post(f"{self.ollama_url}/api/chat",
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False,
                      "options": {"temperature": 0.8, "num_predict": 500}},
                timeout=timeout)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
        return ""

    def _get_model(self) -> str:
        """أول موديل متوفر"""
        import requests
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if models:
                return models[0]
        except:
            pass
        return ""

    def _build_prompt(self, capsule_id: str) -> tuple[str, str]:
        """بناء prompt عشوائي لكبسولة"""
        if capsule_id not in SCOUT_TOPICS:
            # للكبسولات الجديدة (أطفال) — استخدم parent topics
            parent_id = capsule_id.rsplit("_", 1)[0]
            if parent_id in SCOUT_TOPICS:
                capsule_id = parent_id
            else:
                return "", ""

        config = SCOUT_TOPICS[capsule_id]
        template = random.choice(config["prompts"])

        # ملئ المتغيرات
        prompt = template
        for var_name, var_values in config["vars"].items():
            placeholder = "{" + var_name + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, random.choice(var_values))

        return capsule_id, prompt

    def scout_one(self, capsule_id: str, model: str) -> bool:
        """توليد عينة واحدة لكبسولة"""
        _, prompt = self._build_prompt(capsule_id)
        if not prompt:
            return False

        answer = self._ollama_generate(model, prompt)
        if not answer or len(answer) < 20:
            return False

        # حفظ في ملف الكبسولة
        # حماية inbox — لو قيد التدريب الكشافة تكتب بـ inbox/
        cap_dir = self.capsules_dir / capsule_id
        lock_file = cap_dir / ".training_in_progress"
        if lock_file.exists():
            data_dir = cap_dir / "inbox"
        else:
            data_dir = cap_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        scout_file = data_dir / "scout_data.jsonl"

        with open(scout_file, "a") as f:
            f.write(json.dumps({
                "input_text": prompt,
                "output_text": answer,
                "source": "scout",
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")

        self.total_generated += 1
        return True

    def scout_cycle(self, samples_per_capsule: int = 10) -> dict:
        """دورة كشافة واحدة — داخلية + خارجية"""
        self.cycle_count += 1
        logger.info(f"🔍 Scout cycle #{self.cycle_count}")

        results = {}

        # ═══ الكشافة الداخلية أولاً (بيانات حقيقية) ═══
        internal = InternalScout(self.capsules_dir)
        internal_results = internal.scan_all()
        for k, v in internal_results.items():
            results[k] = results.get(k, 0) + v

        # ═══ ثم الكشافة الخارجية (Ollama) ═══
        model = self._get_model()
        if model:
            logger.info(f"🤖 External scout — model: {model}")
            capsule_dirs = sorted(self.capsules_dir.iterdir())
            for capsule_dir in capsule_dirs:
                if not capsule_dir.is_dir():
                    continue
                cid = capsule_dir.name
                meta_path = capsule_dir / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        if meta.get("archived"):
                            continue
                    except:
                        pass
                count = 0
                for _ in range(samples_per_capsule):
                    if self.scout_one(cid, model):
                        count += 1
                if count > 0:
                    results[cid] = results.get(cid, 0) + count
        else:
            logger.warning("No Ollama model — internal scout only")

        total = sum(results.values())
        logger.info(f"🔍 Cycle #{self.cycle_count}: +{total} samples "
                    f"across {len(results)} capsules")
        return results

    def scout_forever(self, samples_per_capsule: int = 10, pause_seconds: int = 60):
        """كشافة بلا توقف — ∞"""
        logger.info("🔍 Scout starting infinite mode...")
        while True:
            try:
                self.scout_cycle(samples_per_capsule)
                logger.info(f"💤 Pause {pause_seconds}s... (total: {self.total_generated})")
                time.sleep(pause_seconds)
            except KeyboardInterrupt:
                logger.info("Scout stopped by user")
                break
            except Exception as e:
                logger.error(f"Scout error: {e}")
                time.sleep(30)


# ═══════════════════════════════════════════════════════════
# الكشّافة الداخلية — تفحص ملفات الجهاز الحقيقية
# ═══════════════════════════════════════════════════════════

# خارطة: امتداد الملف → أي كبسولة تستفيد
FILE_TO_CAPSULE = {
    # كود
    ".py": "code_python",
    ".pyx": "code_python",
    ".pyi": "code_python",
    ".ts": "code_typescript",
    ".tsx": "code_typescript",
    ".js": "code_typescript",
    ".jsx": "code_typescript",
    ".rs": "code_rust",
    ".sql": "code_sql",
    ".css": "code_css",
    ".scss": "code_css",
    ".html": "code_css",
    ".svelte": "code_typescript",
    ".vue": "code_typescript",
    # لغات إضافية — تعلم عميق
    ".c": "code_python",       # C → كبسولة عامة
    ".h": "code_python",
    ".cpp": "code_python",
    ".cc": "code_python",
    ".java": "code_typescript", # Java → كبسولة TS (مشابه)
    ".kt": "code_typescript",  # Kotlin
    ".go": "code_python",      # Go
    ".sh": "devops",
    ".bash": "devops",
    ".zsh": "devops",
    ".ps1": "devops",          # PowerShell
    ".bat": "devops",
    ".cmd": "devops",
    ".rb": "code_python",      # Ruby
    ".php": "code_python",     # PHP
    ".swift": "code_rust",     # Swift → مشابه Rust
    ".asm": "security",        # Assembly → الأمن
    ".s": "security",
    # إعدادات
    ".toml": "devops",
    ".yaml": "devops",
    ".yml": "devops",
    ".ini": "devops",
    ".conf": "devops",
    ".nginx": "devops",
    ".service": "devops",
    ".dockerfile": "devops",
    ".json": "devops",
    ".xml": "devops",
    # قواعد بيانات
    ".prisma": "database_design",
    ".migration": "database_design",
    # أمن
    ".pem": "security",
    ".key": "security",
    ".crt": "security",
    ".rules": "security",      # firewall rules
    # مستندات (تعلم عربي)
    ".md": "conversation_ar",
    ".txt": "conversation_ar",
    ".rst": "conversation_ar",
}

# مجلدات يجب تفحصها — حسب نظام التشغيل
import platform
_OS = platform.system()  # Linux, Darwin, Windows

# مسارات مشتركة
SCAN_DIRS = [
    # المشروع نفسه
    ("~/bi-ide-v8", 5),
    # Home configs
    ("~/.config", 2),
]

# بيانات الطوارئ (لو موجودة)
_EMERGENCY_PATHS = ["/data/emergency", os.path.expanduser("~/emergency_data"), "/mnt/4tb/emergency"]
for _ep in _EMERGENCY_PATHS:
    if os.path.isdir(_ep):
        SCAN_DIRS.append((_ep, 6))
        break

if _OS == "Linux":
    SCAN_DIRS += [
        # كود Python المثبت
        ("/usr/lib/python3", 3),
        ("/usr/local/lib/python3", 3),
        # إعدادات النظام
        ("/etc", 2),
        # === تعلم عميق — كود النظام ===
        ("/usr/src", 4),             # Linux kernel headers
        ("/usr/include", 3),         # C/C++ headers
        ("/usr/share/doc", 2),       # توثيق النظام
        ("/opt", 3),                 # برامج مثبتة
        ("/usr/local/src", 4),       # كود محلي
        ("/var/log", 1),             # لوغات (تعلم استكشاف الأخطاء)
    ]
elif _OS == "Darwin":  # macOS
    SCAN_DIRS += [
        ("/usr/local/lib/python3", 3),
        ("/etc", 2),
        ("/usr/local/Cellar", 3),    # Homebrew
        ("~/Library/Application Support", 2),
        ("/Applications", 1),
        ("/usr/share", 2),
    ]
elif _OS == "Windows":
    SCAN_DIRS += [
        ("C:\\Python3", 3),
        ("C:\\Program Files", 2),
        ("C:\\Windows\\System32\\", 1),
        (os.path.expanduser("~\\AppData\\Local\\Programs"), 2),
        (os.path.expanduser("~\\Documents"), 3),
    ]

# ملفات/مجلدات نتخطاها
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".next", "dist",
    "build", ".cache", "venv", ".venv", "target", ".cargo",
    "capsules",  # لا ندرّب على بيانات التدريب!
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".bin",
    ".whl", ".egg", ".tar", ".gz", ".zip", ".png", ".jpg",
    ".gif", ".ico", ".svg", ".woff", ".ttf", ".map",
    ".lock", ".log",
}


class InternalScout:
    """الكشّافة الداخلية — تتعلم من ملفات الجهاز الحقيقية"""

    def __init__(self, capsules_dir: Path = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        self.seen_hashes_file = self.capsules_dir / ".scout_seen.json"
        self.seen_hashes = self._load_seen()

    def _load_seen(self) -> set:
        """تحميل هاشات الملفات المفحوصة سابقاً"""
        if self.seen_hashes_file.exists():
            try:
                return set(json.loads(self.seen_hashes_file.read_text()))
            except:
                pass
        return set()

    def _save_seen(self):
        """حفظ هاشات الملفات"""
        # نحتفظ بآخر 50000 هاش فقط
        hashes = list(self.seen_hashes)[-50000:]
        self.seen_hashes_file.write_text(json.dumps(hashes))

    def _classify_file(self, filepath: Path) -> str:
        """تصنيف ملف → كبسولة"""
        name = filepath.name.lower()
        suffix = filepath.suffix.lower()

        # ملفات تستنق
        if name.startswith("test_") or "_test." in name or ".test." in name or ".spec." in name:
            return "code_testing"

        # Dockerfile
        if name in ("dockerfile", "docker-compose.yml", "docker-compose.yaml"):
            return "devops"

        # Makefile, CMake
        if name in ("makefile", "cmakelists.txt", "justfile"):
            return "devops"

        # README, docs
        if name.endswith(".md") or name.endswith(".rst"):
            return ""  # skip docs for now

        # بالامتداد
        if suffix in FILE_TO_CAPSULE:
            return FILE_TO_CAPSULE[suffix]

        return ""

    def _extract_training_pair(self, filepath: Path, capsule_id: str) -> dict:
        """استخراج زوج Q&A من ملف"""
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except:
            return {}

        if len(content) < 50 or len(content) > 10000:
            return {}  # صغير جداً أو كبير جداً

        name = filepath.name
        rel_path = str(filepath)

        # بناء الـ Q&A حسب نوع الملف
        if capsule_id.startswith("code_"):
            lang = capsule_id.replace("code_", "")
            input_text = f"Explain what this {lang} code does and how it works:\n\n```{lang}\n{content[:2000]}\n```"
            output_text = f"This is the file `{name}`. Here's the code:\n\n```{lang}\n{content[:3000]}\n```"
        elif capsule_id == "devops":
            input_text = f"Explain this configuration file ({name}):\n\n{content[:2000]}"
            output_text = f"This is the configuration file `{name}`:\n\n{content[:3000]}"
        elif capsule_id == "security":
            input_text = f"Analyze the security aspects of this file ({name})"
            output_text = f"Security analysis of `{name}`:\n\n{content[:3000]}"
        elif capsule_id == "database_design":
            input_text = f"Explain this database schema/migration ({name}):\n\n{content[:2000]}"
            output_text = f"Database file `{name}`:\n\n{content[:3000]}"
        elif capsule_id == "code_testing":
            input_text = f"Explain what this test file tests and how:\n\n{content[:2000]}"
            output_text = f"Test file `{name}`:\n\n{content[:3000]}"
        else:
            return {}

        return {
            "input_text": input_text,
            "output_text": output_text,
            "source": "internal_scout",
            "file": rel_path,
            "timestamp": datetime.now().isoformat(),
        }

    def scan_directory(self, scan_dir: str, max_depth: int = 3,
                       max_files: int = 200) -> dict:
        """فحص مجلد وتوزيع الملفات على الكبسولات"""
        scan_path = Path(scan_dir).expanduser()
        if not scan_path.exists():
            return {}

        results = {}
        count = 0

        for filepath in self._walk(scan_path, max_depth):
            if count >= max_files:
                break

            # تخطي ملفات معينة
            if filepath.suffix.lower() in SKIP_EXTENSIONS:
                continue
            if filepath.stat().st_size > 100_000:  # > 100KB
                continue

            # هاش لتجنب التكرار
            file_hash = hashlib.md5(
                f"{filepath}:{filepath.stat().st_mtime}".encode()
            ).hexdigest()[:12]

            if file_hash in self.seen_hashes:
                continue

            # تصنيف الملف
            capsule_id = self._classify_file(filepath)
            if not capsule_id:
                continue

            # استخراج Q&A
            pair = self._extract_training_pair(filepath, capsule_id)
            if not pair:
                continue

            # حفظ في كبسولة
            # حماية inbox — لو قيد التدريب الكشافة تكتب بـ inbox/
            cap_path = self.capsules_dir / capsule_id
            lock_file = cap_path / ".training_in_progress"
            if lock_file.exists():
                data_dir = cap_path / "inbox"
            else:
                data_dir = cap_path / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_dir / "internal_scout.jsonl", "a") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            self.seen_hashes.add(file_hash)
            results[capsule_id] = results.get(capsule_id, 0) + 1
            count += 1

        return results

    def _walk(self, root: Path, max_depth: int, current_depth: int = 0):
        """مشي بالمجلد مع حد العمق"""
        if current_depth > max_depth:
            return

        try:
            entries = list(root.iterdir())
        except (PermissionError, OSError):
            return

        # ملفات أولاً
        for entry in entries:
            if entry.is_file():
                yield entry

        # ثم المجلدات
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRS:
                yield from self._walk(entry, max_depth, current_depth + 1)

    def scan_all(self) -> dict:
        """فحص كل المجلدات المُعرّفة"""
        logger.info("🔎 Internal Scout — scanning local files...")
        total_results = {}

        for scan_dir, depth in SCAN_DIRS:
            expanded = str(Path(scan_dir).expanduser())
            results = self.scan_directory(expanded, depth)
            for k, v in results.items():
                total_results[k] = total_results.get(k, 0) + v
            if results:
                logger.info(f"  📂 {scan_dir}: +{sum(results.values())} files")

        self._save_seen()
        total = sum(total_results.values())
        logger.info(f"🔎 Internal: +{total} real samples from local files")
        return total_results


# ═══════════════════════════════════════════════════════════
# الكشّافة Offline — تستكشف بدون إنترنت 🔌🔍
# ═══════════════════════════════════════════════════════════

class OfflineScout:
    """
    كشّافة offline — تشتغل بدون إنترنت!

    تستكشف:
      1. الشبكة المحلية (LAN) — أجهزة متصلة
      2. أجهزة USB/Bluetooth
      3. ملفات مشاركة (SMB, NFS)
      4. حزم الشبكة (packet sniffing)
      5. أي مصدر بيانات محلي
    """

    def __init__(self):
        self.discovered_devices: Dict[str, Dict] = {}
        self.discovered_services: List[Dict] = []
        self.scan_history: List[Dict] = []

    def scan_lan(self, subnet: str = None) -> List[Dict]:
        """
        فحص الشبكة المحلية — يكتشف كل الأجهزة

        يستخدم: arp, ping sweep
        """
        devices = []

        # 1. ARP table — أجهزة معروفة
        try:
            result = subprocess.run(
                ["arp", "-a"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and "." in parts[1]:
                    ip = parts[1].strip("()")
                    mac = parts[3] if len(parts) > 3 else "unknown"
                    device = {
                        "ip": ip, "mac": mac, "source": "arp",
                        "timestamp": datetime.now().isoformat(),
                    }
                    devices.append(device)
                    self.discovered_devices[ip] = device
        except Exception as e:
            logger.warning(f"ARP scan failed: {e}")

        # 2. Ping sweep — subnet
        if subnet:
            try:
                base = subnet.rsplit(".", 1)[0]
                for i in range(1, 255):
                    ip = f"{base}.{i}"
                    result = subprocess.run(
                        ["ping", "-c", "1", "-W", "1", ip],
                        capture_output=True, timeout=2,
                    )
                    if result.returncode == 0 and ip not in self.discovered_devices:
                        device = {"ip": ip, "source": "ping", "mac": "",
                                 "timestamp": datetime.now().isoformat()}
                        devices.append(device)
                        self.discovered_devices[ip] = device
            except Exception:
                pass

        self.scan_history.append({
            "type": "lan_scan", "devices": len(devices),
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"🔌 LAN: {len(devices)} devices discovered")
        return devices

    def scan_ports(self, ip: str, ports: List[int] = None) -> List[Dict]:
        """فحص ports مفتوحة — يكتشف خدمات"""
        if not ports:
            ports = [22, 80, 443, 445, 8080, 3306, 5432, 6379, 27017,
                     11434, 8400, 554, 1883, 502]  # SSH, HTTP, SMB, DBs, Ollama, RTSP, MQTT, Modbus

        services = []
        import socket

        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                if result == 0:
                    service_names = {
                        22: "SSH", 80: "HTTP", 443: "HTTPS", 445: "SMB",
                        8080: "HTTP-Alt", 3306: "MySQL", 5432: "PostgreSQL",
                        6379: "Redis", 27017: "MongoDB", 11434: "Ollama",
                        8400: "BrainAPI", 554: "RTSP-Camera", 1883: "MQTT",
                        502: "Modbus",
                    }
                    svc = {
                        "ip": ip, "port": port,
                        "service": service_names.get(port, f"port-{port}"),
                        "open": True,
                    }
                    services.append(svc)
                    self.discovered_services.append(svc)
                sock.close()
            except Exception:
                pass

        logger.info(f"🔍 Ports {ip}: {len(services)} open")
        return services

    def discover_usb(self) -> List[Dict]:
        """اكتشاف أجهزة USB متصلة"""
        devices = []
        try:
            if _OS == "Darwin":
                result = subprocess.run(
                    ["system_profiler", "SPUSBDataType", "-json"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for item in data.get("SPUSBDataType", []):
                        devices.append({
                            "name": item.get("_name", "Unknown USB"),
                            "type": "usb", "source": "system_profiler",
                        })
            elif _OS == "Linux":
                result = subprocess.run(
                    ["lsusb"], capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines():
                    devices.append({
                        "name": line.strip(), "type": "usb", "source": "lsusb",
                    })
        except Exception as e:
            logger.warning(f"USB scan: {e}")

        logger.info(f"🔌 USB: {len(devices)} devices")
        return devices

    def discover_shared_files(self) -> List[Dict]:
        """اكتشاف ملفات مشاركة (SMB/NFS/AirDrop)"""
        shares = []

        # SMB shares
        for ip, device in self.discovered_devices.items():
            try:
                result = subprocess.run(
                    ["smbclient", "-L", ip, "-N"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "Disk" in line:
                            share_name = line.split()[0]
                            shares.append({
                                "ip": ip, "name": share_name,
                                "type": "smb", "path": f"//{ip}/{share_name}",
                            })
            except (FileNotFoundError, Exception):
                pass

        logger.info(f"📁 Shared: {len(shares)} shares found")
        return shares

    def discover_mounted_volumes(self) -> List[Dict]:
        """اكتشاف أقراص خارجية / USB مركّبة"""
        volumes = []

        # macOS: /Volumes
        vol_dir = Path("/Volumes")
        if vol_dir.exists():
            for v in vol_dir.iterdir():
                if v.is_dir() and v.name != "Macintosh HD":
                    try:
                        files = list(v.rglob("*"))[:100]
                        readable = [f for f in files if f.is_file() and f.stat().st_size > 0]
                        volumes.append({
                            "name": v.name, "path": str(v),
                            "type": "volume", "files": len(readable),
                            "size_gb": round(sum(f.stat().st_size for f in readable[:50]) / (1024**3), 2),
                        })
                    except PermissionError:
                        volumes.append({"name": v.name, "path": str(v), "type": "volume", "files": 0, "error": "permission"})

        # Linux: /mnt, /media
        for mnt_dir in [Path("/mnt"), Path("/media")]:
            if mnt_dir.exists():
                for v in mnt_dir.iterdir():
                    if v.is_dir():
                        try:
                            files = list(v.rglob("*"))[:100]
                            readable = [f for f in files if f.is_file()]
                            volumes.append({
                                "name": v.name, "path": str(v),
                                "type": "mount", "files": len(readable),
                            })
                        except PermissionError:
                            pass

        logger.info(f"💾 Mounted volumes: {len(volumes)}")
        return volumes

    def scan_usb_files(self, extensions: List[str] = None, max_files: int = 200) -> List[Dict]:
        """مسح ملفات من USB/أقراص خارجية للتعلّم"""
        if not extensions:
            extensions = [".py", ".json", ".csv", ".txt", ".md", ".yaml", ".yml",
                         ".sql", ".html", ".css", ".js", ".ts", ".rs", ".go",
                         ".toml", ".ini", ".cfg", ".conf", ".xml"]

        files_found = []
        volumes = self.discover_mounted_volumes()

        for vol in volumes:
            vol_path = Path(vol["path"])
            count = 0
            try:
                for f in vol_path.rglob("*"):
                    if count >= max_files // max(len(volumes), 1):
                        break
                    if f.is_file() and f.suffix.lower() in extensions:
                        try:
                            size = f.stat().st_size
                            if 100 < size < 500_000:  # 100B - 500KB
                                files_found.append({
                                    "path": str(f), "name": f.name,
                                    "ext": f.suffix, "size": size,
                                    "volume": vol["name"],
                                })
                                count += 1
                        except PermissionError:
                            pass
            except PermissionError:
                pass

        logger.info(f"📁 USB files scanned: {len(files_found)}")
        return files_found[:max_files]

    def collect_local_knowledge(self) -> Dict:
        """جمع كل المعرفة المحلية المتاحة"""
        knowledge = {
            "lan_devices": len(self.discovered_devices),
            "services": len(self.discovered_services),
            "data_sources": [],
            "mounted_volumes": [],
        }

        # بحث عن مصادر بيانات محلية
        local_sources = [
            "/usr/share/dict/words",          # قاموس
            "/usr/share/doc",                  # توثيق
            "/usr/share/man",                  # manpages
            "/usr/share/info",                 # info pages
        ]

        for src in local_sources:
            if os.path.exists(src):
                knowledge["data_sources"].append(src)

        # أقراص خارجية
        knowledge["mounted_volumes"] = self.discover_mounted_volumes()

        # بحث عن databases محلية
        db_patterns = ["*.db", "*.sqlite", "*.sqlite3"]
        for pattern in db_patterns:
            for db in Path.home().rglob(pattern):
                try:
                    if db.stat().st_size > 1000:  # > 1KB
                        knowledge["data_sources"].append(str(db))
                        if len(knowledge["data_sources"]) > 50:
                            break
                except Exception:
                    pass

        return knowledge

    def full_scan(self) -> Dict:
        """فحص كامل — كل شي متوفر offline"""
        results = {
            "lan": self.scan_lan(),
            "usb": self.discover_usb(),
            "shares": self.discover_shared_files(),
            "volumes": self.discover_mounted_volumes(),
            "usb_files": len(self.scan_usb_files()),
            "local": self.collect_local_knowledge(),
            "timestamp": datetime.now().isoformat(),
        }

        # حفظ بالذاكرة
        memory.save_knowledge(
            topic="Offline Scout Full Scan",
            content=f"LAN: {len(results['lan'])} devices, "
                    f"USB: {len(results['usb'])}, "
                    f"Shares: {len(results['shares'])}, "
                    f"Local sources: {len(results['local']['data_sources'])}",
            source="offline_scout",
        )

        return results


# ═══════════════════════════════════════════════════════════
# الكشّافة الموحدة — online + offline + internal
# ═══════════════════════════════════════════════════════════

class UnifiedScout:
    """
    كشّافة موحدة — تبدّل بين الأوضاع أوتوماتيكياً

    Online:  KnowledgeScout (Ollama + API)
    Offline: OfflineScout (LAN + USB + files)
    Always:  InternalScout (file system)
    """

    def __init__(self):
        self.online = KnowledgeScout()
        self.offline = OfflineScout()
        self.internal = InternalScout()

    def is_online(self) -> bool:
        """فحص الإنترنت"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except (OSError, Exception):
            return False

    def scout_cycle(self, samples_per_capsule: int = 10) -> Dict:
        """دورة كشافة — online أو offline أوتوماتيكي"""
        results = {"mode": "", "online": {}, "offline": {}, "internal": {}}

        # دائماً: ملفات محلية
        results["internal"] = self.internal.scan_all()

        if self.is_online():
            results["mode"] = "online"
            try:
                self.online.scout_cycle(samples_per_capsule)
                results["online"] = {"samples": self.online.total_generated}
            except Exception as e:
                logger.warning(f"Online scout error: {e}")
        else:
            results["mode"] = "offline"
            results["offline"] = {
                "lan": len(self.offline.scan_lan()),
                "usb": len(self.offline.discover_usb()),
            }

        logger.info(f"🔍 Unified Scout: mode={results['mode']}, "
                     f"internal={sum(results['internal'].values())} files")
        return results

    def get_status(self) -> Dict:
        return {
            "online": self.is_online(),
            "total_generated": self.online.total_generated,
            "lan_devices": len(self.offline.discovered_devices),
            "services": len(self.offline.discovered_services),
            "seen_files": len(self.internal.seen_hashes),
        }


# Singletons
scout = KnowledgeScout()
offline_scout = OfflineScout()
unified_scout = UnifiedScout()


if __name__ == "__main__":
    print("🔍 Unified Scout System — Test\n")

    # فحص الإنترنت
    us = UnifiedScout()
    online = us.is_online()
    print(f"Internet: {'✅ Online' if online else '❌ Offline'}\n")

    # فحص LAN
    print("═" * 40)
    print("LAN Scan:")
    devices = us.offline.scan_lan()
    for d in devices[:5]:
        print(f"  📱 {d['ip']} ({d.get('mac', '?')})")
    print(f"  Total: {len(devices)} devices\n")

    # USB
    print("USB Devices:")
    usb = us.offline.discover_usb()
    for d in usb[:5]:
        print(f"  🔌 {d['name']}")
    print(f"  Total: {len(usb)}\n")

    # ملفات محلية
    print("Internal Scan:")
    internal = us.internal.scan_all()
    for capsule, count in sorted(internal.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  📂 {capsule}: +{count} files")

    print(f"\nStatus: {us.get_status()}")

