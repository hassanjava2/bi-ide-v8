#!/usr/bin/env python3
"""
knowledge_scout.py — الكشّافة

تجيب بيانات جديدة بدون توقف لكل الكبسولات:
1. تولّد بيانات Q&A عبر Ollama (local models)
2. تصنّف كل معلومة → أي كبسولة تستفيد
3. توزّع البيانات على الكبسولات
4. تكرر ∞

الكشافة ما تتعب — تشتغل 24/7
"""

import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

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
        data_dir = self.capsules_dir / capsule_id / "data"
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
        """دورة كشافة واحدة — تولّد بيانات لكل الكبسولات"""
        self.cycle_count += 1
        model = self._get_model()
        if not model:
            logger.error("No Ollama model available!")
            return {"error": "no_model"}

        logger.info(f"🔍 Scout cycle #{self.cycle_count} — model: {model}")

        results = {}
        capsule_dirs = sorted(self.capsules_dir.iterdir())

        for capsule_dir in capsule_dirs:
            if not capsule_dir.is_dir():
                continue
            cid = capsule_dir.name

            # تخطي الأرشيف
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
                results[cid] = count
                logger.info(f"  📦 {cid}: +{count} samples")

        logger.info(f"🔍 Cycle #{self.cycle_count} done: "
                    f"+{sum(results.values())} samples across {len(results)} capsules")
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


# Singleton
scout = KnowledgeScout()
