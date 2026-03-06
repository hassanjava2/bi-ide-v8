#!/usr/bin/env python3
"""
capsule_router.py — كبسولة القائد 🧠👑

أهم كبسولة بالنظام:
  1. يفهم سؤال المستخدم
  2. يختار الكبسولات المناسبة
  3. يرسل لكل كبسولة مهمتها
  4. يجمع الأجوبة بجواب واحد متكامل

المعمارية:
  User → Router → [capsule_1, capsule_2, ...] → Router → Answer

التطور:
  - Cycle 1-10: يستخدم قواعد ثابتة (keyword matching)
  - Cycle 10-50: يتعلم من نمط الأسئلة (pattern learning)
  - Cycle 50+: يقرر لحاله بالكامل (autonomous routing)
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("router")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"
ROUTER_LOG = PROJECT_ROOT / "brain" / "capsules" / ".router_log.jsonl"


# ═══════════════════════════════════════════════════════════
# خارطة الكبسولات — أي كبسولة تحسن بأي شي
# ═══════════════════════════════════════════════════════════

CAPSULE_REGISTRY = {
    # === برمجة ===
    "code_python": {
        "name": "Python Expert",
        "skills": ["python", "django", "flask", "fastapi", "torch", "numpy", "pandas"],
        "keywords": ["python", "py", "django", "flask", "pip", "pytorch", "numpy", "class", "def"],
        "priority": 10,
    },
    "code_typescript": {
        "name": "TypeScript Expert",
        "skills": ["typescript", "javascript", "react", "nextjs", "nodejs", "html", "css"],
        "keywords": ["typescript", "javascript", "react", "next", "node", "tsx", "jsx", "vue", "angular", "npm", "frontend", "واجهة"],
        "priority": 10,
    },
    "code_rust": {
        "name": "Rust Expert",
        "skills": ["rust", "systems", "memory", "performance", "wasm"],
        "keywords": ["rust", "cargo", "unsafe", "ownership", "borrow", "wasm", "tauri"],
        "priority": 10,
    },
    "code_sql": {
        "name": "SQL Expert",
        "skills": ["sql", "postgres", "mysql", "sqlite", "prisma"],
        "keywords": ["sql", "database", "query", "select", "insert", "join", "postgres", "mysql", "قاعدة بيانات"],
        "priority": 10,
    },
    "code_css": {
        "name": "CSS/Design Expert",
        "skills": ["css", "tailwind", "animation", "responsive", "design"],
        "keywords": ["css", "style", "animation", "responsive", "tailwind", "design", "color", "تصميم", "لون"],
        "priority": 8,
    },

    # === بنية تحتية ===
    "devops": {
        "name": "DevOps Engineer",
        "skills": ["docker", "nginx", "linux", "deploy", "ci/cd", "ssh"],
        "keywords": ["docker", "deploy", "server", "nginx", "linux", "ssh", "ci", "cd", "kubernetes", "سيرفر", "نشر"],
        "priority": 9,
    },
    "database_design": {
        "name": "Database Architect",
        "skills": ["schema", "migration", "orm", "relations", "indexing"],
        "keywords": ["schema", "migration", "relation", "index", "model", "prisma", "هيكل", "جدول"],
        "priority": 9,
    },

    # === ذكاء ===
    "knowledge_arabic": {
        "name": "Arabic Knowledge",
        "skills": ["arabic", "general_knowledge", "translation", "culture"],
        "keywords": ["عربي", "ترجم", "شنو", "شلون", "ليش", "تاريخ", "ثقافة", "معنى"],
        "priority": 7,
    },
    "conversation_ar": {
        "name": "Arabic Conversation",
        "skills": ["chat", "conversation", "personality"],
        "keywords": ["هلا", "شلونك", "محادثة", "سالفة", "حجي", "قصة"],
        "priority": 6,
    },

    # === أمن ===
    "security": {
        "name": "Security Expert",
        "skills": ["security", "encryption", "vulnerabilities", "pentest", "firewall"],
        "keywords": ["security", "hack", "encrypt", "vulnerability", "cve", "firewall", "ssl", "أمن", "اختراق", "ثغرة", "تشفير"],
        "priority": 10,
    },

    # === مجلس ===
    "sage": {
        "name": "The Sage (الحكيم)",
        "skills": ["strategy", "analysis", "wisdom", "planning"],
        "keywords": ["استراتيجية", "خطة", "تحليل", "رأي", "نصيحة", "strategy", "plan", "analyze"],
        "priority": 5,
    },
    "rebel": {
        "name": "The Rebel (المتمرد)",
        "skills": ["criticism", "alternatives", "challenge", "innovation"],
        "keywords": ["بديل", "مشكلة", "خطأ", "أحسن", "challenge", "alternative", "wrong"],
        "priority": 5,
    },
    "translator": {
        "name": "Translator",
        "skills": ["translation", "languages", "localization"],
        "keywords": ["translate", "ترجم", "translation", "english", "arabic", "إنجليزي"],
        "priority": 8,
    },
    "code_testing": {
        "name": "Testing Expert",
        "skills": ["testing", "unit_test", "integration", "qa"],
        "keywords": ["test", "testing", "unittest", "pytest", "jest", "تجريب", "فحص"],
        "priority": 8,
    },
}


@dataclass
class RoutingDecision:
    """قرار التوجيه"""
    query: str
    selected_capsules: List[str]
    confidence: float
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CapsuleResponse:
    """جواب كبسولة"""
    capsule_id: str
    response: str
    confidence: float
    time_ms: float


class CapsuleRouter:
    """
    القائد — يفهم السؤال ويوجه للكبسولات المناسبة
    
    مراحل التطور:
      1. Rule-based routing (الحين)
      2. Pattern-based (Cycle 10+)  
      3. AI-based (Cycle 50+)
    """

    def __init__(self, capsules_dir: Path = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        self.registry = CAPSULE_REGISTRY.copy()
        self.history: List[RoutingDecision] = []
        self._load_available_capsules()
        logger.info(f"🧠 Router initialized: {len(self.available)} capsules available")

    def _load_available_capsules(self):
        """اكتشاف الكبسولات المتوفرة"""
        self.available = {}
        for capsule_id, info in self.registry.items():
            capsule_dir = self.capsules_dir / capsule_id
            model_dir = capsule_dir / "model"
            has_model = model_dir.exists() and any(model_dir.iterdir()) if model_dir.exists() else False
            has_data = (capsule_dir / "data").exists() if capsule_dir.exists() else False

            self.available[capsule_id] = {
                **info,
                "has_model": has_model,
                "has_data": has_data,
                "ready": has_model,  # جاهز فقط لو عنده موديل متدرب
            }

    def analyze_query(self, query: str) -> Dict:
        """تحليل السؤال — فهم النية والمجال"""
        query_lower = query.lower()

        analysis = {
            "language": "ar" if any(c >= '\u0600' and c <= '\u06FF' for c in query) else "en",
            "is_code_request": any(w in query_lower for w in ["code", "كود", "برمج", "سوي", "اكتب", "function", "class"]),
            "is_question": any(w in query_lower for w in ["?", "شنو", "شلون", "ليش", "كيف", "what", "how", "why"]),
            "is_fix_request": any(w in query_lower for w in ["fix", "bug", "error", "صلح", "خطأ", "مشكلة"]),
            "is_deploy": any(w in query_lower for w in ["deploy", "server", "نشر", "سيرفر"]),
            "is_security": any(w in query_lower for w in ["hack", "security", "أمن", "اختراق", "ثغرة"]),
            "is_design": any(w in query_lower for w in ["design", "ui", "تصميم", "واجهة"]),
            "is_chat": any(w in query_lower for w in ["هلا", "شلونك", "سالفة"]),
            "needs_council": any(w in query_lower for w in ["رأي", "خطة", "استراتيجية", "قرار", "plan", "strategy"]),
            "complexity": self._estimate_complexity(query),
        }

        return analysis

    def _estimate_complexity(self, query: str) -> str:
        """تقدير تعقيد السؤال"""
        words = query.split()
        if len(words) < 5:
            return "simple"
        elif len(words) < 20:
            return "medium"
        else:
            return "complex"

    def route(self, query: str, max_capsules: int = 3) -> RoutingDecision:
        """
        التوجيه الرئيسي — يختار الكبسولات المناسبة
        
        الخوارزمية:
          1. تحليل السؤال
          2. حساب نقاط كل كبسولة
          3. اختيار الأعلى نقاطاً
          4. إرجاع القرار
        """
        query_lower = query.lower()
        analysis = self.analyze_query(query)

        # حساب نقاط كل كبسولة
        scores: Dict[str, float] = {}
        for capsule_id, info in self.available.items():
            score = 0.0

            # مطابقة كلمات مفتاحية
            for keyword in info.get("keywords", []):
                if keyword in query_lower:
                    score += 10.0

            # أولوية الكبسولة
            score += info.get("priority", 5) * 0.5

            # بونص لو جاهزة (عندها موديل)
            if info.get("ready"):
                score += 5.0

            # بونص حسب تحليل السؤال
            if analysis["is_code_request"] and "code" in capsule_id:
                score += 8.0
            if analysis["is_security"] and capsule_id == "security":
                score += 15.0
            if analysis["is_deploy"] and capsule_id == "devops":
                score += 12.0
            if analysis["is_chat"] and capsule_id == "conversation_ar":
                score += 15.0
            if analysis["needs_council"] and capsule_id in ("sage", "rebel"):
                score += 10.0
            if analysis["is_design"] and capsule_id == "code_css":
                score += 10.0
            if analysis["is_fix_request"]:
                score += 3.0  # بونص عام للإصلاح

            if score > 0:
                scores[capsule_id] = score

        # ترتيب حسب النقاط
        sorted_capsules = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # === منطق الاختيار الذكي ===
        selected = []
        reasoning_parts = []

        if analysis["complexity"] == "simple" or analysis["is_chat"]:
            # سؤال بسيط → كبسولة واحدة
            if sorted_capsules:
                selected = [sorted_capsules[0][0]]
                reasoning_parts.append(f"Simple query → single capsule: {sorted_capsules[0][0]}")
        elif analysis["needs_council"]:
            # يحتاج مجلس → sage + rebel + الأعلى
            selected = ["sage", "rebel"]
            for cid, _ in sorted_capsules:
                if cid not in selected and len(selected) < max_capsules:
                    selected.append(cid)
            reasoning_parts.append("Council needed → sage + rebel + specialist")
        else:
            # متوسط/معقد → أعلى 2-3 كبسولات
            for cid, score in sorted_capsules[:max_capsules]:
                if score > 5.0:  # حد أدنى
                    selected.append(cid)
            reasoning_parts.append(f"Multi-capsule routing: {', '.join(selected)}")

        # fallback
        if not selected:
            selected = ["knowledge_arabic"] if analysis["language"] == "ar" else ["code_python"]
            reasoning_parts.append("Fallback: no strong match found")

        # حساب الثقة
        confidence = min(1.0, (sorted_capsules[0][1] / 30.0) if sorted_capsules else 0.3)

        decision = RoutingDecision(
            query=query,
            selected_capsules=selected,
            confidence=round(confidence, 2),
            reasoning=" | ".join(reasoning_parts),
        )

        # تسجيل القرار
        self.history.append(decision)
        self._log_decision(decision)

        logger.info(f"🧠 Route: '{query[:50]}...' → {selected} (conf: {confidence:.0%})")
        return decision

    def combine_responses(self, query: str, responses: List[CapsuleResponse]) -> str:
        """
        دمج أجوبة الكبسولات بجواب واحد متكامل
        
        القائد يجمع، يرتب، ويصيغ الجواب النهائي
        """
        if not responses:
            return "عذراً، ما كدرت أجاوب على هالسؤال."

        if len(responses) == 1:
            return responses[0].response

        # ترتيب حسب الثقة
        responses.sort(key=lambda r: r.confidence, reverse=True)

        # === بناء الجواب المدمج ===
        combined_parts = []

        # المقدمة
        capsule_names = [self.registry.get(r.capsule_id, {}).get("name", r.capsule_id) for r in responses]
        combined_parts.append(
            f"**استشرت {len(responses)} خبراء**: {', '.join(capsule_names)}\n"
        )

        # جواب كل خبير
        for i, resp in enumerate(responses):
            name = self.registry.get(resp.capsule_id, {}).get("name", resp.capsule_id)
            combined_parts.append(f"### {name}:")
            combined_parts.append(resp.response)
            combined_parts.append("")  # سطر فارغ

        # الخلاصة (لو أكثر من جواب)
        if len(responses) >= 2:
            primary = responses[0]
            primary_name = self.registry.get(primary.capsule_id, {}).get("name", primary.capsule_id)
            combined_parts.append(f"---\n**الخلاصة**: الجواب الأقوى من **{primary_name}** (ثقة: {primary.confidence:.0%})")

        return "\n".join(combined_parts)

    def get_capsule_prompt(self, capsule_id: str, query: str) -> str:
        """تحضير prompt مخصص لكل كبسولة"""
        info = self.registry.get(capsule_id, {})
        name = info.get("name", capsule_id)
        skills = ", ".join(info.get("skills", []))

        return (
            f"You are {name}, specialized in: {skills}.\n"
            f"Answer the following question using your expertise.\n"
            f"Be concise, practical, and provide code examples when relevant.\n\n"
            f"Question: {query}"
        )

    def get_status(self) -> Dict:
        """حالة القائد والكبسولات"""
        ready = sum(1 for c in self.available.values() if c.get("ready"))
        total = len(self.available)
        return {
            "total_capsules": total,
            "ready_capsules": ready,
            "total_queries": len(self.history),
            "capsules": {
                cid: {
                    "name": info.get("name"),
                    "ready": info.get("ready", False),
                    "has_data": info.get("has_data", False),
                }
                for cid, info in self.available.items()
            }
        }

    def _log_decision(self, decision: RoutingDecision):
        """تسجيل القرار للتعلم المستقبلي"""
        try:
            with open(ROUTER_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": decision.query[:200],
                    "selected": decision.selected_capsules,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "timestamp": decision.timestamp,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def learn_from_feedback(self, query: str, correct_capsules: List[str]):
        """
        التعلم من التصحيح — المرحلة الثانية من التطور
        
        لو المستخدم قال "الجواب من كبسولة X أحسن":
          → القائد يتعلم ويطور قواعد التوجيه
        """
        log_entry = {
            "type": "feedback",
            "query": query,
            "correct_capsules": correct_capsules,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            feedback_file = self.capsules_dir / ".router_feedback.jsonl"
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.info(f"📝 Feedback recorded: {correct_capsules}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# Singleton + API
# ═══════════════════════════════════════════════════════════

router = CapsuleRouter()


def ask(query: str, max_capsules: int = 3) -> RoutingDecision:
    """نقطة الدخول الرئيسية"""
    return router.route(query, max_capsules)


def status() -> Dict:
    """حالة النظام"""
    return router.get_status()


# ═══════════════════════════════════════════════════════════
# تجربة
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧠 Capsule Router — Test Mode\n")

    test_queries = [
        "سوي لي API بـ Python مع قاعدة بيانات",
        "شنو ثغرات الـ CVE الجديدة؟",
        "شلون أنشر المشروع على السيرفر؟",
        "سوي واجهة مستخدم حلوة بـ React",
        "شنو رأيك بخطة المشروع؟",
        "هلا شلونك؟",
        "Explain buffer overflow vulnerability",
        "Write a Rust function to sort files",
        "ترجم هالنص للإنجليزي",
        "Fix this Python error: IndexError",
    ]

    for q in test_queries:
        decision = ask(q)
        capsules = " + ".join(decision.selected_capsules)
        print(f"  Q: {q}")
        print(f"  → {capsules} (conf: {decision.confidence:.0%})")
        print(f"  💭 {decision.reasoning}")
        print()

    print(f"\n📊 Status: {json.dumps(status(), indent=2, ensure_ascii=False)}")
