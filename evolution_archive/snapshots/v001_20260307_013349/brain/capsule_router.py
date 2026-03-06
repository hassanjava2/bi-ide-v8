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
    mode: str = "fast"  # fast | deep
    research_data: Optional[Dict] = None  # بيانات البحث العميق
    needs_training: List[str] = field(default_factory=list)  # كبسولات تحتاج تدريب
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

    def route(self, query: str, max_capsules: int = 3, mode: str = "fast") -> RoutingDecision:
        """
        التوجيه الرئيسي — يختار الكبسولات المناسبة
        
        mode:
          - 'fast': استخدم الي تعرفه حالياً ⚡
          - 'deep': ابحث عن المنافسين، تعلم، ثم ابني 🔬
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

        # === منطق الاختيار ===
        selected = []
        reasoning_parts = []
        needs_training = []
        research_data = None

        if mode == "deep":
            # === الوضع العميق 🔬 ===
            reasoning_parts.append("🔬 DEEP MODE")

            # 1. بحث عن المنافسين
            research_data = self._deep_research(query, analysis)
            reasoning_parts.append(f"Researched {research_data.get('competitors_found', 0)} competitors")

            # 2. كل الكبسولات ذات الصلة (حتى 5)
            for cid, score in sorted_capsules[:5]:
                if score > 3.0:
                    selected.append(cid)

            # 3. المجلس دائماً بالوضع العميق
            if "sage" not in selected:
                selected.append("sage")
            if "rebel" not in selected:
                selected.append("rebel")

            # 4. اكتشاف الكبسولات الي تحتاج تدريب
            for cid in selected:
                info = self.available.get(cid, {})
                if not info.get("ready") and not info.get("has_data"):
                    needs_training.append(cid)

            if needs_training:
                reasoning_parts.append(f"Need training: {', '.join(needs_training)}")

        elif analysis["complexity"] == "simple" or analysis["is_chat"]:
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
            mode=mode,
            research_data=research_data,
            needs_training=needs_training,
        )

        # تسجيل القرار
        self.history.append(decision)
        self._log_decision(decision)

        logger.info(f"🧠 Route [{mode}]: '{query[:50]}...' → {selected} (conf: {confidence:.0%})")
        return decision

    def _deep_research(self, query: str, analysis: Dict) -> Dict:
        """
        البحث العميق 🔬 — يجمع معلومات عن المنافسين والبدائل
        
        1. يحلل نوع المشروع المطلوب
        2. يبحث عن المنافسين
        3. يستخرج ميزاتهم
        4. يولّد بيانات تدريب للكبسولات
        """
        import urllib.request
        import urllib.parse

        research = {
            "query": query,
            "competitors_found": 0,
            "competitors": [],
            "features": [],
            "tech_stack": [],
            "recommendations": [],
        }

        # استخراج نوع المشروع
        project_keywords = self._extract_project_type(query)
        if not project_keywords:
            return research

        # === بحث GitHub عن مشاريع مشابهة ===
        try:
            search_query = urllib.parse.quote(" ".join(project_keywords))
            url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&per_page=5"
            data = json.loads(urllib.request.urlopen(url, timeout=10).read())

            for repo in data.get("items", [])[:5]:
                competitor = {
                    "name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "language": repo.get("language", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "topics": repo.get("topics", []),
                }
                research["competitors"].append(competitor)
                research["competitors_found"] += 1

                # جمع التقنيات
                if competitor["language"] and competitor["language"] not in research["tech_stack"]:
                    research["tech_stack"].append(competitor["language"])

                # جمع المواضيع كميزات
                for topic in competitor.get("topics", [])[:5]:
                    if topic not in research["features"]:
                        research["features"].append(topic)

            logger.info(f"🔬 Found {research['competitors_found']} competitors on GitHub")

        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")

        # === توصيات بناءً على البحث ===
        if research["competitors"]:
            top = research["competitors"][0]
            research["recommendations"] = [
                f"Top competitor: {top['name']} ({top['stars']} stars)",
                f"Recommended tech: {', '.join(research['tech_stack'][:3])}",
                f"Must-have features: {', '.join(research['features'][:10])}",
            ]

        # === حفظ بيانات البحث كتدريب ===
        self._save_research_as_training(query, research)

        return research

    def _extract_project_type(self, query: str) -> List[str]:
        """استخراج نوع المشروع من السؤال"""
        # كلمات مفتاحية للمشاريع
        project_words = {
            "تطبيق": "app", "موقع": "website", "متجر": "ecommerce store",
            "توصيل": "delivery app", "دردشة": "chat app", "لعبة": "game",
            "مدونة": "blog", "منتدى": "forum", "إدارة": "management system",
            "app": "app", "website": "website", "store": "ecommerce",
            "delivery": "delivery", "chat": "chat", "game": "game",
            "dashboard": "dashboard", "api": "api", "cms": "cms",
            "erp": "erp", "crm": "crm", "social": "social media",
        }

        keywords = []
        query_lower = query.lower()
        for word, english in project_words.items():
            if word in query_lower:
                keywords.append(english)

        return keywords if keywords else query.split()[:3]

    def _save_research_as_training(self, query: str, research: Dict):
        """حفظ نتائج البحث كبيانات تدريب"""
        if not research.get("competitors"):
            return

        samples = []
        for comp in research["competitors"]:
            sample = {
                "input_text": f"What is {comp['name']}? Analyze its features and architecture.",
                "output_text": (
                    f"{comp['name']}: {comp['description']}\n"
                    f"Language: {comp['language']}, Stars: {comp['stars']}\n"
                    f"Topics: {', '.join(comp.get('topics', []))}"
                ),
            }
            samples.append(sample)

        if samples:
            try:
                data_dir = self.capsules_dir / "knowledge_arabic" / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                out_file = data_dir / f"deep_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for s in samples:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                logger.info(f"💾 Saved {len(samples)} research samples")
            except Exception:
                pass

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


def ask(query: str, max_capsules: int = 3, mode: str = "fast") -> RoutingDecision:
    """نقطة الدخول الرئيسية"""
    return router.route(query, max_capsules, mode)


def status() -> Dict:
    """حالة النظام"""
    return router.get_status()


# ═══════════════════════════════════════════════════════════
# تجربة
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧠 Capsule Router — Test Mode\n")

    test_queries = [
        ("سوي لي API بـ Python مع قاعدة بيانات", "fast"),
        ("شنو ثغرات الـ CVE الجديدة؟", "fast"),
        ("سوي تطبيق توصيل أفضل من Uber", "deep"),
        ("سوي واجهة مستخدم حلوة بـ React", "fast"),
        ("سوي متجر إلكتروني", "deep"),
        ("هلا شلونك؟", "fast"),
    ]

    for q, m in test_queries:
        decision = ask(q, mode=m)
        capsules = " + ".join(decision.selected_capsules)
        icon = "⚡" if m == "fast" else "🔬"
        print(f"  {icon} Q: {q}")
        print(f"    → {capsules} (conf: {decision.confidence:.0%})")
        print(f"    💭 {decision.reasoning}")
        if decision.research_data and decision.research_data.get("competitors"):
            print(f"    📊 Competitors: {[c['name'] for c in decision.research_data['competitors'][:3]]}")
        if decision.needs_training:
            print(f"    🎓 Needs training: {decision.needs_training}")
        print()

    print(f"\n📊 Status: {json.dumps(status(), indent=2, ensure_ascii=False)}")
