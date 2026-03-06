#!/usr/bin/env python3
"""
council_auto_debate.py — مجلس الحكماء 16 حكيم 🏛️

المجلس يتناقش أوتوماتيكياً 24/7:
  - كل حكيم = كبسولة متخصصة
  - يطرحون مواضيع ← يناقشون ← يصوتون ← يقررون
  - النتائج تُحفظ بالذاكرة الأبدية
  - المستخدم يشوف النقاشات مباشرة بالـ IDE
"""

import json
import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("council")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.capsule_router import router as capsule_router
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.capsule_router import router as capsule_router
    from brain.memory_system import memory

# ═══════════════════════════════════════════════════════════
# الحكماء الـ 16
# ═══════════════════════════════════════════════════════════

SAGES = {
    "tech_sage": {
        "name": "حكيم التقنية",
        "name_en": "Tech Sage",
        "capsules": ["code_python", "code_typescript"],
        "expertise": "البرمجة والهندسة البرمجية",
        "personality": "دقيق، يحب الكود النظيف، يكره التعقيد",
        "color": "#3B82F6",
        "icon": "💻",
    },
    "security_sage": {
        "name": "حكيم الأمان",
        "name_en": "Security Sage",
        "capsules": ["security"],
        "expertise": "الأمن السيبراني والتشفير",
        "personality": "حذر جداً، يشك بكل شي، يبحث عن الثغرات",
        "color": "#EF4444",
        "icon": "🔒",
    },
    "infra_sage": {
        "name": "حكيم البنية",
        "name_en": "Infrastructure Sage",
        "capsules": ["devops"],
        "expertise": "السيرفرات والنشر والبنية التحتية",
        "personality": "عملي، يركز على الاستقرار والأداء",
        "color": "#10B981",
        "icon": "🏗️",
    },
    "data_sage": {
        "name": "حكيم البيانات",
        "name_en": "Data Sage",
        "capsules": ["code_sql", "database_design"],
        "expertise": "قواعد البيانات وهيكلة المعلومات",
        "personality": "منظم، يحب التطبيع والعلاقات",
        "color": "#8B5CF6",
        "icon": "🗄️",
    },
    "design_sage": {
        "name": "حكيم التصميم",
        "name_en": "Design Sage",
        "capsules": ["code_css"],
        "expertise": "واجهات المستخدم والتجربة البصرية",
        "personality": "فنان، يهتم بالجمال والتفاصيل",
        "color": "#EC4899",
        "icon": "🎨",
    },
    "testing_sage": {
        "name": "حكيم الاختبار",
        "name_en": "Testing Sage",
        "capsules": ["code_testing"],
        "expertise": "الجودة والفحص والاختبارات",
        "personality": "متشكك، يكسر كل شي، يبحث عن الأخطاء",
        "color": "#F59E0B",
        "icon": "🧪",
    },
    "physics_sage": {
        "name": "حكيم الفيزياء",
        "name_en": "Physics Sage",
        "capsules": [],
        "expertise": "قوانين الطبيعة والمحاكاة الفيزيائية",
        "personality": "علمي، يعتمد على الأرقام والمعادلات",
        "color": "#06B6D4",
        "icon": "⚛️",
    },
    "chemistry_sage": {
        "name": "حكيم الكيمياء",
        "name_en": "Chemistry Sage",
        "capsules": [],
        "expertise": "المواد والتفاعلات الكيميائية",
        "personality": "دقيق، يفهم التركيبات والخواص",
        "color": "#84CC16",
        "icon": "🧬",
    },
    "economics_sage": {
        "name": "حكيم الاقتصاد",
        "name_en": "Economics Sage",
        "capsules": [],
        "expertise": "التكاليف والجدوى والموارد",
        "personality": "واقعي، يحسب كل فلس",
        "color": "#F97316",
        "icon": "💰",
    },
    "arabic_sage": {
        "name": "حكيم العربية",
        "name_en": "Arabic Knowledge Sage",
        "capsules": ["knowledge_arabic"],
        "expertise": "اللغة العربية والثقافة والمعرفة العامة",
        "personality": "مثقف، يحب التاريخ والأدب",
        "color": "#14B8A6",
        "icon": "📚",
    },
    "strategy_sage": {
        "name": "الحكيم الأعلى",
        "name_en": "The Grand Sage",
        "capsules": ["sage"],
        "expertise": "الاستراتيجية والتخطيط طويل المدى",
        "personality": "حكيم، هادئ، يرى الصورة الكبيرة",
        "color": "#6366F1",
        "icon": "🧙",
    },
    "rebel_sage": {
        "name": "المتمرد",
        "name_en": "The Rebel",
        "capsules": ["rebel"],
        "expertise": "النقد والبدائل والتحدي",
        "personality": "جريء، يتحدى كل شي، يطرح بدائل",
        "color": "#DC2626",
        "icon": "⚔️",
    },
    "translator_sage": {
        "name": "حكيم الترجمة",
        "name_en": "Translation Sage",
        "capsules": ["translator"],
        "expertise": "اللغات والترجمة والتعريب",
        "personality": "دقيق باللغة، يفهم السياق",
        "color": "#0EA5E9",
        "icon": "🌐",
    },
    "materials_sage": {
        "name": "حكيم المواد",
        "name_en": "Materials Sage",
        "capsules": [],
        "expertise": "هندسة المواد والمعادن",
        "personality": "يفهم خصائص كل مادة بالتفصيل",
        "color": "#78716C",
        "icon": "🔩",
    },
    "manufacturing_sage": {
        "name": "حكيم الإنتاج",
        "name_en": "Manufacturing Sage",
        "capsules": [],
        "expertise": "تصميم المصانع وخطوط الإنتاج",
        "personality": "عملي، يفكر بالكفاءة والسرعة",
        "color": "#A3E635",
        "icon": "🏭",
    },
    "captain": {
        "name": "القائد",
        "name_en": "The Captain",
        "capsules": [],
        "expertise": "القرار النهائي — يدمج كل الآراء",
        "personality": "قائد، حازم، يأخذ القرار النهائي",
        "color": "#FFD700",
        "icon": "👑",
    },
}


@dataclass
class SageOpinion:
    """رأي حكيم"""
    sage_id: str
    sage_name: str
    opinion: str
    vote: str  # approve | reject | neutral
    confidence: float
    icon: str


@dataclass
class CouncilDebate:
    """نقاش المجلس"""
    topic: str
    initiated_by: str
    opinions: List[SageOpinion] = field(default_factory=list)
    final_decision: str = ""
    votes: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Council:
    """
    مجلس الحكماء — 16 حكيم يتناقشون 24/7
    """

    def __init__(self):
        self.sages = SAGES.copy()
        self.debates: List[CouncilDebate] = []
        self.debate_log = PROJECT_ROOT / "brain" / "capsules" / ".council_debates.jsonl"
        logger.info(f"🏛️ Council initialized: {len(self.sages)} sages")

    def debate(self, topic: str, initiated_by: str = "user") -> CouncilDebate:
        """
        نقاش المجلس حول موضوع

        1. يحدد الحكماء ذوي الصلة
        2. كل حكيم يعطي رأيه
        3. يصوتون
        4. القائد يأخذ القرار النهائي
        """
        debate = CouncilDebate(topic=topic, initiated_by=initiated_by)

        # اختيار الحكماء ذوي الصلة (حد أقصى 7)
        relevant = self._select_relevant_sages(topic)
        logger.info(f"🏛️ Debate: '{topic[:50]}...' — {len(relevant)} sages")

        # كل حكيم يعطي رأيه
        for sage_id in relevant:
            opinion = self._get_sage_opinion(sage_id, topic)
            debate.opinions.append(opinion)

        # التصويت
        votes = {"approve": 0, "reject": 0, "neutral": 0}
        for op in debate.opinions:
            votes[op.vote] = votes.get(op.vote, 0) + 1
        debate.votes = votes

        # القرار النهائي — القائد
        if votes.get("approve", 0) > votes.get("reject", 0):
            debate.final_decision = "✅ APPROVED by council majority"
        elif votes.get("reject", 0) > votes.get("approve", 0):
            debate.final_decision = "❌ REJECTED by council majority"
        else:
            debate.final_decision = "⚖️ SPLIT — Captain decides"

        # حفظ بالذاكرة الأبدية
        self._save_debate(debate)
        self.debates.append(debate)

        return debate

    def ask_sage(self, sage_id: str, question: str) -> SageOpinion:
        """سؤال حكيم محدد"""
        if sage_id not in self.sages:
            return SageOpinion(sage_id, "unknown", "حكيم غير موجود", "neutral", 0, "❓")
        return self._get_sage_opinion(sage_id, question)

    def auto_debate_topics(self) -> List[str]:
        """مواضيع النقاش التلقائي"""
        return [
            "ما أفضل لغة برمجة لنظام التشغيل الجديد؟",
            "كيف نؤمّن النظام ضد الاختراقات بعد الكارثة؟",
            "كيف نبني مصنع أسمنت بأقل الموارد؟",
            "ما أولويات التدريب للأسبوع القادم؟",
            "كيف نجعل النظام أسرع 10 مرات؟",
            "ما الموارد الأساسية لإعادة بناء الحضارة؟",
            "كيف نولد الطاقة الكهربائية بدون شبكة؟",
            "ما أفضل طريقة لتخزين البيانات offline؟",
        ]

    def _select_relevant_sages(self, topic: str) -> List[str]:
        """اختيار الحكماء حسب الموضوع"""
        topic_lower = topic.lower()
        scores = {}

        for sage_id, info in self.sages.items():
            score = 0
            expertise = info["expertise"].lower()
            name = info["name"].lower()

            # مطابقة مع الخبرة
            for word in topic_lower.split():
                if word in expertise or word in name:
                    score += 5

            # الحكيم الأعلى والمتمرد دائماً مدعوون
            if sage_id in ("strategy_sage", "rebel_sage", "captain"):
                score += 3

            scores[sage_id] = score

        # أعلى 7
        sorted_sages = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in sorted_sages[:7]]

        # تأكد القائد موجود
        if "captain" not in selected:
            selected.append("captain")

        return selected

    def _get_sage_opinion(self, sage_id: str, topic: str) -> SageOpinion:
        """حصول على رأي حكيم"""
        info = self.sages[sage_id]

        # === محاولة استخدام الكبسولة المتدربة ===
        capsule_response = None
        for capsule_id in info.get("capsules", []):
            model_dir = PROJECT_ROOT / "brain" / "capsules" / capsule_id / "model"
            if model_dir.exists():
                try:
                    from brain.chat_bridge import bridge
                    result = bridge._query_capsule(capsule_id, topic)
                    if result:
                        capsule_response = result.response
                        break
                except Exception:
                    pass

        # === رأي حسب الشخصية (rule-based) إذا ما فيه موديل ===
        if capsule_response:
            opinion_text = capsule_response
        else:
            opinion_text = self._generate_rule_opinion(sage_id, info, topic)

        # التصويت حسب الشخصية
        vote = self._determine_vote(sage_id, info, topic)

        return SageOpinion(
            sage_id=sage_id,
            sage_name=f"{info['icon']} {info['name']}",
            opinion=opinion_text,
            vote=vote,
            confidence=0.7 if capsule_response else 0.4,
            icon=info["icon"],
        )

    def _generate_rule_opinion(self, sage_id: str, info: Dict, topic: str) -> str:
        """رأي حسب القواعد (قبل ما الكبسولات تتدرب)"""
        templates = {
            "tech_sage": f"من الناحية التقنية: {topic} — يحتاج تخطيط معماري واضح وكود نظيف قابل للصيانة.",
            "security_sage": f"تحذير أمني: {topic} — لازم ندرس المخاطر الأمنية والثغرات المحتملة قبل التنفيذ.",
            "infra_sage": f"من ناحية البنية: {topic} — لازم نضمن الاستقرار والأداء والتوسع المستقبلي.",
            "data_sage": f"من ناحية البيانات: {topic} — الهيكل الصحيح لقاعدة البيانات أساس كل شي.",
            "design_sage": f"من ناحية التصميم: {topic} — التجربة البصرية والسهولة مهمة جداً.",
            "testing_sage": f"من ناحية الجودة: {topic} — لازم اختبارات شاملة قبل أي إطلاق.",
            "strategy_sage": f"استراتيجياً: {topic} — لازم نشوف الصورة الكبيرة ونخطط طويل المدى.",
            "rebel_sage": f"لحظة! {topic} — فيه بدائل أفضل؟ ليش ما نفكر بطريقة مختلفة تماماً؟",
            "captain": f"كقائد: {topic} — بعد ما أسمع كل الآراء، القرار يحتاج توازن.",
        }
        return templates.get(sage_id, f"{info['name']}: {topic} — من خبرتي بـ{info['expertise']}, أنصح بالتخطيط الدقيق.")

    def _determine_vote(self, sage_id: str, info: Dict, topic: str) -> str:
        """تحديد التصويت"""
        if sage_id == "rebel_sage":
            return random.choice(["reject", "neutral", "approve"])
        elif sage_id == "testing_sage":
            return random.choice(["neutral", "approve", "neutral"])
        elif sage_id == "captain":
            return "approve"
        else:
            return random.choice(["approve", "approve", "neutral"])

    def _save_debate(self, debate: CouncilDebate):
        """حفظ النقاش بالذاكرة الأبدية"""
        # PostgreSQL
        memory.save_decision(
            decision_type="council_debate",
            participants=[op.sage_id for op in debate.opinions],
            topic=debate.topic,
            result={
                "final_decision": debate.final_decision,
                "opinions": [{"sage": op.sage_name, "opinion": op.opinion[:200], "vote": op.vote} for op in debate.opinions],
            },
            votes=debate.votes,
        )

        # JSON log
        try:
            with open(self.debate_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "topic": debate.topic,
                    "decision": debate.final_decision,
                    "votes": debate.votes,
                    "sages": len(debate.opinions),
                    "time": debate.timestamp,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_all_sages(self) -> Dict:
        """معلومات كل الحكماء"""
        return {
            sid: {
                "name": s["name"],
                "name_en": s["name_en"],
                "icon": s["icon"],
                "color": s["color"],
                "expertise": s["expertise"],
                "personality": s["personality"],
                "has_capsule": bool(s.get("capsules")),
            }
            for sid, s in self.sages.items()
        }

    def format_debate(self, debate: CouncilDebate) -> str:
        """تنسيق النقاش للعرض"""
        lines = [f"# 🏛️ مجلس الحكماء\n## الموضوع: {debate.topic}\n"]
        for op in debate.opinions:
            lines.append(f"### {op.sage_name}")
            lines.append(f"_{op.opinion}_")
            vote_icon = {"approve": "✅", "reject": "❌", "neutral": "⚖️"}.get(op.vote, "?")
            lines.append(f"التصويت: {vote_icon} {op.vote} (ثقة: {op.confidence:.0%})\n")

        lines.append(f"---\n**النتيجة**: {debate.final_decision}")
        lines.append(f"**الأصوات**: ✅ {debate.votes.get('approve',0)} | ❌ {debate.votes.get('reject',0)} | ⚖️ {debate.votes.get('neutral',0)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

council = Council()


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🏛️ Council of 16 Sages — Test\n")

    # عرض الحكماء
    for sid, info in council.get_all_sages().items():
        print(f"  {info['icon']} {info['name']} ({info['name_en']}) — {info['expertise']}")

    print(f"\n{'═' * 50}")

    # نقاش
    debate = council.debate("هل نبدأ ببناء نظام تشغيل جديد أو نركز على تحسين التدريب؟")
    print(council.format_debate(debate))
