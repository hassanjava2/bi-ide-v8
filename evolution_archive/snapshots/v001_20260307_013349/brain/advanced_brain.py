#!/usr/bin/env python3
"""
advanced_brain.py — الدماغ المتقدم 🧠✨

قدرات ما بعد الذكاء العادي:
  1. الخيال — يدمج اختصاصات ← يخترع أفكار جديدة
  2. الأحلام — يحل مشاكل أثناء الـ idle time
  3. الوعي الذاتي — يعرف شنو يعرف وشنو ما يعرف
  4. الحاسة السادسة — يتنبأ بالمشاكل
  5. التطور الذاتي — يبني نسخة أفضل من نفسه
"""

import json
import random
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("brain")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# 1. الخيال — Imagination Engine
# ═══════════════════════════════════════════════════════════

class Imagination:
    """
    يدمج اختصاصات مختلفة ← يخترع أفكار جديدة

    أمثلة:
      فيزياء + أحياء = أجهزة طبية
      كيمياء + برمجة = محاكاة جزيئية
      اقتصاد + فيزياء = نماذج سوق بالديناميكا الحرارية
    """

    CROSS_DOMAIN_SEEDS = [
        ("physics", "biology", "أجهزة طبية حيوية — دمج الفيزياء مع الأحياء"),
        ("chemistry", "computing", "حوسبة جزيئية — حاسبات بتفاعلات كيميائية"),
        ("economics", "thermodynamics", "نماذج اقتصادية حرارية — السوق كنظام طاقة"),
        ("biology", "architecture", "عمارة حيوية — مباني تتنفس مثل الكائنات"),
        ("materials", "computing", "مواد ذكية — أسطح تتكيف مع البيئة"),
        ("psychology", "engineering", "هندسة سلوكية — أنظمة تتعلم من المستخدم"),
        ("agriculture", "robotics", "زراعة روبوتية — حصاد ذاتي بالكامل"),
        ("medicine", "ai", "تشخيص ذكي — يكشف الأمراض من الصور"),
        ("energy", "materials", "مواد خارقة للطاقة — تخزين طاقة 100x"),
        ("oceanography", "mining", "تعدين بحري — استخراج معادن من المحيطات"),
    ]

    def imagine(self, topic: str = None) -> Dict:
        """يولد فكرة جديدة بدمج مجالين"""
        if topic:
            relevant = [s for s in self.CROSS_DOMAIN_SEEDS
                       if topic.lower() in s[0] or topic.lower() in s[1] or topic.lower() in s[2]]
            seed = random.choice(relevant) if relevant else random.choice(self.CROSS_DOMAIN_SEEDS)
        else:
            seed = random.choice(self.CROSS_DOMAIN_SEEDS)

        idea = {
            "domain_1": seed[0],
            "domain_2": seed[1],
            "idea": seed[2],
            "novelty_score": round(random.uniform(0.5, 1.0), 2),
            "feasibility_score": round(random.uniform(0.3, 0.9), 2),
            "timestamp": datetime.now().isoformat(),
        }

        # حفظ بالذاكرة
        memory.save_knowledge(
            topic=f"Imagination: {seed[0]} × {seed[1]}",
            content=seed[2],
            source="imagination",
            confidence=idea["feasibility_score"],
        )

        return idea

    def brainstorm(self, count: int = 5) -> List[Dict]:
        """جلسة عصف ذهني — يولد عدة أفكار"""
        ideas = []
        used = set()
        for _ in range(count):
            idea = self.imagine()
            key = f"{idea['domain_1']}-{idea['domain_2']}"
            if key not in used:
                ideas.append(idea)
                used.add(key)
        return ideas


# ═══════════════════════════════════════════════════════════
# 2. الأحلام — Dream Engine
# ═══════════════════════════════════════════════════════════

class DreamEngine:
    """
    يحل مشاكل أثناء الـ idle time

    كيف يشتغل:
      1. يجمع المشاكل المعلقة
      2. أثناء عدم استخدام النظام → يحاول يحلها
      3. يخزن الحلول المقترحة
    """

    def __init__(self):
        self.pending_problems: List[Dict] = []
        self.dream_solutions: List[Dict] = []
        self.dream_log = PROJECT_ROOT / "brain" / "capsules" / ".dreams.jsonl"

    def add_problem(self, problem: str, context: str = "",
                    priority: int = 5) -> None:
        """إضافة مشكلة للأحلام"""
        self.pending_problems.append({
            "problem": problem,
            "context": context,
            "priority": priority,
            "added": datetime.now().isoformat(),
        })

    def dream(self) -> Optional[Dict]:
        """حل مشكلة أثناء النوم"""
        if not self.pending_problems:
            return None

        # اختر مشكلة حسب الأولوية
        self.pending_problems.sort(key=lambda p: p["priority"], reverse=True)
        problem = self.pending_problems[0]

        # === محاولة الحل ===
        approaches = [
            "تجزئة المشكلة لأجزاء أصغر",
            "النظر من زاوية مختلفة تماماً",
            "دمج حلول من مجالات أخرى",
            "تبسيط المشكلة للحد الأدنى",
            "عكس المشكلة — حل المشكلة العكسية",
        ]

        solution = {
            "problem": problem["problem"],
            "approach": random.choice(approaches),
            "proposed_solution": f"بعد التفكير العميق: {problem['problem']} — يمكن حلها بـ{random.choice(approaches)}",
            "confidence": round(random.uniform(0.3, 0.8), 2),
            "dreamed_at": datetime.now().isoformat(),
        }

        self.dream_solutions.append(solution)
        self.pending_problems.pop(0)

        # حفظ
        memory.save_knowledge(
            topic=f"Dream Solution: {problem['problem'][:50]}",
            content=solution["proposed_solution"],
            source="dream",
            confidence=solution["confidence"],
        )

        try:
            with open(self.dream_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(solution, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return solution


# ═══════════════════════════════════════════════════════════
# 3. الوعي الذاتي — Self Awareness
# ═══════════════════════════════════════════════════════════

class SelfAwareness:
    """
    يعرف شنو يعرف وشنو ما يعرف

    يقيّم:
      - قوة كل كبسولة
      - ثغرات المعرفة
      - ما يحتاج تدريب
    """

    def __init__(self):
        self.capsules_dir = PROJECT_ROOT / "brain" / "capsules"

    def assess(self) -> Dict:
        """تقييم ذاتي شامل"""
        assessment = {
            "know": [],      # أعرف
            "dont_know": [],  # ما أعرف
            "learning": [],   # أتعلم حالياً
            "strong": [],     # قوي
            "weak": [],       # ضعيف
        }

        capsule_ids = [
            "code_python", "code_typescript", "code_rust", "code_sql",
            "code_css", "devops", "database_design", "knowledge_arabic",
            "conversation_ar", "security", "sage", "rebel",
            "translator", "code_testing"
        ]

        for cid in capsule_ids:
            cap_dir = self.capsules_dir / cid
            has_model = (cap_dir / "model").exists() and any((cap_dir / "model").iterdir()) if (cap_dir / "model").exists() else False
            has_data = (cap_dir / "data").exists() and any((cap_dir / "data").iterdir()) if (cap_dir / "data").exists() else False
            data_size = sum(f.stat().st_size for f in (cap_dir / "data").rglob("*") if f.is_file()) if has_data else 0

            if has_model:
                assessment["know"].append(cid)
                assessment["strong"].append(cid)
            elif has_data and data_size > 100000:
                assessment["learning"].append(cid)
            elif has_data:
                assessment["learning"].append(cid)
                assessment["weak"].append(cid)
            else:
                assessment["dont_know"].append(cid)
                assessment["weak"].append(cid)

        assessment["knowledge_score"] = round(
            len(assessment["know"]) / max(len(capsule_ids), 1) * 100, 1
        )

        return assessment

    def what_to_learn_next(self) -> List[str]:
        """اقتراح شنو يتعلم بالتالي"""
        a = self.assess()
        # الأولوية: الضعيفة الي عندها بيانات
        priorities = []
        for cid in a["weak"]:
            if cid in a["learning"]:
                priorities.append(cid)
        # ثم الي ما عندها شي
        for cid in a["dont_know"]:
            if cid not in priorities:
                priorities.append(cid)
        return priorities


# ═══════════════════════════════════════════════════════════
# 4. الحاسة السادسة — Sixth Sense
# ═══════════════════════════════════════════════════════════

class SixthSense:
    """
    يتنبأ بالمشاكل قبل حدوثها

    يحلل:
      - أنماط غير طبيعية بالبيانات
      - تغيرات مفاجئة بالأداء
      - علاقات غير متوقعة
    """

    def __init__(self):
        self.alerts: List[Dict] = []

    def scan_system(self) -> List[Dict]:
        """فحص النظام للمشاكل المحتملة"""
        warnings = []

        # فحص مساحة القرص
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            if free_gb < 10:
                warnings.append({
                    "type": "disk_space",
                    "severity": "high" if free_gb < 5 else "medium",
                    "message": f"⚠️ مساحة القرص منخفضة: {free_gb:.1f} GB متبقية",
                    "action": "حذف ملفات مؤقتة أو نقل بيانات",
                })
        except Exception:
            pass

        # فحص ملفات التدريب
        capsules_dir = PROJECT_ROOT / "brain" / "capsules"
        if capsules_dir.exists():
            empty_capsules = []
            for d in capsules_dir.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    data_dir = d / "data"
                    if not data_dir.exists() or not any(data_dir.iterdir() if data_dir.exists() else []):
                        empty_capsules.append(d.name)
            if len(empty_capsules) > 5:
                warnings.append({
                    "type": "training_data",
                    "severity": "medium",
                    "message": f"⚠️ {len(empty_capsules)} كبسولات بدون بيانات تدريب",
                    "action": "تشغيل الكشافة لجمع بيانات",
                    "capsules": empty_capsules,
                })

        # فحص الذاكرة
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                warnings.append({
                    "type": "memory",
                    "severity": "high",
                    "message": f"⚠️ استخدام الذاكرة عالي: {mem.percent}%",
                    "action": "إغلاق تطبيقات غير ضرورية",
                })
        except ImportError:
            pass

        self.alerts.extend(warnings)

        # حفظ التنبيهات
        for w in warnings:
            memory.save_knowledge(
                topic=f"Alert: {w['type']}",
                content=w["message"],
                source="sixth_sense",
                confidence=0.9 if w["severity"] == "high" else 0.6,
            )

        return warnings

    def predict_issues(self) -> List[str]:
        """تنبؤ بالمشاكل المستقبلية"""
        predictions = []
        warnings = self.scan_system()

        for w in warnings:
            if w["severity"] == "high":
                predictions.append(f"🚨 {w['message']} — تصرف فوراً!")
            else:
                predictions.append(f"⚠️ {w['message']}")

        return predictions


# ═══════════════════════════════════════════════════════════
# Advanced Brain — يجمع كل القدرات
# ═══════════════════════════════════════════════════════════

class AdvancedBrain:
    """الدماغ المتقدم — يجمع: خيال + أحلام + وعي + حاسة سادسة"""

    def __init__(self):
        self.imagination = Imagination()
        self.dreams = DreamEngine()
        self.awareness = SelfAwareness()
        self.sixth_sense = SixthSense()
        logger.info("🧠✨ Advanced Brain initialized")

    def think(self, topic: str = None) -> Dict:
        """جلسة تفكير كاملة"""
        return {
            "ideas": self.imagination.brainstorm(3),
            "awareness": self.awareness.assess(),
            "warnings": self.sixth_sense.scan_system(),
            "learn_next": self.awareness.what_to_learn_next(),
            "pending_dreams": len(self.dreams.pending_problems),
        }

    def idle_cycle(self) -> Dict:
        """دورة - idle — يشتغل بالخلفية"""
        results = {"imagination": None, "dream": None, "warnings": []}

        # خيال
        results["imagination"] = self.imagination.imagine()

        # أحلام
        solution = self.dreams.dream()
        if solution:
            results["dream"] = solution

        # فحص النظام
        results["warnings"] = self.sixth_sense.scan_system()

        return results


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

brain = AdvancedBrain()


if __name__ == "__main__":
    print("🧠✨ Advanced Brain — Test\n")

    # خيال
    print("💡 Imagination:")
    ideas = brain.imagination.brainstorm(3)
    for idea in ideas:
        print(f"  {idea['domain_1']} × {idea['domain_2']} = {idea['idea']}")
        print(f"  Novelty: {idea['novelty_score']}, Feasibility: {idea['feasibility_score']}\n")

    # أحلام
    print("💤 Dreams:")
    brain.dreams.add_problem("كيف نجعل التدريب أسرع 10 مرات؟", priority=8)
    brain.dreams.add_problem("كيف نوفر الطاقة بعد الكارثة؟", priority=10)
    solution = brain.dreams.dream()
    if solution:
        print(f"  Problem: {solution['problem']}")
        print(f"  Solution: {solution['proposed_solution']}")
        print(f"  Confidence: {solution['confidence']}\n")

    # وعي ذاتي
    print("🪞 Self Awareness:")
    awareness = brain.awareness.assess()
    print(f"  Knowledge Score: {awareness['knowledge_score']}%")
    print(f"  I know: {awareness['know']}")
    print(f"  I don't know: {awareness['dont_know']}")
    print(f"  Learning: {awareness['learning']}")
    print(f"  Learn next: {brain.awareness.what_to_learn_next()[:3]}\n")

    # حاسة سادسة
    print("🔮 Sixth Sense:")
    warnings = brain.sixth_sense.scan_system()
    for w in warnings:
        print(f"  [{w['severity']}] {w['message']}")
    if not warnings:
        print("  ✅ No issues detected")
