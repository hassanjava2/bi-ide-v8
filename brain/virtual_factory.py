#!/usr/bin/env python3
"""
virtual_factory.py — المصنع الافتراضي الحي 🏭👷

مصنع كامل يشتغل بعمال AI:
  1. يُبنى حسب خطة (real_life_layer)
  2. كل عامل = agent باختصاصه
  3. يشتغل يوم بيوم — مشاكل تظهر وتُحل
  4. يستقر ← يتطور ← يبني v2 أفضل
  5. مستقبلاً: يدرب بشر حقيقيين
"""

import json
import random
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("factory")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
    from brain.real_life_layer import planner
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory
    from brain.real_life_layer import planner


# ═══════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════

class WorkerRole(Enum):
    MANAGER = "مدير المصنع"
    PRODUCTION_ENGINEER = "مهندس إنتاج"
    MAINTENANCE_TECH = "فني صيانة"
    QUALITY_INSPECTOR = "مراقب جودة"
    WAREHOUSE_MANAGER = "مسؤول مخزون"
    SAFETY_OFFICER = "مسؤول سلامة"
    ACCOUNTANT = "محاسب"
    CHEMIST = "كيميائي"
    ELECTRICIAN = "كهربائي"
    OPERATOR = "مشغّل خط"


class FactoryPhase(Enum):
    CONSTRUCTION = "بناء"
    COMMISSIONING = "تجهيز"
    STARTUP = "بدء تشغيل"
    UNSTABLE = "غير مستقر"
    STABILIZING = "استقرار"
    STABLE = "مستقر"
    OPTIMIZING = "تحسين"
    EVOLVED = "متطور"


class ProblemSeverity(Enum):
    LOW = "بسيط"
    MEDIUM = "متوسط"
    HIGH = "خطير"
    CRITICAL = "حرج"


# ═══════════════════════════════════════════════════════════
# Factory Worker — عامل AI
# ═══════════════════════════════════════════════════════════

@dataclass
class FactoryWorker:
    """عامل AI — agent بذاكرة وتخصص"""
    worker_id: str
    name: str
    role: WorkerRole
    experience_days: int = 0
    problems_solved: int = 0
    shifts_worked: int = 0
    skills: List[str] = field(default_factory=list)
    current_task: str = ""
    morale: float = 0.8        # 0-1
    efficiency: float = 0.5     # يزيد مع الخبرة
    memory: List[Dict] = field(default_factory=list)

    def work_shift(self, problems: List) -> List[Dict]:
        """وردية عمل — يحل مشاكل حسب تخصصه"""
        self.shifts_worked += 1
        self.experience_days += 1
        actions = []

        relevant = [p for p in problems if self._can_handle(p)]
        for problem in relevant[:3]:  # أقصى 3 مشاكل بالوردية
            solved = random.random() < self.efficiency
            if solved:
                self.problems_solved += 1
                self.efficiency = min(self.efficiency + 0.01, 0.99)
                actions.append({
                    "worker": self.name,
                    "role": self.role.value,
                    "action": "solved",
                    "problem": problem["description"],
                    "method": self._solve_method(problem),
                })
            else:
                actions.append({
                    "worker": self.name,
                    "role": self.role.value,
                    "action": "escalated",
                    "problem": problem["description"],
                })

        # ذاكرة العامل
        self.memory.extend(actions[-5:])
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

        return actions

    def _can_handle(self, problem: Dict) -> bool:
        role_problems = {
            WorkerRole.PRODUCTION_ENGINEER: ["production", "output", "speed", "bottleneck"],
            WorkerRole.MAINTENANCE_TECH: ["breakdown", "repair", "maintenance", "wear"],
            WorkerRole.QUALITY_INSPECTOR: ["defect", "quality", "contamination", "spec"],
            WorkerRole.WAREHOUSE_MANAGER: ["inventory", "shortage", "storage", "shipment"],
            WorkerRole.SAFETY_OFFICER: ["safety", "leak", "fire", "hazard", "injury"],
            WorkerRole.ACCOUNTANT: ["cost", "budget", "expense", "profit"],
            WorkerRole.CHEMIST: ["reaction", "temperature", "chemical", "formula"],
            WorkerRole.ELECTRICIAN: ["power", "voltage", "circuit", "motor"],
            WorkerRole.OPERATOR: ["operation", "feed", "control", "valve"],
            WorkerRole.MANAGER: ["any"],
        }
        tags = role_problems.get(self.role, [])
        if "any" in tags:
            return True
        return any(t in problem.get("tags", []) for t in tags)

    def _solve_method(self, problem: Dict) -> str:
        methods = {
            WorkerRole.MAINTENANCE_TECH: "فحص + إصلاح + اختبار",
            WorkerRole.QUALITY_INSPECTOR: "فحص عينات + تعديل معايير + إعادة فحص",
            WorkerRole.PRODUCTION_ENGINEER: "تحليل الخط + تعديل سرعة + موازنة",
            WorkerRole.SAFETY_OFFICER: "تقييم مخاطر + إجراءات وقائية + تدريب",
            WorkerRole.CHEMIST: "تحليل مخبري + تعديل تركيبة + اختبار",
            WorkerRole.ELECTRICIAN: "فحص دارات + استبدال قطعة + تجربة",
            WorkerRole.MANAGER: "تنسيق فرق + تحديد أولويات + متابعة",
        }
        return methods.get(self.role, "تحليل + حل + تقييم")

    def train_human(self, human_name: str, task: str) -> Dict:
        """تدريب موظف بشري"""
        return {
            "trainer": self.name,
            "trainee": human_name,
            "task": task,
            "instructions": f"[{self.role.value}] {task}: " + self._solve_method({"tags": []}),
            "experience_level": "مبتدئ" if self.experience_days < 30 else "متوسط" if self.experience_days < 90 else "خبير",
            "confidence": round(self.efficiency, 2),
        }


# ═══════════════════════════════════════════════════════════
# Production Line — خط إنتاج
# ═══════════════════════════════════════════════════════════

@dataclass
class ProductionLine:
    """خط إنتاج"""
    line_id: str
    product: str
    capacity_per_day: float
    actual_output: float = 0
    efficiency: float = 0.0
    uptime_pct: float = 100
    defect_rate: float = 5.0   # %
    energy_kwh: float = 0


# ═══════════════════════════════════════════════════════════
# Factory Problems — مشاكل المصنع
# ═══════════════════════════════════════════════════════════

PROBLEM_TEMPLATES = [
    {"description": "عطل بالمحرك الرئيسي", "severity": "high", "tags": ["breakdown", "repair", "motor"]},
    {"description": "تسرب مادة كيميائية", "severity": "critical", "tags": ["leak", "chemical", "safety"]},
    {"description": "انخفاض جودة المنتج", "severity": "medium", "tags": ["quality", "defect", "spec"]},
    {"description": "نقص مواد خام", "severity": "high", "tags": ["inventory", "shortage"]},
    {"description": "ارتفاع حرارة الفرن فوق الحد", "severity": "high", "tags": ["temperature", "control"]},
    {"description": "انقطاع كهرباء جزئي", "severity": "medium", "tags": ["power", "voltage", "circuit"]},
    {"description": "تجاوز الميزانية الشهرية", "severity": "low", "tags": ["cost", "budget"]},
    {"description": "عيب بالتغليف", "severity": "low", "tags": ["quality", "defect"]},
    {"description": "بطء خط الإنتاج", "severity": "medium", "tags": ["production", "bottleneck", "speed"]},
    {"description": "تآكل بالأنابيب", "severity": "medium", "tags": ["maintenance", "wear"]},
    {"description": "خطر حريق قرب المخزن", "severity": "critical", "tags": ["fire", "safety", "hazard"]},
    {"description": "تفاعل كيميائي غير متوقع", "severity": "high", "tags": ["reaction", "chemical"]},
    {"description": "تلف بمضخة التبريد", "severity": "medium", "tags": ["breakdown", "repair"]},
    {"description": "تأخر شحنة مواد", "severity": "low", "tags": ["shipment", "inventory"]},
    {"description": "ضوضاء غير طبيعية بالكسارة", "severity": "medium", "tags": ["breakdown", "maintenance"]},
]


# ═══════════════════════════════════════════════════════════
# Virtual Factory — المصنع الافتراضي
# ═══════════════════════════════════════════════════════════

class VirtualFactory:
    """
    مصنع افتراضي حي — يشتغل بعمال AI

    المراحل:
      1. البناء ← حسب خطة real_life_layer
      2. التشغيل ← مشاكل تظهر ← عمال يحلونها
      3. الاستقرار ← إنتاج ثابت
      4. التطوير ← v2 أفضل
      5. تدريب بشر ← كل عامل AI يعلّم بشري
    """

    def __init__(self, product: str, capacity: float = 50000, location: str = "Iraq"):
        self.product = product
        self.capacity = capacity
        self.location = location
        self.phase = FactoryPhase.CONSTRUCTION
        self.day = 0
        self.version = 1

        # خطة المصنع
        plan = planner.plan_factory(product, capacity, location)
        self.plan = plan

        # عمال AI
        self.workers: List[FactoryWorker] = self._create_workers()

        # خط إنتاج
        self.production = ProductionLine(
            line_id=f"{product}_line_1",
            product=product,
            capacity_per_day=capacity / 365,
        )

        # سجلات
        self.daily_logs: List[Dict] = []
        self.problems_history: List[Dict] = []
        self.total_output: float = 0
        self.total_problems: int = 0
        self.total_solved: int = 0

    def _create_workers(self) -> List[FactoryWorker]:
        """إنشاء عمال المصنع"""
        workers = []
        worker_defs = [
            ("mgr_01", "أبو حسين", WorkerRole.MANAGER, ["leadership", "planning"]),
            ("eng_01", "عباس المهندس", WorkerRole.PRODUCTION_ENGINEER, ["optimization", "scheduling"]),
            ("mnt_01", "حيدر الفني", WorkerRole.MAINTENANCE_TECH, ["welding", "electronics"]),
            ("qc_01", "زهراء المفتشة", WorkerRole.QUALITY_INSPECTOR, ["testing", "specs"]),
            ("wh_01", "كرار المخزنجي", WorkerRole.WAREHOUSE_MANAGER, ["inventory", "logistics"]),
            ("saf_01", "ميثم السلامة", WorkerRole.SAFETY_OFFICER, ["first_aid", "fire_safety"]),
            ("acc_01", "نور المحاسبة", WorkerRole.ACCOUNTANT, ["budgeting", "reporting"]),
            ("chem_01", "علي الكيميائي", WorkerRole.CHEMIST, ["analysis", "formulation"]),
            ("elec_01", "مصطفى الكهربائي", WorkerRole.ELECTRICIAN, ["motors", "wiring"]),
            ("op_01", "أحمد المشغّل", WorkerRole.OPERATOR, ["controls", "monitoring"]),
            ("op_02", "محمد المشغّل", WorkerRole.OPERATOR, ["feeding", "valves"]),
        ]
        for wid, name, role, skills in worker_defs:
            workers.append(FactoryWorker(worker_id=wid, name=name, role=role, skills=skills))
        return workers

    def simulate_day(self) -> Dict:
        """محاكاة يوم واحد"""
        self.day += 1
        log = {"day": self.day, "phase": self.phase.value, "events": [], "problems": [], "solutions": []}

        # === تحديد المرحلة ===
        if self.day <= 3:
            self.phase = FactoryPhase.CONSTRUCTION
            log["events"].append("🏗️ أعمال بناء مستمرة")
            return self._finish_day(log)
        elif self.day <= 5:
            self.phase = FactoryPhase.COMMISSIONING
            log["events"].append("⚙️ تجهيز المعدات واختبارها")
            return self._finish_day(log)
        elif self.day <= 10:
            self.phase = FactoryPhase.STARTUP
        elif self.day <= 30:
            self.phase = FactoryPhase.UNSTABLE
        elif self.day <= 60:
            self.phase = FactoryPhase.STABILIZING
        elif self.day <= 90:
            self.phase = FactoryPhase.STABLE
        else:
            self.phase = FactoryPhase.OPTIMIZING

        # === مشاكل اليوم ===
        problem_chance = {
            FactoryPhase.STARTUP: 0.8,
            FactoryPhase.UNSTABLE: 0.6,
            FactoryPhase.STABILIZING: 0.3,
            FactoryPhase.STABLE: 0.1,
            FactoryPhase.OPTIMIZING: 0.05,
        }
        chance = problem_chance.get(self.phase, 0.3)

        problems = []
        num_problems = 0
        for _ in range(5):
            if random.random() < chance:
                num_problems += 1
        if num_problems > 0:
            problems = random.sample(PROBLEM_TEMPLATES, min(num_problems, len(PROBLEM_TEMPLATES)))

        self.total_problems += len(problems)
        log["problems"] = [p["description"] for p in problems]

        # === عمال يحلون المشاكل ===
        for worker in self.workers:
            actions = worker.work_shift(problems)
            for action in actions:
                if action["action"] == "solved":
                    self.total_solved += 1
                    log["solutions"].append(f"{action['worker']}: حل '{action['problem']}' بـ{action['method']}")
                    # إزالة المشكلة المحلولة
                    problems = [p for p in problems if p["description"] != action["problem"]]

        # === إنتاج ===
        base_efficiency = {
            FactoryPhase.STARTUP: 0.2,
            FactoryPhase.UNSTABLE: 0.4,
            FactoryPhase.STABILIZING: 0.7,
            FactoryPhase.STABLE: 0.9,
            FactoryPhase.OPTIMIZING: 0.95,
        }
        eff = base_efficiency.get(self.phase, 0.5)
        problem_penalty = len(problems) * 0.05
        eff = max(0.1, eff - problem_penalty)

        self.production.efficiency = eff
        self.production.actual_output = self.production.capacity_per_day * eff
        self.total_output += self.production.actual_output

        log["production"] = {
            "output_tons": round(self.production.actual_output, 1),
            "efficiency": f"{eff:.0%}",
            "total_tons": round(self.total_output, 1),
        }

        return self._finish_day(log)

    def _finish_day(self, log: Dict) -> Dict:
        self.daily_logs.append(log)

        # حفظ بالذاكرة كل 10 أيام
        if self.day % 10 == 0:
            memory.save_knowledge(
                topic=f"Factory {self.product} Day {self.day}",
                content=f"Phase: {self.phase.value}, Output: {self.total_output:.0f}t, "
                        f"Problems: {self.total_problems}, Solved: {self.total_solved}",
                source="virtual_factory",
            )

        return log

    def simulate_period(self, days: int = 90) -> Dict:
        """محاكاة فترة كاملة"""
        for _ in range(days):
            self.simulate_day()

        return self.get_report()

    def get_report(self) -> Dict:
        """تقرير المصنع"""
        solve_rate = (self.total_solved / max(self.total_problems, 1)) * 100
        return {
            "product": self.product,
            "version": self.version,
            "day": self.day,
            "phase": self.phase.value,
            "total_output_tons": round(self.total_output, 1),
            "total_problems": self.total_problems,
            "total_solved": self.total_solved,
            "solve_rate_pct": round(solve_rate, 1),
            "workers": len(self.workers),
            "avg_efficiency": round(sum(w.efficiency for w in self.workers) / len(self.workers), 2),
            "top_solver": max(self.workers, key=lambda w: w.problems_solved).name,
        }

    def evolve(self) -> "VirtualFactory":
        """تطوير المصنع — v2"""
        self.version += 1
        # تحسينات بناءً على الخبرة
        improved_capacity = self.capacity * 1.15  # 15% أكثر
        new_factory = VirtualFactory(self.product, improved_capacity, self.location)
        new_factory.version = self.version

        # نقل خبرات العمال
        for i, worker in enumerate(new_factory.workers):
            if i < len(self.workers):
                worker.efficiency = min(self.workers[i].efficiency + 0.1, 0.99)
                worker.experience_days = self.workers[i].experience_days

        memory.save_knowledge(
            topic=f"Factory Evolution v{self.version}",
            content=f"{self.product}: Evolved to v{self.version}, capacity +15%",
            source="factory_evolution",
        )

        return new_factory

    def format_report(self) -> str:
        """تقرير منسق"""
        r = self.get_report()
        lines = [
            f"# 🏭 تقرير مصنع: {self.product} v{r['version']}",
            f"**اليوم**: {r['day']} | **المرحلة**: {r['phase']}",
            f"**إنتاج كلي**: {r['total_output_tons']:,.0f} طن",
            f"**مشاكل**: {r['total_problems']} (حُلت {r['total_solved']} — {r['solve_rate_pct']}%)",
            f"**أفضل عامل**: {r['top_solver']}",
            f"\n## 👷 العمال ({r['workers']}):",
        ]
        for w in self.workers:
            bar = "█" * int(w.efficiency * 10) + "░" * (10 - int(w.efficiency * 10))
            lines.append(f"  {w.name:15s} [{bar}] {w.efficiency:.0%} ({w.problems_solved} حلول)")
        return "\n".join(lines)


if __name__ == "__main__":
    print("🏭 Virtual Factory — Test\n")

    factory = VirtualFactory("cement", capacity=50000)
    print(f"Workers: {len(factory.workers)}")
    for w in factory.workers:
        print(f"  👷 {w.name} — {w.role.value}")

    print(f"\n{'═' * 50}")
    print("محاكاة 90 يوم...\n")

    report = factory.simulate_period(90)
    print(factory.format_report())

    print(f"\n{'═' * 50}")
    print("تطوير للـ v2...\n")

    factory_v2 = factory.evolve()
    report_v2 = factory_v2.simulate_period(30)
    print(f"v2 بعد 30 يوم: إنتاج {report_v2['total_output_tons']:,.0f} طن, "
          f"مشاكل: {report_v2['total_problems']}, حلول: {report_v2['total_solved']}")
