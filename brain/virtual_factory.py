#!/usr/bin/env python3
"""
virtual_factory.py — نظام المصانع الافتراضية الحي 🏭👷

مصانع لكلشي حرفياً:
  - أسمنت، حديد، زجاج، طوب، صابون، أدوية، إلكترونيات...
  - كل مصنع يُبنى + يشتغل + يحل مشاكله + يتطور
  - 11+ عامل AI لكل مصنع
  - الكشافة تكتشف مصانع جديدة
  - المجلس يقرر أوتوماتيكياً
  - المستخدم عنده حق الفيتو على كلشي

المستقبل:
  - كل عامل AI يدرب بشري حقيقي
  - يراقب من كاميرات ← يوجه ← يصحح
"""

import json
import random
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("factory")
PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
    from brain.real_life_layer import planner
    from brain.capsule_tree import tree as capsule_tree
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory
    from brain.real_life_layer import planner
    from brain.capsule_tree import tree as capsule_tree

# ═══════════════════════════════════════════════════════════
# خريطة ربط: دور العامل ← كبسولات التخصص
# ═══════════════════════════════════════════════════════════
ROLE_TO_CAPSULES = {
    "MANAGER": ["beam", "foundation", "highway"],        # civil/structures
    "PRODUCTION_ENGINEER": ["furnace", "ore", "mold"],   # manufacturing
    "MAINTENANCE_TECH": ["piston", "diesel", "cooling"],  # mechanical
    "QUALITY_INSPECTOR": ["cement", "steel", "glass"],    # industrial chemistry
    "WAREHOUSE_MANAGER": ["asphalt", "pavement", "alloy"],
    "SAFETY_OFFICER": ["heat", "temperature", "entropy"],  # thermodynamics
    "ACCOUNTANT": ["carbon", "polymer", "mineral"],        # chemistry
    "CHEMIST": ["cement", "steel", "glass"],               # industrial chem
    "ELECTRICIAN": ["grid", "transformer", "generator"],   # power_systems
    "OPERATOR": ["foundry", "furnace", "blast"],           # metals
}


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


# ═══════════════════════════════════════════════════════════
# Factory Worker — عامل AI
# ═══════════════════════════════════════════════════════════

@dataclass
class FactoryWorker:
    worker_id: str
    name: str
    role: WorkerRole
    experience_days: int = 0
    problems_solved: int = 0
    efficiency: float = 0.5
    memory: List[Dict] = field(default_factory=list)
    assigned_capsules: List[str] = field(default_factory=list)

    def work_shift(self, problems: List) -> List[Dict]:
        self.experience_days += 1
        # كبسولات مدرّبة ترفع الكفاءة
        capsule_bonus = min(len(self.assigned_capsules) * 0.02, 0.1)
        effective_eff = min(self.efficiency + capsule_bonus, 0.99)

        actions = []
        relevant = [p for p in problems if self._can_handle(p)]
        for problem in relevant[:3]:
            solved = random.random() < effective_eff
            if solved:
                self.problems_solved += 1
                self.efficiency = min(self.efficiency + 0.01, 0.99)
                actions.append({"worker": self.name, "role": self.role.value,
                              "action": "solved", "problem": problem["description"]})
            else:
                actions.append({"worker": self.name, "role": self.role.value,
                              "action": "escalated", "problem": problem["description"]})
        self.memory.extend(actions[-5:])
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]
        return actions

    def _can_handle(self, problem: Dict) -> bool:
        role_tags = {
            WorkerRole.PRODUCTION_ENGINEER: ["production", "output", "speed", "bottleneck"],
            WorkerRole.MAINTENANCE_TECH: ["breakdown", "repair", "maintenance", "wear"],
            WorkerRole.QUALITY_INSPECTOR: ["defect", "quality", "contamination", "spec"],
            WorkerRole.WAREHOUSE_MANAGER: ["inventory", "shortage", "storage", "shipment"],
            WorkerRole.SAFETY_OFFICER: ["safety", "leak", "fire", "hazard"],
            WorkerRole.ACCOUNTANT: ["cost", "budget", "expense", "profit"],
            WorkerRole.CHEMIST: ["reaction", "temperature", "chemical", "formula"],
            WorkerRole.ELECTRICIAN: ["power", "voltage", "circuit", "motor"],
            WorkerRole.OPERATOR: ["operation", "feed", "control", "valve"],
            WorkerRole.MANAGER: ["any"],
        }
        tags = role_tags.get(self.role, [])
        return "any" in tags or any(t in problem.get("tags", []) for t in tags)

    def train_human(self, human_name: str, task: str) -> Dict:
        return {
            "trainer": self.name, "trainee": human_name, "task": task,
            "role": self.role.value,
            "level": "مبتدئ" if self.experience_days < 30 else "متوسط" if self.experience_days < 90 else "خبير",
            "confidence": round(self.efficiency, 2),
        }


# ═══════════════════════════════════════════════════════════
# مشاكل المصنع
# ═══════════════════════════════════════════════════════════

PROBLEM_TEMPLATES = [
    {"description": "عطل بالمحرك الرئيسي", "tags": ["breakdown", "repair", "motor"]},
    {"description": "تسرب مادة كيميائية", "tags": ["leak", "chemical", "safety"]},
    {"description": "انخفاض جودة المنتج", "tags": ["quality", "defect", "spec"]},
    {"description": "نقص مواد خام", "tags": ["inventory", "shortage"]},
    {"description": "ارتفاع حرارة فوق الحد", "tags": ["temperature", "control"]},
    {"description": "انقطاع كهرباء جزئي", "tags": ["power", "voltage", "circuit"]},
    {"description": "تجاوز ميزانية", "tags": ["cost", "budget"]},
    {"description": "بطء خط إنتاج", "tags": ["production", "bottleneck", "speed"]},
    {"description": "تآكل أنابيب", "tags": ["maintenance", "wear"]},
    {"description": "خطر حريق", "tags": ["fire", "safety", "hazard"]},
    {"description": "تفاعل كيميائي غير متوقع", "tags": ["reaction", "chemical"]},
    {"description": "تلف مضخة تبريد", "tags": ["breakdown", "repair"]},
    {"description": "ضوضاء غير طبيعية", "tags": ["breakdown", "maintenance"]},
]


# ═══════════════════════════════════════════════════════════
# مصنع واحد
# ═══════════════════════════════════════════════════════════

class VirtualFactory:
    def __init__(self, product: str, capacity: float = 50000, location: str = "Iraq"):
        self.product = product
        self.capacity = capacity
        self.location = location
        self.factory_id = f"factory_{product}_{int(time.time())}"
        self.phase = FactoryPhase.CONSTRUCTION
        self.day = 0
        self.version = 1
        self.workers = self._create_workers()
        self.total_output = 0.0
        self.total_problems = 0
        self.total_solved = 0
        self.daily_logs: List[Dict] = []

    def _create_workers(self) -> List[FactoryWorker]:
        names = [
            ("mgr", "أبو حسين", WorkerRole.MANAGER),
            ("eng", "عباس", WorkerRole.PRODUCTION_ENGINEER),
            ("mnt", "حيدر", WorkerRole.MAINTENANCE_TECH),
            ("qc", "زهراء", WorkerRole.QUALITY_INSPECTOR),
            ("wh", "كرار", WorkerRole.WAREHOUSE_MANAGER),
            ("saf", "ميثم", WorkerRole.SAFETY_OFFICER),
            ("acc", "نور", WorkerRole.ACCOUNTANT),
            ("chem", "علي", WorkerRole.CHEMIST),
            ("elec", "مصطفى", WorkerRole.ELECTRICIAN),
            ("op1", "أحمد", WorkerRole.OPERATOR),
            ("op2", "محمد", WorkerRole.OPERATOR),
        ]
        return [FactoryWorker(f"{self.product}_{wid}", name, role) for wid, name, role in names]

    def simulate_day(self) -> Dict:
        self.day += 1
        log = {"day": self.day, "phase": self.phase.value, "events": [], "solutions": []}

        if self.day <= 3:
            self.phase = FactoryPhase.CONSTRUCTION
            return log
        elif self.day <= 5:
            self.phase = FactoryPhase.COMMISSIONING
            return log
        elif self.day <= 10: self.phase = FactoryPhase.STARTUP
        elif self.day <= 30: self.phase = FactoryPhase.UNSTABLE
        elif self.day <= 60: self.phase = FactoryPhase.STABILIZING
        elif self.day <= 90: self.phase = FactoryPhase.STABLE
        else: self.phase = FactoryPhase.OPTIMIZING

        chance = {FactoryPhase.STARTUP: 0.8, FactoryPhase.UNSTABLE: 0.6,
                  FactoryPhase.STABILIZING: 0.3, FactoryPhase.STABLE: 0.1,
                  FactoryPhase.OPTIMIZING: 0.05}.get(self.phase, 0.3)

        problems = [p for p in PROBLEM_TEMPLATES if random.random() < chance][:5]
        self.total_problems += len(problems)

        for worker in self.workers:
            for action in worker.work_shift(problems):
                if action["action"] == "solved":
                    self.total_solved += 1
                    problems = [p for p in problems if p["description"] != action["problem"]]
                    log["solutions"].append(f"{action['worker']}: {action['problem']}")

        eff = {FactoryPhase.STARTUP: 0.2, FactoryPhase.UNSTABLE: 0.4,
               FactoryPhase.STABILIZING: 0.7, FactoryPhase.STABLE: 0.9,
               FactoryPhase.OPTIMIZING: 0.95}.get(self.phase, 0.5)
        eff = max(0.1, eff - len(problems) * 0.05)
        output = (self.capacity / 365) * eff
        self.total_output += output
        log["output"] = round(output, 1)
        log["efficiency"] = f"{eff:.0%}"

        self.daily_logs.append(log)
        return log

    def simulate_period(self, days: int = 90) -> Dict:
        for _ in range(days):
            self.simulate_day()
        return self.get_report()

    def get_report(self) -> Dict:
        return {
            "factory_id": self.factory_id, "product": self.product,
            "version": self.version, "day": self.day, "phase": self.phase.value,
            "total_output": round(self.total_output, 1),
            "problems": self.total_problems, "solved": self.total_solved,
            "solve_rate": round(self.total_solved / max(self.total_problems, 1) * 100, 1),
            "workers": len(self.workers),
            "top_solver": max(self.workers, key=lambda w: w.problems_solved).name,
        }

    def evolve(self) -> "VirtualFactory":
        new = VirtualFactory(self.product, self.capacity * 1.15, self.location)
        new.version = self.version + 1
        for i, w in enumerate(new.workers):
            if i < len(self.workers):
                w.efficiency = min(self.workers[i].efficiency + 0.1, 0.99)
                w.experience_days = self.workers[i].experience_days
        return new


# ═══════════════════════════════════════════════════════════
# مدير المصانع — كل المصانع!
# ═══════════════════════════════════════════════════════════

# أنواع المصانع المعروفة
FACTORY_CATALOG = {
    "cement": {"name_ar": "أسمنت", "capacity": 50000, "category": "بناء"},
    "steel": {"name_ar": "حديد", "capacity": 30000, "category": "بناء"},
    "glass": {"name_ar": "زجاج", "capacity": 10000, "category": "بناء"},
    "brick": {"name_ar": "طوب", "capacity": 100000, "category": "بناء"},
    "concrete": {"name_ar": "خرسانة", "capacity": 80000, "category": "بناء"},
    "soap": {"name_ar": "صابون", "capacity": 5000, "category": "استهلاكي"},
    "medicine": {"name_ar": "أدوية", "capacity": 1000, "category": "صحة"},
    "electronics": {"name_ar": "إلكترونيات", "capacity": 500, "category": "تقنية"},
    "furniture": {"name_ar": "أثاث", "capacity": 20000, "category": "استهلاكي"},
    "textile": {"name_ar": "نسيج", "capacity": 15000, "category": "استهلاكي"},
    "plastic": {"name_ar": "بلاستيك", "capacity": 10000, "category": "صناعي"},
    "paper": {"name_ar": "ورق", "capacity": 20000, "category": "استهلاكي"},
    "fertilizer": {"name_ar": "أسمدة", "capacity": 25000, "category": "زراعة"},
    "food_processing": {"name_ar": "تعليب غذائي", "capacity": 15000, "category": "غذاء"},
    "water_treatment": {"name_ar": "معالجة مياه", "capacity": 50000, "category": "بنية تحتية"},
    "solar_panel": {"name_ar": "ألواح شمسية", "capacity": 5000, "category": "طاقة"},
    "battery": {"name_ar": "بطاريات", "capacity": 3000, "category": "طاقة"},
    "wire_cable": {"name_ar": "أسلاك وكابلات", "capacity": 10000, "category": "كهرباء"},
    "pipe": {"name_ar": "أنابيب", "capacity": 20000, "category": "بنية تحتية"},
    "tools": {"name_ar": "أدوات يدوية", "capacity": 5000, "category": "صناعي"},
}


class FactoryManager:
    """
    مدير كل المصانع 🏭🏭🏭

    - ينشئ مصانع لكلشي
    - الكشافة تكتشف مصانع جديدة
    - المجلس يقرر أوتوماتيكياً
    - المستخدم عنده حق الفيتو
    """

    def __init__(self):
        self.factories: Dict[str, VirtualFactory] = {}
        self.pending_proposals: List[Dict] = []   # مقترحات تنتظر المجلس
        self.vetoed: List[str] = []                # مرفوضة بالفيتو
        self.user_commands: List[Dict] = []        # أوامر المستخدم

    # ═══════════════════════════════════════════════════════
    # إنشاء مصانع
    # ═══════════════════════════════════════════════════════

    def create_factory(self, product: str, capacity: float = None,
                       location: str = "Iraq") -> VirtualFactory:
        """إنشاء مصنع جديد"""
        catalog = FACTORY_CATALOG.get(product, {})
        if not capacity:
            capacity = catalog.get("capacity", 10000)

        factory = VirtualFactory(product, capacity, location)
        self.factories[factory.factory_id] = factory

        # ربط العمال بكبسولات التخصص
        self._assign_capsules_to_workers(factory)

        memory.save_knowledge(
            topic=f"Factory Created: {product}",
            content=f"مصنع {catalog.get('name_ar', product)}: {capacity:,} طن/سنة @ {location}",
            source="factory_manager",
        )
        logger.info(f"🏭 Created: {product} ({capacity:,} tons/year)")
        return factory

    def _assign_capsules_to_workers(self, factory: VirtualFactory):
        """ربط كل عامل بكبسولات التخصص من capsule_tree"""
        for worker in factory.workers:
            role_name = worker.role.name
            desired = ROLE_TO_CAPSULES.get(role_name, [])
            for keyword in desired:
                matches = capsule_tree.find_capsules(keyword, top_k=1)
                if matches:
                    cap_id = matches[0].node_id
                    if cap_id not in worker.assigned_capsules:
                        worker.assigned_capsules.append(cap_id)

    def create_all_essential(self) -> List[str]:
        """إنشاء كل المصانع الأساسية"""
        created = []
        for product in FACTORY_CATALOG:
            if not any(f.product == product for f in self.factories.values()):
                self.create_factory(product)
                created.append(product)
        return created

    # ═══════════════════════════════════════════════════════
    # الكشافة → مقترحات مصانع جديدة
    # ═══════════════════════════════════════════════════════

    def scout_propose_factory(self, product: str, reason: str,
                               source: str = "scout") -> Dict:
        """الكشافة أو أي طبقة تقترح مصنع جديد"""
        proposal = {
            "id": f"prop_{product}_{int(time.time())}",
            "product": product, "reason": reason,
            "source": source, "status": "pending",
            "timestamp": datetime.now().isoformat(),
        }
        self.pending_proposals.append(proposal)

        memory.save_decision(
            decision_type="factory_proposal",
            participants=[source],
            topic=f"مقترح مصنع: {product}",
            result=proposal,
        )

        return proposal

    def council_review_proposals(self) -> List[Dict]:
        """المجلس يراجع المقترحات أوتوماتيكياً"""
        results = []
        for proposal in self.pending_proposals:
            if proposal["status"] != "pending":
                continue
            if proposal["product"] in self.vetoed:
                proposal["status"] = "vetoed"
                results.append(proposal)
                continue

            # المجلس يوافق أوتوماتيكياً (إلا لو المستخدم رفض)
            proposal["status"] = "approved"
            factory = self.create_factory(proposal["product"])
            proposal["factory_id"] = factory.factory_id
            results.append(proposal)

        self.pending_proposals = [p for p in self.pending_proposals if p["status"] == "pending"]
        return results

    # ═══════════════════════════════════════════════════════
    # حق الفيتو — المستخدم يسيطر على كلشي
    # ═══════════════════════════════════════════════════════

    def user_veto(self, factory_id: str = None, product: str = None, reason: str = ""):
        """المستخدم يرفض — أمره فوق الكل"""
        cmd = {"type": "veto", "factory_id": factory_id, "product": product,
               "reason": reason, "timestamp": datetime.now().isoformat()}
        self.user_commands.append(cmd)

        if product:
            self.vetoed.append(product)
        if factory_id and factory_id in self.factories:
            del self.factories[factory_id]

        memory.save_decision(
            decision_type="user_veto", participants=["user"],
            topic=f"فيتو: {product or factory_id}",
            result=cmd,
        )
        logger.info(f"🚫 VETO: {product or factory_id} — {reason}")

    def user_command(self, command: str, target: str = None, params: Dict = None):
        """
        أمر مستخدم — يسري على الكل فوراً

        أمثلة:
          user_command("create", "cement", {"capacity": 100000})
          user_command("stop", factory_id)
          user_command("evolve", factory_id)
          user_command("create_all")
        """
        cmd = {"command": command, "target": target,
               "params": params or {}, "timestamp": datetime.now().isoformat()}
        self.user_commands.append(cmd)

        if command == "create" and target:
            cap = (params or {}).get("capacity")
            return self.create_factory(target, cap)
        elif command == "create_all":
            return self.create_all_essential()
        elif command == "stop" and target in self.factories:
            del self.factories[target]
        elif command == "evolve" and target in self.factories:
            old = self.factories[target]
            new = old.evolve()
            self.factories[new.factory_id] = new
            return new
        elif command == "veto":
            self.user_veto(product=target, reason=str(params))

    # ═══════════════════════════════════════════════════════
    # محاكاة + تقارير
    # ═══════════════════════════════════════════════════════

    def simulate_all(self, days: int = 30) -> Dict:
        """محاكاة كل المصانع"""
        results = {}
        for fid, factory in self.factories.items():
            factory.simulate_period(days)
            results[fid] = factory.get_report()
        return results

    def get_summary(self) -> Dict:
        """ملخص كل المصانع"""
        total_output = sum(f.total_output for f in self.factories.values())
        total_workers = sum(len(f.workers) for f in self.factories.values())
        return {
            "factories": len(self.factories),
            "products": list(set(f.product for f in self.factories.values())),
            "total_output_tons": round(total_output, 1),
            "total_workers": total_workers,
            "total_problems": sum(f.total_problems for f in self.factories.values()),
            "total_solved": sum(f.total_solved for f in self.factories.values()),
            "vetoed": self.vetoed,
            "user_commands": len(self.user_commands),
            "pending_proposals": len(self.pending_proposals),
        }

    def format_summary(self) -> str:
        """تقرير منسق"""
        s = self.get_summary()
        lines = [
            f"# 🏭 مدير المصانع — {s['factories']} مصنع\n",
            f"إنتاج كلي: **{s['total_output_tons']:,.0f} طن**",
            f"عمال AI: **{s['total_workers']}**",
            f"مشاكل: {s['total_problems']} (حُلت {s['total_solved']})",
            f"أوامر مستخدم: {s['user_commands']}",
            f"مرفوضة بالفيتو: {s['vetoed']}\n",
        ]

        for fid, f in self.factories.items():
            r = f.get_report()
            cat = FACTORY_CATALOG.get(f.product, {})
            lines.append(f"  🏭 {cat.get('name_ar', f.product):15s} v{r['version']} "
                        f"| يوم {r['day']:3d} | {r['phase']:12s} "
                        f"| {r['total_output']:8,.0f}ط | حل {r['solve_rate']}%")
        return "\n".join(lines)


# Singleton
factory_manager = FactoryManager()


if __name__ == "__main__":
    print("🏭 Virtual Factory System — Test\n")

    # إنشاء 5 مصانع أساسية
    for product in ["cement", "steel", "glass", "soap", "solar_panel"]:
        factory_manager.create_factory(product)

    print(f"Created {len(factory_manager.factories)} factories\n")

    # الكشافة تقترح مصنع جديد
    factory_manager.scout_propose_factory("battery", "needed for solar storage", "scout_tech")
    factory_manager.scout_propose_factory("plastic", "essential material", "scout_industry")

    # المجلس يراجع
    approved = factory_manager.council_review_proposals()
    print(f"Council approved: {len(approved)} proposals")

    # المستخدم يستخدم الفيتو
    factory_manager.user_veto(product="plastic", reason="not environmentally friendly")

    # محاكاة 30 يوم
    print("\nSimulating 30 days...\n")
    factory_manager.simulate_all(30)

    print(factory_manager.format_summary())

    # كتالوج كل المصانع المتاحة
    print(f"\n📋 Factory Catalog ({len(FACTORY_CATALOG)} types):")
    for prod, info in FACTORY_CATALOG.items():
        print(f"  {info['name_ar']:15s} ({prod:20s}) — {info['capacity']:>8,} طن — {info['category']}")
