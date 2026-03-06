#!/usr/bin/env python3
"""
real_life_layer.py — طبقة الحياة الواقعية 🏭⚛️

"كيف أبني مصنع أسمنت؟" → خطة كاملة واقعية:
  - فيزياء: حرارة، ضغط، قوى
  - كيمياء: تفاعلات، مواد، خواص
  - اقتصاد: تكلفة، جدوى، موارد
  - إنتاج: خطوط، محاكاة
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("real_life")

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Physics Engine
# ═══════════════════════════════════════════════════════════

class PhysicsEngine:
    """محاكاة فيزيائية — حرارة، ضغط، قوى، كهرباء"""

    # ثوابت فيزيائية
    GRAVITY = 9.81          # m/s²
    BOLTZMANN = 1.38e-23    # J/K
    STEFAN = 5.67e-8        # W/m²K⁴
    ATM = 101325            # Pa

    def heat_transfer(self, temp_hot: float, temp_cold: float,
                      area: float, thickness: float,
                      conductivity: float) -> float:
        """حساب انتقال الحرارة (واط)"""
        return conductivity * area * (temp_hot - temp_cold) / thickness

    def energy_to_heat(self, mass: float, specific_heat: float,
                       temp_change: float) -> float:
        """طاقة لرفع الحرارة (جول)"""
        return mass * specific_heat * temp_change

    def pressure_at_depth(self, depth: float, fluid_density: float = 1000) -> float:
        """ضغط عند عمق (باسكال)"""
        return self.ATM + fluid_density * self.GRAVITY * depth

    def structural_load(self, force: float, area: float) -> float:
        """إجهاد هيكلي (باسكال)"""
        return force / area if area > 0 else float('inf')

    def power_generation(self, flow_rate: float, head: float,
                         efficiency: float = 0.85) -> float:
        """طاقة كهرومائية (واط)"""
        return 1000 * self.GRAVITY * flow_rate * head * efficiency

    def solar_power(self, area: float, irradiance: float = 1000,
                    efficiency: float = 0.20) -> float:
        """طاقة شمسية (واط)"""
        return area * irradiance * efficiency

    def wind_power(self, blade_radius: float, wind_speed: float,
                   air_density: float = 1.225, efficiency: float = 0.40) -> float:
        """طاقة رياح (واط)"""
        area = math.pi * blade_radius ** 2
        return 0.5 * air_density * area * wind_speed ** 3 * efficiency


# ═══════════════════════════════════════════════════════════
# Chemistry Engine
# ═══════════════════════════════════════════════════════════

class ChemistryEngine:
    """تفاعلات كيميائية — مواد، تفاعلات، خواص"""

    REACTIONS = {
        "cement": {
            "name": "صناعة الأسمنت",
            "equation": "CaCO₃ → CaO + CO₂",
            "temp_required": 1450,  # °C
            "energy_per_ton": 3.5,  # GJ/ton
            "raw_materials": {"limestone": 1.5, "clay": 0.3, "gypsum": 0.05},  # tons per ton cement
            "byproducts": {"CO2": 0.6},  # tons
        },
        "steel": {
            "name": "صناعة الحديد",
            "equation": "Fe₂O₃ + 3CO → 2Fe + 3CO₂",
            "temp_required": 1600,
            "energy_per_ton": 20,
            "raw_materials": {"iron_ore": 1.6, "coal": 0.5, "limestone": 0.3},
            "byproducts": {"CO2": 1.8, "slag": 0.3},
        },
        "glass": {
            "name": "صناعة الزجاج",
            "equation": "SiO₂ + Na₂CO₃ → Na₂SiO₃ + CO₂",
            "temp_required": 1700,
            "energy_per_ton": 8,
            "raw_materials": {"sand": 0.7, "soda_ash": 0.25, "limestone": 0.15},
            "byproducts": {"CO2": 0.2},
        },
        "brick": {
            "name": "صناعة الطوب",
            "equation": "Clay → Fired Brick",
            "temp_required": 1000,
            "energy_per_ton": 2,
            "raw_materials": {"clay": 1.2, "sand": 0.1},
            "byproducts": {},
        },
        "concrete": {
            "name": "صناعة الخرسانة",
            "equation": "Cement + Sand + Gravel + Water",
            "temp_required": 25,
            "energy_per_ton": 1,
            "raw_materials": {"cement": 0.12, "sand": 0.4, "gravel": 0.6, "water": 0.18},
            "byproducts": {},
        },
        "soap": {
            "name": "صناعة الصابون",
            "equation": "Fat + NaOH → Soap + Glycerol",
            "temp_required": 80,
            "energy_per_ton": 0.5,
            "raw_materials": {"fat_oil": 0.8, "sodium_hydroxide": 0.15, "water": 0.3},
            "byproducts": {"glycerol": 0.1},
        },
    }

    MATERIALS = {
        "steel": {"density": 7850, "melting": 1538, "tensile_strength": 400e6},
        "concrete": {"density": 2400, "compressive_strength": 30e6},
        "wood": {"density": 600, "tensile_strength": 50e6},
        "aluminum": {"density": 2700, "melting": 660, "tensile_strength": 300e6},
        "copper": {"density": 8960, "melting": 1085, "conductivity": 401},
        "glass": {"density": 2500, "melting": 1700, "tensile_strength": 33e6},
        "brick": {"density": 1900, "compressive_strength": 15e6},
    }

    def get_reaction(self, product: str) -> Optional[Dict]:
        return self.REACTIONS.get(product)

    def get_material(self, name: str) -> Optional[Dict]:
        return self.MATERIALS.get(name)

    def calculate_production(self, product: str, quantity_tons: float) -> Dict:
        """حساب متطلبات الإنتاج"""
        reaction = self.REACTIONS.get(product)
        if not reaction:
            return {"error": f"Unknown product: {product}"}

        raw = {k: v * quantity_tons for k, v in reaction["raw_materials"].items()}
        byp = {k: v * quantity_tons for k, v in reaction.get("byproducts", {}).items()}
        energy = reaction["energy_per_ton"] * quantity_tons

        return {
            "product": product,
            "quantity_tons": quantity_tons,
            "reaction": reaction["equation"],
            "temp_required_c": reaction["temp_required"],
            "energy_gj": round(energy, 2),
            "raw_materials_tons": {k: round(v, 2) for k, v in raw.items()},
            "byproducts_tons": {k: round(v, 2) for k, v in byp.items()},
        }


# ═══════════════════════════════════════════════════════════
# Economics Engine
# ═══════════════════════════════════════════════════════════

class EconomicsEngine:
    """تكاليف — جدوى — موارد"""

    COST_PER_TON = {
        "limestone": 15, "clay": 10, "gypsum": 30, "sand": 12,
        "iron_ore": 100, "coal": 80, "soda_ash": 200,
        "fat_oil": 500, "sodium_hydroxide": 300, "cement": 100,
        "gravel": 15, "water": 0.5,
    }

    ENERGY_COST = 20  # $/GJ

    def estimate_cost(self, raw_materials: Dict, energy_gj: float) -> Dict:
        """تقدير التكلفة"""
        material_cost = sum(
            self.COST_PER_TON.get(m, 50) * qty
            for m, qty in raw_materials.items()
        )
        energy_cost = energy_gj * self.ENERGY_COST
        labor_cost = material_cost * 0.3
        overhead = (material_cost + energy_cost) * 0.15
        total = material_cost + energy_cost + labor_cost + overhead

        return {
            "material_cost": round(material_cost, 2),
            "energy_cost": round(energy_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "overhead": round(overhead, 2),
            "total_cost": round(total, 2),
        }


# ═══════════════════════════════════════════════════════════
# Factory Planner — مخطط المصانع
# ═══════════════════════════════════════════════════════════

@dataclass
class FactoryPlan:
    """خطة مصنع"""
    product: str
    capacity_tons_per_year: float
    location: str
    chemistry: Dict
    physics: Dict
    economics: Dict
    timeline_months: int
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FactoryPlanner:
    """مخطط المصانع — يصمم مصنع كامل واقعياً"""

    def __init__(self):
        self.physics = PhysicsEngine()
        self.chemistry = ChemistryEngine()
        self.economics = EconomicsEngine()

    def plan_factory(self, product: str, capacity: float = 100000,
                     location: str = "Iraq") -> FactoryPlan:
        """
        تخطيط مصنع كامل

        Args:
            product: نوع المنتج (cement, steel, glass, etc)
            capacity: الطاقة الإنتاجية (طن/سنة)
            location: الموقع
        """
        # كيمياء
        chem = self.chemistry.calculate_production(product, capacity)
        if "error" in chem:
            return FactoryPlan(product=product, capacity_tons_per_year=capacity,
                              location=location, chemistry=chem, physics={},
                              economics={}, timeline_months=0)

        # فيزياء — الطاقة
        daily_tons = capacity / 365
        reaction = self.chemistry.get_reaction(product)
        temp = reaction["temp_required"] if reaction else 1000

        furnace_power = self.physics.heat_transfer(temp, 25, 50, 0.3, 1.5)
        solar = self.physics.solar_power(5000)
        wind = self.physics.wind_power(25, 8)

        phys = {
            "furnace_temp_c": temp,
            "furnace_power_kw": round(furnace_power / 1000, 1),
            "daily_production_tons": round(daily_tons, 1),
            "solar_backup_kw": round(solar / 1000, 1),
            "wind_backup_kw": round(wind / 1000, 1),
        }

        # اقتصاد
        econ = self.economics.estimate_cost(chem["raw_materials_tons"], chem["energy_gj"])
        econ["cost_per_ton"] = round(econ["total_cost"] / capacity, 2) if capacity > 0 else 0

        # الجدول الزمني
        if capacity < 10000:
            timeline = 6
        elif capacity < 100000:
            timeline = 12
        else:
            timeline = 18

        # توصيات
        recommendations = [
            f"الطاقة الإنتاجية: {capacity:,.0f} طن/سنة ({daily_tons:.0f} طن/يوم)",
            f"درجة حرارة الفرن: {temp}°C",
            f"الطاقة المطلوبة: {chem['energy_gj']:,.0f} GJ/سنة",
        ]

        if solar > 100000:
            recommendations.append("✅ الطاقة الشمسية كافية كمصدر احتياطي")
        if temp > 1400:
            recommendations.append("⚠️ يحتاج طوب حراري عالي الجودة (alumina-silica)")

        plan = FactoryPlan(
            product=product,
            capacity_tons_per_year=capacity,
            location=location,
            chemistry=chem,
            physics=phys,
            economics=econ,
            timeline_months=timeline,
            recommendations=recommendations,
        )

        # حفظ بالذاكرة الأبدية
        memory.save_knowledge(
            topic=f"Factory: {product} in {location}",
            content=json.dumps({"capacity": capacity, "cost": econ["total_cost"],
                               "timeline": timeline}, ensure_ascii=False),
            source="factory_planner",
        )

        return plan

    def format_plan(self, plan: FactoryPlan) -> str:
        """تنسيق الخطة للعرض"""
        lines = [
            f"# 🏭 خطة مصنع: {plan.product}",
            f"**الموقع**: {plan.location}",
            f"**الطاقة**: {plan.capacity_tons_per_year:,.0f} طن/سنة",
            f"**مدة البناء**: {plan.timeline_months} شهر\n",
            "## ⚛️ الكيمياء",
            f"**التفاعل**: `{plan.chemistry.get('reaction', 'N/A')}`",
            f"**الحرارة**: {plan.chemistry.get('temp_required_c', 0)}°C",
            f"**الطاقة**: {plan.chemistry.get('energy_gj', 0):,.0f} GJ/سنة\n",
            "**المواد الخام** (طن/سنة):",
        ]
        for mat, qty in plan.chemistry.get("raw_materials_tons", {}).items():
            lines.append(f"  - {mat}: {qty:,.0f}")

        lines.extend([
            f"\n## ⚡ الفيزياء",
            f"حرارة الفرن: {plan.physics.get('furnace_temp_c', 0)}°C",
            f"إنتاج يومي: {plan.physics.get('daily_production_tons', 0)} طن",
            f"طاقة شمسية احتياطية: {plan.physics.get('solar_backup_kw', 0)} kW",
            f"\n## 💰 الاقتصاد",
            f"تكلفة المواد: ${plan.economics.get('material_cost', 0):,.0f}",
            f"تكلفة الطاقة: ${plan.economics.get('energy_cost', 0):,.0f}",
            f"عمالة: ${plan.economics.get('labor_cost', 0):,.0f}",
            f"**إجمالي**: ${plan.economics.get('total_cost', 0):,.0f}",
            f"**تكلفة/طن**: ${plan.economics.get('cost_per_ton', 0):.2f}",
            f"\n## 📋 توصيات",
        ])
        for r in plan.recommendations:
            lines.append(f"- {r}")

        return "\n".join(lines)


planner = FactoryPlanner()


if __name__ == "__main__":
    print("🏭 Factory Planner — Test\n")

    for product in ["cement", "steel", "glass", "soap"]:
        plan = planner.plan_factory(product, capacity=50000)
        print(f"{'═' * 50}")
        print(planner.format_plan(plan))
        print()
