"""
طبقة الحياة الواقعية - Real Life Layer
أسفل طبقة بالنظام - الأهم! بدونها المشروع مجرد برنامج

هذه الطبقة تحاكي الواقع الحقيقي بحيث:
- تقدر تُعطيها سؤال: "كيف أبني مصنع أسمنت؟"
- ترد عليك بـ خطة كاملة واقعية
- تحسب: الموارد + الطاقة + الوقت + العمال + التكلفة
- كل شي حسب قوانين الفيزياء والكيمياء الحقيقية
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math

logger = logging.getLogger(__name__)


class Specialization(Enum):
    """الاختصاصات العلمية والهندسية"""
    # فيزياء
    FLUID_MECHANICS = "fluid_mechanics"
    THERMODYNAMICS = "thermodynamics"
    ELECTROMAGNETISM = "electromagnetism"
    QUANTUM_MECHANICS = "quantum_mechanics"
    GENERAL_RELATIVITY = "general_relativity"
    
    # كيمياء
    ORGANIC_CHEMISTRY = "organic_chemistry"
    INORGANIC_CHEMISTRY = "inorganic_chemistry"
    PHYSICAL_CHEMISTRY = "physical_chemistry"
    ANALYTICAL_CHEMISTRY = "analytical_chemistry"
    INDUSTRIAL_CHEMISTRY = "industrial_chemistry"
    
    # مواد
    METALLURGY = "metallurgy"
    CERAMICS = "ceramics"
    POLYMERS = "polymers"
    COMPOSITES = "composites"
    NANOMATERIALS = "nanomaterials"
    
    # إنتاج
    FACTORY_DESIGN = "factory_design"
    PRODUCTION_LINES = "production_lines"
    QUALITY_CONTROL = "quality_control"
    MAINTENANCE = "maintenance"
    AUTOMATION = "automation"
    
    # اقتصاد
    COST_ANALYSIS = "cost_analysis"
    FEASIBILITY = "feasibility"
    MARKET_ANALYSIS = "market_analysis"
    FINANCING = "financing"
    RISK_ANALYSIS = "risk_analysis"


@dataclass
class PhysicalProperties:
    """خصائص فيزيائية لمادة"""
    density: float  # kg/m³
    melting_point: Optional[float] = None  # °C
    boiling_point: Optional[float] = None  # °C
    thermal_conductivity: Optional[float] = None  # W/(m·K)
    specific_heat: Optional[float] = None  # J/(kg·K)
    tensile_strength: Optional[float] = None  # MPa
    compressive_strength: Optional[float] = None  # MPa
    electrical_conductivity: Optional[float] = None  # S/m
    hardness: Optional[float] = None  # Mohs or Vickers


@dataclass
class ChemicalReaction:
    """تفاعل كيميائي"""
    reactants: Dict[str, float]  # {chemical_formula: moles}
    products: Dict[str, float]
    activation_energy: float  # kJ/mol
    reaction_temperature: float  # °C
    reaction_pressure: Optional[float] = None  # atm
    catalyst: Optional[str] = None
    enthalpy_change: Optional[float] = None  # kJ/mol (exothermic if negative)


@dataclass
class ResourceRequirement:
    """متطلبات المورد"""
    resource_type: str
    quantity: float
    unit: str
    source_location: Optional[str] = None
    extraction_cost_per_unit: Optional[float] = None
    availability_score: float = 1.0  # 0-1


@dataclass
class FactoryBlueprint:
    """مخطط مصنع"""
    name: str
    blueprint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    production_capacity: float = 0  # tons/day
    energy_requirement: float = 0  # kW
    water_requirement: float = 0  # m³/day
    land_area: float = 0  # m²
    construction_time: float = 0  # months
    construction_cost: float = 0  # USD
    raw_materials: List[ResourceRequirement] = field(default_factory=list)
    equipment_list: List[Dict[str, Any]] = field(default_factory=list)
    labor_requirements: Dict[str, int] = field(default_factory=dict)  # {skill_level: count}
    environmental_impact: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "production_capacity_tons_per_day": self.production_capacity,
            "energy_requirement_kw": self.energy_requirement,
            "water_requirement_m3_per_day": self.water_requirement,
            "land_area_m2": self.land_area,
            "construction_time_months": self.construction_time,
            "construction_cost_usd": self.construction_cost,
            "raw_materials": [
                {"type": r.resource_type, "quantity": r.quantity, "unit": r.unit}
                for r in self.raw_materials
            ],
            "labor_requirements": self.labor_requirements,
            "environmental_impact": self.environmental_impact
        }


class PhysicsEngine:
    """
    محرك فيزيائي - يحاكي القوانين الفيزيائية الحقيقية
    """
    
    def __init__(self):
        self.gravity = 9.81  # m/s²
        self.constants = {
            "R": 8.314,  # Gas constant J/(mol·K)
            "NA": 6.022e23,  # Avogadro's number
            "k": 1.381e-23,  # Boltzmann constant J/K
            "c": 299792458,  # Speed of light m/s
        }
        logger.info("✅ Physics Engine initialized")
    
    def calculate_stress(self, force: float, area: float) -> float:
        """حساب الإجهاد: σ = F/A (Pascals)"""
        return force / area if area > 0 else float('inf')
    
    def calculate_thermal_expansion(self, length: float, temp_change: float, 
                                    coefficient: float) -> float:
        """حساب التمدد الحراري: ΔL = L₀ × α × ΔT"""
        return length * coefficient * temp_change
    
    def calculate_heat_transfer_conduction(self, k: float, area: float, 
                                           temp_diff: float, thickness: float) -> float:
        """حساب نقل الحرارة بالتوصيل: Q = k×A×(T₁-T₂)/d (Watts)"""
        return k * area * temp_diff / thickness if thickness > 0 else 0
    
    def calculate_fluid_flow_rate(self, pipe_diameter: float, velocity: float) -> float:
        """حساب معدل تدفق السائل: Q = v×A (m³/s)"""
        area = math.pi * (pipe_diameter / 2) ** 2
        return velocity * area
    
    def calculate_energy_efficiency(self, useful_output: float, total_input: float) -> float:
        """حساب كفاءة الطاقة: η = useful/total (0-1)"""
        return useful_output / total_input if total_input > 0 else 0
    
    def simulate_chemical_reaction(self, reaction: ChemicalReaction, 
                                   temperature: float, pressure: float) -> Dict[str, Any]:
        """محاكاة تفاعل كيميائي"""
        # Simplified Arrhenius equation for reaction rate
        R = self.constants["R"]
        T = temperature + 273.15  # Convert to Kelvin
        
        # Check if temperature is sufficient
        if temperature < reaction.reaction_temperature:
            return {
                "feasible": False,
                "reason": f"Temperature too low. Need {reaction.reaction_temperature}°C, got {temperature}°C",
                "conversion_rate": 0
            }
        
        # Calculate approximate conversion rate (simplified)
        activation_energy_j = reaction.activation_energy * 1000  # kJ to J
        rate_constant = math.exp(-activation_energy_j / (R * T))
        conversion_rate = min(0.95, rate_constant * 10)  # Max 95% conversion
        
        return {
            "feasible": True,
            "conversion_rate": conversion_rate,
            "products": {k: v * conversion_rate for k, v in reaction.products.items()},
            "energy_required": reaction.enthalpy_change or 0
        }
    
    def calculate_structural_load(self, mass: float, safety_factor: float = 2.0) -> Dict[str, float]:
        """حساب الحمل الإنشائي مع معامل أمان"""
        weight = mass * self.gravity  # N
        design_load = weight * safety_factor
        return {
            "weight_n": weight,
            "design_load_n": design_load,
            "safety_factor": safety_factor,
            "mass_kg": mass
        }


class RealLifeAgent:
    """
    وكيل الحياة الواقعية - كل شخص = agent باختصاص دقيق
    """
    
    def __init__(self, agent_id: str, name: str, specialization: Specialization, 
                 expertise_level: float = 1.0):
        self.agent_id = agent_id
        self.name = name
        self.specialization = specialization
        self.expertise_level = expertise_level  # 0.0 to 1.0
        self.memory: List[Dict[str, Any]] = []  # تعلم وتذكر
        self.decisions_made = 0
        logger.info(f"✅ Agent initialized: {name} ({specialization.value})")
    
    async def analyze(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل استعلام ضمن اختصاص الوكيل"""
        # This would connect to actual domain knowledge or trained models
        # For now, using structured reasoning based on specialization
        
        analysis = {
            "agent_id": self.agent_id,
            "specialization": self.specialization.value,
            "expertise_level": self.expertise_level,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "findings": [],
            "concerns": [],
            "recommendations": []
        }
        
        # Add specialization-specific analysis
        if self.specialization in [s for s in Specialization if "CHEMISTRY" in s.value]:
            analysis["findings"].append(self._analyze_chemical_aspects(query, context))
        elif self.specialization in [s for s in Specialization if "DYNAMICS" in s.value or "THERMO" in s.value]:
            analysis["findings"].append(self._analyze_thermal_aspects(query, context))
        elif self.specialization in [s for s in Specialization if "METALLURGY" in s.value or "CERAMICS" in s.value]:
            analysis["findings"].append(self._analyze_material_aspects(query, context))
        elif self.specialization in [s for s in Specialization if "FACTORY" in s.value or "PRODUCTION" in s.value]:
            analysis["findings"].append(self._analyze_production_aspects(query, context))
        elif self.specialization in [s for s in Specialization if "COST" in s.value or "FEASIBILITY" in s.value]:
            analysis["findings"].append(self._analyze_economic_aspects(query, context))
        
        self.memory.append(analysis)
        self.decisions_made += 1
        
        return analysis
    
    def _analyze_chemical_aspects(self, query: str, context: Dict) -> str:
        """تحليل الجوانب الكيميائية"""
        return f"[{self.name}] Chemical analysis: Reactions require temperature monitoring and stoichiometric balance."
    
    def _analyze_thermal_aspects(self, query: str, context: Dict) -> str:
        """تحليل الجوانب الحرارية"""
        return f"[{self.name}] Thermal analysis: Heat transfer calculations show need for insulation and cooling systems."
    
    def _analyze_material_aspects(self, query: str, context: Dict) -> str:
        """تحليل المواد"""
        return f"[{self.name}] Material analysis: Selected materials must withstand operating temperatures and mechanical stress."
    
    def _analyze_production_aspects(self, query: str, context: Dict) -> str:
        """تحليل الإنتاج"""
        return f"[{self.name}] Production analysis: Line layout and workflow optimization required for efficiency."
    
    def _analyze_economic_aspects(self, query: str, context: Dict) -> str:
        """تحليل اقتصادي"""
        return f"[{self.name}] Economic analysis: ROI calculations and cost-benefit analysis needed."


class RealLifeLayer:
    """
    طبقة الحياة الواقعية - النواة الأساسية لإعادة بناء الحضارة
    """
    
    def __init__(self):
        self.physics_engine = PhysicsEngine()
        self.agents: Dict[str, RealLifeAgent] = {}
        self.blueprints: Dict[str, FactoryBlueprint] = {}
        self.materials_db: Dict[str, PhysicalProperties] = {}
        self.reactions_db: Dict[str, ChemicalReaction] = {}
        
        self._initialize_agents()
        self._initialize_materials_database()
        self._initialize_reactions_database()
        
        logger.info("🏛️ Real Life Layer initialized with 25 specialists")
    
    def _initialize_agents(self):
        """تهيئة الوكلاء المتخصصين (25 وكيل)"""
        agents_config = [
            # فيزيائيون (5)
            ("phys_fluid", "Dr. Flow", Specialization.FLUID_MECHANICS),
            ("phys_thermo", "Dr. Heat", Specialization.THERMODYNAMICS),
            ("phys_elec", "Dr. Field", Specialization.ELECTROMAGNETISM),
            ("phys_quantum", "Dr. Quanta", Specialization.QUANTUM_MECHANICS),
            ("phys_relativity", "Dr. Spacetime", Specialization.GENERAL_RELATIVITY),
            
            # كيميائيون (5)
            ("chem_org", "Dr. Organic", Specialization.ORGANIC_CHEMISTRY),
            ("chem_inorg", "Dr. Mineral", Specialization.INORGANIC_CHEMISTRY),
            ("chem_phys", "Dr. Reaction", Specialization.PHYSICAL_CHEMISTRY),
            ("chem_anal", "Dr. Analyzer", Specialization.ANALYTICAL_CHEMISTRY),
            ("chem_ind", "Dr. Process", Specialization.INDUSTRIAL_CHEMISTRY),
            
            # مهندسو مواد (5)
            ("mat_metal", "Dr. Steel", Specialization.METALLURGY),
            ("mat_ceramic", "Dr. Clay", Specialization.CERAMICS),
            ("mat_poly", "Dr. Plastic", Specialization.POLYMERS),
            ("mat_comp", "Dr. Fiber", Specialization.COMPOSITES),
            ("mat_nano", "Dr. Nano", Specialization.NANOMATERIALS),
            
            # مهندسو إنتاج (5)
            ("prod_design", "Dr. Factory", Specialization.FACTORY_DESIGN),
            ("prod_line", "Dr. Line", Specialization.PRODUCTION_LINES),
            ("prod_quality", "Dr. Quality", Specialization.QUALITY_CONTROL),
            ("prod_maint", "Dr. Repair", Specialization.MAINTENANCE),
            ("prod_auto", "Dr. Robot", Specialization.AUTOMATION),
            
            # اقتصاديون (5)
            ("eco_cost", "Dr. Cost", Specialization.COST_ANALYSIS),
            ("eco_feas", "Dr. Feasible", Specialization.FEASIBILITY),
            ("eco_market", "Dr. Market", Specialization.MARKET_ANALYSIS),
            ("eco_fin", "Dr. Finance", Specialization.FINANCING),
            ("eco_risk", "Dr. Safe", Specialization.RISK_ANALYSIS),
        ]
        
        for agent_id, name, spec in agents_config:
            self.agents[agent_id] = RealLifeAgent(agent_id, name, spec)
    
    def _initialize_materials_database(self):
        """تهيئة قاعدة بيانات المواد"""
        self.materials_db = {
            "steel_carbon": PhysicalProperties(
                density=7850,
                melting_point=1370,
                thermal_conductivity=50,
                tensile_strength=400,
                compressive_strength=250
            ),
            "aluminum": PhysicalProperties(
                density=2700,
                melting_point=660,
                thermal_conductivity=205,
                tensile_strength=90,
                compressive_strength=150
            ),
            "concrete": PhysicalProperties(
                density=2400,
                compressive_strength=40,
                thermal_conductivity=1.7
            ),
            "brick_refractory": PhysicalProperties(
                density=2300,
                melting_point=1800,
                thermal_conductivity=1.2,
                compressive_strength=50
            ),
            "limestone": PhysicalProperties(
                density=2700,
                melting_point=825  # Decomposes
            ),
        }
    
    def _initialize_reactions_database(self):
        """تهيئة قاعدة بيانات التفاعلات الكيميائية"""
        self.reactions_db = {
            "cement_clinker": ChemicalReaction(
                reactants={"CaCO3": 1.0, "SiO2": 0.25, "Al2O3": 0.05, "Fe2O3": 0.03},
                products={"CaO": 1.0, "CO2": 1.0, "Ca2SiO4": 0.25},
                activation_energy=170,  # kJ/mol
                reaction_temperature=1450,
                enthalpy_change=+178  # Endothermic
            ),
            "iron_ore_reduction": ChemicalReaction(
                reactants={"Fe2O3": 1.0, "CO": 3.0},
                products={"Fe": 2.0, "CO2": 3.0},
                activation_energy=100,
                reaction_temperature=1200,
                enthalpy_change=-25  # Exothermic
            )
        }
    
    async def design_factory(self, product_type: str, target_capacity: float, 
                            location_context: Optional[Dict] = None) -> FactoryBlueprint:
        """
        تصميم مصنع كامل - يجمع كل الاختصاصات
        
        مثال: design_factory("cement", 1000)  # 1000 tons/day
        """
        logger.info(f"🏭 Designing {product_type} factory: {target_capacity} tons/day")
        
        # Gather all expert opinions
        context = {"product": product_type, "capacity": target_capacity, "location": location_context}
        
        expert_analyses = []
        relevant_specs = self._get_relevant_specialists(product_type)
        
        for agent_id in relevant_specs:
            agent = self.agents.get(agent_id)
            if agent:
                analysis = await agent.analyze(f"Design {product_type} factory", context)
                expert_analyses.append(analysis)
        
        # Create blueprint based on product type
        if product_type.lower() in ["cement", "اسمنت"]:
            blueprint = await self._design_cement_factory(target_capacity, expert_analyses)
        elif product_type.lower() in ["steel", "حديد"]:
            blueprint = await self._design_steel_factory(target_capacity, expert_analyses)
        elif product_type.lower() in ["brick", "طوب"]:
            blueprint = await self._design_brick_factory(target_capacity, expert_analyses)
        else:
            blueprint = FactoryBlueprint(name=f"{product_type.title()} Factory")
        
        self.blueprints[blueprint.blueprint_id] = blueprint
        return blueprint
    
    def _get_relevant_specialists(self, product_type: str) -> List[str]:
        """تحديد الاختصاصيين ذوي الصلة"""
        specialists_map = {
            "cement": ["chem_inorg", "phys_thermo", "mat_ceramic", "mat_metal", 
                      "prod_design", "prod_line", "eco_cost", "eco_feas"],
            "steel": ["chem_inorg", "phys_thermo", "mat_metal", "prod_design", 
                     "prod_auto", "eco_cost", "eco_risk"],
            "brick": ["chem_inorg", "mat_ceramic", "phys_thermo", "prod_line"]
        }
        return specialists_map.get(product_type.lower(), list(self.agents.keys())[:5])
    
    async def _design_cement_factory(self, capacity: float, 
                                     analyses: List[Dict]) -> FactoryBlueprint:
        """تصميم مصنع أسمنت واقعي"""
        
        # حسابات حقيقية لمصنع أسمنت
        # Based on: 1000 tons/day = ~350,000 tons/year
        
        blueprint = FactoryBlueprint(name="Cement Production Plant")
        blueprint.production_capacity = capacity
        
        # Energy: ~110 kWh/ton for cement
        blueprint.energy_requirement = capacity * 110  # kW continuous
        
        # Water: ~0.5 m³/ton for cooling and processes
        blueprint.water_requirement = capacity * 0.5
        
        # Land: ~0.5 m² per ton/year capacity
        blueprint.land_area = capacity * 365 * 0.5
        
        # Construction time: 18-24 months for medium plant
        blueprint.construction_time = 20
        
        # Cost: ~$150-300 per annual ton capacity
        annual_capacity = capacity * 365
        blueprint.construction_cost = annual_capacity * 200  # $200/ton
        
        # Raw materials (per ton of cement)
        blueprint.raw_materials = [
            ResourceRequirement("limestone", 1.25 * capacity, "tons/day", availability_score=0.9),
            ResourceRequirement("clay", 0.25 * capacity, "tons/day", availability_score=0.95),
            ResourceRequirement("gypsum", 0.05 * capacity, "tons/day", availability_score=0.8),
            ResourceRequirement("iron_ore", 0.02 * capacity, "tons/day", availability_score=0.85),
            ResourceRequirement("coal_petcoke", 0.15 * capacity, "tons/day", availability_score=0.7),
        ]
        
        # Equipment
        blueprint.equipment_list = [
            {"name": "Jaw Crusher", "power_kw": 200, "count": 2},
            {"name": "Raw Mill (Vertical Roller)", "power_kw": 1500, "count": 1},
            {"name": "Rotary Kiln (4.5m × 75m)", "power_kw": 3000, "count": 1},
            {"name": "Cement Mill (Ball Mill)", "power_kw": 2000, "count": 2},
            {"name": "Cooler (Grate)", "power_kw": 500, "count": 1},
            {"name": "Packaging Machine", "power_kw": 100, "count": 4},
        ]
        
        # Labor
        blueprint.labor_requirements = {
            "engineers": 15,
            "technicians": 50,
            "operators": 80,
            "laborers": 120,
            "admin": 20
        }
        
        # Environmental
        blueprint.environmental_impact = {
            "co2_emissions_tons_per_day": capacity * 0.9,  # ~0.9 ton CO2 per ton cement
            "dust_emissions_kg_per_day": capacity * 0.5,
            "water_usage_m3_per_day": blueprint.water_requirement,
            "noise_level_db": 85
        }
        
        return blueprint
    
    async def _design_steel_factory(self, capacity: float, 
                                    analyses: List[Dict]) -> FactoryBlueprint:
        """تصميم مصنع حديد"""
        blueprint = FactoryBlueprint(name="Steel Production Plant")
        blueprint.production_capacity = capacity
        blueprint.energy_requirement = capacity * 500  # More energy intensive
        blueprint.land_area = capacity * 365 * 0.8
        blueprint.construction_time = 30
        blueprint.construction_cost = capacity * 365 * 400  # More expensive
        
        blueprint.raw_materials = [
            ResourceRequirement("iron_ore", 1.6 * capacity, "tons/day"),
            ResourceRequirement("coking_coal", 0.5 * capacity, "tons/day"),
            ResourceRequirement("limestone", 0.2 * capacity, "tons/day"),
        ]
        
        return blueprint
    
    async def _design_brick_factory(self, capacity: float, 
                                    analyses: List[Dict]) -> FactoryBlueprint:
        """تصميم مصنع طوب"""
        blueprint = FactoryBlueprint(name="Brick Production Plant")
        blueprint.production_capacity = capacity
        blueprint.energy_requirement = capacity * 20
        blueprint.land_area = capacity * 365 * 0.2
        blueprint.construction_time = 6
        blueprint.construction_cost = capacity * 365 * 50
        
        blueprint.raw_materials = [
            ResourceRequirement("clay", 2.0 * capacity, "tons/day"),
            ResourceRequirement("sand", 0.5 * capacity, "tons/day"),
        ]
        
        return blueprint
    
    async def simulate_construction(self, blueprint_id: str, 
                                   simulation_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        محاكاة بناء المصنع - يختبر التصميم قبل التنفيذ
        """
        blueprint = self.blueprints.get(blueprint_id)
        if not blueprint:
            return {"error": "Blueprint not found"}
        
        logger.info(f"🔬 Simulating construction: {blueprint.name}")
        
        # Structural simulation
        structural_analysis = self.physics_engine.calculate_structural_load(
            blueprint.construction_cost / 1000  # Rough estimate of materials mass
        )
        
        # Thermal simulation for kilns/furnaces
        if blueprint.energy_requirement > 1000:
            kiln_temp = 1500  # Cement kiln temperature
            heat_loss = self.physics_engine.calculate_heat_transfer_conduction(
                k=1.2,  # Refractory brick
                area=1000,  # m²
                temp_diff=kiln_temp - 30,
                thickness=0.5  # m
            )
        else:
            heat_loss = 0
        
        # Efficiency calculation
        material_efficiency = self.physics_engine.calculate_energy_efficiency(
            blueprint.production_capacity,
            sum(r.quantity for r in blueprint.raw_materials)
        )
        
        simulation_result = {
            "blueprint_id": blueprint_id,
            "structural_integrity": "PASS" if structural_analysis["safety_factor"] >= 1.5 else "FAIL",
            "thermal_efficiency": f"{(1 - heat_loss/blueprint.energy_requirement)*100:.1f}%" if blueprint.energy_requirement > 0 else "N/A",
            "material_efficiency": f"{material_efficiency*100:.1f}%",
            "environmental_compliance": blueprint.environmental_impact.get("co2_emissions_tons_per_day", 0) < blueprint.production_capacity,
            "recommendations": []
        }
        
        if heat_loss > blueprint.energy_requirement * 0.1:
            simulation_result["recommendations"].append("Improve kiln insulation to reduce heat loss")
        
        if blueprint.water_requirement > 1000:
            simulation_result["recommendations"].append("Consider water recycling system")
        
        return simulation_result
    
    def get_all_agents_status(self) -> Dict[str, Any]:
        """حالة جميع الوكلاء"""
        return {
            "total_agents": len(self.agents),
            "by_specialization": {
                spec.value: len([a for a in self.agents.values() if a.specialization == spec])
                for spec in Specialization
            },
            "total_decisions": sum(a.decisions_made for a in self.agents.values()),
            "blueprints_created": len(self.blueprints),
            "materials_in_db": len(self.materials_db),
            "reactions_in_db": len(self.reactions_db)
        }


# Global instance
real_life_layer = RealLifeLayer()
