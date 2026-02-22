"""
خبراء المجالات - 8-16 خبير متخصص
Domain Experts - Specialized Knowledge

ERP:
- خبير المحاسبة (Accounting)
- خبير المخازن (Inventory)
- خبير الموارد البشرية (HR)
- خبير المبيعات (Sales)
- خبير المشتريات (Purchasing)
- خبير العملاء (CRM)

Technical:
- خبير البرمجة (Rust, Python, TS)
- خبير قواعد البيانات (PostgreSQL)
- خبير الذكاء الاصطناعي (AI/ML)
- خبير الأمان (Security)
"""
import sys; sys.path.insert(0, '.'); import encoding_fix; encoding_fix.safe_print("")

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio


class DomainType(Enum):
    """أنواع المجالات"""
    ACCOUNTING = "accounting"       # محاسبة
    INVENTORY = "inventory"         # مخازن
    HR = "hr"                       # موارد بشرية
    SALES = "sales"                 # مبيعات
    PURCHASING = "purchasing"       # مشتريات
    CRM = "crm"                     # عملاء
    RUST = "rust"                   # برمجة Rust
    PYTHON = "python"               # برمجة Python
    TYPESCRIPT = "typescript"       # برمجة TS
    DATABASE = "database"           # قواعد بيانات
    AI_ML = "ai_ml"                 # ذكاء اصطناعي
    SECURITY = "security"           # أمان


@dataclass
class Expertise:
    """خبرة في مجال معين"""
    domain: DomainType
    level: int  # 1-10
    years_experience: float
    certifications: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)


@dataclass
class DomainExpert:
    """خبير مجال"""
    id: str
    name: str
    expertise: Expertise
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    is_active: bool = True
    
    # إحصائيات
    queries_answered: int = 0
    success_rate: float = 1.0
    
    async def answer_query(self, query: str, context: Dict = None) -> Dict:
        """
        الإجابة على استعلام في المجال
        
        Args:
            query: السؤال
            context: سياق إضافي
            
        Returns:
            إجابة + ثقة + خطوات
        """
        # تحليل الاستعلام
        parsed = self._parse_query(query)
        
        # البحث في قاعدة المعرفة
        knowledge = self._search_knowledge(parsed)
        
        # تطبيق القواعد
        rules_applied = self._apply_rules(parsed, knowledge)
        
        # توليد الإجابة
        answer = self._generate_answer(knowledge, rules_applied, context)
        
        self.queries_answered += 1
        
        return {
            'answer': answer,
            'confidence': self._calculate_confidence(knowledge),
            'steps': rules_applied,
            'domain': self.expertise.domain.value,
            'expert_id': self.id
        }
    
    def _parse_query(self, query: str) -> Dict:
        """تحليل الاستعلام"""
        return {
            'intent': self._detect_intent(query),
            'entities': self._extract_entities(query),
            'complexity': self._assess_complexity(query)
        }
    
    def _detect_intent(self, query: str) -> str:
        """كشف النية"""
        intents = {
            'how': 'procedural',      # كيف
            'what': 'information',    # ما
            'why': 'causal',          # لماذا
            'when': 'temporal',       # متى
            'calculate': 'computation' # احسب
        }
        for key, intent in intents.items():
            if key in query.lower():
                return intent
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """استخراج الكيانات"""
        # في الواقع: استخدم NER
        return []
    
    def _assess_complexity(self, query: str) -> int:
        """تقييم التعقيد"""
        words = len(query.split())
        if words < 5:
            return 1  # بسيط
        elif words < 15:
            return 2  # متوسط
        else:
            return 3  # معقد
    
    def _search_knowledge(self, parsed: Dict) -> List[Dict]:
        """البحث في قاعدة المعرفة"""
        # محاكاة - في الواقع: vector search
        return [{'topic': 'general', 'content': 'base knowledge'}]
    
    def _apply_rules(self, parsed: Dict, knowledge: List[Dict]) -> List[str]:
        """تطبيق القواعد"""
        steps = []
        for rule in self.rules:
            if self._rule_matches(rule, parsed):
                steps.append(f"Applied: {rule}")
        return steps
    
    def _rule_matches(self, rule: str, parsed: Dict) -> bool:
        """هل القاعدة تنطبق؟"""
        return True  # محاكاة
    
    def _generate_answer(self, knowledge: List[Dict], 
                        rules_applied: List[str],
                        context: Dict) -> str:
        """توليد الإجابة"""
        return f"Based on {self.expertise.domain.value} expertise"
    
    def _calculate_confidence(self, knowledge: List[Dict]) -> float:
        """حساب الثقة"""
        base = self.expertise.level / 10
        if knowledge:
            base *= 0.9
        return min(base, 1.0)


class DomainExpertTeam:
    """
    فريق خبراء المجالات (8-16 خبير)
    """
    
    def __init__(self):
        self.experts: Dict[DomainType, List[DomainExpert]] = {}
        self._initialize_experts()
        print(f"🏛️ Domain Expert Team initialized: {len(self.experts)} domains")
    
    def _initialize_experts(self):
        """تهيئة الخبراء"""
        
        # ERP Experts
        self.experts[DomainType.ACCOUNTING] = [
            DomainExpert(
                id="ACC001",
                name="خبير المحاسبة العامة",
                expertise=Expertise(
                    domain=DomainType.ACCOUNTING,
                    level=9,
                    years_experience=15,
                    certifications=["CPA", "CMA"],
                    specializations=["financial_accounting", "cost_accounting", "tax"]
                ),
                rules=[
                    "الأصول = الخصوم + حقوق الملكية",
                    "كل معاملة تؤثر على جانبين",
                    "القيد المزدوج إلزامي"
                ]
            ),
            DomainExpert(
                id="ACC002",
                name="خبير الضرائب",
                expertise=Expertise(
                    domain=DomainType.ACCOUNTING,
                    level=8,
                    years_experience=10,
                    specializations=["tax_planning", "vat", "corporate_tax"]
                )
            )
        ]
        
        self.experts[DomainType.INVENTORY] = [
            DomainExpert(
                id="INV001",
                name="خبير المخازن",
                expertise=Expertise(
                    domain=DomainType.INVENTORY,
                    level=9,
                    years_experience=12,
                    specializations=["warehouse_management", "fifo_lifo", "reorder_points"]
                ),
                rules=[
                    "FIFO للمنتجات قصيرة الأجل",
                    "LIFO للمنتجات طويلة الأجل",
                    "نقطة إعادة الطلب = (استهلاك يومي × فترة التوريد) + مخزون أمان"
                ]
            )
        ]
        
        self.experts[DomainType.HR] = [
            DomainExpert(
                id="HR001",
                name="خبير الموارد البشرية",
                expertise=Expertise(
                    domain=DomainType.HR,
                    level=8,
                    years_experience=10,
                    specializations=["payroll", "attendance", "performance"]
                )
            )
        ]
        
        self.experts[DomainType.SALES] = [
            DomainExpert(
                id="SAL001",
                name="خبير المبيعات",
                expertise=Expertise(
                    domain=DomainType.SALES,
                    level=9,
                    years_experience=14,
                    specializations=["b2b_sales", "pricing", "customer_relations"]
                )
            )
        ]
        
        self.experts[DomainType.CRM] = [
            DomainExpert(
                id="CRM001",
                name="خبير العملاء",
                expertise=Expertise(
                    domain=DomainType.CRM,
                    level=8,
                    years_experience=11,
                    specializations=["customer_retention", "loyalty", "support"]
                )
            )
        ]
        
        # Technical Experts
        self.experts[DomainType.RUST] = [
            DomainExpert(
                id="RUST001",
                name="خبير Rust",
                expertise=Expertise(
                    domain=DomainType.RUST,
                    level=9,
                    years_experience=8,
                    certifications=["Rust Certified"],
                    specializations=["systems_programming", "async", "webassembly"]
                ),
                rules=[
                    "Ownership is key",
                    "Borrow checker is your friend",
                    "Zero-cost abstractions"
                ]
            )
        ]
        
        self.experts[DomainType.PYTHON] = [
            DomainExpert(
                id="PY001",
                name="خبير Python",
                expertise=Expertise(
                    domain=DomainType.PYTHON,
                    level=10,
                    years_experience=12,
                    specializations=["data_science", "ai_ml", "fastapi"]
                )
            )
        ]
        
        self.experts[DomainType.TYPESCRIPT] = [
            DomainExpert(
                id="TS001",
                name="خبير TypeScript",
                expertise=Expertise(
                    domain=DomainType.TYPESCRIPT,
                    level=9,
                    years_experience=9,
                    specializations=["react", "node", "type_system"]
                )
            )
        ]
        
        self.experts[DomainType.DATABASE] = [
            DomainExpert(
                id="DB001",
                name="خبير PostgreSQL",
                expertise=Expertise(
                    domain=DomainType.DATABASE,
                    level=9,
                    years_experience=13,
                    specializations=[["postgresql", "optimization", "partitioning"]]
                )
            )
        ]
        
        self.experts[DomainType.AI_ML] = [
            DomainExpert(
                id="AI001",
                name="خبير الذكاء الاصطناعي",
                expertise=Expertise(
                    domain=DomainType.AI_ML,
                    level=10,
                    years_experience=10,
                    certifications=["TensorFlow", "PyTorch"],
                    specializations=["llm", "training", "optimization"]
                )
            )
        ]
        
        self.experts[DomainType.SECURITY] = [
            DomainExpert(
                id="SEC001",
                name="خبير الأمان",
                expertise=Expertise(
                    domain=DomainType.SECURITY,
                    level=9,
                    years_experience=11,
                    certifications=["CISSP", "CEH"],
                    specializations=["cryptography", "penetration_testing", "compliance"]
                )
            )
        ]
    
    async def route_query(self, query: str, context: Dict = None) -> Dict:
        """
        توجيه الاستعلام للخبير المناسب
        
        Args:
            query: السؤال
            context: سياق إضافي
            
        Returns:
            إجابة من أفضل خبير
        """
        # تحديد المجال
        domain = self._detect_domain(query)
        
        if domain not in self.experts:
            return {
                'error': 'No expert available for this domain',
                'detected_domain': domain
            }
        
        # اختيار أفضل خبير
        experts = self.experts[domain]
        best_expert = max(experts, key=lambda e: e.expertise.level * e.success_rate)
        
        # الإجابة
        result = await best_expert.answer_query(query, context)
        
        return {
            'expert': best_expert.name,
            'domain': domain.value,
            'result': result
        }
    
    def _detect_domain(self, query: str) -> DomainType:
        """كشف المجال من الاستعلام"""
        query_lower = query.lower()
        
        keywords = {
            DomainType.ACCOUNTING: ['فاتورة', 'قيد', 'حساب', ' debit', 'credit', 'balance'],
            DomainType.INVENTORY: ['مخزن', 'جرد', 'تالف', 'stock', 'inventory', 'warehouse'],
            DomainType.HR: ['موظف', 'راتب', 'حضور', 'payroll', 'attendance', 'employee'],
            DomainType.SALES: ['بيع', 'زبون', 'عمولة', 'sale', 'customer', 'invoice'],
            DomainType.CRM: ['شكوى', 'ولاء', 'نقاط', 'loyalty', 'complaint', 'support'],
            DomainType.RUST: ['rust', 'cargo', 'ownership', 'borrow'],
            DomainType.PYTHON: ['python', 'pandas', 'numpy', 'fastapi'],
            DomainType.TYPESCRIPT: ['typescript', 'react', 'node'],
            DomainType.DATABASE: ['database', 'sql', 'postgres', 'query'],
            DomainType.AI_ML: ['ai', 'ml', 'training', 'model', 'neural'],
            DomainType.SECURITY: ['security', 'encryption', 'auth', 'vulnerability']
        }
        
        scores = {}
        for domain, words in keywords.items():
            score = sum(1 for word in words if word in query_lower)
            if score > 0:
                scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return DomainType.ACCOUNTING  # default
    
    def get_all_experts(self) -> List[DomainExpert]:
        """جمع كل الخبراء"""
        all_experts = []
        for experts in self.experts.values():
            all_experts.extend(experts)
        return all_experts
    
    def get_stats(self) -> Dict:
        """إحصائيات الفريق"""
        total_experts = sum(len(experts) for experts in self.experts.values())
        total_domains = len(self.experts)
        total_queries = sum(
            e.queries_answered 
            for experts in self.experts.values() 
            for e in experts
        )
        
        return {
            'total_experts': total_experts,
            'total_domains': total_domains,
            'total_queries_answered': total_queries,
            'domains': [d.value for d in self.experts.keys()]
        }


# Singleton
domain_team = DomainExpertTeam()
