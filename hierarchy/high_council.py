"""
الطبقة السادسة: المجلس الدائم للحكماء (16 حكيم)
The Eternal Council - 24/7 Continuous Meeting

Fixed:
- Duplicate ID S002 corrected
- get_status() uses safe attribute access
- wise_men_count is dynamic
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict
import asyncio
from collections import deque


class SageRole(Enum):
    """أدوار الحكماء الثمانية - المجلس العالي"""
    IDENTITY = "identity"           # حكيم الهوية
    STRATEGY = "strategy"           # حكيم الاستراتيجيا
    ETHICS = "ethics"               # حكيم الأخلاق
    BALANCE = "balance"             # حكيم التوازن
    KNOWLEDGE = "knowledge"         # حكيم المعرفة
    RELATIONS = "relations"         # حكيم العلاقات
    INNOVATION = "innovation"       # حكيم الابتكار
    PROTECTION = "protection"       # حكيم الحماية


class OperationsRole(Enum):
    """أدوار مجلس العمليات الثمانية"""
    SYSTEM = "system"               # حكيم النظام الشامل
    EXECUTION = "execution"         # حكيم التنفيذ السريع
    BRIDGE = "bridge"               # حكيم الربط بين الطبقات
    REPORTS = "reports"             # حكيم التقارير الفورية
    COORDINATION = "coordination"   # حكيم التنسيق
    MONITORING = "monitoring"       # حكيم المتابعة
    VERIFICATION = "verification"   # حكيم التدقيق
    EMERGENCY = "emergency"         # حكيم الطوارئ


@dataclass
class Sage:
    """حكيم من المجلس"""
    id: str
    name: str
    role: SageRole
    current_focus: str = ""
    is_active: bool = True
    expertise: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.expertise:
            self.expertise = [self.role.value, "wisdom", "counsel"]


@dataclass
class OperationsSage:
    """حكيم عمليات"""
    id: str
    name: str
    role: OperationsRole
    assigned_tasks: List = field(default_factory=list)
    is_active: bool = True


@dataclass
class Discussion:
    """نقاش داخل المجلس"""
    topic: str
    initiator: str
    opinions: Dict[str, str] = field(default_factory=dict)
    votes: Dict[str, float] = field(default_factory=dict)
    consensus: Optional[str] = None
    consensus_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HighCouncil:
    """
    المجلس العالي (8 حكماء)
    
    يجتمعون 24/7، يتناقشون، ويصدرون القرارات
    """
    
    def __init__(self):
        self.sages: Dict[SageRole, Sage] = self._initialize_sages()
        self.discussion_history: deque = deque(maxlen=1000)
        self.current_discussion: Optional[Discussion] = None
        self.meeting_active: bool = True
        self.president_present: bool = False
        self._deliberation_count: int = 0
        
    def _initialize_sages(self) -> Dict[SageRole, Sage]:
        """تهيئة الـ 8 حكماء - FIXED: إصلاح ID المكرر"""
        return {
            SageRole.IDENTITY: Sage(
                "S001", "حكيم الهوية", SageRole.IDENTITY,
                expertise=["identity", "culture", "values", "purpose"]
            ),
            SageRole.STRATEGY: Sage(
                "S002", "حكيم الاستراتيجيا", SageRole.STRATEGY,
                expertise=["strategy", "planning", "vision", "long-term"]
            ),
            # FIXED: Changed from S002 to S003
            SageRole.ETHICS: Sage(
                "S003", "حكيم الأخلاق", SageRole.ETHICS,
                expertise=["ethics", "morality", "fairness", "justice"]
            ),
            SageRole.BALANCE: Sage(
                "S004", "حكيم التوازن", SageRole.BALANCE,
                expertise=["balance", "harmony", "equilibrium", "moderation"]
            ),
            SageRole.KNOWLEDGE: Sage(
                "S005", "حكيم المعرفة", SageRole.KNOWLEDGE,
                expertise=["knowledge", "wisdom", "learning", "research"]
            ),
            SageRole.RELATIONS: Sage(
                "S006", "حكيم العلاقات", SageRole.RELATIONS,
                expertise=["relations", "diplomacy", "communication", "collaboration"]
            ),
            SageRole.INNOVATION: Sage(
                "S007", "حكيم الابتكار", SageRole.INNOVATION,
                expertise=["innovation", "creativity", "invention", "progress"]
            ),
            SageRole.PROTECTION: Sage(
                "S008", "حكيم الحماية", SageRole.PROTECTION,
                expertise=["protection", "security", "defense", "safety"]
            ),
        }
    
    async def start_eternal_meeting(self):
        """بدء الاجتماع الدائم (24/7)"""
        print("🏛️ المجلس العالي يبدأ اجتماعه الدائم...")
        self.meeting_active = True
        
        while self.meeting_active:
            await self._monitor_system()
            await self._discuss_pending_issues()
            await self._report_to_president()
            await asyncio.sleep(60)  # كل دقيقة
    
    async def _monitor_system(self):
        """مراقبة حالة النظام"""
        inactive_count = 0
        for sage in self.sages.values():
            if not sage.is_active:
                inactive_count += 1
                print(f"⚠️ {sage.name} غير نشط!")
        
        if inactive_count > 0:
            print(f"⚠️ {inactive_count} حكماء غير نشطين")
    
    async def _discuss_pending_issues(self):
        """مناقشة القضايا العالقة"""
        issues = await self._fetch_issues()
        
        for issue in issues:
            await self._conduct_deliberation(issue)
    
    async def _conduct_deliberation(self, topic: str) -> Discussion:
        """إجراء مناقشة مع حساب توافق حقيقي"""
        discussion = Discussion(topic=topic, initiator="المجلس")
        self.current_discussion = discussion
        
        # جمع آراء الحكماء مع أوزان
        weighted_opinions = []
        for sage in self.sages.values():
            if not sage.is_active:
                continue
            
            opinion = await self._get_sage_opinion(sage, topic)
            weight = self._calculate_sage_weight(sage, topic)
            
            discussion.opinions[sage.id] = opinion
            discussion.votes[sage.id] = weight
            weighted_opinions.append((weight, opinion))
        
        # حساب التوافق بدلاً من القيمة الثابتة
        consensus_result = self._calculate_consensus(weighted_opinions)
        discussion.consensus_score = consensus_result["score"]
        
        if consensus_result["has_consensus"]:
            discussion.consensus = consensus_result["decision"]
            print(f"✅ توافق على: {topic} (النسبة: {consensus_result['score']:.2f})")
            await self._dispatch_to_operations(topic, consensus_result["decision"])
        else:
            print(f"⚠️ لا توافق على: {topic} (النسبة: {consensus_result['score']:.2f}) - يرفع للرئيس")
            await self._escalate_to_president(topic, discussion)
        
        self.discussion_history.append(discussion)
        self._deliberation_count += 1
        
        return discussion
    
    def _calculate_consensus(self, weighted_opinions: List[tuple]) -> Dict:
        """
        حساب التوافق بناءً على آراء مُوزنة
        
        Returns:
            dict with score, has_consensus, decision
        """
        if not weighted_opinions:
            return {"score": 0.0, "has_consensus": False, "decision": None}
        
        total_weight = sum(w for w, _ in weighted_opinions)
        if total_weight == 0:
            return {"score": 0.0, "has_consensus": False, "decision": None}
        
        # حساب النسبة الموزونة للموافقين
        positive_weight = sum(
            w for w, op in weighted_opinions 
            if any(kw in op.lower() for kw in ["صح", "مقبول", "موافق", "good", "agree", "support"])
        )
        
        consensus_ratio = positive_weight / total_weight
        
        # يعتبر توافقاً إذا كانت النسبة >= 0.75 (6/8)
        has_consensus = consensus_ratio >= 0.75
        
        return {
            "score": round(consensus_ratio, 2),
            "has_consensus": has_consensus,
            "decision": "موافقة المجلس بالإجماع" if has_consensus else None
        }
    
    def _calculate_sage_weight(self, sage: Sage, topic: str) -> float:
        """حساب وزن الحكيم بناءً على خبرته في الموضوع"""
        topic_lower = topic.lower()
        weight = 1.0  # الوزن الأساسي
        
        # زيادة الوزن إذا كانت خبرته متناسبة
        for expertise in sage.expertise:
            if expertise.lower() in topic_lower:
                weight += 0.5
        
        return min(weight, 3.0)  # الحد الأقصى للوزن
    
    async def _get_sage_opinion(self, sage: Sage, topic: str) -> str:
        """جلب رأي حكيم مع تحليل ذكي للموضوع"""
        topic_lower = topic.lower()
        
        # آراء مخصصة حسب الدور والموضوع
        opinions = {
            SageRole.IDENTITY: self._analyze_identity(topic_lower),
            SageRole.STRATEGY: self._analyze_strategy(topic_lower),
            SageRole.ETHICS: self._analyze_ethics(topic_lower),
            SageRole.BALANCE: self._analyze_balance(topic_lower),
            SageRole.KNOWLEDGE: self._analyze_knowledge(topic_lower),
            SageRole.RELATIONS: self._analyze_relations(topic_lower),
            SageRole.INNOVATION: self._analyze_innovation(topic_lower),
            SageRole.PROTECTION: self._analyze_protection(topic_lower),
        }
        
        return opinions.get(sage.role, "محايد")
    
    def _analyze_identity(self, topic: str) -> str:
        if any(kw in topic for kw in ["هوية", "identity", "brand", "value"]):
            return "يتوافق مع هويتنا وقيمنا الأساسية"
        return "محايد من حيث الهوية"
    
    def _analyze_strategy(self, topic: str) -> str:
        if any(kw in topic for kw in ["plan", "strategy", "خطة", "استراتيجية", "future"]):
            return "استراتيجياً صحيح ومتناسب مع الأهداف طويلة المدى"
        return "يحتاج تقييم استراتيجي أعمق"
    
    def _analyze_ethics(self, topic: str) -> str:
        if any(kw in topic for kw in ["ethical", "أخلاق", "privacy", "خاصية"]):
            return "أخلاقياً مقبول مع مراعاة الخصوصية والشفافية"
        return "لا يوجد تعارض أخلاقي واضح"
    
    def _analyze_balance(self, topic: str) -> str:
        if any(kw in topic for kw in ["risk", "balance", "مخاطر", "توازن"]):
            return "متوازن بين المخاطر والفرص"
        return "متوازن نسبياً"
    
    def _analyze_knowledge(self, topic: str) -> str:
        if any(kw in topic for kw in ["knowledge", "تعلم", "learning", "research", "بحث"]):
            return "يعزز المعرفة ويسهم في التعلم المستمر"
        return "يستفيد من المعرفة المتاحة"
    
    def _analyze_relations(self, topic: str) -> str:
        if any(kw in topic for kw in ["relation", "علاقة", "collaboration", "تعاون"]):
            return "يعزز العلاقات ويحسن التعاون"
        return "تأثير محايد على العلاقات"
    
    def _analyze_innovation(self, topic: str) -> str:
        if any(kw in topic for kw in ["innovation", "ابتكار", "creative", "جديد"]):
            return "مبتكر ويفتح آفاقاً جديدة"
        return "يمكن تحسينه من حيث الابتكار"
    
    def _analyze_protection(self, topic: str) -> str:
        if any(kw in topic for kw in ["security", "أمان", "protection", "safety", "حماية"]):
            return "يحتاج مراجعة أمنية دقيقة"
        return "لا يشكل خطراً أمنياً واضحاً"
    
    async def continuous_deliberation(self):
        """منفذ للحلقة المستمرة للمجلس"""
        while self.meeting_active:
            await self._discuss_pending_issues()
            await asyncio.sleep(60)
    
    async def _dispatch_to_operations(self, topic: str, decision: str):
        """إرسال للعمليات للتنفيذ"""
        pass
    
    async def _escalate_to_president(self, topic: str, discussion: Discussion):
        """رفع للرئيس في حال عدم التوافق"""
        pass
    
    async def _report_to_president(self):
        """رفع تقرير دوري للرئيس"""
        pass
    
    async def _fetch_issues(self) -> List[str]:
        """جلب القضايا من الطبقات السفلى"""
        return ["موضوع 1", "موضوع 2"]
    
    def president_entered(self):
        """الرئيس دخل المجلس"""
        self.president_present = True
        print("👑 الرئيس في المجلس - الانتباه!")
    
    def president_exited(self):
        """الرئيس غادر المجلس"""
        self.president_present = False
        print("👑 الرئيس غادر - نستمر بالعمل")
    
    def get_status(self) -> dict:
        """
        الحصول على حالة المجلس - FIXED: استخدام سمات آمنة
        """
        active_sages = sum(1 for s in self.sages.values() if s.is_active)
        total_sages = len(self.sages)
        
        return {
            'is_meeting': self.meeting_active,
            'wise_men_count': total_sages,  # FIXED: العدد الفعلي
            'active_sages': active_sages,
            'meeting_status': 'continuous' if self.meeting_active else 'paused',
            'president_present': self.president_present,
            'topics_discussed': len(self.discussion_history),
            'current_topic': self.current_discussion.topic if self.current_discussion else None,
            'consensus_rate': self._calculate_average_consensus(),
        }
    
    def _calculate_average_consensus(self) -> float:
        """حساب معدل التوافق المتوسط"""
        if not self.discussion_history:
            return 0.0
        
        total_score = sum(d.consensus_score for d in self.discussion_history)
        return round(total_score / len(self.discussion_history), 2)
    
    def get_all_sages(self) -> List[Dict]:
        """Get all sages as dictionaries for API serialization"""
        return [
            {
                "id": sage.id,
                "name": sage.name,
                "role": sage.role.value,
                "is_active": sage.is_active,
                "current_focus": sage.current_focus,
                "expertise": sage.expertise,
            }
            for sage in self.sages.values()
        ]


class OperationsCouncil:
    """
    مجلس العمليات (8 حكماء)
    
    ينفذون قرارات المجلس العالي
    """
    
    def __init__(self, high_council: HighCouncil):
        self.high_council = high_council
        self.sages: Dict[OperationsRole, OperationsSage] = self._initialize_sages()
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
    def _initialize_sages(self) -> Dict[OperationsRole, OperationsSage]:
        """تهيئة الـ 8 حكماء عمليات"""
        return {
            OperationsRole.SYSTEM: OperationsSage("O001", "حكيم النظام", OperationsRole.SYSTEM),
            OperationsRole.EXECUTION: OperationsSage("O002", "حكيم التنفيذ", OperationsRole.EXECUTION),
            OperationsRole.BRIDGE: OperationsSage("O003", "حكيم الربط", OperationsRole.BRIDGE),
            OperationsRole.REPORTS: OperationsSage("O004", "حكيم التقارير", OperationsRole.REPORTS),
            OperationsRole.COORDINATION: OperationsSage("O005", "حكيم التنسيق", OperationsRole.COORDINATION),
            OperationsRole.MONITORING: OperationsSage("O006", "حكيم المتابعة", OperationsRole.MONITORING),
            OperationsRole.VERIFICATION: OperationsSage("O007", "حكيم التدقيق", OperationsRole.VERIFICATION),
            OperationsRole.EMERGENCY: OperationsSage("O008", "حكيم الطوارئ", OperationsRole.EMERGENCY),
        }
    
    async def start_execution_loop(self):
        """حلقة التنفيذ المستمرة"""
        print("⚙️ مجلس العمليات يبدأ التنفيذ...")
        
        while True:
            task = await self.execution_queue.get()
            await self._distribute_task(task)
            await self._execute_task(task)
            await self._report_completion(task)
    
    async def receive_decision(self, topic: str, decision: str):
        """استلام قرار من المجلس العالي"""
        task = {
            "topic": topic,
            "decision": decision,
            "timestamp": datetime.now(timezone.utc)
        }
        await self.execution_queue.put(task)
        print(f"⚙️ استلمنا مهمة: {topic}")
    
    async def _distribute_task(self, task: dict):
        """توزيع المهمة على الحكماء"""
        pass
    
    async def _execute_task(self, task: dict):
        """تنفيذ المهمة"""
        print(f"⚙️ ننفذ: {task['topic']}")
        await asyncio.sleep(1)
    
    async def _report_completion(self, task: dict):
        """رفع تقرير الإنجاز"""
        print(f"✅ اكتمل: {task['topic']}")


# Singleton instances
high_council = HighCouncil()
operations_council = OperationsCouncil(high_council)
