"""
النظام الهرمي المتكامل - Integrated Hierarchical AI System

🏛️ الهيكل التنظيمي الكامل:

    الرئيس (المستخدم)
         ↓
    البعد السابع (4 مخططون - 100 سنة)
         ↓
    مجلس الحكماء (16 حكيم - 24/7)
         ↓
    فرق الظل والنور (8 متوازنون)
         ↓
    الكشافة (4 كشافة)
         ↓
    الفريق الميتا (16 مدير)
         ↓
    خبراء المجالات (12 خبير)
         ↓
    فرق التنفيذ (مؤقتة)

الاستخدام:
    from src.core.hierarchy import ai_hierarchy
    
    # دخول المجلس
    status = ai_hierarchy.enter_council()
    
    # إصدار أمر
    result = await ai_hierarchy.execute_command("analyze market")
"""

from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timezone

# استيراد الطبقات الأساسية
from .president import PresidentInterface, AlertLevel, PresidentialCommand, CommandType
from .seventh_dimension import SeventhDimension, seventh_dimension
from .high_council import HighCouncil, high_council
from .shadow_light import BalanceCouncil, balance_council
from .scouts import ScoutManager, scout_manager
from .meta_team import MetaTeam, meta_team
from .domain_experts import DomainExpertTeam, domain_team
from .execution_team import ExecutionManager, execution_manager, TaskPriority

# استيراد الطبقات الفوقية الجديدة
from .meta_architect import (
    MetaArchitectLayer, 
    get_meta_architect_layer,
    BuilderCouncil,
    ExecutiveController,
    DynamicLayerGenerator
)


class AIHierarchy:
    """
    🏛️ النظام الهرمي المتكامل
    
    يدير التفاعل بين كل الطبقات
    """
    
    def __init__(self):
        # الطبقات الأساسية (الـ 7)
        self.president = PresidentInterface()
        self.seventh = seventh_dimension
        self.council = high_council
        self.balance = balance_council
        self.scouts = scout_manager
        self.meta = meta_team
        self.experts = domain_team
        self.execution = execution_manager
        
        # الطبقات الفوقية الجديدة (3 طبقات)
        self.meta_architect = None  # يتم تهيئته لاحقاً
        self.builder_council = None
        self.executive_controller = None
        
        # الحالة
        self.is_initialized = False
        self.active_mode = "normal"  # normal, crisis, innovation, construction
        
        print("""
🏛️ ╔══════════════════════════════════════════════════════╗
    ║   AI HIERARCHY SYSTEM - النظام الهرمي المتكامل     ║
    ╠══════════════════════════════════════════════════════╣
    ║  Core Layers: 7                                      ║
    ║  Meta Layers: 3 (Builder, Architect, Controller)     ║
    ║  Total Entities: 100+                                ║
    ║  President: User (24/7 Access)                       ║
    ║  High Council: 16 Wise Men (Always Meeting)          ║
    ║  Executive Controller: Awaiting Orders               ║
    ║  Status: Ready                                       ║
    ╚══════════════════════════════════════════════════════╝
        """)
    
    async def initialize(self):
        """تهيئة النظام"""
        if self.is_initialized:
            return {'status': 'already_initialized'}
        
        print("🚀 Initializing AI Hierarchy...")
        
        # 1. تحميل الخطة طويلة المدى
        century_plan = await self.seventh.develop_century_plan()
        print(f"   ✓ Century Plan loaded: {century_plan['milestones'][0]['goal']}")
        
        # 2. تفعيل المجلس
        council_status = self.council.get_status()
        print(f"   ✓ High Council: {council_status['meeting_status']}")
        
        # 3. تفعيل الكشافة
        intel = await self.scouts.gather_all_intel()
        print(f"   ✓ Scouts: {intel['total_reports']} reports ready")
        
        # 4. تهيئة الخبراء
        print(f"   ✓ Domain Experts: {len(self.experts.experts)} experts ready")
        
        # 5. تهيئة الطبقات الفوقية (الجديدة)
        print("\n🏗️ Initializing Meta Layers...")
        self.meta_architect = get_meta_architect_layer(self.council)
        self.builder_council = self.meta_architect.builder_council
        self.executive_controller = self.meta_architect.executive_controller
        print(f"   ✓ Meta Architect: Active")
        print(f"   ✓ Builder Council: {sum(len(team) for team in self.builder_council.teams.values())} specialists")
        print(f"   ✓ Executive Controller: {self.executive_controller.title}")
        
        self.is_initialized = True
        print("\n✅ AI Hierarchy Fully Initialized (10 Layers Total)")
        
        return {
            'status': 'initialized',
            'layers_active': 10,  # 7 core + 3 meta
            'entities_ready': 100  # 80 core + 20 meta
        }
    
    def enter_council(self) -> Dict:
        """
        دخول المجلس (24/7)
        
        يدخل المستخدم للمجلس للإشراف المباشر
        """
        return self.president.enter_council()
    
    def get_council_status(self) -> Dict:
        """الحصول على حالة المجلس"""
        return self.council.get_status()
    
    async def execute_command(self, command: str, 
                              alert_level: AlertLevel = AlertLevel.GREEN,
                              context: Optional[Dict] = None) -> Dict:
        """
        تنفيذ أمر من الرئيس
        
        المسار:
        1. الرئيس يصدر الأمر
        2. المجلس يناقش (إذا لزم)
        3. الخبراء يحللون
        4. التنفيذ
        """
        print(f"\n📜 Command: '{command}' | Level: {alert_level.name}")
        
        # 1. إصدار الأمر
        cmd_type = CommandType.EXECUTE if alert_level in [AlertLevel.RED, AlertLevel.BLACK] else CommandType.WAIT
        cmd_obj = PresidentialCommand(
            command_type=cmd_type,
            target_layer=0,  # All layers
            description=command,
            timestamp=datetime.now(timezone.utc),
            requires_confirmation=(alert_level == AlertLevel.BLACK)
        )
        order = await self.president.issue_command(cmd_obj)
        
        # 2. تنفيذ فوري للحرج
        if alert_level in [AlertLevel.RED, AlertLevel.BLACK]:
            print("   ⚡ IMMEDIATE EXECUTION")
            immediate_result = await self._immediate_execute(command, context)
            return {
                'command': command,
                'decision': {'execute': True, 'reasoning': 'Immediate execution (critical alert)'},
                'result': immediate_result,
            }
        
        # 3. استشارة المجلس
        print("   🏛️ Consulting High Council...")
        
        # ⚠️ WARNING: MOCK DATA - NOT REAL AI CONSENSUS
        # TODO: Implement real consensus algorithm with HighCouncil
        # This is placeholder data for demonstration purposes only
        # The consensus score (0.75) is hardcoded and not based on actual AI evaluation
        consensus = {
            '_warning': 'MOCK DATA - DO NOT USE FOR REAL DECISIONS',
            '_note': 'This is placeholder data. Real AI consensus not implemented.',
            'consensus': 0.75,  # ⬅️ HARDCODED VALUE - NOT REAL
            'rounds': 3,
            'decision': f'Proceed with: {command}',
            'confidence': 0.8,  # ⬅️ PLACEHOLDER
            'timestamp': '2026-02-24',
            'status': 'mock_implementation'
        }
        
        # 4. توازن الظل والنور
        print("   ⚖️ Shadow/Light evaluation...")
        balance = await self.balance.evaluate_proposal({
            'name': command,
            'type': 'execution'
        })
        
        # 5. جلب معلومات من الكشافة
        print("   🕵️ Gathering intelligence...")
        intel = await self.scouts.gather_all_intel()
        
        # 6. تحليل الخبراء
        print("   👥 Consulting domain experts...")
        expert_opinion = await self.experts.route_query(command, context or {})
        
        # 7. قرار نهائي
        decision = self._make_final_decision(
            consensus, balance, expert_opinion, alert_level
        )
        
        # 8. التنفيذ
        if decision['execute']:
            print(f"   ✅ EXECUTING: {decision['reasoning']}")
            result = await self._execute_with_team(command, decision)
        else:
            print(f"   ❌ REJECTED: {decision['reasoning']}")
            result = {'status': 'rejected', 'reason': decision['reasoning']}
        
        return {
            'command': command,
            'decision': decision,
            'result': result,
            'council_consensus': consensus.get('consensus'),
            'balance_score': balance.get('balance_score'),
            'expert_recommendation': expert_opinion.get('recommendation')
        }
    
    def _make_final_decision(self, consensus: Dict, balance: Dict,
                            expert: Dict, alert_level: AlertLevel) -> Dict:
        """اتخاذ القرار النهائي"""
        # عوامل القرار
        council_agreement = consensus.get('consensus', 0.5)
        balance_score = balance.get('balance_score', 0)
        expert_confidence = expert.get('confidence', 0.5)
        
        # وزن القرار
        weights = {
            'council': 0.4,
            'balance': 0.3,
            'expert': 0.3
        }
        
        # درجة التنفيذ
        execution_score = (
            council_agreement * weights['council'] +
            (balance_score + 1) / 2 * weights['balance'] +  # normalize -1,1 to 0,1
            expert_confidence * weights['expert']
        )
        
        # حدود التنفيذ
        threshold = 0.6
        if alert_level == AlertLevel.ORANGE:
            threshold = 0.4
        elif alert_level == AlertLevel.YELLOW:
            threshold = 0.5
        
        execute = execution_score >= threshold
        
        return {
            'execute': execute,
            'confidence': execution_score,
            'threshold': threshold,
            'reasoning': f"Score: {execution_score:.2f} vs {threshold} threshold",
            'factors': {
                'council': council_agreement,
                'balance': balance_score,
                'expert': expert_confidence
            }
        }
    
    async def _immediate_execute(self, command: str, context: Optional[Dict]) -> Dict:
        """تنفيذ فوري (للأوامر الحرجة)"""
        # إنشاء قوة مهمة
        force = await self.execution.create_task_force(
            f"URGENT: {command}",
            ['crisis_responder_1', 'crisis_responder_2']
        )
        
        await force.assign_task(command, 'crisis_responder_1', 
                               priority=TaskPriority.CRITICAL,
                               deadline_hours=1)
        
        report = await force.execute_mission()
        
        return {
            'status': 'executed_immediately',
            'mission_report': report
        }
    
    async def _execute_with_team(self, command: str, decision: Dict) -> Dict:
        """التنفيذ مع فريق"""
        # إنشاء قوة مهمة مناسبة
        force = await self.execution.create_task_force(
            command,
            ['executor_1', 'executor_2', 'qa_checker']
        )
        
        await force.assign_task(command, 'executor_1')
        
        report = await force.execute_mission()
        
        return {
            'status': 'executed',
            'mission_report': report
        }
    
    def veto_destruction(self, decision_id: str) -> Dict:
        """
        الفيتو على قرار التدمير الذاتي
        
        يستخدم فقط للقرارات الحرجة جداً
        """
        return self.president.veto_destruction(decision_id)
    
    async def start_continuous_operations(self):
        """بدء العمليات المستمرة"""
        print("\n🔄 Starting continuous operations...")
        
        # تشغيل كل الطبقات بالتوازي
        await asyncio.gather(
            # المجلس الدائم
            self._council_meeting_loop(),
            
            # الكشافة
            self.scouts.continuous_intelligence(self.council),
            
            # الميتا
            self.meta.continuous_self_improvement(),
            
            # البعد السابع
            self._seventh_dimension_loop()
        )
    
    async def _council_meeting_loop(self):
        """حلقة اجتماع المجلس"""
        while True:
            # المجلس يجتمون باستمرار
            await self.council.continuous_deliberation()
            await asyncio.sleep(60)  # كل دقيقة
    
    async def _seventh_dimension_loop(self):
        """حلقة البعد السابع"""
        while True:
            # مراجعة استراتيجية
            review = await self.seventh.annual_strategic_review()
            print(f"🔮 Strategic Review: {review['year']}")
            await asyncio.sleep(86400)  # كل يوم (محاكاة للسنة)
    
    def get_full_status(self) -> Dict:
        """الحالة الكاملة للنظام"""
        return {
            'president': {
                'in_meeting': self.president.is_present,
                'veto_power': self.president.veto_power_active
            },
            'council': self.council.get_status(),
            'scouts': {
                'intel_buffer_size': len(self.scouts.intel_buffer),
                'high_priority_queue': len(self.scouts.high_priority_queue)
            },
            'meta': self.meta.get_system_health(),
            'experts': {
                'total': len(self.experts.experts),
                'domains': [d.value for d in self.experts.experts.keys()]
            },
            'execution': self.execution.get_execution_stats()
        }
    
    def get_wisdom(self) -> str:
        """حكمة من النظام"""
        return self.seventh.get_wisdom_for_today()

    # ==================== Smart Council compatibility ====================

    def get_all_wise_men(self):
        """Compatibility API expected by `api/routes/council.py`."""
        wise_men = []
        try:
            sages = getattr(self.council, "sages", {})
            for role, sage in sages.items():
                wise_men.append(
                    {
                        "id": getattr(sage, "id", None),
                        "name": getattr(sage, "name", str(role)),
                        "role": getattr(getattr(sage, "role", None), "value", str(role)),
                        "is_active": getattr(sage, "is_active", True),
                        "current_focus": getattr(sage, "current_focus", ""),
                    }
                )
        except Exception:
            pass
        return wise_men

    def ask(self, message: str) -> Dict[str, Any]:
        """Synchronous ask() used by council endpoints.

        This is a lightweight compatibility layer; the full async pipeline is
        available via `execute_command()`.
        """
        first_sage = None
        try:
            sages = list(getattr(self.council, "sages", {}).values())
            if sages:
                first_sage = sages[0]
        except Exception:
            first_sage = None

        wise_man_name = getattr(first_sage, "name", "حكيم القرار") if first_sage else "حكيم القرار"
        response = f"تم استلام رسالتك: {message}"
        return {
            "response": response,
            "wise_man": wise_man_name,
            "confidence": 0.4,
            "evidence": [],
            "response_source": "hierarchy-local",
        }

    def discuss(self, topic: str):
        """Synchronous discuss() used by council endpoints."""
        discussion = []
        for item in self.get_all_wise_men():
            discussion.append(
                {
                    "wise_man": item.get("name"),
                    "role": item.get("role"),
                    "opinion": f"رأي مبدئي حول: {topic}",
                }
            )
        return discussion
    
    # ==================== الطبقات الفوقية - Meta Layers ====================
    
    async def send_presidential_order(self, order: str, params: dict = None) -> dict:
        """
        إرسال أمر رئاسي مباشر للحكيم التنفيذي
        
        الأمور المتاحة:
        - build_layer: بناء طبقة جديدة
        - destroy_layer: تدمير طبقة
        - connect: ربط طبقتين
        - disconnect: فك ربط
        - rebuild: إعادة بناء الهيكل
        - emergency: تجاوز طارئ
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.executive_controller:
            return {"error": "Meta layers not initialized"}
        
        return await self.executive_controller.receive_presidential_order(order, params or {})
    
    async def create_new_layer(self, name: str, layer_type: str = "EXECUTIVE", 
                               components: list = None, connections: list = None) -> dict:
        """بناء طبقة جديدة"""
        return await self.send_presidential_order("build_layer", {
            "name": name,
            "type": layer_type,
            "components": components or [],
            "connections": connections or []
        })
    
    async def destroy_layer(self, layer_id: str, force: bool = False) -> dict:
        """تدمير طبقة"""
        return await self.send_presidential_order("destroy_layer", {
            "layer_id": layer_id,
            "force": force
        })
    
    async def create_new_hierarchy(self, name: str, layers: int = 3) -> dict:
        """إنشاء هيكل هرمي جديد منفصل"""
        if not self.is_initialized:
            await self.initialize()
        return await self.meta_architect.create_new_hierarchy({
            "name": name,
            "layers": layers
        })
    
    def get_meta_status(self) -> dict:
        """حالة الطبقات الفوقية"""
        if not self.executive_controller:
            return {"status": "not_initialized"}
        
        return {
            "executive_controller": self.executive_controller.get_status(),
            "builder_teams": {
                team: len(members) 
                for team, members in self.builder_council.teams.items()
            } if self.builder_council else {},
            "can_create_layers": True,
            "can_destroy_layers": True,
            "can_rebuild_hierarchy": True
        }


# Singleton
ai_hierarchy = AIHierarchy()

# تصدير الأساسي
__all__ = [
    'ai_hierarchy',
    'AIHierarchy',
    'PresidentInterface',
    'AlertLevel',
    'HighCouncil',
    'BalanceCouncil',
    'ScoutManager',
    'MetaTeam',
    'DomainExpertTeam',
    'ExecutionManager',
    'SeventhDimension'
]
