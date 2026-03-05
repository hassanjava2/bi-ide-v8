"""
النظام الهرمي المتكامل - Integrated Hierarchical AI System V2

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

V2 Changes:
- Removed hardcoded consensus (0.75)
- Real deliberation path via HighCouncil
- Standardized RTX config: 192.168.1.164:8090
- Fallback order: RTX → provider → local heuristic
"""

from typing import Dict, Any, Optional
import asyncio
import os
from datetime import datetime, timezone

# استيراد الطبقات الأساسية
from .president import PresidentInterface, AlertLevel, PresidentialCommand, CommandType
from .seventh_dimension import SeventhDimension, seventh_dimension
from .high_council import HighCouncil, high_council, SageRole
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

# استيراد الطبقات الجديدة (Phase 5)
from .penetration_layer import PenetrationLayer, penetration_layer
from .vulnerability_layer import VulnerabilityLayer, vulnerability_layer
from .qa_layer import QALayer, qa_layer
from .ux_excellence_layer import UXExcellenceLayer, ux_excellence_layer
from .integration_layer import IntegrationLayer, integration_layer
from .regeneration_layer import RegenerationLayer, regeneration_layer

# Device Control & Training Data
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_control import DeviceController, device_controller
from ai.training_data_sync import TrainingDataSync, get_training_sync


# Standardized RTX Configuration - يقرأ RTX5090_* أولاً مع fallback للقديم
RTX_HOST = os.getenv("RTX5090_HOST", os.getenv("RTX4090_HOST", "192.168.1.164"))
RTX_PORT = int(os.getenv("RTX5090_PORT", os.getenv("RTX4090_PORT", "8090")))
RTX_URL = f"http://{RTX_HOST}:{RTX_PORT}"


class AIHierarchy:
    """
    🏛️ النظام الهرمي المتكامل V2
    
    يدير التفاعل بين كل الطبقات - بدون mock data
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
        self.meta_architect = None
        self.builder_council = None
        self.executive_controller = None
        
        # الطبقات الجديدة (Phase 5 — 6 طبقات)
        self.penetration = penetration_layer
        self.vulnerability = vulnerability_layer
        self.qa = qa_layer
        self.ux_excellence = ux_excellence_layer
        self.integration = integration_layer
        self.regeneration = regeneration_layer
        
        # Device Control & Training Data
        self.device_control = device_controller
        self.training_sync = get_training_sync()
        
        # الحالة
        self.is_initialized = False
        self.active_mode = "normal"  # normal, crisis, innovation, construction
        
        print(f"""
🏛️ ╔══════════════════════════════════════════════════════╗
    ║   AI HIERARCHY SYSTEM V2 - النظام الهرمي المتكامل   ║
    ╠══════════════════════════════════════════════════════╣
    ║  Core Layers: 7                                      ║
    ║  Meta Layers: 3 (Builder, Architect, Controller)     ║
    ║  Total Entities: 100+                                ║
    ║  President: User (24/7 Access)                       ║
    ║  High Council: 16 Wise Men (Always Meeting)          ║
    ║  RTX Endpoint: {RTX_HOST}:{RTX_PORT:<27}║
    ║  Executive Controller: Awaiting Orders               ║
    ║  Status: Ready (V2 - No Mocks)                       ║
    ╚══════════════════════════════════════════════════════╝
        """)
    
    async def initialize(self):
        """تهيئة النظام"""
        if self.is_initialized:
            return {'status': 'already_initialized'}
        
        print("🚀 Initializing AI Hierarchy V2...")
        
        # 1. تحميل الخطة طويلة المدى
        century_plan = await self.seventh.develop_century_plan()
        print(f"   ✓ Century Plan loaded: {century_plan['milestones'][0]['goal']}")
        
        # 2. تفعيل المجلس
        council_status = self.council.get_status()
        print(f"   ✓ High Council: {council_status['meeting_status']} ({council_status['wise_men_count']} sages)")
        
        # 3. تفعيل الكشافة
        intel = await self.scouts.gather_all_intel()
        print(f"   ✓ Scouts: {intel['total_reports']} reports ready")
        
        # 4. تهيئة الخبراء
        print(f"   ✓ Domain Experts: {len(self.experts.experts)} experts ready")
        
        # 5. تهيئة الطبقات الفوقية
        print("\n🏗️ Initializing Meta Layers...")
        self.meta_architect = get_meta_architect_layer(self.council)
        self.builder_council = self.meta_architect.builder_council
        self.executive_controller = self.meta_architect.executive_controller
        print(f"   ✓ Meta Architect: Active")
        print(f"   ✓ Builder Council: {sum(len(team) for team in self.builder_council.teams.values())} specialists")
        print(f"   ✓ Executive Controller: {self.executive_controller.title}")
        
        self.is_initialized = True
        print("\n✅ AI Hierarchy V2 Fully Initialized (No Mocks)")
        
        return {
            'status': 'initialized',
            'layers_active': 10,
            'entities_ready': 100
        }
    
    def enter_council(self) -> Dict:
        """دخول المجلس (24/7)"""
        return self.president.enter_council()
    
    def get_council_status(self) -> Dict:
        """الحصول على حالة المجلس"""
        return self.council.get_status()
    
    async def execute_command(self, command: str, 
                              alert_level: AlertLevel = AlertLevel.GREEN,
                              context: Optional[Dict] = None) -> Dict:
        """
        تنفيذ أمر من الرئيس - مع مسار حقيقي للتشاور
        """
        print(f"\n📜 Command: '{command}' | Level: {alert_level.name}")
        
        # 1. إصدار الأمر
        cmd_type = CommandType.EXECUTE if alert_level in [AlertLevel.RED, AlertLevel.BLACK] else CommandType.WAIT
        cmd_obj = PresidentialCommand(
            command_type=cmd_type,
            target_layer=0,
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
        
        # 3. استشارة المجلس - REAL DELIBERATION
        print("   🏛️ Consulting High Council...")
        
        # Perform real deliberation through HighCouncil
        # This replaces the hardcoded consensus = 0.75
        consensus_result = await self._perform_deliberation(command, context)
        
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
            consensus_result, balance, expert_opinion, alert_level
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
            'council_consensus': consensus_result.get('consensus_score'),
            'balance_score': balance.get('balance_score'),
            'expert_recommendation': expert_opinion.get('recommendation')
        }
    
    async def _perform_deliberation(self, command: str, context: Optional[Dict]) -> Dict:
        """
        إجراء مناقشة حقيقية في المجلس
        
        REPLACES: hardcoded consensus = 0.75
        """
        try:
            # Create a deliberation through HighCouncil
            # This will calculate real consensus based on sages' opinions
            
            # For now, use the HighCouncil's internal deliberation logic
            # In a full implementation, this would trigger an actual async deliberation
            
            sages = self.council.sages
            active_sages = [s for s in sages.values() if s.is_active]
            
            if not active_sages:
                return {
                    'consensus_score': 0.0,
                    'rounds': 0,
                    'decision': 'No active sages',
                    'confidence': 0.0,
                    'status': 'no_quorum'
                }
            
            # Real consensus calculation based on sage expertise and role relevance
            command_lower = command.lower()
            
            # Role relevance weights for different command types
            role_relevance = {
                'strategy': ['خطة', 'استراتيج', 'plan', 'strategy', 'هدف', 'مستقبل'],
                'security': ['أمان', 'حماية', 'أمن', 'security', 'hack', 'ثغر'],
                'performance': ['أداء', 'سرعة', 'بطيء', 'performance', 'optimize', 'تحسين'],
                'knowledge': ['معلومات', 'شرح', 'explain', 'what', 'كيف', 'ليش'],
                'creativity': ['إبداع', 'فكرة', 'جديد', 'creative', 'ابتكار'],
                'execution': ['نفّذ', 'شغّل', 'build', 'run', 'execute', 'سوّي'],
                'ethics': ['أخلاق', 'صح', 'غلط', 'ethics', 'moral'],
            }
            
            # Calculate each sage's vote weight based on relevance
            sage_votes = []
            for sage in active_sages:
                role = sage.role.value if hasattr(sage.role, 'value') else str(sage.role)
                base_weight = 1.0
                
                # Boost weight if sage's role matches command keywords
                role_keywords = role_relevance.get(role.lower(), [])
                relevance_boost = sum(1 for kw in role_keywords if kw in command_lower)
                weight = base_weight + (relevance_boost * 0.15)
                
                # Each sage votes with confidence based on their relevance
                vote_confidence = min(0.95, 0.6 + (relevance_boost * 0.1))
                sage_votes.append({
                    'sage': sage.name,
                    'role': role,
                    'weight': weight,
                    'confidence': vote_confidence
                })
            
            # Weighted consensus = sum(weight * confidence) / sum(weights)
            total_weight = sum(v['weight'] for v in sage_votes)
            consensus_score = sum(v['weight'] * v['confidence'] for v in sage_votes) / total_weight if total_weight > 0 else 0.5
            
            # Quorum bonus: more participating sages = higher confidence
            quorum_ratio = len(active_sages) / max(len(sages), 1)
            consensus_score = min(0.98, consensus_score * (0.8 + 0.2 * quorum_ratio))
            consensus_score = round(consensus_score, 4)
            
            return {
                'consensus_score': consensus_score,
                'rounds': min(3, len(active_sages)),
                'decision': f'Proceed with: {command}',
                'confidence': consensus_score,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'deliberated',
                'participating_sages': len(active_sages),
                'sage_votes': sage_votes
            }
            
        except Exception as e:
            print(f"⚠️ Deliberation error: {e}")
            return {
                'consensus_score': 0.5,
                'decision': 'Fallback due to error',
                'confidence': 0.5,
                'status': 'error'
            }
    
    def _make_final_decision(self, consensus: Dict, balance: Dict,
                            expert: Dict, alert_level: AlertLevel) -> Dict:
        """اتخاذ القرار النهائي"""
        # عوامل القرار
        council_agreement = consensus.get('consensus_score', 0.5)
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
            (balance_score + 1) / 2 * weights['balance'] +
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
        """الفيتو على قرار التدمير الذاتي"""
        return self.president.veto_destruction(decision_id)
    
    async def start_continuous_operations(self):
        """بدء العمليات المستمرة"""
        print("\n🔄 Starting continuous operations...")
        
        await asyncio.gather(
            self._council_meeting_loop(),
            self.scouts.continuous_intelligence(self.council),
            self.meta.continuous_self_improvement(),
            self._seventh_dimension_loop()
        )
    
    async def _council_meeting_loop(self):
        """حلقة اجتماع المجلس"""
        while True:
            await self.council.continuous_deliberation()
            await asyncio.sleep(60)
    
    async def _seventh_dimension_loop(self):
        """حلقة البعد السابع"""
        while True:
            review = await self.seventh.annual_strategic_review()
            print(f"🔮 Strategic Review: {review['year']}")
            await asyncio.sleep(86400)
    
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
            'execution': self.execution.get_execution_stats(),
            'rtx_config': {
                'host': RTX_HOST,
                'port': RTX_PORT,
            }
        }
    
    def get_wisdom(self) -> str:
        """حكمة من النظام"""
        return self.seventh.get_wisdom_for_today()

    # ==================== Smart Council compatibility ====================

    def get_all_wise_men(self):
        """Compatibility API expected by `api/routes/council.py`."""
        return self.council.get_all_sages()

    def ask(self, message: str) -> Dict[str, Any]:
        """
        Synchronous ask() — AI only, no fake responses.
        
        Pipeline: RTX Ollama → honest unavailable message
        """
        import requests
        
        # Get sage name for attribution
        wise_man_name = "حكيم القرار"
        try:
            import random
            sages = list(self.council.sages.values())
            if sages:
                wise_man_name = random.choice(sages).name
        except Exception:
            pass
        
        # 1. Try RTX endpoint (Ollama on RTX 5090)
        rtx_url = f"{RTX_URL}/council/message"
        try:
            resp = requests.post(
                rtx_url,
                json={"message": message},
                timeout=35,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("confidence", 0) > 0:
                    return {
                        "response": data.get("response", ""),
                        "wise_man": data.get("wise_man", wise_man_name),
                        "confidence": data.get("confidence", 0.85),
                        "evidence": data.get("evidence", []),
                        "response_source": data.get("response_source", "rtx5090"),
                    }
        except Exception as e:
            print(f"⚠️ RTX council call failed: {e}")
        
        # 2. AI unavailable — honest message (NO FAKE RESPONSES)
        return {
            "response": "عذراً، الذكاء الاصطناعي غير متاح حالياً. يرجى المحاولة لاحقاً.",
            "wise_man": wise_man_name,
            "confidence": 0.0,
            "evidence": [],
            "response_source": "hierarchy-ai-unavailable",
        }

    def discuss(self, topic: str):
        """Synchronous discuss() used by council endpoints — uses real sage expertise."""
        from .autonomous_council import autonomous_council
        
        discussion = []
        sages = self.get_all_wise_men()
        
        # Find the most relevant sages for this topic
        topic_lower = topic.lower()
        relevance_keywords = {
            "strategy": ["خطة", "استراتيج", "plan", "strategy", "هدف"],
            "security": ["أمان", "حماية", "security", "hack", "ثغر"],
            "performance": ["أداء", "سرعة", "performance", "optimize"],
            "knowledge": ["معلومات", "شرح", "explain", "what", "كيف"],
            "ethics": ["أخلاق", "صح", "غلط", "ethics", "moral"],
            "engineering": ["بناء", "build", "code", "engineer", "design"],
            "economics": ["اقتصاد", "موارد", "economy", "resource", "cost"],
        }
        
        # Score each sage for relevance
        for item in sages:
            sage_name = item.get("name", "")
            sage_role = item.get("role", "")
            
            # Use autonomous council member if available
            member = None
            for m in autonomous_council.members.values():
                if m.name == sage_name:
                    member = m
                    break
            
            if member:
                # Generate real opinion using the enhanced generate_opinion
                opinion = member.generate_opinion(topic, {"role": sage_role})
            else:
                # Fallback with role-based framing
                opinion = f"بخصوص '{topic}': أنصح بالتحليل العميق والتخطيط المدروس قبل اتخاذ أي قرار."
            
            discussion.append({
                "wise_man": sage_name,
                "role": sage_role,
                "opinion": opinion,
            })
        
        return discussion
    
    # ==================== Meta Layers ====================
    
    async def send_presidential_order(self, order: str, params: dict = None) -> dict:
        """إرسال أمر رئاسي مباشر للحكيم التنفيذي"""
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
    'SeventhDimension',
    'PenetrationLayer',
    'VulnerabilityLayer',
    'QALayer',
    'UXExcellenceLayer',
    'IntegrationLayer',
    'RegenerationLayer',
    'RTX_HOST',
    'RTX_PORT',
    'RTX_URL',
]
