"""
24/7 Autonomous Council Loop - المجلس المستقل الدائم

المجلس يشتغل أوتوماتيكي 24/7 بدون توقف:
1. يتناقشون بمواضيع حقيقية أوتوماتيكياً
2. ينتجون أفكار + قرارات + خطط بدون تدخلي
3. أشوف مناقشاتهم مباشرة على شكل دردشة حقيقية
4. كل حكيم يطرح رأيه باختصاصه

وضعين:
- وضع النقاش (أوتوماتيكي): المجلس يتناقش بينهم
- وضع التفاعل (أنا أتدخل): أتناقش وياهم
"""

import asyncio
import random
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class DiscussionMode(Enum):
    """أوضاع النقاش"""
    AUTONOMOUS = "autonomous"  # المجلس يتناقش لوحده
    INTERACTIVE = "interactive"  # الرئيس يتفاعل
    DECISION = "decision"  # اتخاذ قرار محدد


class DiscussionTopic(Enum):
    """مواضيع النقاش"""
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    ECONOMY = "economy"
    SOCIETY = "society"
    PHILOSOPHY = "philosophy"
    STRATEGY = "strategy"
    TRAINING = "training"
    SELF_IMPROVEMENT = "self_improvement"
    POST_CATASTROPHE = "post_catastrophe"
    RESOURCE_MANAGEMENT = "resource_management"


@dataclass
class CouncilMember:
    """عضو مجلس"""
    member_id: str
    name: str
    title: str
    expertise: List[str]
    personality: str  # "optimist", "pessimist", "pragmatic", "visionary", "cautious"
    voice_color: str  # For UI display
    
    def generate_opinion(self, topic: str, context: Dict) -> str:
        """توليد رأي حسب الشخصية"""
        templates = {
            "optimist": [
                "This is a tremendous opportunity! {detail}",
                "I see great potential here. {detail}",
                "We can achieve excellence if we {action}."
            ],
            "pessimist": [
                "We must be cautious. {risk}",
                "I see significant risks: {risk}",
                "This could fail if we don't address {concern}."
            ],
            "pragmatic": [
                "Let's focus on practical steps: {action}",
                "The most efficient approach is {action}.",
                "We should prioritize {priority}."
            ],
            "visionary": [
                "This aligns with our long-term vision. {future}",
                "Imagine the possibilities: {future}",
                "In 50 years, this will {impact}."
            ],
            "cautious": [
                "We need more data before deciding. {question}",
                "Have we considered {question}?",
                "Let's verify {question} first."
            ]
        }
        
        template = random.choice(templates.get(self.personality, templates["pragmatic"]))
        
        # Fill in context (simplified)
        opinion = template.format(
            detail="The data supports this direction.",
            risk="Resource constraints and timeline pressures.",
            action="implement incremental improvements.",
            future="transform how we operate entirely.",
            impact="be seen as a pivotal moment",
            priority="stability and reliability.",
            concern="the unknown variables",
            question="all potential failure modes"
        )
        
        return opinion


@dataclass
class DiscussionMessage:
    """رسالة في المناقشة"""
    message_id: str
    member_id: str
    member_name: str
    content: str
    timestamp: datetime
    topic: str
    reply_to: Optional[str] = None
    importance: int = 3  # 1-5


@dataclass
class CouncilDecision:
    """قرار المجلس"""
    decision_id: str
    topic: str
    question: str
    votes: Dict[str, str]  # member_id -> vote (yes/no/abstain)
    reasoning: str
    confidence: float
    timestamp: datetime
    executed: bool = False
    execution_result: Optional[str] = None


class AutonomousCouncil:
    """
    المجلس المستقل الدائم - يشتغل 24/7
    """
    
    def __init__(self):
        self.members: Dict[str, CouncilMember] = {}
        self.discussion_history: List[DiscussionMessage] = []
        self.decisions: List[CouncilDecision] = []
        self.current_mode = DiscussionMode.AUTONOMOUS
        self.active_topic: Optional[str] = None
        self.running = False
        self.discussion_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "discussions_count": 0,
            "decisions_made": 0,
            "topics_covered": set(),
            "start_time": None
        }
        
        self._initialize_members()
        logger.info("🏛️ Autonomous Council initialized with 16 sages")
    
    def _initialize_members(self):
        """تهيئة أعضاء المجلس (16 حكيم)"""
        members_data = [
            ("sage_1", "Ibn Sina", "The Physician", ["medicine", "biology", "philosophy"], "pragmatic", "#4CAF50"),
            ("sage_2", "Al-Khwarizmi", "The Mathematician", ["mathematics", "algorithms", "astronomy"], "pragmatic", "#2196F3"),
            ("sage_3", "Ibn Rushd", "The Philosopher", ["philosophy", "logic", "law"], "visionary", "#9C27B0"),
            ("sage_4", "Al-Haytham", "The Optics Master", ["physics", "optics", "engineering"], "cautious", "#FF9800"),
            ("sage_5", "Al-Biruni", "The Polymath", ["geography", "astronomy", "history"], "optimist", "#00BCD4"),
            ("sage_6", "Maimonides", "The Wise", ["medicine", "theology", "ethics"], "cautious", "#795548"),
            ("sage_7", "Al-Farabi", "The Second Teacher", ["philosophy", "music", "sociology"], "visionary", "#E91E63"),
            ("sage_8", "Ibn Khaldun", "The Historian", ["history", "sociology", "economics"], "pessimist", "#607D8B"),
            ("sage_9", "Al-Razi", "The Experimenter", ["chemistry", "medicine", "experiments"], "pragmatic", "#8BC34A"),
            ("sage_10", "Al-Kindi", "The Philosopher of Arabs", ["philosophy", "cryptography", "physics"], "optimist", "#3F51B5"),
            ("sage_11", "Al-Tusi", "The Astronomer", ["astronomy", "mathematics", "engineering"], "visionary", "#009688"),
            ("sage_12", "Ibn Battuta", "The Traveler", ["geography", "anthropology", "logistics"], "optimist", "#FFC107"),
            ("sage_13", "Al-Jazari", "The Engineer", ["engineering", "mechanics", "automation"], "pragmatic", "#FF5722"),
            ("sage_14", "Fatima Al-Fihri", "The Founder", ["education", "institution building", "vision"], "visionary", "#673AB7"),
            ("sage_15", "Al-Masudi", "The Historian", ["history", "geography", "culture"], "pessimist", "#795548"),
            ("sage_16", "Al-Khazini", "The Mechanic", ["physics", "mechanics", "measurement"], "cautious", "#9E9E9E"),
        ]
        
        for data in members_data:
            member = CouncilMember(*data)
            self.members[member.member_id] = member
    
    async def start(self):
        """بدء المجلس (24/7)"""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        self.discussion_task = asyncio.create_task(self._discussion_loop())
        
        logger.info("🚀 Autonomous Council started (24/7 mode)")
    
    async def stop(self):
        """إيقاف المجلس"""
        self.running = False
        if self.discussion_task:
            self.discussion_task.cancel()
            try:
                await self.discussion_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Autonomous Council stopped")
    
    async def _discussion_loop(self):
        """حلقة النقاش الرئيسية"""
        while self.running:
            try:
                # Select a random topic
                topic = random.choice(list(DiscussionTopic))
                self.active_topic = topic.value
                self.stats["topics_covered"].add(topic.value)
                
                logger.info(f"🗣️ Council discussing: {topic.value}")
                
                # Generate a discussion on this topic
                await self._conduct_discussion(topic)
                
                self.stats["discussions_count"] += 1
                
                # Wait before next discussion (5-15 minutes)
                wait_time = random.randint(300, 900)
                await asyncio.sleep(wait_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discussion loop: {e}")
                await asyncio.sleep(60)
    
    async def _conduct_discussion(self, topic: DiscussionTopic):
        """إجراء مناقشة على موضوع"""
        # Generate discussion prompt based on topic
        prompts = self._get_topic_prompts(topic)
        prompt = random.choice(prompts)
        
        # Select participants (random subset or all)
        participants = list(self.members.values())
        random.shuffle(participants)
        
        # Each member contributes
        for member in participants[:random.randint(5, 16)]:
            opinion = member.generate_opinion(prompt, {})
            
            message = DiscussionMessage(
                message_id=str(uuid.uuid4()),
                member_id=member.member_id,
                member_name=member.name,
                content=opinion,
                timestamp=datetime.now(timezone.utc),
                topic=topic.value
            )
            
            self.discussion_history.append(message)
            
            # Log for visibility
            logger.info(f"💬 [{member.name}] {opinion[:100]}...")
            
            # Simulate thinking time
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Make a decision
        decision = await self._make_decision(topic, prompt)
        if decision:
            self.decisions.append(decision)
            self.stats["decisions_made"] += 1
            logger.info(f"📊 Decision made on {topic.value}: {decision.confidence:.2f} confidence")
    
    def _get_topic_prompts(self, topic: DiscussionTopic) -> List[str]:
        """الحصول على محفزات للموضوع"""
        prompts = {
            DiscussionTopic.TECHNOLOGY: [
                "What technology should we prioritize for training?",
                "How can we improve our code generation capabilities?",
                "Should we develop our own operating system?"
            ],
            DiscussionTopic.SCIENCE: [
                "What scientific domain needs more training data?",
                "How should we balance breadth vs depth in scientific knowledge?"
            ],
            DiscussionTopic.STRATEGY: [
                "What is our priority for the next phase?",
                "How should we allocate our computational resources?"
            ],
            DiscussionTopic.POST_CATASTROPHE: [
                "What knowledge is most critical for rebuilding?",
                "How should we organize information for post-catastrophe access?"
            ],
            DiscussionTopic.SELF_IMPROVEMENT: [
                "How can we improve our own architecture?",
                "What capabilities are we missing?"
            ],
            DiscussionTopic.RESOURCE_MANAGEMENT: [
                "Are we using our training data efficiently?",
                "Should we expand to more hardware?"
            ]
        }
        
        return prompts.get(topic, ["What should we focus on?"])
    
    async def _make_decision(self, topic: DiscussionTopic, 
                            question: str) -> Optional[CouncilDecision]:
        """اتخاذ قرار بالتصويت"""
        # Simple voting mechanism
        votes = {}
        for member_id, member in self.members.items():
            # Vote based on personality
            if member.personality == "optimist":
                vote = random.choices(["yes", "no", "abstain"], weights=[0.7, 0.2, 0.1])[0]
            elif member.personality == "pessimist":
                vote = random.choices(["yes", "no", "abstain"], weights=[0.3, 0.6, 0.1])[0]
            else:
                vote = random.choices(["yes", "no", "abstain"], weights=[0.5, 0.3, 0.2])[0]
            
            votes[member_id] = vote
        
        # Calculate confidence based on agreement
        yes_votes = sum(1 for v in votes.values() if v == "yes")
        no_votes = sum(1 for v in votes.values() if v == "no")
        total_votes = yes_votes + no_votes
        
        if total_votes == 0:
            confidence = 0.5
        else:
            confidence = max(yes_votes, no_votes) / total_votes
        
        decision = CouncilDecision(
            decision_id=str(uuid.uuid4()),
            topic=topic.value,
            question=question,
            votes=votes,
            reasoning=f"Council vote: {yes_votes} yes, {no_votes} no",
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
        )
        
        return decision
    
    async def user_interact(self, user_message: str, 
                           mentioned_member: Optional[str] = None) -> List[DiscussionMessage]:
        """
        تفاعل المستخدم مع المجلس
        
        mentioned_member: إذا ذكر عضو بالاسم، يجاوب هو بالذات
        """
        responses = []
        
        # If specific member mentioned
        if mentioned_member and mentioned_member in self.members:
            member = self.members[mentioned_member]
            opinion = member.generate_opinion(user_message, {})
            
            message = DiscussionMessage(
                message_id=str(uuid.uuid4()),
                member_id=member.member_id,
                member_name=member.name,
                content=opinion,
                timestamp=datetime.now(timezone.utc),
                topic="user_interaction"
            )
            
            self.discussion_history.append(message)
            responses.append(message)
        
        else:
            # Get relevant experts
            relevant = self._find_relevant_experts(user_message)
            
            for member_id in relevant[:3]:
                member = self.members[member_id]
                opinion = member.generate_opinion(user_message, {})
                
                message = DiscussionMessage(
                    message_id=str(uuid.uuid4()),
                    member_id=member.member_id,
                    member_name=member.name,
                    content=opinion,
                    timestamp=datetime.now(timezone.utc),
                    topic="user_interaction"
                )
                
                self.discussion_history.append(message)
                responses.append(message)
        
        return responses
    
    def _find_relevant_experts(self, query: str) -> List[str]:
        """البحث عن الخبراء ذوي الصلة"""
        query_lower = query.lower()
        scores = {}
        
        for member_id, member in self.members.items():
            score = 0
            for expertise in member.expertise:
                if expertise.lower() in query_lower:
                    score += 1
            scores[member_id] = score
        
        # Return top scorers
        sorted_members = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_members if m[1] > 0][:5]
    
    async def command_execution(self, command: str) -> CouncilDecision:
        """
        تنفيذ أمر من الرئيس
        مثال: "سوولي ERP" → المجلس يقرر وينفذ
        """
        logger.info(f"👑 Presidential command: {command}")
        
        # Create decision context
        decision = CouncilDecision(
            decision_id=str(uuid.uuid4()),
            topic="presidential_command",
            question=command,
            votes={m: "yes" for m in self.members.keys()},  # All agree
            reasoning=f"Presidential command: {command}",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.decisions.append(decision)
        
        # Here would trigger actual execution
        logger.info(f"🎯 Executing: {command}")
        
        return decision
    
    def get_discussion_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """الحصول على سجل المناقشات"""
        recent = self.discussion_history[-limit:]
        return [
            {
                "member": m.member_name,
                "content": m.content,
                "topic": m.topic,
                "timestamp": m.timestamp.isoformat()
            }
            for m in recent
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """حالة المجلس"""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds()
        
        return {
            "running": self.running,
            "mode": self.current_mode.value,
            "active_topic": self.active_topic,
            "discussions_count": self.stats["discussions_count"],
            "decisions_made": self.stats["decisions_made"],
            "topics_covered": len(self.stats["topics_covered"]),
            "uptime_seconds": uptime,
            "members_active": len(self.members),
            "history_size": len(self.discussion_history)
        }


# Global instance
autonomous_council = AutonomousCouncil()
