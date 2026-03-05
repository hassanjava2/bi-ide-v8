"""
Council AI Bridge - ربط المجلس بـ RTX5090

يُمكّن الحكماء من الحصول على آراء حقيقية من AI
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass


RTX5090_URL = os.getenv("RTX5090_URL", "http://192.168.1.164:8090")


@dataclass
class SageOpinion:
    """رأي حكيم"""
    sage_id: str
    sage_name: str
    role: str
    opinion: str
    confidence: float
    reasoning: str
    source: str  # "rtx5090" أو "local"


class CouncilAIBridge:
    """
    جسر الذكاء الاصطناعي للمجلس
    
    يطلب آراء من RTX5090، وإذا لم يكن متوفراً يستخدم local heuristics
    """
    
    def __init__(self, rtx_url: str = None):
        self.rtx_url = rtx_url or RTX5090_URL
        self._rtx_available = None  # سيتم التحقق منه
        self._cache: Dict[str, SageOpinion] = {}
        
    async def _check_rtx_availability(self) -> bool:
        """التحقق من توفر RTX5090"""
        if self._rtx_available is not None:
            return self._rtx_available
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.rtx_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    self._rtx_available = resp.status == 200
                    return self._rtx_available
        except:
            self._rtx_available = False
            return False
    
    async def get_sage_opinion(
        self,
        sage_id: str,
        sage_name: str,
        sage_role: str,
        topic: str,
        context: Dict[str, Any] = None
    ) -> SageOpinion:
        """
        الحصول على رأي حكيم - حقيقي من AI أو fallback محلي
        """
        cache_key = f"{sage_id}:{topic}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # محاولة الحصول على رأي حقيقي من RTX5090
        if await self._check_rtx_availability():
            try:
                opinion = await self._fetch_from_rtx(
                    sage_name, sage_role, topic, context
                )
                self._cache[cache_key] = opinion
                return opinion
            except Exception as e:
                print(f"⚠️ RTX5090 failed for {sage_name}: {e}")
        
        # Fallback: رأي محلي مُحسّن
        opinion = self._generate_local_opinion(
            sage_id, sage_name, sage_role, topic
        )
        self._cache[cache_key] = opinion
        return opinion
    
    async def _fetch_from_rtx(
        self,
        sage_name: str,
        sage_role: str,
        topic: str,
        context: Dict[str, Any] = None
    ) -> SageOpinion:
        """طلب رأي من RTX5090"""
        
        prompt = f"""أنت {sage_name}، حكيم متخصص في {sage_role}.

الموضوع المطروح للنقاش: {topic}

بصفتك خبيراً في {sage_role}، ما هو رأيك في هذا الموضوع؟
هل تؤيد أم تعارض؟ وما هو التبرير؟

قدم رأيك باختصار (2-3 جمل) مع ذكر السبب."""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rtx_url}/council/message",
                json={
                    "message": prompt,
                    "context": context or {},
                    "sage_role": sage_role,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SageOpinion(
                        sage_id=f"rtx_{sage_role}",
                        sage_name=sage_name,
                        role=sage_role,
                        opinion=data.get("response", "لا رأي"),
                        confidence=data.get("confidence", 0.7),
                        reasoning="استنتاج من RTX5090",
                        source="rtx5090"
                    )
                else:
                    raise Exception(f"RTX returned {resp.status}")
    
    def _generate_local_opinion(
        self,
        sage_id: str,
        sage_name: str,
        sage_role: str,
        topic: str
    ) -> SageOpinion:
        """توليد رأي محلي (fallback)"""
        topic_lower = topic.lower()
        
        # تحليل أكثر ذكاءً من الثابت
        role_keywords = {
            "identity": ["هوية", "identity", "brand", "قيم", "value"],
            "strategy": ["خطة", "plan", "استراتيجية", "strategy", "future", "مستقبل"],
            "ethics": ["أخلاق", "ethics", "privacy", "خصوصية", "عدل", "justice"],
            "balance": ["توازن", "balance", "risk", "مخاطر", "harmony"],
            "knowledge": ["معرفة", "knowledge", "تعلم", "learning", "بحث", "research"],
            "relations": ["علاقات", "relations", "تعاون", "collaboration", "communication"],
            "innovation": ["ابتكار", "innovation", "إبداع", "creative", "new", "جديد"],
            "protection": ["أمان", "security", "حماية", "protection", "safety", "safety"],
        }
        
        # حساب مدى تطابق الموضوع مع الدور
        relevance = 0
        keywords = role_keywords.get(sage_role.lower(), [])
        for kw in keywords:
            if kw in topic_lower:
                relevance += 1
        
        confidence = min(0.6 + (relevance * 0.1), 0.95)
        
        if relevance > 0:
            opinion_text = f"كخبير في {sage_role}، أرى أن هذا الموضوع يتطلب {sage_role}ة مُحكمة. التحليل يشير إلى جوانب إيجابية مع الحاجة لتوخي الحذر."
        else:
            opinion_text = f"من منظور {sage_role}، الموضوع يبدو محايداً نسبياً. لا مخاطر واضحة ولا مكاسب استثنائية."
        
        return SageOpinion(
            sage_id=sage_id,
            sage_name=sage_name,
            role=sage_role,
            opinion=opinion_text,
            confidence=confidence,
            reasoning=f"تطابق {relevance} كلمات مفتاحية مع خبرة {sage_role}",
            source="local"
        )
    
    async def batch_get_opinions(
        self,
        sages: list,
        topic: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, SageOpinion]:
        """الحصول على آراء مجموعة من الحكماء بشكل متوازي"""
        tasks = []
        for sage in sages:
            task = self.get_sage_opinion(
                sage.id,
                sage.name,
                sage.role.value if hasattr(sage.role, 'value') else str(sage.role),
                topic,
                context
            )
            tasks.append(task)
        
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for sage, opinion in zip(sages, opinions):
            if isinstance(opinion, Exception):
                # Fallback على الخطأ
                opinion = self._generate_local_opinion(
                    sage.id, sage.name,
                    sage.role.value if hasattr(sage.role, 'value') else str(sage.role),
                    topic
                )
            result[sage.id] = opinion
        
        return result


# Singleton
_bridge: Optional[CouncilAIBridge] = None


def get_ai_bridge() -> CouncilAIBridge:
    """الحصول على الجسر الموحد"""
    global _bridge
    if _bridge is None:
        _bridge = CouncilAIBridge()
    return _bridge
