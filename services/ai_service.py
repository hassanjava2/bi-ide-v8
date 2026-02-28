"""
خدمة الذكاء الاصطناعي
AI Service for code generation and inference
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """نموذج حدود المعدل"""
    requests: int = 0
    reset_time: datetime = field(default_factory=datetime.now)
    window_seconds: int = 60
    max_requests: int = 100
    
    def is_allowed(self) -> bool:
        """التحقق مما إذا كان الطلب مسموحاً"""
        now = datetime.now()
        if now - self.reset_time > timedelta(seconds=self.window_seconds):
            self.requests = 0
            self.reset_time = now
        return self.requests < self.max_requests
    
    def consume(self) -> bool:
        """استهلاك طلب واحد"""
        if self.is_allowed():
            self.requests += 1
            return True
        return False


@dataclass
class ContextEntry:
    """نموذج مدخل السياق"""
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """نموذج السياق"""
    user_id: str
    entries: List[ContextEntry] = field(default_factory=list)
    max_entries: int = 20
    ttl_seconds: int = 3600  # ساعة واحدة
    
    def add_entry(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """إضافة مدخل للسياق"""
        self._cleanup_expired()
        
        entry = ContextEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        
        # الاحتفاظ فقط بآخر المدخلات
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_context(self, limit: int = 10) -> str:
        """الحصول على نص السياق"""
        self._cleanup_expired()
        
        recent_entries = self.entries[-limit:]
        return "\n".join(e.content for e in recent_entries)
    
    def _cleanup_expired(self) -> None:
        """إزالة المدخلات منتهية الصلاحية"""
        cutoff = datetime.now() - timedelta(seconds=self.ttl_seconds)
        self.entries = [e for e in self.entries if e.timestamp > cutoff]


@dataclass
class AIResponse:
    """نموذج رد الذكاء الاصطناعي"""
    content: str
    tokens_used: int
    model: str
    confidence: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class AIService:
    """
    خدمة الذكاء الاصطناعي
    
    تدير توليد الأكواد والإكمال والمراجعة مع إدارة السياق وحدود المعدل
    """
    
    def __init__(
        self,
        default_model: str = "gpt-4",
        rate_limit_per_minute: int = 100
    ):
        """
        تهيئة خدمة الذكاء الاصطناعي
        
        المعاملات:
            default_model: النموذج الافتراضي
            rate_limit_per_minute: حد الطلبات في الدقيقة
        """
        self._default_model = default_model
        self._rate_limits: Dict[str, RateLimit] = {}
        self._contexts: Dict[str, Context] = {}
        self._rate_limit_per_minute = rate_limit_per_minute
        self._context_lock = asyncio.Lock()
        
        logger.info("تم تهيئة خدمة الذكاء الاصطناعي")
    
    def _get_user_rate_limit(self, user_id: str) -> RateLimit:
        """الحصول على حدود المعدل للمستخدم"""
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = RateLimit(
                max_requests=self._rate_limit_per_minute,
                window_seconds=60
            )
        return self._rate_limits[user_id]
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """التحقق من حدود المعدل"""
        rate_limit = self._get_user_rate_limit(user_id)
        return rate_limit.consume()
    
    async def _get_context(self, user_id: str) -> Context:
        """الحصول على سياق المستخدم"""
        async with self._context_lock:
            if user_id not in self._contexts:
                self._contexts[user_id] = Context(user_id=user_id)
            return self._contexts[user_id]
    
    async def generate_code(
        self,
        user_id: str,
        prompt: str,
        language: Optional[str] = None,
        use_context: bool = True
    ) -> AIResponse:
        """
        توليد كود جديد
        
        المعاملات:
            user_id: معرف المستخدم
            prompt: الوصف
            language: لغة البرمجة
            use_context: استخدام السياق السابق
            
        العائد:
            AIResponse: الرد المولد
            
        الاستثناءات:
            RuntimeError: إذا تجاوز المستخدم حدود المعدل
        """
        start_time = time.time()
        
        try:
            # التحقق من حدود المعدل
            if not self._check_rate_limit(user_id):
                raise RuntimeError("تم تجاوز حدود المعدل. الرجاء المحاولة لاحقاً")
            
            # بناء السياق
            context = ""
            if use_context:
                user_context = await self._get_context(user_id)
                context = user_context.get_context()
            
            # محاكاة توليد الكود
            await asyncio.sleep(0.5)
            
            lang = language or "python"
            generated_code = self._mock_generate_code(prompt, lang, context)
            
            # تحديث السياق
            if use_context:
                user_context = await self._get_context(user_id)
                user_context.add_entry(
                    content=f"Prompt: {prompt}\nGenerated: {generated_code[:100]}...",
                    metadata={"type": "generation", "language": lang}
                )
            
            processing_time = time.time() - start_time
            
            response = AIResponse(
                content=generated_code,
                tokens_used=len(generated_code.split()),
                model=self._default_model,
                confidence=0.92,
                processing_time=processing_time
            )
            
            logger.info(f"تم توليد كود للمستخدم: {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"خطأ في توليد الكود: {e}")
            raise
    
    async def complete_code(
        self,
        user_id: str,
        partial_code: str,
        cursor_position: Optional[int] = None,
        use_context: bool = True
    ) -> AIResponse:
        """
        إكمال كود جزئي
        
        المعاملات:
            user_id: معرف المستخدم
            partial_code: الكود الجزئي
            cursor_position: موقع المؤشر
            use_context: استخدام السياق
            
        العائد:
            AIResponse: الإكمال المقترح
        """
        start_time = time.time()
        
        try:
            if not self._check_rate_limit(user_id):
                raise RuntimeError("تم تجاوز حدود المعدل")
            
            context = ""
            if use_context:
                user_context = await self._get_context(user_id)
                context = user_context.get_context()
            
            await asyncio.sleep(0.3)
            
            completion = self._mock_complete_code(partial_code, cursor_position, context)
            
            if use_context:
                user_context = await self._get_context(user_id)
                user_context.add_entry(
                    content=f"Partial: {partial_code[:50]}...\nCompletion: {completion[:50]}...",
                    metadata={"type": "completion"}
                )
            
            processing_time = time.time() - start_time
            
            return AIResponse(
                content=completion,
                tokens_used=len(completion.split()),
                model=self._default_model,
                confidence=0.88,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"خطأ في إكمال الكود: {e}")
            raise
    
    async def explain_code(
        self,
        user_id: str,
        code: str,
        detail_level: str = "medium",
        use_context: bool = True
    ) -> AIResponse:
        """
        شرح كود معين
        
        المعاملات:
            user_id: معرف المستخدم
            code: الكود المراد شرحه
            detail_level: مستوى التفصيل (low/medium/high)
            use_context: استخدام السياق
            
        العائد:
            AIResponse: الشرح
        """
        start_time = time.time()
        
        try:
            if not self._check_rate_limit(user_id):
                raise RuntimeError("تم تجاوز حدود المعدل")
            
            await asyncio.sleep(0.4)
            
            explanation = self._mock_explain_code(code, detail_level)
            
            if use_context:
                user_context = await self._get_context(user_id)
                user_context.add_entry(
                    content=f"Explained code: {code[:50]}...",
                    metadata={"type": "explanation", "detail": detail_level}
                )
            
            processing_time = time.time() - start_time
            
            return AIResponse(
                content=explanation,
                tokens_used=len(explanation.split()),
                model=self._default_model,
                confidence=0.90,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"خطأ في شرح الكود: {e}")
            raise
    
    async def review_code(
        self,
        user_id: str,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        use_context: bool = True
    ) -> AIResponse:
        """
        مراجعة كود
        
        المعاملات:
            user_id: معرف المستخدم
            code: الكود المراد مراجعته
            language: لغة البرمجة
            focus_areas: مجالات التركيز
            use_context: استخدام السياق
            
        العائد:
            AIResponse: المراجعة والملاحظات
        """
        start_time = time.time()
        
        try:
            if not self._check_rate_limit(user_id):
                raise RuntimeError("تم تجاوز حدود المعدل")
            
            await asyncio.sleep(0.5)
            
            review = self._mock_review_code(code, language, focus_areas)
            
            if use_context:
                user_context = await self._get_context(user_id)
                user_context.add_entry(
                    content=f"Reviewed code: {code[:50]}...",
                    metadata={"type": "review", "focus": focus_areas}
                )
            
            processing_time = time.time() - start_time
            
            return AIResponse(
                content=review,
                tokens_used=len(review.split()),
                model=self._default_model,
                confidence=0.85,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"خطأ في مراجعة الكود: {e}")
            raise
    
    def clear_context(self, user_id: str) -> bool:
        """
        مسح سياق المستخدم
        
        المعاملات:
            user_id: معرف المستخدم
            
        العائد:
            bool: True إذا نجحت العملية
        """
        try:
            if user_id in self._contexts:
                del self._contexts[user_id]
                logger.info(f"تم مسح سياق المستخدم: {user_id}")
            return True
        except Exception as e:
            logger.error(f"خطأ في مسح السياق: {e}")
            return False
    
    async def get_rate_limit_status(self, user_id: str) -> Dict[str, Any]:
        """
        الحصول على حالة حدود المعدل
        
        المعاملات:
            user_id: معرف المستخدم
            
        العائد:
            Dict: معلومات حدود المعدل
        """
        rate_limit = self._get_user_rate_limit(user_id)
        
        return {
            "requests_made": rate_limit.requests,
            "max_requests": rate_limit.max_requests,
            "remaining": rate_limit.max_requests - rate_limit.requests,
            "reset_time": rate_limit.reset_time.isoformat(),
            "window_seconds": rate_limit.window_seconds
        }
    
    # دوال محاكاة للردود
    def _mock_generate_code(self, prompt: str, language: str, context: str) -> str:
        """محاكاة توليد كود"""
        templates = {
            "python": f"""# Generated Python code
# Based on: {prompt[:30]}...

def main():
    # TODO: Implement based on requirements
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""",
            "javascript": f"""// Generated JavaScript code
// Based on: {prompt[:30]}...

function main() {{
    console.log("Hello, World!");
    return 0;
}}

main();
"""
        }
        return templates.get(language, templates["python"])
    
    def _mock_complete_code(
        self,
        partial_code: str,
        cursor_position: Optional[int],
        context: str
    ) -> str:
        """محاكاة إكمال كود"""
        return "    # Auto-completed line\n    pass"
    
    def _mock_explain_code(self, code: str, detail_level: str) -> str:
        """محاكاة شرح كود"""
        explanations = {
            "low": "هذا كود بسيط يقوم بطباعة رسالة.",
            "medium": "هذا الكود يعرف دالة main() تقوم بطباعة 'Hello, World!' ثم تعيد 0.",
            "high": """
الشرح التفصيلي:
1. يتم تعريف دالة main() كنقطة دخول البرنامج
2. تستخدم الدالة print() لطباعة رسالة للمستخدم
3. تعيد الدالة القيمة 0 للإشارة إلى النجاح
4. يتحقق الشرط if __name__ == "__main__" من أن الكود يعمل مباشرة وليس مستورداً
"""
        }
        return explanations.get(detail_level, explanations["medium"])
    
    def _mock_review_code(
        self,
        code: str,
        language: Optional[str],
        focus_areas: Optional[List[str]]
    ) -> str:
        """محاكاة مراجعة كود"""
        return """
مراجعة الكود:

الإيجابيات:
- الهيكل العام جيد
- الأسامي واضحة

الملاحظات:
- يفضل إضافة docstrings للدوال
- يمكن تحسين معالجة الأخطاء
- يُنصح بإضافة اختبارات وحدة

التقييم: 7/10
"""
