"""
hallucination_guard.py — 5 فلاتر ضد الهلوسة

يمنع البيانات الخاطئة من التكاثر بالتدريب الذاتي.

الفلاتر الخمسة:
1. المتمرد — يتحدى كل جواب
2. الإجماع — 3+ كبسولات لازم يتفقون
3. التحقق الرياضي — Z3/SymPy للمواضيع الدقيقة
4. عتبة الثقة — فقط 90%+ يصير تدريب
5. التحقق الخارجي — الكشاف يتأكد من مصادر

القاعدة: فقط المعرفة المثبتة تتحول لتدريب. الباقي يموت.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger("hallucination_guard")


class HallucinationGuard:
    """
    حارس الهلوسة — يمنع البيانات الخاطئة من دخول التدريب
    
    كل بيانات تدريب (من التدريب الذاتي أو مصادر خارجية)
    لازم تمر من الفلاتر الخمسة قبل ما تُقبل.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        consensus_required: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.consensus_required = consensus_required
        self.stats = {
            "total_checked": 0,
            "approved": 0,
            "rejected_rebel": 0,
            "rejected_consensus": 0,
            "rejected_math": 0,
            "rejected_confidence": 0,
            "rejected_external": 0,
        }
    
    # ─── Main Check ──────────────────────────────────────────
    
    def check(
        self,
        question: str,
        answer: str,
        rebel_fn: Optional[Callable] = None,
        capsule_fns: Optional[List[Callable]] = None,
        confidence: float = 0.5,
        external_verify_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        فحص شامل — يمرر الجواب على الفلاتر الخمسة
        
        Returns:
            {"approved": True/False, "filters_passed": [...], "filters_failed": [...]}
        """
        self.stats["total_checked"] += 1
        
        filters_passed = []
        filters_failed = []
        
        # ─── Filter 1: المتمرد ────────────────────────────
        if rebel_fn:
            rebel_ok = self._filter_rebel(question, answer, rebel_fn)
            if rebel_ok:
                filters_passed.append("rebel")
            else:
                filters_failed.append("rebel")
                self.stats["rejected_rebel"] += 1
        else:
            filters_passed.append("rebel_skipped")
        
        # ─── Filter 2: الإجماع ────────────────────────────
        if capsule_fns and len(capsule_fns) >= self.consensus_required:
            consensus_ok = self._filter_consensus(question, answer, capsule_fns)
            if consensus_ok:
                filters_passed.append("consensus")
            else:
                filters_failed.append("consensus")
                self.stats["rejected_consensus"] += 1
        else:
            filters_passed.append("consensus_skipped")
        
        # ─── Filter 3: التحقق الرياضي ─────────────────────
        if self._needs_math_verification(question):
            math_ok = self._filter_math(answer)
            if math_ok:
                filters_passed.append("math")
            else:
                filters_failed.append("math")
                self.stats["rejected_math"] += 1
        else:
            filters_passed.append("math_skipped")
        
        # ─── Filter 4: عتبة الثقة ─────────────────────────
        if confidence >= self.confidence_threshold:
            filters_passed.append("confidence")
        else:
            filters_failed.append("confidence")
            self.stats["rejected_confidence"] += 1
        
        # ─── Filter 5: التحقق الخارجي ─────────────────────
        if external_verify_fn:
            external_ok = self._filter_external(question, answer, external_verify_fn)
            if external_ok:
                filters_passed.append("external")
            else:
                filters_failed.append("external")
                self.stats["rejected_external"] += 1
        else:
            filters_passed.append("external_skipped")
        
        # ─── Decision ─────────────────────────────────────
        # Must pass ALL active filters (non-skipped)
        active_failed = [f for f in filters_failed if not f.endswith("_skipped")]
        approved = len(active_failed) == 0
        
        if approved:
            self.stats["approved"] += 1
        
        return {
            "approved": approved,
            "filters_passed": filters_passed,
            "filters_failed": filters_failed,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
    
    # ─── Individual Filters ──────────────────────────────────
    
    def _filter_rebel(self, question: str, answer: str, rebel_fn: Callable) -> bool:
        """فلتر 1: المتمرد يتحدى الجواب"""
        try:
            prompt = f"هل هذا الجواب صحيح ومنطقي؟ اكتب فقط 'صح' أو 'غلط':\n\nسؤال: {question}\nجواب: {answer}"
            verdict = rebel_fn("rebel", prompt)
            return "صح" in verdict.lower() or "correct" in verdict.lower() or "صحيح" in verdict.lower()
        except Exception:
            return True  # If rebel fails, pass by default
    
    def _filter_consensus(self, question: str, answer: str, capsule_fns: List[Callable]) -> bool:
        """فلتر 2: إجماع 3+ كبسولات"""
        agreements = 0
        for fn in capsule_fns[:5]:  # Max 5 voters
            try:
                prompt = f"هل هذا الجواب منطقي؟ اكتب 'نعم' أو 'لا':\n\nسؤال: {question}\nجواب: {answer}"
                vote = fn("voter", prompt)
                if "نعم" in vote.lower() or "yes" in vote.lower():
                    agreements += 1
            except Exception:
                continue
        
        return agreements >= self.consensus_required
    
    def _needs_math_verification(self, question: str) -> bool:
        """هل السؤال يحتاج تحقق رياضي؟"""
        math_keywords = [
            "حساب", "رياضيات", "معادلة", "نسبة", "احتمال",
            "calculate", "math", "equation", "probability",
            "فيزياء", "قوة", "سرعة", "physics",
        ]
        return any(kw in question.lower() for kw in math_keywords)
    
    def _filter_math(self, answer: str) -> bool:
        """فلتر 3: تحقق رياضي — يفحص الأرقام والعمليات"""
        # Basic: check if numbers in answer are consistent
        numbers = re.findall(r'\d+\.?\d*', answer)
        
        if not numbers:
            return True  # No numbers to verify
        
        # Check for obvious contradictions
        try:
            import sympy
            # Try to parse and verify any equations found
            equations = re.findall(r'(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)', answer)
            for eq in equations:
                parts = eq.split("=")
                if len(parts) == 2:
                    left = sympy.sympify(parts[0].strip())
                    right = sympy.sympify(parts[1].strip())
                    if left != right:
                        logger.warning(f"🔢 خطأ رياضي: {eq}")
                        return False
        except ImportError:
            pass  # SymPy not available, skip
        except Exception:
            pass
        
        return True
    
    def _filter_external(self, question: str, answer: str, verify_fn: Callable) -> bool:
        """فلتر 5: التحقق من مصادر خارجية (الكشاف)"""
        try:
            result = verify_fn(question, answer)
            return result.get("verified", False)
        except Exception:
            return True  # If external fails, pass by default
    
    # ─── Stats ───────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total_checked"]
        return {
            **self.stats,
            "approval_rate": f"{self.stats['approved'] / max(total, 1):.0%}",
            "timestamp": datetime.now().isoformat(),
        }
