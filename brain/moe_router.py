#!/usr/bin/env python3
"""
moe_router.py — Mixture of Experts Router 🧠⚡

بدل موديل واحد كبير → 8+ موديلات صغيرة متخصصة:
  - كل سؤال → الخبير المناسب يجاوب
  - أسرع + أدق من موديل واحد كبير
  - يحمل فقط الموديل المطلوب (RAM efficient)

الهيكل:
  Gate Network → يختار الخبير
  Expert 1: code_python
  Expert 2: code_typescript
  Expert 3: security
  Expert 4: devops
  Expert 5: knowledge_arabic
  Expert 6: conversation_ar
  Expert 7: code_sql
  Expert 8: code_css
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("moe")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Expert Definition
# ═══════════════════════════════════════════════════════════

@dataclass
class Expert:
    """خبير واحد"""
    expert_id: str
    name: str
    capsule_id: str
    keywords: List[str]
    weight: float = 1.0           # وزن — يزيد مع النجاح
    queries_handled: int = 0
    avg_confidence: float = 0.5
    model_loaded: bool = False
    model_size_mb: float = 0


@dataclass
class MoEDecision:
    """قرار MoE"""
    query: str
    top_expert: str
    all_scores: Dict[str, float]
    confidence: float
    response: str = ""
    time_ms: float = 0


# ═══════════════════════════════════════════════════════════
# Gate Network — شبكة البوابة
# ═══════════════════════════════════════════════════════════

class GateNetwork:
    """
    شبكة البوابة — تختار الخبير المناسب

    Phase 1: Rule-based (keyword matching)
    Phase 2: Pattern-based (after Cycle 10+)
    Phase 3: Neural gate (after Cycle 50+)
    """

    def __init__(self, experts: Dict[str, Expert]):
        self.experts = experts

    def route(self, query: str) -> List[Tuple[str, float]]:
        """
        توجيه السؤال للخبير المناسب

        Returns:
            [(expert_id, score), ...] مرتب من الأعلى
        """
        query_lower = query.lower()
        scores = {}

        for eid, expert in self.experts.items():
            score = 0

            # مطابقة كلمات مفتاحية
            for kw in expert.keywords:
                if kw in query_lower:
                    score += 3

            # وزن الخبير
            score *= expert.weight

            # مكافأة الي جاوب أكثر بثقة عالية
            if expert.avg_confidence > 0.7:
                score *= 1.2

            scores[eid] = score

        # ترتيب
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores


# ═══════════════════════════════════════════════════════════
# MoE Router
# ═══════════════════════════════════════════════════════════

class MoERouter:
    """
    Mixture of Experts Router

    يوجّه كل سؤال للخبير المناسب:
      1. Gate Network يسجّل النقاط
      2. يختار أعلى 1-2 خبير
      3. يحمل الموديل المتدرب
      4. يجاوب
      5. يتعلم من النتيجة (يحدّث الأوزان)
    """

    def __init__(self):
        self.experts: Dict[str, Expert] = {}
        self._init_experts()
        self.gate = GateNetwork(self.experts)
        self.history: List[MoEDecision] = []
        logger.info(f"🧠⚡ MoE Router: {len(self.experts)} experts")

    def _init_experts(self):
        """تهيئة الخبراء"""
        expert_defs = [
            ("python", "Python Expert", "code_python",
             ["python", "py", "pip", "django", "flask", "fastapi", "pandas", "numpy",
              "def ", "class ", "import ", "pytorch", "tensorflow"]),
            ("typescript", "TypeScript Expert", "code_typescript",
             ["typescript", "javascript", "react", "next", "vue", "angular", "node",
              "npm", "tsx", "jsx", "component", "frontend", "ui"]),
            ("security", "Security Expert", "security",
             ["security", "أمن", "hack", "vulnerability", "cve", "exploit", "firewall",
              "encrypt", "password", "ssl", "tls", "auth", "xss", "injection"]),
            ("devops", "DevOps Expert", "devops",
             ["docker", "kubernetes", "k8s", "deploy", "ci/cd", "server", "nginx",
              "linux", "ssh", "cloud", "aws", "terraform", "ansible", "yaml"]),
            ("arabic", "Arabic Knowledge", "knowledge_arabic",
             ["عربي", "arabic", "عراق", "تاريخ", "ثقافة", "أدب", "إسلام",
              "شنو", "شلون", "ليش", "هسه", "اريد", "سوي"]),
            ("conversation", "Conversation Expert", "conversation_ar",
             ["مرحبا", "شلونك", "كيف", "شكرا", "hello", "hi", "chat",
              "tell me", "explain", "help", "why"]),
            ("sql", "Database Expert", "code_sql",
             ["sql", "database", "query", "select", "insert", "table", "join",
              "postgresql", "mysql", "migration", "schema", "بيانات"]),
            ("css", "Design Expert", "code_css",
             ["css", "style", "design", "color", "font", "layout", "flexbox",
              "grid", "responsive", "animation", "تصميم", "ألوان"]),
        ]

        for eid, name, capsule, keywords in expert_defs:
            # فحص الموديل
            model_dir = CAPSULES_ROOT / capsule / "model"
            has_model = model_dir.exists() and any(model_dir.iterdir()) if model_dir.exists() else False
            model_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024*1024) if has_model else 0

            self.experts[eid] = Expert(
                expert_id=eid,
                name=name,
                capsule_id=capsule,
                keywords=keywords,
                model_loaded=has_model,
                model_size_mb=round(model_size, 1),
            )

    def route(self, query: str, top_k: int = 2) -> MoEDecision:
        """
        توجيه السؤال

        Args:
            query: السؤال
            top_k: عدد الخبراء (1-3)
        """
        start = time.time()

        # Gate يسجّل النقاط
        scores = self.gate.route(query)
        all_scores = {eid: score for eid, score in scores}

        # أعلى خبير
        top = scores[0] if scores else ("python", 0)
        top_expert_id = top[0]
        top_expert = self.experts.get(top_expert_id)

        # الاستجابة
        response = ""
        confidence = 0.5

        if top_expert and top_expert.model_loaded:
            # === استخدام الموديل المتدرب ===
            try:
                from brain.chat_bridge import bridge
                result = bridge._query_capsule(top_expert.capsule_id, query)
                if result:
                    response = result.response
                    confidence = result.confidence
            except Exception:
                pass

        if not response:
            response = f"[{top_expert.name if top_expert else 'Unknown'}]: Processing '{query[:50]}...'"
            confidence = 0.3

        elapsed = (time.time() - start) * 1000

        decision = MoEDecision(
            query=query,
            top_expert=top_expert_id,
            all_scores=all_scores,
            confidence=confidence,
            response=response,
            time_ms=elapsed,
        )

        # تحديث الخبير
        if top_expert:
            top_expert.queries_handled += 1
            top_expert.avg_confidence = (
                top_expert.avg_confidence * 0.9 + confidence * 0.1
            )

        self.history.append(decision)
        return decision

    def feedback(self, query: str, was_good: bool):
        """تعلم من ردود الفعل — يحدّث أوزان الخبراء"""
        if self.history:
            last = self.history[-1]
            expert = self.experts.get(last.top_expert)
            if expert:
                if was_good:
                    expert.weight = min(expert.weight * 1.05, 3.0)
                else:
                    expert.weight = max(expert.weight * 0.95, 0.3)

    def get_status(self) -> Dict:
        """حالة MoE"""
        return {
            "total_experts": len(self.experts),
            "experts_with_models": sum(1 for e in self.experts.values() if e.model_loaded),
            "total_queries": sum(e.queries_handled for e in self.experts.values()),
            "experts": [
                {
                    "id": e.expert_id,
                    "name": e.name,
                    "capsule": e.capsule_id,
                    "has_model": e.model_loaded,
                    "model_mb": e.model_size_mb,
                    "weight": round(e.weight, 2),
                    "queries": e.queries_handled,
                    "confidence": round(e.avg_confidence, 2),
                }
                for e in self.experts.values()
            ],
        }


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

moe = MoERouter()


if __name__ == "__main__":
    print("🧠⚡ Mixture of Experts — Test\n")

    status = moe.get_status()
    print(f"Experts: {status['total_experts']} ({status['experts_with_models']} with models)\n")

    for e in status["experts"]:
        model_icon = "✅" if e["has_model"] else "⏳"
        print(f"  {model_icon} {e['name']:20s} weight={e['weight']:.2f} model={e['model_mb']:.0f}MB")

    print(f"\n{'═' * 50}\nTest Queries:\n")

    queries = [
        "كيف أسوي API بـ FastAPI؟",
        "شنو XSS vulnerability؟",
        "سوي Docker compose",
        "شلون أصمم button",
        "شلونك؟",
        "SELECT * FROM users",
    ]

    for q in queries:
        d = moe.route(q)
        print(f"  Q: {q}")
        print(f"  → Expert: {d.top_expert} (confidence: {d.confidence:.0%})")
        print(f"  → Top 3: {list(d.all_scores.items())[:3]}")
        print()
