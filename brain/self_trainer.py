"""
self_trainer.py — التدريب الذاتي (الكبسولات تدرّب بعضها)

المبدأ التاسع: النقاش بين الكبسولات = بيانات تدريب جديدة

الآلية:
  1. كبسولة A تسأل كبسولة B سؤال صعب
  2. B تجاوب
  3. C (المتمرد) يقيّم: "هل الجواب صح؟"
  4. إذا نجح الفلاتر الخمسة → يصير بيانات تدريب
  5. الكبسولات تتدرب على النقاشات المفلترة
  = بيانات لانهائية بدون مصادر خارجية
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger("self_trainer")


class SelfTrainer:
    """
    التدريب الذاتي — كبسولات تدرّب بعضها عبر النقاش
    
    كل جلسة نقاش تمر بـ 3 مراحل:
    1. السؤال: كبسولة تولّد سؤال صعب
    2. الجواب: كبسولة ثانية تجاوب
    3. التقييم: المتمرد + الفلاتر يقيّمون
    
    فقط النقاشات اللي تنجح من الفلاتر = بيانات تدريب
    """
    
    def __init__(self, hallucination_guard=None):
        self.guard = hallucination_guard
        self.sessions: List[Dict[str, Any]] = []
        self.approved_data: List[Dict[str, str]] = []  # Filtered training data
        self.rejected_count = 0
    
    # ─── Question Generation ─────────────────────────────────
    
    def generate_question(
        self,
        topic: str,
        difficulty: str = "hard",
        inference_fn: Optional[Callable] = None,
    ) -> str:
        """
        توليد سؤال صعب عن موضوع معين
        
        إذا ما اكو inference_fn → يستخدم قوالب أساسية
        """
        if inference_fn:
            prompt = f"ولّد سؤال {difficulty} عن {topic}. السؤال لازم يحتاج تفكير عميق. فقط السؤال بدون جواب."
            try:
                return inference_fn("question_generator", prompt)
            except Exception:
                pass
        
        # Fallback: template-based questions
        templates = [
            f"ما هي أفضل طريقة لتطبيق {topic} في مشروع حقيقي؟ وليش؟",
            f"ما هي أكبر المخاطر في {topic}؟ وكيف نتجنبها؟",
            f"قارن بين أفضل 3 طرق في {topic}. أيهن أفضل ولماذا؟",
            f"لو عندك مشروع يحتاج {topic} — شلون تبدي من الصفر؟",
            f"ما هو أكبر خطأ شائع في {topic}؟ وشلون نتجنبه؟",
        ]
        
        import random
        return random.choice(templates)
    
    # ─── Debate Session ──────────────────────────────────────
    
    def run_debate_session(
        self,
        topic: str,
        answerer_fn: Optional[Callable] = None,
        rebel_fn: Optional[Callable] = None,
        num_rounds: int = 3,
    ) -> Dict[str, Any]:
        """
        جلسة نقاش كاملة:
        
        1. ولّد أسئلة عن الموضوع
        2. الكبسولة المختصة تجاوب
        3. المتمرد يقيّم ويتحدى
        4. الفلاتر تمرر أو ترفض
        5. المقبول → بيانات تدريب
        """
        session = {
            "topic": topic,
            "rounds": [],
            "approved": 0,
            "rejected": 0,
            "started_at": datetime.now().isoformat(),
        }
        
        for round_num in range(num_rounds):
            # 1. Generate question
            question = self.generate_question(topic, inference_fn=answerer_fn)
            
            # 2. Get answer
            answer = ""
            if answerer_fn:
                try:
                    answer = answerer_fn("answerer", question)
                except Exception as e:
                    answer = f"⚠️ فشل: {e}"
            
            if not answer or "فشل" in answer or "خطأ" in answer:
                session["rounds"].append({
                    "round": round_num + 1,
                    "question": question,
                    "answer": answer,
                    "status": "failed",
                })
                session["rejected"] += 1
                continue
            
            # 3. Rebel evaluation
            rebel_verdict = "approved"
            rebel_comment = ""
            if rebel_fn:
                try:
                    rebel_response = rebel_fn(
                        "rebel",
                        f"قيّم هذا الجواب. هل هو صحيح ومنطقي؟\n\nالسؤال: {question}\nالجواب: {answer}\n\nاكتب: مقبول أو مرفوض + السبب"
                    )
                    if "مرفوض" in rebel_response.lower() or "rejected" in rebel_response.lower():
                        rebel_verdict = "rejected"
                    rebel_comment = rebel_response
                except Exception:
                    rebel_verdict = "no_rebel"
            
            # 4. Hallucination guard
            guard_passed = True
            if self.guard and rebel_verdict != "rejected":
                guard_result = self.guard.check(question, answer)
                guard_passed = guard_result.get("approved", False)
                if not guard_passed:
                    rebel_verdict = "filtered"
            
            # 5. Record result
            round_data = {
                "round": round_num + 1,
                "question": question,
                "answer": answer,
                "rebel_verdict": rebel_verdict,
                "rebel_comment": rebel_comment[:200],
                "guard_passed": guard_passed,
                "status": "approved" if rebel_verdict == "approved" and guard_passed else "rejected",
            }
            session["rounds"].append(round_data)
            
            if round_data["status"] == "approved":
                # Convert to training data
                training_sample = {
                    "instruction": question,
                    "output": answer,
                    "source": "self_training",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                }
                self.approved_data.append(training_sample)
                session["approved"] += 1
            else:
                session["rejected"] += 1
                self.rejected_count += 1
        
        session["finished_at"] = datetime.now().isoformat()
        self.sessions.append(session)
        
        logger.info(
            f"📚 جلسة نقاش '{topic}': "
            f"{session['approved']} مقبول / {session['rejected']} مرفوض"
        )
        
        return session
    
    # ─── Export Training Data ────────────────────────────────
    
    def export_training_data(self, output_path: str) -> Dict[str, Any]:
        """
        تصدير البيانات المعتمدة لملف JSONL للتدريب
        """
        if not self.approved_data:
            return {"status": "empty", "message": "ما اكو بيانات معتمدة بعد"}
        
        from pathlib import Path
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            for sample in self.approved_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        return {
            "status": "exported",
            "path": str(path),
            "total_samples": len(self.approved_data),
            "total_sessions": len(self.sessions),
            "rejection_rate": f"{self.rejected_count / max(self.rejected_count + len(self.approved_data), 1):.0%}",
        }
    
    # ─── Status ──────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "total_sessions": len(self.sessions),
            "approved_samples": len(self.approved_data),
            "rejected_count": self.rejected_count,
            "rejection_rate": f"{self.rejected_count / max(self.rejected_count + len(self.approved_data), 1):.0%}",
            "timestamp": datetime.now().isoformat(),
        }
