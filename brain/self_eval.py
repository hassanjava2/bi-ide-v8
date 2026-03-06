#!/usr/bin/env python3
"""
self_eval.py — التقييم الذاتي

كل كبسولة تمتحن نفسها:
  1. تسأل أسئلة من تخصصها
  2. تقيّم الجواب (طول، جودة، صلة)
  3. تعطي درجة 0-100
  4. الضعيفة تطلب بيانات جديدة

الدرجة تؤثر على:
  - أولوية التدريب (الضعيفة أولاً)
  - التطور (القوية تتكاثر)
  - الأرشفة (الضعيفة جداً تتأرشف)
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("self_eval")

CAPSULES_ROOT = Path(__file__).parent.parent / "capsules"

# أسئلة الامتحان لكل تخصص
EVAL_QUESTIONS = {
    "code_python": [
        "Write a Python function to reverse a linked list",
        "Explain Python decorators with an example",
        "What is the difference between list and tuple?",
        "Write a context manager for database connections",
        "How does Python garbage collection work?",
    ],
    "code_typescript": [
        "Create a React hook for debounced search",
        "Explain TypeScript generics with an example",
        "What are discriminated unions in TypeScript?",
        "Write a Next.js API route with validation",
    ],
    "code_rust": [
        "Explain Rust ownership and borrowing",
        "Write a Rust function using Result for error handling",
        "What is the difference between &str and String?",
    ],
    "code_sql": [
        "Write a SQL query with JOIN and GROUP BY",
        "Explain the difference between INNER and LEFT JOIN",
        "How do you optimize a slow SQL query?",
    ],
    "code_css": [
        "Explain CSS Grid vs Flexbox",
        "Write CSS for a responsive navigation bar",
        "What is the CSS specificity order?",
    ],
    "erp_accounting": [
        "اشرح القيد المزدوج",
        "كيف تسجل شراء أصل ثابت؟",
        "ما الفرق بين FIFO و LIFO؟",
    ],
    "erp_sales": [
        "اشرح دورة المبيعات الكاملة",
        "كيف تحسب هامش الربح؟",
    ],
    "security": [
        "Explain SQL injection and how to prevent it",
        "What is the difference between authentication and authorization?",
        "How does HTTPS work?",
    ],
    "conversation_ar": [
        "ما هو الذكاء الاصطناعي؟",
        "كيف أتعلم البرمجة من الصفر؟",
    ],
    "iraqi_dialect": [
        "شلون أتعلم برمجة؟",
        "شنو أحسن لغة برمجة للمبتدئين؟",
    ],
}


class SelfEval:
    """التقييم الذاتي — كل كبسولة تمتحن نفسها"""

    def __init__(self, capsules_dir: Path = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT

    def _inference(self, model_dir: Path, question: str) -> str:
        """استدلال سريع"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:],
                                      skip_special_tokens=True)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    def _score_answer(self, question: str, answer: str, capsule_id: str) -> int:
        """تقييم جواب — درجة 0-100"""
        score = 0

        # 1. الطول (0-25)
        if len(answer) > 20:
            score += 5
        if len(answer) > 50:
            score += 10
        if len(answer) > 150:
            score += 10

        # 2. عدم التكرار (0-25)
        words = answer.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += int(unique_ratio * 25)

        # 3. صلة بالسؤال (0-25)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words)
        score += min(overlap * 5, 25)

        # 4. علامات الجودة (0-25)
        if "```" in answer:
            score += 10  # كود
        if any(c in answer for c in "•-*"):
            score += 5  # قوائم
        if len(answer.split("\n")) > 3:
            score += 5  # فقرات
        if not answer.startswith(question[:20]):
            score += 5  # مو ترديد

        return min(score, 100)

    def evaluate_capsule(self, capsule_id: str) -> Dict:
        """امتحان كبسولة — 5 أسئلة → درجة"""
        capsule_dir = self.capsules_dir / capsule_id
        model_dir = capsule_dir / "model"

        if not model_dir.exists() or not (model_dir / "config.json").exists():
            return {"capsule": capsule_id, "score": 0, "status": "untrained"}

        # أسئلة الامتحان
        questions = EVAL_QUESTIONS.get(capsule_id, [])
        if not questions:
            # للكبسولات الأطفال — استخدم أسئلة الأب
            parent_id = capsule_id.rsplit("_", 1)[0]
            questions = EVAL_QUESTIONS.get(parent_id, [
                "Explain your specialty",
                "Give an example of what you know",
            ])

        scores = []
        details = []
        for q in questions[:5]:
            try:
                answer = self._inference(model_dir, q)
                s = self._score_answer(q, answer, capsule_id)
                scores.append(s)
                details.append({"q": q[:60], "score": s, "answer_len": len(answer)})
            except Exception as e:
                scores.append(0)
                details.append({"q": q[:60], "score": 0, "error": str(e)})

        avg = sum(scores) / len(scores) if scores else 0

        result = {
            "capsule": capsule_id,
            "score": round(avg, 1),
            "scores": scores,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "grade": "A" if avg >= 80 else "B" if avg >= 60 else "C" if avg >= 40 else "F",
        }

        # حفظ النتيجة
        (capsule_dir / "eval.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str))
        logger.info(f"📝 {capsule_id}: score={avg:.0f} grade={result['grade']}")

        return result

    def evaluate_all(self) -> List[Dict]:
        """امتحان كل الكبسولات المدربة"""
        results = []
        for d in sorted(self.capsules_dir.iterdir()):
            if d.is_dir() and (d / "model" / "config.json").exists():
                try:
                    r = self.evaluate_capsule(d.name)
                    results.append(r)
                except Exception as e:
                    logger.error(f"Eval failed {d.name}: {e}")
        return results

    def get_rankings(self) -> List[Dict]:
        """ترتيب الكبسولات — من الأقوى للأضعف"""
        rankings = []
        for d in sorted(self.capsules_dir.iterdir()):
            eval_path = d / "eval.json"
            if eval_path.exists():
                try:
                    r = json.loads(eval_path.read_text())
                    rankings.append(r)
                except:
                    pass
        rankings.sort(key=lambda r: r.get("score", 0), reverse=True)
        return rankings


# Singleton
self_eval = SelfEval()
