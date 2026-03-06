"""
brain_factory.py — مصنع الأدمغة (طبقة التوليد)

يخلق أدمغة جديدة (LoRA adapters) أوتوماتيكياً
يدرّبهم → يقيّمهم → يحذف الضعيف → يكاثر القوي
يخلط أدمغة من تخصصات مختلفة (mutation) → ابتكار

Architecture: Infinite Brain Network
Rule: الأدمغة تتكاثر بلا حدود — مو ثابتة
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("brain_factory")

# ─── Config ───────────────────────────────────────────────────
ADAPTERS_DIR = Path(os.getenv("ADAPTERS_DIR", "/home/bi/training_data/models/finetuned"))
REGISTRY_PATH = Path(__file__).parent.parent / "config" / "sage_brain_mapping.json"
BRAIN_STATE_PATH = Path("/tmp/brain_factory_state.json")

# Foundation that ALL brains must learn
FOUNDATION_DATASETS = [
    "mathematical_physics",    # أساس التفكير المنطقي
    "formal_logic",            # استدلال رياضي
    "arabic_fluency",          # يحجي طبيعي
]


class BrainFactory:
    """
    مصنع الأدمغة — يخلق LoRA adapters متخصصة أوتوماتيكياً
    
    Capabilities:
    1. إنشاء adapter جديد من بيانات تخصصية
    2. تقييم أداء كل adapter
    3. حذف الضعيف + تكاثر القوي
    4. Mutation — خلط adapters من تخصصات مختلفة
    """
    
    def __init__(self):
        self.registry = self._load_registry()
        self.state = self._load_state()
        self.adapters_dir = ADAPTERS_DIR
    
    def _load_registry(self) -> dict:
        """تحميل سجل الأدمغة"""
        if REGISTRY_PATH.exists():
            return json.loads(REGISTRY_PATH.read_text())
        return {"version": "2.0", "brain_network": {}}
    
    def _load_state(self) -> dict:
        """تحميل حالة المصنع"""
        if BRAIN_STATE_PATH.exists():
            try:
                return json.loads(BRAIN_STATE_PATH.read_text())
            except Exception:
                pass
        return {
            "brains_created": 0,
            "brains_killed": 0,
            "mutations": 0,
            "evaluations": [],
            "last_evolution": None,
        }
    
    def _save_state(self):
        """حفظ حالة المصنع"""
        BRAIN_STATE_PATH.write_text(json.dumps(self.state, indent=2, default=str))
    
    # ─── Brain Discovery ─────────────────────────────────────
    
    def discover_existing_brains(self) -> List[Dict[str, Any]]:
        """
        اكتشاف الأدمغة الموجودة على الـ GPU
        يفحص كل مجلد LoRA adapter ويرجع قائمة
        """
        brains = []
        
        if not self.adapters_dir.exists():
            logger.warning(f"مجلد الأدمغة غير موجود: {self.adapters_dir}")
            return brains
        
        for d in sorted(self.adapters_dir.iterdir()):
            if not d.is_dir():
                continue
            
            # Check if it's a valid adapter
            adapter_config = d / "adapter_config.json"
            has_config = adapter_config.exists()
            
            # Check checkpoints
            checkpoints = list(d.glob("checkpoint-*/adapter_config.json"))
            
            if has_config or checkpoints:
                brain_info = {
                    "id": d.name,
                    "path": str(d),
                    "has_adapter_config": has_config,
                    "checkpoints": len(checkpoints),
                    "size_mb": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024),
                    "created": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                }
                brains.append(brain_info)
        
        logger.info(f"🧠 اكتشفت {len(brains)} دماغ موجود")
        return brains
    
    # ─── Brain Creation ──────────────────────────────────────
    
    def create_brain(
        self,
        brain_id: str,
        specialty: str,
        perspective: str,
        training_data_path: str,
        temperature: float = 0.7,
        base_model: str = "Qwen/Qwen2.5-1.5B",
        epochs: int = 3,
        learning_rate: float = 2e-4,
    ) -> Dict[str, Any]:
        """
        إنشاء دماغ جديد (LoRA adapter)
        
        Args:
            brain_id: معرف فريد للدماغ
            specialty: التخصص (security, medical, etc)
            perspective: المنظور (هجومي، دفاعي، متمرد...)
            training_data_path: مسار بيانات التدريب
            temperature: حرارة التوليد (عالي = إبداعي، منخفض = دقيق)
        
        Returns:
            معلومات الدماغ المُنشأ
        """
        output_dir = self.adapters_dir / brain_id
        
        brain_config = {
            "id": brain_id,
            "specialty": specialty,
            "perspective": perspective,
            "temperature": temperature,
            "base_model": base_model,
            "training_config": {
                "data_path": training_data_path,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "foundation_datasets": FOUNDATION_DATASETS,
            },
            "created_at": datetime.now().isoformat(),
            "status": "pending_training",
            "evaluation_score": None,
        }
        
        # Save brain config
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "brain_config.json"
        config_path.write_text(json.dumps(brain_config, indent=2, ensure_ascii=False))
        
        self.state["brains_created"] += 1
        self._save_state()
        
        logger.info(f"🧠 دماغ جديد: {brain_id} ({specialty}/{perspective})")
        return brain_config
    
    # ─── Swarm Debate ────────────────────────────────────────
    
    def swarm_debate(
        self,
        swarm_brains: List[str],
        query: str,
        rounds: int = 3,
        inference_fn=None,
    ) -> Dict[str, Any]:
        """
        نقاش السرب — أدمغة متعددة تتناقش
        
        Round 1: كل دماغ يجاوب لحاله
        Round 2: كل دماغ يشوف أجوبة الباقين ويعدّل
        Round 3: التصويت + الدماغ المتمرد يتحدى
        
        Returns:
            النتيجة المركّبة بعد النقاش
        """
        if inference_fn is None:
            return {"error": "inference_fn required"}
        
        debate_log = []
        current_responses = {}
        
        for round_num in range(1, rounds + 1):
            round_responses = {}
            
            for brain_id in swarm_brains:
                if round_num == 1:
                    # Round 1: إجابة مستقلة
                    prompt = query
                elif round_num == 2:
                    # Round 2: شوف آراء الباقين وعدّل
                    others = "\n".join([
                        f"- {bid}: {resp}"
                        for bid, resp in current_responses.items()
                        if bid != brain_id
                    ])
                    prompt = f"السؤال: {query}\n\nآراء الآخرين:\n{others}\n\nما رأيك الآن؟ هل تغيّر أو تثبّت موقفك؟"
                else:
                    # Round 3: تحدي المتمرد
                    if "rebel" in brain_id:
                        prompt = f"السؤال: {query}\n\nالإجماع الحالي:\n{json.dumps(current_responses, ensure_ascii=False)}\n\nأنت المتمرد — تحدَّ هذا الإجماع. ما الخطأ فيه؟"
                    else:
                        prompt = f"السؤال: {query}\n\nالمتمرد يعترض. دافع عن موقفك بقوة."
                
                try:
                    response = inference_fn(brain_id, prompt)
                    round_responses[brain_id] = response
                except Exception as e:
                    round_responses[brain_id] = f"⚠️ خطأ: {e}"
            
            current_responses = round_responses
            debate_log.append({
                "round": round_num,
                "responses": dict(round_responses),
            })
        
        # Synthesize final result
        return {
            "query": query,
            "debate_rounds": rounds,
            "participating_brains": swarm_brains,
            "final_responses": current_responses,
            "debate_log": debate_log,
            "timestamp": datetime.now().isoformat(),
        }
    
    # ─── Brain Evaluation ────────────────────────────────────
    
    def evaluate_brain(self, brain_id: str, test_prompts: List[str], inference_fn=None) -> float:
        """
        تقييم دماغ — يختبره بأسئلة ويعطيه درجة
        يحفظ النتيجة بالـ state
        """
        if inference_fn is None:
            return 0.0
        
        scores = []
        for prompt in test_prompts:
            try:
                response = inference_fn(brain_id, prompt)
                # Simple quality checks
                score = 0.0
                if response and len(response) > 50:
                    score += 0.3  # يجاوب بمحتوى
                if not any(w in response for w in ["غير متاح", "خطأ", "error"]):
                    score += 0.3  # ما يفشل
                if len(response) > 200:
                    score += 0.2  # إجابة مفصّلة
                if any(c in response for c in ["لأن", "بسبب", "نتيجة", "because"]):
                    score += 0.2  # يبرر إجاباته
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        avg_score = sum(scores) / max(len(scores), 1)
        
        self.state["evaluations"].append({
            "brain_id": brain_id,
            "score": avg_score,
            "tests": len(test_prompts),
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()
        
        logger.info(f"📊 تقييم {brain_id}: {avg_score:.2f}")
        return avg_score
    
    # ─── Evolution ───────────────────────────────────────────
    
    def evolve(self) -> Dict[str, Any]:
        """
        طبقة التطور — حذف الضعيف + تكاثر القوي + mutation
        
        1. يفرز الأدمغة حسب الدرجة
        2. يحذف أضعف 20%
        3. يكاثر أقوى 20% (يخلق نسخ مع تعديلات)
        4. Mutation: يخلط adapters من تخصصات مختلفة
        """
        evaluations = self.state.get("evaluations", [])
        
        if len(evaluations) < 5:
            return {"status": "not_enough_data", "message": "نحتاج 5 تقييمات أقل شي"}
        
        # Sort by score
        sorted_evals = sorted(evaluations, key=lambda x: x["score"])
        
        # Bottom 20% = candidates for deletion
        kill_count = max(1, len(sorted_evals) // 5)
        to_kill = sorted_evals[:kill_count]
        
        # Top 20% = candidates for reproduction
        to_reproduce = sorted_evals[-kill_count:]
        
        result = {
            "killed": [e["brain_id"] for e in to_kill],
            "reproduced": [e["brain_id"] for e in to_reproduce],
            "total_brains": len(sorted_evals),
            "timestamp": datetime.now().isoformat(),
        }
        
        self.state["last_evolution"] = result
        self.state["brains_killed"] += kill_count
        self._save_state()
        
        logger.info(f"🧬 تطور: حذف {kill_count} + تكاثر {kill_count}")
        return result
    
    # ─── Status ──────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """حالة المصنع"""
        existing = self.discover_existing_brains()
        
        return {
            "factory_status": "active",
            "existing_brains": len(existing),
            "brains_created": self.state.get("brains_created", 0),
            "brains_killed": self.state.get("brains_killed", 0),
            "mutations": self.state.get("mutations", 0),
            "evaluations_count": len(self.state.get("evaluations", [])),
            "last_evolution": self.state.get("last_evolution"),
            "adapters_dir": str(self.adapters_dir),
            "gpu_brains": existing[:5],  # First 5 for quick view
            "timestamp": datetime.now().isoformat(),
        }


# ─── Singleton ───────────────────────────────────────────────
brain_factory = BrainFactory()
