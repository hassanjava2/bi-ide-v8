"""
capsule.py — كبسولة الدماغ (Brain Capsule)

كل كبسولة = وحدة مغلقة فيها:
- الموديل (LoRA adapter)
- البيانات (training data)
- الإعدادات (config)
- سجل التعلم (what it knows)
- واجهة تواصل مع كبسولات أخرى

مبادئ:
1. Encapsulation — بيانات + موديل + تدريب = داخل الكبسولة
2. Inheritance — تخصص جديد = يرث من أب (+ أم اختياري)
3. Communication — كل كبسولة تعلن عن معرفتها + تسأل كبسولات أخرى
"""

import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger("brain_capsule")


class BrainCapsule:
    """
    كبسولة دماغ مغلقة — self-contained AI brain unit
    
    كل شي خاص بالدماغ محتوى داخل الكبسولة:
    - LoRA adapter weights
    - بيانات التدريب الخاصة
    - قائمة المعرفة (شنو يعرف)  
    - سجل التعلم والتطور
    - واجهة التواصل مع الباقين
    """
    
    def __init__(
        self,
        capsule_id: str,
        name: str,
        specialty: str,
        base_dir: Path,
        parent_ids: Optional[List[str]] = None,
        temperature: float = 0.7,
    ):
        self.capsule_id = capsule_id
        self.name = name
        self.specialty = specialty
        self.base_dir = base_dir / capsule_id
        self.parent_ids = parent_ids or []
        self.temperature = temperature
        
        # Create capsule directory structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "data").mkdir(exist_ok=True)
        (self.base_dir / "adapter").mkdir(exist_ok=True)
        
        # Load or create config
        self.config_path = self.base_dir / "config.json"
        self.knowledge_path = self.base_dir / "knowledge.json"
        self.learn_log_path = self.base_dir / "learn.log"
        self.eval_path = self.base_dir / "eval.json"
        
        self.config = self._load_or_create_config()
        self.knowledge = self._load_or_create_knowledge()
    
    # ─── Encapsulation ──────────────────────────────────────
    
    def _load_or_create_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        config = {
            "capsule_id": self.capsule_id,
            "name": self.name,
            "specialty": self.specialty,
            "parents": self.parent_ids,
            "temperature": self.temperature,
            "created_at": datetime.now().isoformat(),
            "version": 1,
            "status": "initialized",
            "total_training_samples": 0,
            "evaluation_score": 0.0,
        }
        self.config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        return config
    
    def _load_or_create_knowledge(self) -> dict:
        """ما تعرفه الكبسولة — قائمة المواضيع + مستوى الثقة"""
        if self.knowledge_path.exists():
            return json.loads(self.knowledge_path.read_text())
        knowledge = {
            "capsule_id": self.capsule_id,
            "topics": {},          # topic -> confidence (0-1)
            "languages": ["ar"],
            "can_answer": [],      # أنواع الأسئلة
            "last_updated": datetime.now().isoformat(),
        }
        self.knowledge_path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False))
        return knowledge
    
    def _save_knowledge(self):
        self.knowledge["last_updated"] = datetime.now().isoformat()
        self.knowledge_path.write_text(json.dumps(self.knowledge, indent=2, ensure_ascii=False))
    
    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    # ─── Knowledge Management ────────────────────────────────
    
    def register_knowledge(self, topic: str, confidence: float = 0.5):
        """إعلان: أنا أعرف عن هذا الموضوع"""
        self.knowledge["topics"][topic] = min(confidence, 1.0)
        self._save_knowledge()
        logger.info(f"📝 [{self.name}] أعرف عن '{topic}' بثقة {confidence:.0%}")
    
    def knows_about(self, topic: str) -> float:
        """هل أعرف عن هذا الموضوع؟ يرجع مستوى الثقة"""
        # Exact match
        if topic in self.knowledge["topics"]:
            return self.knowledge["topics"][topic]
        # Partial match
        topic_lower = topic.lower()
        for known_topic, conf in self.knowledge["topics"].items():
            if topic_lower in known_topic.lower() or known_topic.lower() in topic_lower:
                return conf * 0.8  # Partial match = less confident
        return 0.0
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """ملخص — شنو أعرف (للسجل العام)"""
        return {
            "capsule_id": self.capsule_id,
            "name": self.name,
            "specialty": self.specialty,
            "topics": list(self.knowledge["topics"].keys()),
            "topic_count": len(self.knowledge["topics"]),
            "avg_confidence": (
                sum(self.knowledge["topics"].values()) / max(len(self.knowledge["topics"]), 1)
            ),
            "parents": self.parent_ids,
            "status": self.config.get("status", "unknown"),
        }
    
    # ─── Training ────────────────────────────────────────────
    
    def ingest_data(self, data: List[Dict[str, str]], source: str = "unknown"):
        """إضافة بيانات تدريب جديدة للكبسولة"""
        data_file = self.base_dir / "data" / f"ingest_{int(time.time())}.jsonl"
        with open(data_file, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        self.config["total_training_samples"] = self.config.get("total_training_samples", 0) + len(data)
        self._save_config()
        
        # Log
        self._log_learn(f"ingested {len(data)} samples from {source}")
        logger.info(f"📥 [{self.name}] استلم {len(data)} عينة من {source}")
    
    def request_knowledge(self, topic: str) -> Dict[str, Any]:
        """
        ما أعرف عن هذا الموضوع — أحتاج مساعدة!
        يرجع طلب رسمي يتم إرساله عبر الـ Bus
        """
        request = {
            "type": "knowledge_request",
            "from_capsule": self.capsule_id,
            "from_name": self.name,
            "topic": topic,
            "urgency": "normal",
            "timestamp": datetime.now().isoformat(),
        }
        self._log_learn(f"requested knowledge about: {topic}")
        return request
    
    def _log_learn(self, message: str):
        """تسجيل بسجل التعلم"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.learn_log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    # ─── Communication ───────────────────────────────────────
    
    def ask(self, question: str, inference_fn=None) -> Dict[str, Any]:
        """
        اسأل الكبسولة سؤال
        إذا تعرف → تجاوب
        إذا ما تعرف → ترجع طلب معرفة
        """
        # Check if we have knowledge about this topic
        max_confidence = 0.0
        best_topic = None
        for word in question.split():
            conf = self.knows_about(word)
            if conf > max_confidence:
                max_confidence = conf
                best_topic = word
        
        if max_confidence > 0.3 and inference_fn:
            # We know about this — generate response
            try:
                response = inference_fn(self.capsule_id, question)
                return {
                    "status": "answered",
                    "capsule_id": self.capsule_id,
                    "capsule_name": self.name,
                    "response": response,
                    "confidence": max_confidence,
                    "relevant_topic": best_topic,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "capsule_id": self.capsule_id,
                    "error": str(e),
                }
        
        # Don't know — request knowledge
        return {
            "status": "unknown",
            "capsule_id": self.capsule_id,
            "capsule_name": self.name,
            "message": f"ما أعرف عن '{question[:50]}...' — أحتاج بيانات",
            "knowledge_request": self.request_knowledge(question[:100]),
        }
    
    # ─── Inheritance ─────────────────────────────────────────
    
    @classmethod
    def create_child(
        cls,
        child_id: str,
        child_name: str,
        child_specialty: str,
        base_dir: Path,
        parent_capsule: 'BrainCapsule',
        second_parent: Optional['BrainCapsule'] = None,
        temperature: float = 0.7,
    ) -> 'BrainCapsule':
        """
        وراثة — إنشاء كبسولة ترث من أب (+ أم اختياري)
        
        الابن يرث:
        - معرفة الأب/الأم (topics)
        - بيانات التدريب الأساسية
        - ثم يتخصص بمجاله
        """
        parent_ids = [parent_capsule.capsule_id]
        if second_parent:
            parent_ids.append(second_parent.capsule_id)
        
        child = cls(
            capsule_id=child_id,
            name=child_name,
            specialty=child_specialty,
            base_dir=base_dir,
            parent_ids=parent_ids,
            temperature=temperature,
        )
        
        # Inherit knowledge from parent(s) with reduced confidence
        for topic, conf in parent_capsule.knowledge["topics"].items():
            child.register_knowledge(topic, conf * 0.7)  # 70% inherited confidence
        
        if second_parent:
            for topic, conf in second_parent.knowledge["topics"].items():
                existing = child.knows_about(topic)
                # Take the higher confidence if both parents know it
                child.register_knowledge(topic, max(existing, conf * 0.7))
        
        child._log_learn(f"inherited from parents: {parent_ids}")
        child.config["status"] = "inherited"
        child._save_config()
        
        logger.info(f"🧬 [{child_name}] ولدت من {[p.name for p in [parent_capsule] + ([second_parent] if second_parent else [])]}")
        return child
    
    # ─── Status ──────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """حالة الكبسولة"""
        adapter_exists = (self.base_dir / "adapter" / "adapter_config.json").exists()
        data_files = list((self.base_dir / "data").glob("*.jsonl"))
        
        return {
            "capsule_id": self.capsule_id,
            "name": self.name,
            "specialty": self.specialty,
            "parents": self.parent_ids,
            "status": self.config.get("status", "unknown"),
            "has_adapter": adapter_exists,
            "data_files": len(data_files),
            "training_samples": self.config.get("total_training_samples", 0),
            "known_topics": len(self.knowledge.get("topics", {})),
            "evaluation_score": self.config.get("evaluation_score", 0.0),
            "temperature": self.temperature,
        }
