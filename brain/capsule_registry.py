"""
capsule_registry.py — سجل الكبسولات (Knowledge Directory)

كل كبسولة تعلن عن معرفتها هنا.
لما كبسولة تحتاج مساعدة → تبحث بالسجل: "منو يعرف عن [الموضوع]؟"

الغرض:
- اكتشاف الكبسولات المتاحة
- البحث عن أفضل كبسولة لموضوع معين
- تتبع حالة كل الكبسولات
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("capsule_registry")


class CapsuleRegistry:
    """
    سجل مركزي — كل كبسولة تسجل نفسها + معرفتها
    
    مثل DNS بس للأدمغة:
    "أريد أحد يعرف عن التشفير" → "capsule-crypto بثقة 92%"
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("/tmp/capsule_registry.json")
        self.capsules: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self):
        if self.registry_path.exists():
            try:
                self.capsules = json.loads(self.registry_path.read_text())
            except Exception:
                self.capsules = {}
    
    def _save(self):
        self.registry_path.write_text(
            json.dumps(self.capsules, indent=2, ensure_ascii=False, default=str)
        )
    
    # ─── Registration ────────────────────────────────────────
    
    def register(self, capsule_summary: Dict[str, Any]):
        """كبسولة تسجل نفسها بالسجل"""
        capsule_id = capsule_summary["capsule_id"]
        self.capsules[capsule_id] = {
            **capsule_summary,
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
        }
        self._save()
        logger.info(f"📋 [{capsule_summary['name']}] مسجلة — تعرف {capsule_summary.get('topic_count', 0)} مواضيع")
    
    def unregister(self, capsule_id: str):
        """إزالة كبسولة من السجل"""
        if capsule_id in self.capsules:
            name = self.capsules[capsule_id].get("name", capsule_id)
            del self.capsules[capsule_id]
            self._save()
            logger.info(f"🗑️ [{name}] أُزيلت من السجل")
    
    def heartbeat(self, capsule_id: str):
        """نبضة — الكبسولة لسه شغالة"""
        if capsule_id in self.capsules:
            self.capsules[capsule_id]["last_heartbeat"] = datetime.now().isoformat()
            self._save()
    
    # ─── Discovery ───────────────────────────────────────────
    
    def find_expert(self, topic: str, min_confidence: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        بحث: "منو يعرف عن [الموضوع]؟"
        
        Returns: [(capsule_id, capsule_name, confidence), ...]
        مرتب من الأعلى ثقة للأقل
        """
        results = []
        topic_lower = topic.lower()
        
        for cid, info in self.capsules.items():
            topics = info.get("topics", [])
            best_match = 0.0
            
            for known_topic in topics:
                if topic_lower in known_topic.lower() or known_topic.lower() in topic_lower:
                    # Topic match — use average confidence
                    best_match = max(best_match, info.get("avg_confidence", 0.5))
                elif topic_lower == known_topic.lower():
                    best_match = max(best_match, info.get("avg_confidence", 0.8))
            
            # Also check specialty
            if topic_lower in info.get("specialty", "").lower():
                best_match = max(best_match, 0.7)
            
            if best_match >= min_confidence:
                results.append((cid, info.get("name", cid), best_match))
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def find_by_specialty(self, specialty: str) -> List[Dict[str, Any]]:
        """بحث حسب التخصص"""
        return [
            info for info in self.capsules.values()
            if specialty.lower() in info.get("specialty", "").lower()
        ]
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """كل الكبسولات المسجلة"""
        return dict(self.capsules)
    
    def get_capsule_info(self, capsule_id: str) -> Optional[Dict[str, Any]]:
        """معلومات كبسولة محددة"""
        return self.capsules.get(capsule_id)
    
    # ─── Stats ───────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات السجل"""
        specialties = set()
        total_topics = 0
        
        for info in self.capsules.values():
            specialties.add(info.get("specialty", "unknown"))
            total_topics += info.get("topic_count", 0)
        
        return {
            "total_capsules": len(self.capsules),
            "specialties": list(specialties),
            "total_topics": total_topics,
            "registry_path": str(self.registry_path),
            "timestamp": datetime.now().isoformat(),
        }


# ─── Singleton ───────────────────────────────────────────────
capsule_registry = CapsuleRegistry()
