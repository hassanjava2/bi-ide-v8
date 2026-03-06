#!/usr/bin/env python3
"""
layer_system.py — نظام الطبقات الكاملة 🏛️

10+ طبقة + طبقة تضيف طبقات أوتوماتيكياً

الطبقات:
  1. المجلس الأعلى (16 حكيم) ← council_auto_debate.py
  2. البُعد السابع (تخطيط 100 سنة)
  3. فريق الظل والنور (مؤيد ومعارض)
  4. الكشافة (online + offline)
  5. الفريق الفوقي (Meta: Architect + Guardian)
  6. خبراء المجال (11+ مجال = شجرات)
  7. فريق التنفيذ (يبرمج ويبني)
  8. طبقة الربط (تجمع أفكار كل الطبقات)
  9. طبقة الحياة الواقعية (مصانع + فيزياء)
  10. طبقة إضافة الطبقات (ذاتية!)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("layers")
PROJECT_ROOT = Path(__file__).parent.parent

try:
    from brain.memory_system import memory
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


@dataclass
class Layer:
    """طبقة بالنظام"""
    layer_id: str
    name: str
    name_ar: str
    level: int              # 1 = أعلى
    role: str               # وصف الدور
    active: bool = True
    auto_created: bool = False
    processes: List[str] = field(default_factory=list)
    reports_to: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class LayerManager:
    """
    مدير الطبقات — يدير 10+ طبقة

    كل طبقة عندها:
      - دور محدد
      - عمليات (processes)
      - تقارير لطبقة أعلى
    """

    def __init__(self):
        self.layers: Dict[str, Layer] = {}
        self._init_default_layers()

    def _init_default_layers(self):
        """الطبقات الافتراضية"""
        defaults = [
            ("supreme_council", "Supreme Council", "المجلس الأعلى", 1,
             "قرارات استراتيجية — 16 حكيم يتناقشون ويصوتون",
             ["debate", "vote", "strategic_planning"]),

            ("seventh_dimension", "7th Dimension", "البُعد السابع", 2,
             "تخطيط 100 سنة — سيناريوهات مستقبلية",
             ["future_planning", "scenario_analysis", "long_term_strategy"]),

            ("shadow_light", "Shadow & Light", "فريق الظل والنور", 3,
             "كل قرار يُفحص — مؤيد ومعارض يتناقشون",
             ["advocate", "critic", "balance_check"]),

            ("scouts", "Scout System", "الكشافة", 4,
             "استكشاف online + offline + LAN — عيون النظام",
             ["web_search", "lan_scan", "file_scan", "data_collection"]),

            ("meta_team", "Meta Team", "الفريق الفوقي", 5,
             "MetaArchitect + Guardian + Security + Compliance",
             ["architecture", "security_audit", "compliance_check"]),

            ("domain_experts", "Domain Experts", "خبراء المجال", 6,
             "11+ مجال — كل مجال = شجرة من الأهرام",
             ["expert_consultation", "domain_analysis", "specialized_knowledge"]),

            ("execution_team", "Execution Team", "فريق التنفيذ", 7,
             "يبرمج ← يبني ← يختبر ← يصلح",
             ["coding", "building", "testing", "fixing"]),

            ("linking_layer", "Linking Layer", "طبقة الربط", 8,
             "تجمع أفكار كل الطبقات ← تدمج ← تقرر",
             ["idea_synthesis", "cross_layer_communication", "decision_merge"]),

            ("real_life", "Real Life Layer", "طبقة الحياة الواقعية", 9,
             "فيزياء + كيمياء + مصانع + محاكاة + تدريب بشر",
             ["physics_sim", "chemistry_sim", "factory_sim", "human_training"]),

            ("layer_factory", "Layer Factory", "طبقة إضافة الطبقات", 10,
             "تكتشف الحاجة لطبقة جديدة ← تنشئها أوتوماتيكياً",
             ["layer_detection", "layer_creation", "auto_expansion"]),
        ]

        for i, (lid, name, name_ar, level, role, procs) in enumerate(defaults):
            reports = defaults[max(0, i-1)][0] if i > 0 else ""
            self.layers[lid] = Layer(
                layer_id=lid, name=name, name_ar=name_ar,
                level=level, role=role, processes=procs,
                reports_to=reports,
            )

    def add_layer(self, layer_id: str, name: str, name_ar: str,
                  level: int, role: str, processes: List[str] = None,
                  auto: bool = False) -> Layer:
        """إضافة طبقة جديدة"""
        layer = Layer(
            layer_id=layer_id, name=name, name_ar=name_ar,
            level=level, role=role, processes=processes or [],
            auto_created=auto,
        )
        self.layers[layer_id] = layer

        memory.save_knowledge(
            topic=f"New Layer: {name}",
            content=f"{'Auto-' if auto else ''}Created layer: {name_ar} (level {level})",
            source="layer_system",
        )
        logger.info(f"{'🤖' if auto else '✅'} New layer: {name_ar} (L{level})")
        return layer

    def auto_detect_needed_layers(self, context: str) -> List[str]:
        """يكتشف طبقات ناقصة أوتوماتيكياً"""
        new_layers = []
        suggestions = {
            "vision": ("vision_layer", "Vision Layer", "طبقة الرؤية", 5,
                      "تحليل صور/فيديو + كاميرات + مراقبة",
                      ["image_analysis", "video_analysis", "camera_monitoring"]),
            "language": ("language_layer", "Language Layer", "طبقة اللغة", 6,
                     "ترجمة + تحليل نصوص + توليد محتوى",
                     ["translation", "text_analysis", "content_generation"]),
            "ethics": ("ethics_layer", "Ethics Layer", "طبقة الأخلاق", 3,
                     "فحص أخلاقي لكل قرار — حلال/حرام/مناسب",
                     ["ethical_review", "moral_check", "cultural_sensitivity"]),
            "education": ("education_layer", "Education Layer", "طبقة التعليم", 7,
                      "تعليم البشر — مناهج + تدريبات + تقييم",
                      ["curriculum_design", "teaching", "assessment"]),
        }

        words = context.lower().split()
        for key, (lid, name, name_ar, level, role, procs) in suggestions.items():
            if key in words and lid not in self.layers:
                self.add_layer(lid, name, name_ar, level, role, procs, auto=True)
                new_layers.append(lid)

        return new_layers

    def process_request(self, query: str, mode: str = "fast") -> Dict:
        """معالجة طلب عبر كل الطبقات"""
        results = {"query": query, "mode": mode, "layers_activated": [], "responses": []}

        for lid, layer in sorted(self.layers.items(), key=lambda x: x[1].level):
            if not layer.active:
                continue

            relevance = self._check_relevance(query, layer)
            if relevance > 0.3:
                results["layers_activated"].append(layer.name_ar)
                results["responses"].append({
                    "layer": layer.name_ar,
                    "level": layer.level,
                    "role": layer.role,
                    "relevance": round(relevance, 2),
                })

        return results

    def _check_relevance(self, query: str, layer: Layer) -> float:
        words = set(query.lower().split())
        proc_words = set()
        for p in layer.processes:
            proc_words.update(p.lower().replace("_", " ").split())
        overlap = len(words.intersection(proc_words))
        return min(overlap / max(len(proc_words), 1), 1.0)

    def get_hierarchy(self) -> str:
        """عرض التسلسل الهرمي"""
        lines = ["🏛️ نظام الطبقات\n"]
        for lid, layer in sorted(self.layers.items(), key=lambda x: x[1].level):
            icon = "🤖" if layer.auto_created else "🏛️"
            status = "✅" if layer.active else "⛔"
            lines.append(f"  L{layer.level:2d} {icon} {layer.name_ar:20s} ({layer.name}) {status}")
            lines.append(f"       └ {layer.role}")
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "total_layers": len(self.layers),
            "active": sum(1 for l in self.layers.values() if l.active),
            "auto_created": sum(1 for l in self.layers.values() if l.auto_created),
            "max_level": max(l.level for l in self.layers.values()),
        }


# Singleton
layer_manager = LayerManager()


if __name__ == "__main__":
    print(layer_manager.get_hierarchy())
    print(f"\nStats: {layer_manager.stats()}")

    # اكتشاف طبقات جديدة
    print("\n" + "═" * 50)
    new = layer_manager.auto_detect_needed_layers("need vision and ethics layers")
    print(f"Auto-detected: {new}")
    print(f"\nAfter: {layer_manager.stats()}")
