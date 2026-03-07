#!/usr/bin/env python3
"""
capsule_bridge.py — الجسر الكامل بين الكبسولات والنظام كله

يربط:
  capsule_500.py (498 كبسولة) ← → capsule_tree.py (نظام الوراثة)
  ← → hierarchy (المجلس + الحكماء)
  ← → real_life_layer (المصانع + الفيزياء)
  ← → data_preprocessor (بيانات التدريب)
  ← → IDE API endpoints

الاستخدام:
  from brain.capsule_bridge import bridge
  bridge.sync_all()  # ربط كامل
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CapsuleBridge:
    """الجسر المركزي — يربط الكبسولات بكل شي"""

    def __init__(self):
        self.tree = None
        self.registry = {}
        self.layer_connections = {}
        self._initialized = False

    def initialize(self):
        """تهيئة الجسر — import lazy لتجنب circular"""
        if self._initialized:
            return

        # 1. Load capsule registry
        try:
            from .capsule_500 import CAPSULE_REGISTRY
            self.registry = CAPSULE_REGISTRY
            logger.info(f"📦 Loaded {len(self.registry)} capsules from registry")
        except ImportError:
            logger.warning("⚠️ capsule_500 not found")
            self.registry = {}

        # 2. Load capsule tree
        try:
            from .capsule_tree import tree as capsule_tree
            self.tree = capsule_tree
            logger.info(f"🌳 Loaded capsule tree: {len(capsule_tree.nodes)} nodes")
        except ImportError:
            logger.warning("⚠️ capsule_tree not found")
            self.tree = None

        self._initialized = True

    # ═══════════════════════════════════════════════
    # 1. Sync capsule_500 → capsule_tree
    # ═══════════════════════════════════════════════

    def sync_registry_to_tree(self) -> Dict[str, int]:
        """مزامنة كل الكبسولات من capsule_500 إلى capsule_tree مع وراثة"""
        self.initialize()
        if not self.tree:
            return {"error": "capsule_tree not available"}

        stats = {"added": 0, "existing": 0, "inherited": 0, "categories": set()}

        for capsule_id, (name_ar, keywords) in self.registry.items():
            # Parse hierarchy: "software.languages.python" → ["software", "software.languages", "software.languages.python"]
            parts = capsule_id.split(".")
            hierarchy = []
            for i in range(len(parts)):
                hierarchy.append(".".join(parts[:i + 1]))

            # Ensure all intermediate nodes exist
            for idx, node_id in enumerate(hierarchy):
                if node_id not in self.tree.nodes:
                    parent_ids = [hierarchy[idx - 1]] if idx > 0 else []
                    node_name = parts[idx]

                    # Determine node type
                    if idx == 0:
                        node_type = "category"
                        node_name_ar = self._translate_category(node_name)
                    elif idx == len(hierarchy) - 1:
                        node_type = "capsule"
                        node_name_ar = name_ar
                    else:
                        node_type = "tree"
                        node_name_ar = node_name

                    # Use keywords only for leaf nodes
                    kws = keywords if idx == len(hierarchy) - 1 else []

                    self.tree.add_node(
                        node_id=node_id,
                        name=node_name,
                        name_ar=node_name_ar,
                        node_type=node_type,
                        parent_ids=parent_ids,
                        keywords=kws,
                        inherit_from_parents=True,
                    )
                    stats["added"] += 1

                    if parent_ids:
                        stats["inherited"] += 1
                else:
                    stats["existing"] += 1

                if idx == 0:
                    stats["categories"].add(node_id)

        stats["categories"] = len(stats["categories"])
        logger.info(
            f"✅ Sync complete: {stats['added']} added, "
            f"{stats['inherited']} inherited, {stats['categories']} categories"
        )
        return stats

    # ═══════════════════════════════════════════════
    # 2. Connect to Hierarchy (Council + Sages)
    # ═══════════════════════════════════════════════

    def connect_to_hierarchy(self):
        """ربط الكبسولات بالمجلس — كل حكيم يعرف كبسولاته"""
        self.initialize()

        try:
            from hierarchy.autonomous_council import autonomous_council
        except ImportError:
            logger.warning("⚠️ autonomous_council not available")
            return {}

        # Map sage expertise → capsules
        sage_capsules = {}
        for member_id, member in autonomous_council.members.items():
            matched = []
            for capsule_id, (name_ar, keywords) in self.registry.items():
                # Match sage expertise to capsule keywords
                for exp in member.expertise:
                    exp_lower = exp.lower()
                    capsule_cat = capsule_id.split(".")[0]
                    if (exp_lower in capsule_id.lower() or
                        exp_lower in capsule_cat or
                        any(exp_lower in kw.lower() for kw in keywords)):
                        matched.append(capsule_id)
                        break
            sage_capsules[member.name] = matched[:50]  # Top 50 per sage

        self.layer_connections["hierarchy"] = sage_capsules
        logger.info(f"🏛️ Connected {len(sage_capsules)} sages to capsules")

        # Log mapping
        for sage, caps in sage_capsules.items():
            logger.info(f"   {sage}: {len(caps)} capsules")

        return sage_capsules

    # ═══════════════════════════════════════════════
    # 3. Connect to Real Life Layer
    # ═══════════════════════════════════════════════

    def connect_to_real_life(self):
        """ربط الكبسولات بطبقة الحياة الواقعية — كل agent يعرف كبسولاته"""
        self.initialize()

        try:
            from hierarchy.real_life_layer import real_life_layer
        except ImportError:
            logger.warning("⚠️ real_life_layer not available")
            return {}

        # Map specialization → capsules
        agent_capsules = {}
        spec_keywords = {
            "fluid_mechanics": ["fluid", "pipe", "flow", "hydraulic"],
            "thermodynamics": ["heat", "thermal", "temperature", "energy"],
            "metallurgy": ["steel", "iron", "metal", "alloy", "welding"],
            "ceramics": ["cement", "brick", "ceramic", "glass", "kiln"],
            "factory_design": ["factory", "plant", "production", "manufacturing"],
            "cost_analysis": ["cost", "budget", "price", "economics"],
            "organic_chemistry": ["organic", "polymer", "chemical"],
            "electromagnetism": ["electric", "motor", "generator", "power"],
            "automation": ["robot", "PLC", "automation", "CNC"],
        }

        for agent_id, agent in real_life_layer.agents.items():
            spec = agent.specialization.value
            search_kws = spec_keywords.get(spec, [spec.replace("_", " ")])
            matched = []

            for capsule_id, (name_ar, keywords) in self.registry.items():
                kws_lower = [k.lower() for k in keywords]
                cap_lower = capsule_id.lower()
                if any(sk in cap_lower or any(sk in kw for kw in kws_lower)
                       for sk in search_kws):
                    matched.append(capsule_id)

            agent_capsules[agent_id] = matched[:30]

        self.layer_connections["real_life"] = agent_capsules
        logger.info(f"🌍 Connected {len(agent_capsules)} agents to capsules")
        return agent_capsules

    # ═══════════════════════════════════════════════
    # 4. Connect to Data Preprocessor
    # ═══════════════════════════════════════════════

    def connect_to_preprocessing(self):
        """ربط الكبسولات بمعالج البيانات — capsule matching يستخدم الشجرة"""
        self.initialize()

        try:
            from .data_preprocessor import CAPSULE_KEYWORDS, load_capsule_keywords
            # Update the preprocessor's keyword map from our registry
            for capsule_id, (name_ar, keywords) in self.registry.items():
                CAPSULE_KEYWORDS[capsule_id] = keywords
            logger.info(f"📊 Connected {len(CAPSULE_KEYWORDS)} capsule keywords to preprocessor")
            return {"capsules_connected": len(CAPSULE_KEYWORDS)}
        except ImportError:
            logger.warning("⚠️ data_preprocessor not available")
            return {}

    # ═══════════════════════════════════════════════
    # 5. API Status — for IDE integration
    # ═══════════════════════════════════════════════

    def get_full_status(self) -> Dict[str, Any]:
        """حالة الجسر الكاملة — تُعرض بالـ IDE"""
        self.initialize()

        tree_stats = self.tree.stats() if self.tree else {}

        return {
            "registry_capsules": len(self.registry),
            "tree_nodes": tree_stats.get("total_nodes", 0),
            "tree_trees": tree_stats.get("trees", 0),
            "tree_linked": tree_stats.get("linked", 0),
            "tree_orphans": tree_stats.get("orphans", 0),
            "layer_connections": {
                layer: len(mapping)
                for layer, mapping in self.layer_connections.items()
            },
            "categories": self._get_category_summary(),
        }

    def find_capsules_for_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """بحث ذكي — يستخدم الشجرة + الكلمات المفتاحية"""
        self.initialize()
        results = []

        # Use tree search (includes inherited keywords)
        if self.tree:
            tree_results = self.tree.find_capsules(query, top_k=top_k)
            for node in tree_results:
                results.append({
                    "capsule_id": node.node_id,
                    "name_ar": node.name_ar,
                    "keywords": node.keywords,
                    "level": node.level,
                    "inherited_data": node.inherited_data_count,
                    "source": "tree",
                })

        # Fallback: direct registry search (improved)
        if not results:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            scored = []
            for cid, (name_ar, kws) in self.registry.items():
                score = 0
                cid_lower = cid.lower()
                # Score by capsule ID match (strongest signal)
                for w in query_words:
                    if w in cid_lower:
                        score += 3
                # Score by keyword match
                for kw in kws:
                    kw_lower = kw.lower()
                    if kw_lower in query_lower:
                        score += 2
                    elif any(w in kw_lower or kw_lower in w for w in query_words):
                        score += 1
                # Score by Arabic name match
                if any(w in name_ar for w in query_words):
                    score += 2
                if score > 0:
                    scored.append((cid, name_ar, kws, score))
            scored.sort(key=lambda x: -x[3])
            for cid, name_ar, kws, score in scored[:top_k]:
                results.append({
                    "capsule_id": cid,
                    "name_ar": name_ar,
                    "keywords": kws,
                    "score": score,
                    "source": "registry",
                })

        return results

    # ═══════════════════════════════════════════════
    # Sync All
    # ═══════════════════════════════════════════════

    def sync_all(self) -> Dict[str, Any]:
        """ربط كامل — كل شي مع كل شي"""
        self.initialize()
        results = {}

        logger.info("🔗 === Full Capsule Sync Starting ===")

        # 1. Sync to tree
        results["tree_sync"] = self.sync_registry_to_tree()

        # 2. Fix orphans
        if self.tree:
            orphans = self.tree.check_orphans()
            if orphans:
                fixed = self.tree.fix_orphans()
                results["orphans_fixed"] = fixed

        # 3. Cascade inheritance from roots
        if self.tree:
            for node_id, node in self.tree.nodes.items():
                if node.level == 0 and node.node_type == "category":
                    self.tree.cascade_inherit(node_id)
            results["inheritance"] = "cascaded"

        # 4. Connect to layers
        results["hierarchy"] = self.connect_to_hierarchy()
        results["real_life"] = self.connect_to_real_life()
        results["preprocessing"] = self.connect_to_preprocessing()

        # 5. Status
        results["status"] = self.get_full_status()

        logger.info("🔗 === Full Capsule Sync Complete ===")
        return results

    # ═══════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════

    def _translate_category(self, cat: str) -> str:
        translations = {
            "software": "برمجيات", "brain": "أدمغة", "hacking": "اختراق",
            "engineering": "هندسة", "science": "علوم", "manufacturing": "تصنيع",
            "medicine": "طب", "agriculture": "زراعة", "crafts": "حرف",
            "energy": "طاقة", "water": "مياه", "society": "مجتمع",
            "business": "أعمال", "governance": "حوكمة", "survival": "بقاء",
            "military": "عسكري", "wisdom": "حكمة", "communication": "اتصالات",
            "transport": "نقل", "computing": "حوسبة", "vision": "رؤية",
            "robotics": "روبوتات", "space": "فضاء", "marine": "بحري",
            "advanced": "متقدم", "knowledge": "معرفة", "food_security": "أمن غذائي",
            "other": "أخرى",
        }
        return translations.get(cat, cat)

    def _get_category_summary(self) -> Dict[str, int]:
        counts = {}
        for cid in self.registry:
            cat = cid.split(".")[0]
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# Singleton
bridge = CapsuleBridge()
