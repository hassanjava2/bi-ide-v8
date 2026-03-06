#!/usr/bin/env python3
"""
capsule_tree.py — نظام الأشجار والأهرام ∞ 🌳🔺

كل اختصاص = شجرة
كل فرع = هرم
كل هرم = كبسولات متخصصة
التوسع أوتوماتيكي — لا حدود!

مثال:
  هندسة (شجرة)
  ├── كهربائية (هرم)
  │   ├── طاقة (كبسولة)
  │   │   ├── شمسية
  │   │   ├── رياح
  │   │   └── نووية ← يُضاف أوتوماتيكياً
  │   ├── إلكترونيات
  │   └── اتصالات
  ├── ميكانيكية (هرم)
  └── مدنية (هرم)
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("capsule_tree")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"
TREES_FILE = CAPSULES_ROOT / ".trees.json"

try:
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


@dataclass
class CapsuleNode:
    """عقدة بالشجرة — كبسولة أو فرع"""
    node_id: str
    name: str
    name_ar: str
    parent_id: str = ""
    level: int = 0              # عمق بالشجرة
    is_leaf: bool = True        # ورقة = كبسولة نهائية
    children: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    data_count: int = 0
    model_trained: bool = False
    auto_created: bool = False  # أُنشئ أوتوماتيكياً؟
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SpecializationTree:
    """
    شجرة اختصاصات — تتوسع ∞

    كل شجرة = مجال كبير (هندسة، طب، علوم...)
    كل فرع = هرم من التخصصات
    أوراق الشجرة = كبسولات AI متدربة
    """

    def __init__(self):
        self.nodes: Dict[str, CapsuleNode] = {}
        self.roots: List[str] = []
        self._load()
        if not self.roots:
            self._init_default_trees()

    def _load(self):
        if TREES_FILE.exists():
            try:
                data = json.loads(TREES_FILE.read_text())
                for nid, nd in data.get("nodes", {}).items():
                    self.nodes[nid] = CapsuleNode(**nd)
                self.roots = data.get("roots", [])
            except Exception:
                pass

    def _save(self):
        try:
            data = {
                "roots": self.roots,
                "nodes": {nid: {
                    "node_id": n.node_id, "name": n.name, "name_ar": n.name_ar,
                    "parent_id": n.parent_id, "level": n.level,
                    "is_leaf": n.is_leaf, "children": n.children,
                    "keywords": n.keywords, "data_count": n.data_count,
                    "model_trained": n.model_trained, "auto_created": n.auto_created,
                    "created_at": n.created_at,
                } for nid, n in self.nodes.items()},
            }
            TREES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Save error: {e}")

    def _init_default_trees(self):
        """الأشجار الافتراضية"""
        trees = {
            "engineering": ("Engineering", "هندسة", [
                ("electrical", "Electrical", "كهربائية", ["power", "electronics", "telecom", "solar", "circuit"]),
                ("mechanical", "Mechanical", "ميكانيكية", ["engine", "turbine", "gear", "fluid", "thermal"]),
                ("civil", "Civil", "مدنية", ["building", "bridge", "concrete", "foundation", "road"]),
                ("software", "Software", "برمجيات", ["code", "algorithm", "api", "database", "frontend"]),
                ("chemical", "Chemical", "كيميائية", ["reactor", "polymer", "catalyst", "refinery"]),
            ]),
            "science": ("Science", "علوم", [
                ("physics", "Physics", "فيزياء", ["force", "energy", "wave", "quantum", "gravity"]),
                ("chemistry", "Chemistry", "كيمياء", ["reaction", "element", "molecule", "bond", "acid"]),
                ("biology", "Biology", "أحياء", ["cell", "dna", "protein", "evolution", "ecology"]),
                ("math", "Mathematics", "رياضيات", ["calculus", "algebra", "geometry", "statistics"]),
            ]),
            "medicine": ("Medicine", "طب", [
                ("surgery", "Surgery", "جراحة", ["operation", "scalpel", "wound", "suture"]),
                ("pharmacy", "Pharmacy", "صيدلة", ["drug", "dose", "medicine", "vaccine"]),
                ("diagnostics", "Diagnostics", "تشخيص", ["xray", "mri", "blood", "symptom"]),
            ]),
            "manufacturing": ("Manufacturing", "صناعة", [
                ("metals", "Metals", "معادن", ["steel", "iron", "aluminum", "copper", "welding"]),
                ("ceramics", "Ceramics", "سيراميك", ["clay", "kiln", "glaze", "brick", "tile"]),
                ("textiles", "Textiles", "نسيج", ["fabric", "cotton", "weaving", "dye"]),
                ("food", "Food", "غذائية", ["processing", "canning", "fermentation", "pasteurize"]),
            ]),
            "agriculture": ("Agriculture", "زراعة", [
                ("crops", "Crops", "محاصيل", ["wheat", "rice", "corn", "irrigation"]),
                ("livestock", "Livestock", "ثروة حيوانية", ["cattle", "poultry", "dairy", "feed"]),
                ("aquaculture", "Aquaculture", "أسماك", ["fish", "shrimp", "pond", "net"]),
            ]),
            "computing": ("Computing", "حوسبة", [
                ("ai_ml", "AI/ML", "ذكاء اصطناعي", ["neural", "training", "model", "inference"]),
                ("security", "Security", "أمن سيبراني", ["encrypt", "firewall", "exploit", "hash"]),
                ("systems", "Systems", "أنظمة", ["kernel", "os", "driver", "scheduler"]),
                ("networking", "Networking", "شبكات", ["tcp", "http", "dns", "router", "packet"]),
            ]),
        }

        for tree_id, (name, name_ar, branches) in trees.items():
            root = CapsuleNode(node_id=tree_id, name=name, name_ar=name_ar,
                              level=0, is_leaf=False)
            self.nodes[tree_id] = root
            self.roots.append(tree_id)

            for branch_id, bname, bname_ar, kws in branches:
                full_id = f"{tree_id}.{branch_id}"
                branch = CapsuleNode(
                    node_id=full_id, name=bname, name_ar=bname_ar,
                    parent_id=tree_id, level=1, is_leaf=True,
                    keywords=kws,
                )
                self.nodes[full_id] = branch
                root.children.append(full_id)

                # إنشاء مجلد الكبسولة
                cap_dir = CAPSULES_ROOT / full_id.replace(".", "_")
                (cap_dir / "data").mkdir(parents=True, exist_ok=True)

        self._save()
        logger.info(f"🌳 Initialized {len(self.roots)} trees, {len(self.nodes)} nodes")

    def add_branch(self, parent_id: str, name: str, name_ar: str,
                   keywords: List[str] = None, auto: bool = False) -> CapsuleNode:
        """إضافة فرع جديد — يدوي أو أوتوماتيكي"""
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent not found: {parent_id}")

        branch_id = f"{parent_id}.{name.lower().replace(' ', '_')}"
        if branch_id in self.nodes:
            return self.nodes[branch_id]

        parent.is_leaf = False

        branch = CapsuleNode(
            node_id=branch_id, name=name, name_ar=name_ar,
            parent_id=parent_id, level=parent.level + 1,
            is_leaf=True, keywords=keywords or [],
            auto_created=auto,
        )
        self.nodes[branch_id] = branch
        parent.children.append(branch_id)

        # مجلد كبسولة
        cap_dir = CAPSULES_ROOT / branch_id.replace(".", "_")
        (cap_dir / "data").mkdir(parents=True, exist_ok=True)

        self._save()

        memory.save_knowledge(
            topic=f"New Branch: {branch_id}",
            content=f"{'Auto-' if auto else ''}Created: {name_ar} under {parent_id}",
            source="capsule_tree",
        )

        logger.info(f"{'🤖' if auto else '✅'} New branch: {branch_id} ({name_ar})")
        return branch

    def auto_expand(self, text: str) -> List[str]:
        """
        توسع أوتوماتيكي — يكتشف تخصصات جديدة من النص

        يحلل الكلمات ← إذا ما تنتمي لأي كبسولة ← ينشئ كبسولة جديدة
        """
        words = set(text.lower().split())
        new_branches = []

        # كلمات محتملة لتخصصات جديدة
        potential = {
            "robotics": ("engineering", "Robotics", "روبوتات", ["robot", "servo", "actuator", "sensor"]),
            "aerospace": ("engineering", "Aerospace", "فضائية", ["rocket", "satellite", "orbit", "thrust"]),
            "biotech": ("science", "Biotechnology", "تقنية حيوية", ["gene", "crispr", "clone", "stem"]),
            "renewable": ("engineering.electrical", "Renewable Energy", "طاقة متجددة", ["solar", "wind", "hydrogen", "geothermal"]),
            "3dprint": ("manufacturing", "3D Printing", "طباعة ثلاثية", ["3d", "print", "filament", "layer"]),
            "quantum_comp": ("computing", "Quantum Computing", "حوسبة كمية", ["qubit", "quantum", "superposition", "entangle"]),
            "nanotechnology": ("science", "Nanotechnology", "نانو", ["nano", "molecule", "atomic", "graphene"]),
        }

        for key, (parent, name, name_ar, kws) in potential.items():
            full_id = f"{parent}.{key}"
            if full_id not in self.nodes and words.intersection(set(kws)):
                try:
                    self.add_branch(parent, name, name_ar, kws, auto=True)
                    new_branches.append(full_id)
                except Exception:
                    pass

        return new_branches

    def find_capsule(self, query: str) -> List[CapsuleNode]:
        """بحث عن كبسولة مناسبة للسؤال"""
        query_words = set(query.lower().split())
        scores = []

        for nid, node in self.nodes.items():
            if not node.keywords:
                continue
            match = len(query_words.intersection(set(node.keywords)))
            if match > 0:
                scores.append((node, match))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:5]]

    def get_tree_view(self, root_id: str = None) -> str:
        """عرض الشجرة"""
        lines = []
        roots = [root_id] if root_id else self.roots

        def _render(nid, indent=0):
            node = self.nodes.get(nid)
            if not node:
                return
            prefix = "  " * indent + ("├── " if indent > 0 else "")
            icon = "🍃" if node.is_leaf else "🌳" if indent == 0 else "🔺"
            trained = " ✅" if node.model_trained else ""
            auto = " 🤖" if node.auto_created else ""
            data = f" ({node.data_count})" if node.data_count > 0 else ""
            lines.append(f"{prefix}{icon} {node.name_ar} ({node.name}){data}{trained}{auto}")
            for child_id in node.children:
                _render(child_id, indent + 1)

        for rid in roots:
            _render(rid)
            lines.append("")

        return "\n".join(lines)

    def stats(self) -> Dict:
        """إحصائيات"""
        leaves = [n for n in self.nodes.values() if n.is_leaf]
        return {
            "trees": len(self.roots),
            "total_nodes": len(self.nodes),
            "capsules": len(leaves),
            "trained": sum(1 for n in leaves if n.model_trained),
            "auto_created": sum(1 for n in self.nodes.values() if n.auto_created),
        }


# Singleton
tree = SpecializationTree()


if __name__ == "__main__":
    print("🌳 Capsule Tree System\n")

    s = tree.stats()
    print(f"Trees: {s['trees']}, Nodes: {s['total_nodes']}, Capsules: {s['capsules']}\n")

    # عرض كل الأشجار
    print(tree.get_tree_view())

    # توسع أوتوماتيكي
    print("═" * 50)
    print("Auto-expand test:")
    new = tree.auto_expand("I need to build a robot with servo motors and sensors for solar energy")
    print(f"  New branches: {new}")

    print(f"\nAfter expansion: {tree.stats()}")
