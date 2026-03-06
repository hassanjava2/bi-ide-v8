#!/usr/bin/env python3
"""
capsule_tree.py — نظام الأشجار والأهرام مع الوراثة ∞ 🌳🔺🧬

البنية:
  طبقة → كبسولات → أشجار → أهرام → كبسولات → أشجار → ... ∞

الوراثة:
  - كبسولة أب تورّث معلوماتها للأبناء
  - الابن يتدرب أكثر بتخصصه
  - وراثة متعددة: كبسولة ترث من أكثر من أب
  - أب يورّث لأكثر من ابن
  - كلشي أوتوماتيكي

لا كبسولة بدون ربط — كل عقدة مرتبطة بالشبكة
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger("capsule_tree")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"
TREES_FILE = CAPSULES_ROOT / ".trees.json"

try:
    from brain.memory_system import memory
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Capsule Node — عقدة كبسولة مع وراثة
# ═══════════════════════════════════════════════════════════

@dataclass
class CapsuleNode:
    """عقدة بالشجرة — كبسولة بوراثة"""
    node_id: str
    name: str
    name_ar: str
    node_type: str = "capsule"     # layer | tree | pyramid | capsule
    parent_ids: List[str] = field(default_factory=list)   # وراثة متعددة
    children_ids: List[str] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)  # يرث معرفة من
    inherits_to: List[str] = field(default_factory=list)    # يورّث معرفة إلى
    level: int = 0
    keywords: List[str] = field(default_factory=list)
    data_count: int = 0
    inherited_data_count: int = 0  # بيانات موروثة
    model_trained: bool = False
    auto_created: bool = False
    linked: bool = True            # مرتبطة بالشبكة؟
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════
# Specialization Tree — الشجرة مع الوراثة ∞
# ═══════════════════════════════════════════════════════════

class SpecializationTree:
    """
    نظام أشجار + أهرام + وراثة ∞

    البنية:
      طبقة (layer)
      └── كبسولات
          └── أشجار (tree)
              └── أهرام (pyramid)
                  └── كبسولات
                      └── أشجار → ... ∞

    الوراثة:
      كبسولة أب → تورّث بيانات + أوزان للأبناء
      الابن → يتدرب أكثر على تخصصه
      متعددة: كبسولة ← ترث من عدة آباء
    """

    def __init__(self):
        self.nodes: Dict[str, CapsuleNode] = {}
        self.roots: List[str] = []
        self._load()
        if not self.roots:
            self._init_defaults()

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
            data = {"roots": self.roots, "nodes": {}}
            for nid, n in self.nodes.items():
                data["nodes"][nid] = {
                    "node_id": n.node_id, "name": n.name, "name_ar": n.name_ar,
                    "node_type": n.node_type,
                    "parent_ids": n.parent_ids, "children_ids": n.children_ids,
                    "inherits_from": n.inherits_from, "inherits_to": n.inherits_to,
                    "level": n.level, "keywords": n.keywords,
                    "data_count": n.data_count, "inherited_data_count": n.inherited_data_count,
                    "model_trained": n.model_trained, "auto_created": n.auto_created,
                    "linked": n.linked, "created_at": n.created_at,
                }
            TREES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Save error: {e}")

    def _init_defaults(self):
        """إنشاء البنية الافتراضية"""
        trees = {
            # ═══ هندسة ═══
            "engineering": ("Engineering", "هندسة", "tree", [
                ("electrical", "Electrical", "كهربائية", "pyramid", ["power", "electronics", "telecom", "solar", "circuit"], [
                    ("power_systems", "Power Systems", "أنظمة طاقة", ["grid", "transformer", "generator"]),
                    ("solar_energy", "Solar Energy", "طاقة شمسية", ["panel", "inverter", "solar"]),
                    ("electronics", "Electronics", "إلكترونيات", ["pcb", "chip", "ic", "resistor"]),
                ]),
                ("mechanical", "Mechanical", "ميكانيكية", "pyramid", ["engine", "turbine", "gear", "thermal"], [
                    ("engines", "Engines", "محركات", ["piston", "diesel", "combustion"]),
                    ("hvac", "HVAC", "تكييف", ["cooling", "heating", "ventilation"]),
                ]),
                ("civil", "Civil", "مدنية", "pyramid", ["building", "bridge", "concrete", "road"], [
                    ("structures", "Structures", "إنشاءات", ["beam", "column", "foundation"]),
                    ("roads", "Roads", "طرق", ["asphalt", "highway", "pavement"]),
                ]),
                ("software", "Software", "برمجيات", "pyramid", ["code", "algorithm", "api", "database"], [
                    ("web", "Web Dev", "ويب", ["html", "css", "react", "node"]),
                    ("mobile", "Mobile", "موبايل", ["android", "ios", "flutter"]),
                    ("backend", "Backend", "خلفية", ["rest", "graphql", "microservice"]),
                    ("devops", "DevOps", "عمليات", ["docker", "k8s", "ci", "cd"]),
                ]),
                ("chemical", "Chemical", "كيميائية", "pyramid", ["reactor", "polymer", "catalyst"], []),
            ]),
            # ═══ علوم ═══
            "science": ("Science", "علوم", "tree", [
                ("physics", "Physics", "فيزياء", "pyramid", ["force", "energy", "wave", "quantum", "gravity"], [
                    ("thermodynamics", "Thermodynamics", "ديناميكا حرارية", ["heat", "entropy", "temperature"]),
                    ("mechanics", "Mechanics", "ميكانيكا", ["force", "motion", "velocity"]),
                    ("quantum", "Quantum", "كمّ", ["qubit", "superposition", "entangle"]),
                ]),
                ("chemistry", "Chemistry", "كيمياء", "pyramid", ["reaction", "element", "molecule", "acid"], [
                    ("organic", "Organic", "عضوية", ["carbon", "hydrocarbon", "polymer"]),
                    ("inorganic", "Inorganic", "غير عضوية", ["metal", "mineral", "oxide"]),
                    ("industrial", "Industrial", "صناعية", ["cement", "steel", "glass", "soap"]),
                ]),
                ("biology", "Biology", "أحياء", "pyramid", ["cell", "dna", "protein", "ecology"], []),
                ("math", "Mathematics", "رياضيات", "pyramid", ["calculus", "algebra", "geometry"], []),
            ]),
            # ═══ طب ═══
            "medicine": ("Medicine", "طب", "tree", [
                ("surgery", "Surgery", "جراحة", "pyramid", ["operation", "scalpel", "suture"], []),
                ("pharmacy", "Pharmacy", "صيدلة", "pyramid", ["drug", "dose", "vaccine"], []),
                ("diagnostics", "Diagnostics", "تشخيص", "pyramid", ["xray", "mri", "blood"], []),
                ("emergency", "Emergency", "طوارئ", "pyramid", ["trauma", "cpr", "triage"], []),
            ]),
            # ═══ صناعة ═══
            "manufacturing": ("Manufacturing", "صناعة", "tree", [
                ("metals", "Metals", "معادن", "pyramid", ["steel", "iron", "aluminum", "welding"], [
                    ("steelmaking", "Steelmaking", "صناعة حديد", ["blast", "furnace", "ore"]),
                    ("casting", "Casting", "صب", ["mold", "foundry", "alloy"]),
                ]),
                ("ceramics", "Ceramics", "سيراميك", "pyramid", ["clay", "kiln", "brick", "tile"], []),
                ("textiles", "Textiles", "نسيج", "pyramid", ["fabric", "cotton", "weaving"], []),
                ("food", "Food", "غذائية", "pyramid", ["processing", "canning", "fermentation"], []),
                ("chemicals", "Chemicals", "كيماويات", "pyramid", ["soap", "detergent", "plastic"], []),
                ("construction_materials", "Construction", "مواد بناء", "pyramid", ["cement", "concrete", "glass"], []),
            ]),
            # ═══ زراعة ═══
            "agriculture": ("Agriculture", "زراعة", "tree", [
                ("crops", "Crops", "محاصيل", "pyramid", ["wheat", "rice", "irrigation"], []),
                ("livestock", "Livestock", "ثروة حيوانية", "pyramid", ["cattle", "poultry", "dairy"], []),
                ("aquaculture", "Aquaculture", "أسماك", "pyramid", ["fish", "shrimp", "pond"], []),
            ]),
            # ═══ حوسبة ═══
            "computing": ("Computing", "حوسبة", "tree", [
                ("ai_ml", "AI/ML", "ذكاء اصطناعي", "pyramid", ["neural", "training", "model"], [
                    ("nlp", "NLP", "معالجة لغات", ["tokenizer", "embedding", "transformer"]),
                    ("vision_ai", "Computer Vision", "رؤية حاسوبية", ["yolo", "detection", "segmentation"]),
                ]),
                ("security", "Security", "أمن سيبراني", "pyramid", ["encrypt", "firewall", "exploit"], [
                    ("crypto", "Cryptography", "تشفير", ["aes", "rsa", "hash"]),
                    ("pentest", "Pentesting", "اختبار اختراق", ["vulnerability", "exploit", "scan"]),
                ]),
                ("systems", "Systems", "أنظمة", "pyramid", ["kernel", "os", "driver"], []),
                ("networking", "Networking", "شبكات", "pyramid", ["tcp", "http", "dns", "router"], []),
            ]),
        }

        for tree_id, (name, name_ar, ntype, branches) in trees.items():
            # إنشاء الشجرة الجذرية
            root = CapsuleNode(node_id=tree_id, name=name, name_ar=name_ar,
                              node_type="tree", level=0)
            self.nodes[tree_id] = root
            self.roots.append(tree_id)

            for branch_id, bname, bname_ar, btype, kws, sub_branches in branches:
                full_id = f"{tree_id}.{branch_id}"
                branch = CapsuleNode(
                    node_id=full_id, name=bname, name_ar=bname_ar,
                    node_type=btype, parent_ids=[tree_id], level=1,
                    keywords=kws, inherits_from=[tree_id],
                )
                self.nodes[full_id] = branch
                root.children_ids.append(full_id)
                root.inherits_to.append(full_id)

                # إنشاء فروع فرعية (كبسولات نهائية ترث من الهرم)
                for sub_id, sname, sname_ar, skws in sub_branches:
                    sub_full_id = f"{full_id}.{sub_id}"
                    sub = CapsuleNode(
                        node_id=sub_full_id, name=sname, name_ar=sname_ar,
                        node_type="capsule", parent_ids=[full_id], level=2,
                        keywords=skws, inherits_from=[full_id],
                    )
                    self.nodes[sub_full_id] = sub
                    branch.children_ids.append(sub_full_id)
                    branch.inherits_to.append(sub_full_id)

                    # مجلد البيانات
                    cap_dir = CAPSULES_ROOT / sub_full_id.replace(".", "_")
                    (cap_dir / "data").mkdir(parents=True, exist_ok=True)

                # مجلد الهرم نفسه
                cap_dir = CAPSULES_ROOT / full_id.replace(".", "_")
                (cap_dir / "data").mkdir(parents=True, exist_ok=True)

        self._save()
        logger.info(f"🌳 Init: {len(self.roots)} trees, {len(self.nodes)} total nodes")

    # ═══════════════════════════════════════════════════════
    # الوراثة 🧬 — أهم شي بالنظام
    # ═══════════════════════════════════════════════════════

    def inherit(self, parent_id: str, child_id: str):
        """وراثة: الأب يورّث بياناته للابن"""
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if not parent or not child:
            return

        # ربط الوراثة
        if parent_id not in child.inherits_from:
            child.inherits_from.append(parent_id)
        if child_id not in parent.inherits_to:
            parent.inherits_to.append(child_id)

        # نقل البيانات
        child.inherited_data_count += parent.data_count

        # نسخ الكلمات المفتاحية
        for kw in parent.keywords:
            if kw not in child.keywords:
                child.keywords.append(kw)

        child.linked = True
        self._save()

        memory.save_knowledge(
            topic=f"Inheritance: {parent_id} → {child_id}",
            content=f"{parent.name_ar} ورّث {parent.data_count} عينة لـ {child.name_ar}",
            source="capsule_inheritance",
        )

    def multi_inherit(self, parent_ids: List[str], child_id: str):
        """وراثة متعددة: الابن يرث من عدة آباء"""
        for pid in parent_ids:
            self.inherit(pid, child_id)

    def broadcast_inherit(self, parent_id: str, child_ids: List[str]):
        """أب يورّث لعدة أبناء"""
        for cid in child_ids:
            self.inherit(parent_id, cid)

    def cascade_inherit(self, root_id: str):
        """وراثة متسلسلة: من الجذر لكل الأحفاد"""
        node = self.nodes.get(root_id)
        if not node:
            return
        for child_id in node.children_ids:
            self.inherit(root_id, child_id)
            self.cascade_inherit(child_id)  # recursion ∞

    # ═══════════════════════════════════════════════════════
    # إضافة عقد
    # ═══════════════════════════════════════════════════════

    def add_node(self, node_id: str, name: str, name_ar: str,
                 node_type: str = "capsule", parent_ids: List[str] = None,
                 keywords: List[str] = None, auto: bool = False,
                 inherit_from_parents: bool = True) -> CapsuleNode:
        """إضافة عقدة جديدة مع ربط ووراثة"""
        if node_id in self.nodes:
            return self.nodes[node_id]

        parent_ids = parent_ids or []
        level = 0
        for pid in parent_ids:
            p = self.nodes.get(pid)
            if p:
                level = max(level, p.level + 1)

        node = CapsuleNode(
            node_id=node_id, name=name, name_ar=name_ar,
            node_type=node_type, parent_ids=parent_ids,
            level=level, keywords=keywords or [],
            auto_created=auto, linked=len(parent_ids) > 0,
        )
        self.nodes[node_id] = node

        # ربط بالآباء
        for pid in parent_ids:
            parent = self.nodes.get(pid)
            if parent:
                parent.children_ids.append(node_id)
                if inherit_from_parents:
                    self.inherit(pid, node_id)

        # مجلد البيانات
        cap_dir = CAPSULES_ROOT / node_id.replace(".", "_")
        (cap_dir / "data").mkdir(parents=True, exist_ok=True)

        self._save()
        logger.info(f"{'🤖' if auto else '✅'} New: {name_ar} ({node_type}) L{level}")
        return node

    def auto_expand(self, text: str) -> List[str]:
        """توسع أوتوماتيكي — يكتشف تخصصات جديدة ∞"""
        words = set(text.lower().split())
        new_nodes = []

        expansions = {
            "robotics": ("engineering", "Robotics", "روبوتات", "pyramid",
                        ["robot", "servo", "actuator", "sensor"]),
            "aerospace": ("engineering", "Aerospace", "فضائية", "pyramid",
                         ["rocket", "satellite", "orbit"]),
            "biotech": ("science.biology", "Biotechnology", "تقنية حيوية", "capsule",
                       ["gene", "crispr", "clone"]),
            "renewable": ("engineering.electrical", "Renewable Energy", "طاقة متجددة", "capsule",
                         ["solar", "wind", "hydrogen"]),
            "3dprint": ("manufacturing", "3D Printing", "طباعة ثلاثية", "capsule",
                       ["3d", "print", "filament"]),
            "nanotech": ("science", "Nanotechnology", "نانو", "capsule",
                        ["nano", "graphene", "atomic"]),
        }

        for key, (parent, name, name_ar, ntype, kws) in expansions.items():
            full_id = f"{parent}.{key}"
            if full_id not in self.nodes and words.intersection(set(kws)):
                self.add_node(full_id, name, name_ar, ntype,
                             parent_ids=[parent], keywords=kws, auto=True)
                new_nodes.append(full_id)

        return new_nodes

    # ═══════════════════════════════════════════════════════
    # فحص الربط — لا كبسولة طايفة!
    # ═══════════════════════════════════════════════════════

    def check_orphans(self) -> List[str]:
        """فحص الكبسولات الغير مرتبطة"""
        orphans = []
        for nid, node in self.nodes.items():
            if nid in self.roots:
                continue
            if not node.parent_ids and not node.inherits_from:
                node.linked = False
                orphans.append(nid)
        return orphans

    def fix_orphans(self) -> int:
        """ربط الكبسولات اليتيمة بأقرب شجرة"""
        orphans = self.check_orphans()
        fixed = 0
        for oid in orphans:
            node = self.nodes[oid]
            # بحث عن أقرب شجرة حسب الكلمات المفتاحية
            best_parent = None
            best_score = 0
            for pid, pnode in self.nodes.items():
                if pid == oid:
                    continue
                overlap = len(set(node.keywords).intersection(set(pnode.keywords)))
                if overlap > best_score:
                    best_score = overlap
                    best_parent = pid

            if best_parent:
                node.parent_ids.append(best_parent)
                self.inherit(best_parent, oid)
                node.linked = True
                fixed += 1

        self._save()
        return fixed

    # ═══════════════════════════════════════════════════════
    # بحث
    # ═══════════════════════════════════════════════════════

    def find_capsules(self, query: str, top_k: int = 5) -> List[CapsuleNode]:
        """بحث عن كبسولات مناسبة — يشمل الموروثة"""
        query_words = set(query.lower().split())
        scores = []
        for nid, node in self.nodes.items():
            all_kw = set(node.keywords)
            # إضافة كلمات الآباء (موروثة)
            for pid in node.inherits_from:
                p = self.nodes.get(pid)
                if p:
                    all_kw.update(p.keywords)
            match = len(query_words.intersection(all_kw))
            if match > 0:
                scores.append((node, match))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k]]

    # ═══════════════════════════════════════════════════════
    # عرض
    # ═══════════════════════════════════════════════════════

    def get_tree_view(self, root_id: str = None) -> str:
        lines = []
        roots = [root_id] if root_id else self.roots

        type_icons = {"tree": "🌳", "pyramid": "🔺", "capsule": "🍃", "layer": "🏛️"}

        def _render(nid, indent=0):
            node = self.nodes.get(nid)
            if not node:
                return
            prefix = "  " * indent + ("├── " if indent > 0 else "")
            icon = type_icons.get(node.node_type, "📦")
            trained = " ✅" if node.model_trained else ""
            auto = " 🤖" if node.auto_created else ""
            inherited = f" 🧬{len(node.inherits_from)}" if node.inherits_from and nid not in self.roots else ""
            orphan = " ⚠️ORPHAN" if not node.linked and nid not in self.roots else ""
            lines.append(f"{prefix}{icon} {node.name_ar} ({node.name}){inherited}{trained}{auto}{orphan}")
            for cid in node.children_ids:
                _render(cid, indent + 1)

        for rid in roots:
            _render(rid)
            lines.append("")
        return "\n".join(lines)

    def stats(self) -> Dict:
        nodes = list(self.nodes.values())
        return {
            "trees": len(self.roots),
            "total_nodes": len(nodes),
            "layers": sum(1 for n in nodes if n.node_type == "layer"),
            "pyramids": sum(1 for n in nodes if n.node_type == "pyramid"),
            "capsules": sum(1 for n in nodes if n.node_type == "capsule"),
            "trained": sum(1 for n in nodes if n.model_trained),
            "auto_created": sum(1 for n in nodes if n.auto_created),
            "orphans": len(self.check_orphans()),
            "inheritance_links": sum(len(n.inherits_from) for n in nodes),
        }


# Singleton
tree = SpecializationTree()


if __name__ == "__main__":
    print("🌳🧬 Capsule Tree + Inheritance System\n")

    s = tree.stats()
    print(f"Trees: {s['trees']}, Nodes: {s['total_nodes']}, "
          f"Pyramids: {s['pyramids']}, Capsules: {s['capsules']}, "
          f"Inheritance links: {s['inheritance_links']}\n")

    # عرض شجرة هندسة
    print(tree.get_tree_view("engineering"))

    # وراثة متسلسلة
    print("═" * 50)
    print("Cascade inheritance from engineering:")
    tree.cascade_inherit("engineering")
    s2 = tree.stats()
    print(f"  Inheritance links after cascade: {s2['inheritance_links']}")

    # توسع أوتوماتيكي
    new = tree.auto_expand("building robot with servo and solar panels nano graphene")
    print(f"\nAuto-expanded: {new}")

    # فحص يتيمة
    orphans = tree.check_orphans()
    print(f"Orphans: {orphans}")
    if orphans:
        fixed = tree.fix_orphans()
        print(f"Fixed {fixed} orphans")

    print(f"\nFinal: {tree.stats()}")
