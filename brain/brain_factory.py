#!/usr/bin/env python3
"""
brain_factory.py — مصنع الأدمغة اللامتناهي

النظام القديم (LoRA): محذوف
النظام الجديد (Capsules):
- يولّد كبسولات أطفال من الأقوياء (وراثة)
- يجمع كبسولتين → كبسولة هجينة (تزاوج)
- يحذف الضعيف ← يكاثر القوي (داروين)
- الشجرة تنمو تلقائياً — بلا حدود
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("brain_factory")

CAPSULES_ROOT = Path(__file__).parent.parent / "capsules"

# مخطط التخصصات الفرعية — كل كبسولة قوية تولّد أطفال
SPECIALIZATION_MAP = {
    "code_python": ["fastapi", "django", "pytorch", "automation", "data_science"],
    "code_typescript": ["react_advanced", "nextjs", "nodejs", "graphql"],
    "code_rust": ["tauri_advanced", "async_systems", "networking"],
    "code_sql": ["postgresql", "optimization", "analytics"],
    "security": ["encryption", "pentesting", "network_security", "forensics"],
    "erp_accounting": ["tax_iraq", "audit", "budgeting", "financial_analysis"],
    "erp_sales": ["crm", "analytics", "forecasting"],
    "erp_inventory": ["warehouse", "logistics", "supply_chain"],
    "erp_hr": ["payroll", "recruitment", "performance"],
    "erp_purchasing": ["vendor_management", "procurement", "contracts"],
    "conversation_ar": ["customer_service", "technical_support", "education"],
    "iraqi_dialect": ["baghdadi", "basrawi", "mosulawi"],
    "devops": ["docker", "kubernetes", "ci_cd", "monitoring"],
    "database_design": ["nosql", "graph_db", "time_series"],
    "sage": ["risk_analysis", "long_term_planning", "decision_science"],
    "rebel": ["critical_thinking", "devil_advocate", "stress_testing"],
}

# مخطط التزاوج — كبسولات هجينة مفيدة
BREEDING_MAP = [
    ("code_python", "security", "secure_python"),
    ("code_python", "devops", "python_devops"),
    ("code_python", "code_sql", "python_database"),
    ("code_typescript", "code_css", "fullstack_web"),
    ("erp_accounting", "security", "financial_security"),
    ("sage", "code_python", "ai_architect"),
]


class BrainFactory:
    """مصنع الأدمغة — يولّد ويطوّر الكبسولات بلا حدود"""

    def __init__(self, capsules_dir: Optional[Path] = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        self.capsules_dir.mkdir(parents=True, exist_ok=True)
        self.evolution_count = 0

    def get_all_capsules(self) -> list[dict]:
        """جلب كل الكبسولات مع معلوماتها"""
        capsules = []
        for d in sorted(self.capsules_dir.iterdir()):
            if not d.is_dir():
                continue
            info = self._load_capsule_info(d)
            if info:
                capsules.append(info)
        return capsules

    def _load_capsule_info(self, capsule_dir: Path) -> dict:
        """تحميل معلومات كبسولة"""
        capsule_id = capsule_dir.name
        result_path = capsule_dir / "result.json"
        meta_path = capsule_dir / "meta.json"

        info = {
            "id": capsule_id,
            "dir": str(capsule_dir),
            "has_model": (capsule_dir / "model" / "config.json").exists(),
            "layer": 0,
            "parent": None,
            "children": [],
            "status": "untrained",
            "loss": None,
            "training_minutes": None,
            "archived": False,
        }

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                info["status"] = result.get("status", "unknown")
                info["loss"] = result.get("loss")
                info["training_minutes"] = result.get("minutes")
            except:
                pass

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                info["layer"] = meta.get("layer", 0)
                info["parent"] = meta.get("parent")
                info["children"] = meta.get("children", [])
                info["archived"] = meta.get("archived", False)
            except:
                pass

        data_dir = capsule_dir / "data"
        if data_dir.exists():
            total = 0
            for f in data_dir.glob("*.jsonl"):
                try:
                    total += sum(1 for _ in open(f))
                except:
                    pass
            info["data_samples"] = total
        else:
            info["data_samples"] = 0

        return info

    def get_strong(self, threshold: float = 0.15) -> list[dict]:
        """كبسولات قوية"""
        return [c for c in self.get_all_capsules()
                if c["loss"] is not None and c["loss"] < threshold
                and c["has_model"] and not c["archived"]]

    def get_weak(self, threshold: float = 0.5) -> list[dict]:
        """كبسولات ضعيفة"""
        return [c for c in self.get_all_capsules()
                if c["loss"] is not None and c["loss"] > threshold
                and not c["archived"]]

    def get_untrained(self) -> list[dict]:
        """كبسولات ما اتدربت بعد"""
        return [c for c in self.get_all_capsules()
                if not c["has_model"] and c["data_samples"] > 30
                and not c["archived"]]

    def create_child(self, parent_id: str, specialty: str, description: str = "") -> str:
        """إنشاء كبسولة ابن ترث موديل الأب"""
        child_id = f"{parent_id}_{specialty}"
        child_dir = self.capsules_dir / child_id
        parent_dir = self.capsules_dir / parent_id

        if child_dir.exists():
            return child_id
        if not parent_dir.exists():
            logger.error(f"Parent {parent_id} not found!")
            return ""

        child_dir.mkdir(parents=True)
        (child_dir / "data").mkdir()
        (child_dir / "model").mkdir()

        # نسخ موديل الأب
        parent_model = parent_dir / "model"
        if parent_model.exists() and (parent_model / "config.json").exists():
            for f in parent_model.iterdir():
                if f.is_file():
                    shutil.copy2(f, child_dir / "model" / f.name)

        # معلومات الأب
        parent_layer = 0
        parent_meta_path = parent_dir / "meta.json"
        if parent_meta_path.exists():
            try:
                pm = json.loads(parent_meta_path.read_text())
                parent_layer = pm.get("layer", 0)
            except:
                pass

        # حفظ meta الابن
        (child_dir / "meta.json").write_text(json.dumps({
            "id": child_id, "parent": parent_id, "specialty": specialty,
            "description": description, "layer": parent_layer + 1,
            "children": [], "created": datetime.now().isoformat(),
        }, indent=2, ensure_ascii=False))

        # تحديث children الأب
        parent_meta = {"layer": parent_layer, "children": []}
        if parent_meta_path.exists():
            try:
                parent_meta = json.loads(parent_meta_path.read_text())
            except:
                pass
        if child_id not in parent_meta.get("children", []):
            parent_meta.setdefault("children", []).append(child_id)
            parent_meta_path.write_text(json.dumps(parent_meta, indent=2, ensure_ascii=False))

        logger.info(f"🧒 Child: {child_id} (L{parent_layer + 1}) from {parent_id}")
        return child_id

    def breed(self, p1: str, p2: str, name: str) -> str:
        """تزاوج — موديل الأول + بيانات الثاني"""
        child_id = f"{p1}_{name}"
        if (self.capsules_dir / child_id).exists():
            return child_id

        self.create_child(p1, name, f"Hybrid: {p1} × {p2}")

        # نسخ بيانات الأب الثاني
        p2_data = self.capsules_dir / p2 / "data"
        child_data = self.capsules_dir / child_id / "data"
        if p2_data.exists():
            for f in p2_data.glob("*.jsonl"):
                shutil.copy2(f, child_data / f"from_{p2}_{f.name}")

        logger.info(f"🧬 Bred: {p1} × {p2} → {child_id}")
        return child_id

    def evolve(self) -> dict:
        """دورة تطور واحدة — الأقوياء يتكاثرون، الضعفاء يتأرشفون"""
        self.evolution_count += 1
        strong = self.get_strong(0.15)
        weak = self.get_weak(0.5)
        created = []
        archived = []

        # 1. الأقوياء يولّدون أطفال
        for capsule in strong:
            cid = capsule["id"]
            existing_children = set(capsule.get("children", []))

            if cid in SPECIALIZATION_MAP:
                for spec in SPECIALIZATION_MAP[cid]:
                    child_id = f"{cid}_{spec}"
                    if child_id not in existing_children:
                        self.create_child(cid, spec, f"Evolution #{self.evolution_count}")
                        created.append(child_id)
                        break  # طفل واحد كل دورة

        # 2. تزاوج إذا الأبوين قويين
        strong_ids = {c["id"] for c in strong}
        for p1, p2, name in BREEDING_MAP:
            child_id = f"{p1}_{name}"
            if p1 in strong_ids and p2 in strong_ids:
                if not (self.capsules_dir / child_id).exists():
                    self.breed(p1, p2, name)
                    created.append(child_id)

        # 3. أرشفة الضعفاء (layer > 0 فقط — لا نحذف الجذور)
        for capsule in weak:
            if capsule["layer"] > 0:
                meta_path = Path(capsule["dir"]) / "meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except:
                        pass
                meta["archived"] = True
                meta["archived_at"] = datetime.now().isoformat()
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
                archived.append(capsule["id"])

        logger.info(f"🧬 Evolution #{self.evolution_count}: +{len(created)} -{len(archived)}")
        return {"cycle": self.evolution_count, "created": created, "archived": archived}

    def get_tree(self) -> dict:
        """بناء شجرة الكبسولات"""
        capsules = {c["id"]: c for c in self.get_all_capsules()}
        roots = [c for c in capsules.values() if c["parent"] is None and not c["archived"]]

        def node(c):
            n = {"id": c["id"], "L": c["layer"], "loss": c["loss"],
                 "data": c["data_samples"], "status": c["status"]}
            kids = [node(capsules[kid]) for kid in c.get("children", []) if kid in capsules]
            if kids:
                n["children"] = kids
            return n

        return {"roots": [node(r) for r in roots], "total": len(capsules)}

    def get_status(self) -> dict:
        all_c = self.get_all_capsules()
        return {
            "total_capsules": len(all_c),
            "trained": len([c for c in all_c if c["has_model"]]),
            "untrained": len([c for c in all_c if not c["has_model"] and not c["archived"]]),
            "archived": len([c for c in all_c if c["archived"]]),
            "evolution_cycles": self.evolution_count,
            "max_layer": max((c["layer"] for c in all_c), default=0),
        }


# Singleton
brain_factory = BrainFactory()
