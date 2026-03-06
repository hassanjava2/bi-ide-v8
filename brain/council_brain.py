#!/usr/bin/env python3
"""
council_brain.py — مجلس الأدمغة 🏛️

الحكيم (sage) + المتمرد (rebel) يقررون:
  - شنو كبسولات جديدة نصنع؟
  - إي كبسولات تتأرشف؟
  - وين نركز التدريب؟
  - شنو المهارات الناقصة؟

المجلس يشتغل بدون تدخل بشري.
يقرأ الحالة → يحلل → يصوّت → ينفّذ.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("council")

CAPSULES_ROOT = Path(__file__).parent.parent / "capsules"


class CouncilBrain:
    """مجلس الأدمغة — حكيم + متمرد يقررون الاستراتيجية"""

    def __init__(self, capsules_dir: Path = None):
        self.capsules_dir = capsules_dir or CAPSULES_ROOT
        self.decisions_log = self.capsules_dir / ".council_log.json"

    def _get_state(self) -> Dict:
        """جمع حالة كل الكبسولات"""
        capsules = []
        for d in sorted(self.capsules_dir.iterdir()):
            if not d.is_dir():
                continue

            info = {"id": d.name, "has_model": False, "loss": None,
                    "score": None, "archived": False, "layer": 0,
                    "data_count": 0, "children": []}

            # نتيجة التدريب
            result_path = d / "result.json"
            if result_path.exists():
                try:
                    r = json.loads(result_path.read_text())
                    info["loss"] = r.get("loss")
                    info["has_model"] = True
                except:
                    pass

            # نتيجة الامتحان
            eval_path = d / "eval.json"
            if eval_path.exists():
                try:
                    e = json.loads(eval_path.read_text())
                    info["score"] = e.get("score")
                except:
                    pass

            # meta
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    m = json.loads(meta_path.read_text())
                    info["layer"] = m.get("layer", 0)
                    info["children"] = m.get("children", [])
                    info["archived"] = m.get("archived", False)
                except:
                    pass

            # بيانات
            data_dir = d / "data"
            if data_dir.exists():
                for f in data_dir.glob("*.jsonl"):
                    try:
                        info["data_count"] += sum(1 for _ in open(f))
                    except:
                        pass

            if not (d / "model" / "config.json").exists():
                info["has_model"] = False

            capsules.append(info)

        return {
            "capsules": capsules,
            "total": len(capsules),
            "trained": len([c for c in capsules if c["has_model"]]),
            "archived": len([c for c in capsules if c["archived"]]),
            "avg_loss": self._avg([c["loss"] for c in capsules if c["loss"]]),
            "avg_score": self._avg([c["score"] for c in capsules if c["score"]]),
        }

    def _avg(self, values):
        return round(sum(values) / len(values), 4) if values else None

    def decide(self) -> Dict:
        """
        المجلس يحلل ويقرر:
        1. الحكيم (sage): تحليل هادئ + توصيات
        2. المتمرد (rebel): نقد + تحديات
        3. القرار النهائي: مزيج
        """
        state = self._get_state()
        capsules = state["capsules"]
        active = [c for c in capsules if not c["archived"]]

        decisions = {
            "timestamp": datetime.now().isoformat(),
            "state_summary": {
                "total": state["total"],
                "trained": state["trained"],
                "avg_loss": state["avg_loss"],
                "avg_score": state["avg_score"],
            },
            "sage_analysis": [],
            "rebel_challenges": [],
            "actions": [],
        }

        # ═══ الحكيم — تحليل هادئ ═══

        # 1. كبسولات ما اتدربت
        untrained = [c for c in active if not c["has_model"] and c["data_count"] >= 30]
        if untrained:
            decisions["sage_analysis"].append(
                f"يوجد {len(untrained)} كبسولة بحاجة تدريب: "
                + ", ".join(c["id"] for c in untrained[:5])
            )
            for c in untrained[:3]:
                decisions["actions"].append({
                    "action": "prioritize_training",
                    "capsule": c["id"],
                    "reason": f"{c['data_count']} samples waiting",
                })

        # 2. كبسولات ضعيفة (score < 30)
        weak = [c for c in active if c["score"] is not None and c["score"] < 30]
        if weak:
            decisions["sage_analysis"].append(
                f"{len(weak)} كبسولة ضعيفة تحت 30: "
                + ", ".join(c["id"] for c in weak)
            )
            for c in weak:
                decisions["actions"].append({
                    "action": "request_more_data",
                    "capsule": c["id"],
                    "reason": f"score={c['score']}",
                })

        # 3. كبسولات قوية بدون أطفال
        strong = [c for c in active
                  if c["loss"] and c["loss"] < 0.1 and not c["children"]]
        if strong:
            decisions["sage_analysis"].append(
                f"{len(strong)} كبسولة قوية ممكن تتكاثر: "
                + ", ".join(c["id"] for c in strong)
            )
            for c in strong[:2]:
                decisions["actions"].append({
                    "action": "spawn_children",
                    "capsule": c["id"],
                    "reason": f"loss={c['loss']:.4f}, no children yet",
                })

        # 4. فجوات معرفية — تخصصات ناقصة
        existing = {c["id"] for c in active}
        needed = {
            "code_go", "code_java", "ml_basics", "data_engineering",
            "api_design", "mobile_dev", "cloud_aws", "cloud_gcp",
        }
        missing = needed - existing
        if missing:
            decisions["sage_analysis"].append(
                f"تخصصات ناقصة: {', '.join(list(missing)[:5])}"
            )
            for m in list(missing)[:2]:
                decisions["actions"].append({
                    "action": "create_new_capsule",
                    "capsule": m,
                    "reason": "knowledge gap",
                })

        # ═══ المتمرد — نقد ═══

        # 1. كبسولات كثيرة أرشيف
        if state["archived"] > state["total"] * 0.3:
            decisions["rebel_challenges"].append(
                f"⚠️ {state['archived']}/{state['total']} مؤرشفة! النظام يقتل أكثر مما يصنع"
            )

        # 2. كل الكبسولات نفس المستوى
        layers = set(c["layer"] for c in active)
        if len(layers) <= 1 and len(active) > 10:
            decisions["rebel_challenges"].append(
                "⚠️ كل الكبسولات Layer 0! ما في تخصص عميق — أين الأطفال؟"
            )

        # 3. بيانات قليلة
        low_data = [c for c in active if c["data_count"] < 50 and c["has_model"]]
        if low_data:
            decisions["rebel_challenges"].append(
                f"⚠️ {len(low_data)} كبسولة مدربة على بيانات قليلة — "
                "التدريب على بيانات قليلة = نتائج ضعيفة"
            )

        # ═══ حفظ القرارات ═══
        self._log_decision(decisions)
        logger.info(f"🏛️ Council: {len(decisions['actions'])} actions, "
                    f"{len(decisions['rebel_challenges'])} warnings")

        return decisions

    def _log_decision(self, decision: Dict):
        """حفظ قرار المجلس"""
        log = []
        if self.decisions_log.exists():
            try:
                log = json.loads(self.decisions_log.read_text())
            except:
                pass
        log.append(decision)
        # آخر 50 قرار فقط
        log = log[-50:]
        self.decisions_log.write_text(
            json.dumps(log, indent=2, ensure_ascii=False, default=str))

    def execute_decisions(self, decisions: Dict = None):
        """تنفيذ قرارات المجلس"""
        from brain.brain_factory import BrainFactory

        if decisions is None:
            decisions = self.decide()

        factory = BrainFactory(self.capsules_dir)
        executed = []

        for action in decisions.get("actions", []):
            act = action["action"]
            cid = action.get("capsule", "")

            if act == "spawn_children":
                # إنتاج أطفال من الأقوياء
                result = factory.evolve()
                executed.append(f"evolve: +{len(result.get('created', []))}")

            elif act == "create_new_capsule":
                # كبسولة جديدة من الصفر
                new_dir = self.capsules_dir / cid
                if not new_dir.exists():
                    new_dir.mkdir(parents=True)
                    (new_dir / "data").mkdir()
                    (new_dir / "model").mkdir()
                    (new_dir / "meta.json").write_text(json.dumps({
                        "id": cid, "layer": 0, "children": [],
                        "created": datetime.now().isoformat(),
                        "created_by": "council",
                    }, indent=2))
                    factory._register_capsule(cid, cid)
                    executed.append(f"created: {cid}")

            elif act == "prioritize_training":
                executed.append(f"priority: {cid}")

            elif act == "request_more_data":
                executed.append(f"more_data: {cid}")

        logger.info(f"🏛️ Executed: {executed}")
        return executed


# Singleton
council = CouncilBrain()
