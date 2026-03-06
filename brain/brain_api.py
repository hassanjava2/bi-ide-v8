#!/usr/bin/env python3
"""
brain_api.py — API موحد لكل وحدات الدماغ 🧠🌐

سيرفر واحد يربط الـ IDE بكل شي:
  Chat, Council, AutoProgrammer, Memory, MoE, Curriculum,
  BI-OS, Real Life, Advanced Brain, File Learner

Port: 8400
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("brain_api")
PROJECT_ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from brain.memory_system import memory
from brain.chat_bridge import bridge as chat_bridge
from brain.council_auto_debate import council
from brain.auto_programmer import programmer
from brain.moe_router import moe
from brain.curriculum_learning import curriculum
from brain.bi_os_kernel import kernel as bios_kernel
from brain.real_life_layer import planner as factory_planner
from brain.advanced_brain import brain as advanced_brain
from brain.ide_file_learner import learner as file_learner
from brain.capsule_tree import tree as capsule_tree
from brain.virtual_factory import factory_manager, FACTORY_CATALOG
from brain.layer_system import layer_manager
from brain.vision_layer import vision
from brain.self_evolution_loop import evolver
from brain.knowledge_scout import unified_scout


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        print("❌ pip install fastapi uvicorn pydantic")
        return None

    app = FastAPI(title="BI-IDE Brain API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class ChatReq(BaseModel):
        message: str
        session_id: str = None
        mode: str = "fast"

    class CouncilReq(BaseModel):
        topic: str

    class SageReq(BaseModel):
        sage_id: str
        question: str

    class ProjectReq(BaseModel):
        description: str
        output_dir: str = None
        mode: str = "fast"

    class FactoryReq(BaseModel):
        product: str
        capacity: float = 50000
        location: str = "Iraq"

    class FileLearnReq(BaseModel):
        file_path: str
        content: str = None

    class EditLearnReq(BaseModel):
        file_path: str
        before: str
        after: str

    class TrainingReq(BaseModel):
        capsule_id: str
        samples: int
        accuracy: float
        cycle: int = 0

    class ShellReq(BaseModel):
        command: str

    class VetoReq(BaseModel):
        factory_id: str = None
        product: str = None
        reason: str = ""

    class UserCommandReq(BaseModel):
        command: str
        target: str = None
        params: dict = None

    class InheritReq(BaseModel):
        parent_id: str
        child_id: str

    class NodeReq(BaseModel):
        node_id: str
        name: str
        name_ar: str
        node_type: str = "capsule"
        parent_ids: list = []
        keywords: list = []

    class AnalyzeReq(BaseModel):
        filepath: str

    class EvolutionReq(BaseModel):
        reason: str = ""

    # ─── Chat ───
    @app.post("/api/chat")
    async def chat(r: ChatReq):
        return chat_bridge.chat(r.message, r.session_id, r.mode)

    @app.get("/api/chat/history/{session_id}")
    async def chat_history(session_id: str):
        return {"messages": chat_bridge.get_history(session_id)}

    # ─── Council ───
    @app.post("/api/council/debate")
    async def council_debate(r: CouncilReq):
        d = council.debate(r.topic)
        return {
            "topic": d.topic,
            "opinions": [{"sage_id": o.sage_id, "sage_name": o.sage_name,
                          "opinion": o.opinion, "vote": o.vote,
                          "confidence": o.confidence, "icon": o.icon} for o in d.opinions],
            "votes": d.votes, "decision": d.final_decision,
            "formatted": council.format_debate(d),
        }

    @app.get("/api/council/sages")
    async def council_sages():
        return council.get_all_sages()

    @app.post("/api/council/ask")
    async def ask_sage(r: SageReq):
        o = council.ask_sage(r.sage_id, r.question)
        return {"sage_id": o.sage_id, "sage_name": o.sage_name,
                "opinion": o.opinion, "vote": o.vote,
                "confidence": o.confidence, "icon": o.icon}

    # ─── AutoProgrammer ───
    @app.post("/api/programmer/create")
    async def create_project(r: ProjectReq):
        p = programmer.create_project(r.description, r.output_dir, r.mode)
        return {"name": p.name, "type": p.project_type,
                "files": [{"path": t.file_path, "lang": t.language,
                           "status": t.status} for t in p.tasks],
                "output_dir": p.output_dir, "status": p.status}

    @app.get("/api/programmer/status")
    async def prog_status():
        return programmer.get_status()

    # ─── MoE ───
    @app.post("/api/moe/route")
    async def moe_route(r: ChatReq):
        d = moe.route(r.message)
        return {"expert": d.top_expert, "scores": d.all_scores,
                "confidence": d.confidence, "response": d.response}

    @app.get("/api/moe/status")
    async def moe_st():
        return moe.get_status()

    # ─── Curriculum ───
    @app.get("/api/curriculum/status")
    async def curr_status():
        return {"capsules": curriculum.get_all_status()}

    @app.get("/api/curriculum/next/{capsule_id}")
    async def curr_next(capsule_id: str):
        return curriculum.get_next_training(capsule_id)

    @app.post("/api/curriculum/report")
    async def curr_report(r: TrainingReq):
        curriculum.report_training(r.capsule_id, r.samples, r.accuracy, r.cycle)
        return {"status": "ok"}

    @app.get("/api/curriculum/class-report")
    async def class_report():
        return {"report": curriculum.get_class_report()}

    # ─── Memory ───
    @app.get("/api/memory/stats")
    async def mem_stats():
        return memory.get_stats()

    @app.post("/api/memory/search")
    async def mem_search(r: ChatReq):
        return {"results": memory.search_knowledge(r.message)}

    # ─── Factory ───
    @app.post("/api/factory/plan")
    async def factory_plan(r: FactoryReq):
        p = factory_planner.plan_factory(r.product, r.capacity, r.location)
        return {"product": p.product, "capacity": p.capacity_tons_per_year,
                "timeline_months": p.timeline_months,
                "chemistry": p.chemistry, "physics": p.physics,
                "economics": p.economics, "recommendations": p.recommendations,
                "formatted": factory_planner.format_plan(p)}

    # ─── Brain ───
    @app.get("/api/brain/think")
    async def think():
        return advanced_brain.think()

    @app.get("/api/brain/imagine")
    async def imagine():
        return advanced_brain.imagination.imagine()

    @app.get("/api/brain/awareness")
    async def awareness():
        return advanced_brain.awareness.assess()

    @app.get("/api/brain/warnings")
    async def warnings():
        return {"warnings": advanced_brain.sixth_sense.scan_system()}

    # ─── File Learner ───
    @app.post("/api/learner/file")
    async def learn_file(r: FileLearnReq):
        return file_learner.learn_from_file(r.file_path, r.content)

    @app.post("/api/learner/edit")
    async def learn_edit(r: EditLearnReq):
        return file_learner.learn_from_edit(r.file_path, r.before, r.after)

    # ─── BI-OS ───
    @app.post("/api/bios/shell")
    async def shell(r: ShellReq):
        return {"output": bios_kernel.shell(r.command)}

    @app.get("/api/bios/status")
    async def bios_st():
        return bios_kernel.get_status()

    @app.get("/api/bios/ps")
    async def bios_ps():
        return {"processes": bios_kernel.processes.list_processes()}

    # ─── Master Status ───
    @app.get("/api/status")
    async def master():
        return {
            "brain_api": "active",
            "memory": memory.get_stats(),
            "moe": {"experts": moe.get_status()["total_experts"],
                    "with_models": moe.get_status()["experts_with_models"]},
            "council": {"sages": len(council.get_all_sages())},
            "bios": bios_kernel.get_status(),
            "capsule_tree": capsule_tree.stats(),
            "factories": factory_manager.get_summary(),
            "layers": layer_manager.stats(),
            "vision": vision.get_status(),
            "evolution": evolver.get_status(),
            "scout": unified_scout.get_status(),
        }

    # ═══════════════════════════════════════════════════════
    # أشجار الكبسولات 🌳
    # ═══════════════════════════════════════════════════════

    @app.get("/api/capsules/tree")
    async def capsules_tree_view(root_id: str = None):
        return {"tree": capsule_tree.get_tree_view(root_id), "stats": capsule_tree.stats()}

    @app.get("/api/capsules/search")
    async def capsules_search(q: str, top_k: int = 5):
        results = capsule_tree.find_capsules(q, top_k)
        return {"results": [{"id": n.node_id, "name": n.name, "name_ar": n.name_ar,
                             "type": n.node_type, "keywords": n.keywords} for n in results]}

    @app.get("/api/capsules/stats")
    async def capsules_stats():
        return capsule_tree.stats()

    @app.post("/api/capsules/inherit")
    async def capsules_inherit(r: InheritReq):
        capsule_tree.inherit(r.parent_id, r.child_id)
        return {"status": "ok"}

    @app.post("/api/capsules/cascade/{root_id}")
    async def capsules_cascade(root_id: str):
        capsule_tree.cascade_inherit(root_id)
        return {"status": "ok", "stats": capsule_tree.stats()}

    @app.post("/api/capsules/add")
    async def capsules_add(r: NodeReq):
        n = capsule_tree.add_node(r.node_id, r.name, r.name_ar, r.node_type, r.parent_ids, r.keywords)
        return {"id": n.node_id, "name": n.name_ar, "level": n.level}

    @app.post("/api/capsules/expand")
    async def capsules_expand(r: ChatReq):
        new_nodes = capsule_tree.auto_expand(r.message)
        return {"new_nodes": new_nodes, "stats": capsule_tree.stats()}

    @app.get("/api/capsules/orphans")
    async def capsules_orphans():
        orphans = capsule_tree.check_orphans()
        return {"orphans": orphans, "count": len(orphans)}

    @app.post("/api/capsules/fix-orphans")
    async def capsules_fix_orphans():
        fixed = capsule_tree.fix_orphans()
        return {"fixed": fixed}

    # ═══════════════════════════════════════════════════════
    # المصانع 🏭
    # ═══════════════════════════════════════════════════════

    @app.get("/api/factories")
    async def factories_list():
        return factory_manager.get_summary()

    @app.get("/api/factories/catalog")
    async def factories_catalog():
        return {"catalog": FACTORY_CATALOG, "count": len(FACTORY_CATALOG)}

    @app.get("/api/factories/report")
    async def factories_report():
        return {"report": factory_manager.format_summary()}

    @app.post("/api/factories/create")
    async def factories_create(r: FactoryReq):
        f = factory_manager.create_factory(r.product, r.capacity, r.location)
        return f.get_report()

    @app.post("/api/factories/create-all")
    async def factories_create_all():
        created = factory_manager.create_all_essential()
        return {"created": created, "count": len(created)}

    @app.post("/api/factories/simulate")
    async def factories_simulate(days: int = 30):
        results = factory_manager.simulate_all(days)
        return {"results": results, "summary": factory_manager.get_summary()}

    @app.post("/api/factories/veto")
    async def factories_veto(r: VetoReq):
        factory_manager.user_veto(r.factory_id, r.product, r.reason)
        return {"status": "vetoed", "product": r.product}

    @app.post("/api/factories/command")
    async def factories_command(r: UserCommandReq):
        result = factory_manager.user_command(r.command, r.target, r.params)
        return {"status": "executed", "result": str(result)}

    @app.post("/api/factories/propose")
    async def factories_propose(r: FactoryReq):
        p = factory_manager.scout_propose_factory(r.product, f"Proposed by user: {r.location}")
        return p

    @app.post("/api/factories/council-review")
    async def factories_council_review():
        results = factory_manager.council_review_proposals()
        return {"reviewed": len(results), "results": results}

    # ═══════════════════════════════════════════════════════
    # الطبقات 🏛️
    # ═══════════════════════════════════════════════════════

    @app.get("/api/layers")
    async def layers_list():
        return {"hierarchy": layer_manager.get_hierarchy(), "stats": layer_manager.stats()}

    @app.get("/api/layers/stats")
    async def layers_stats():
        return layer_manager.stats()

    @app.post("/api/layers/detect")
    async def layers_detect(r: ChatReq):
        new_layers = layer_manager.auto_detect_needed_layers(r.message)
        return {"new_layers": new_layers, "stats": layer_manager.stats()}

    @app.post("/api/layers/process")
    async def layers_process(r: ChatReq):
        return layer_manager.process_request(r.message, r.mode)

    # ═══════════════════════════════════════════════════════
    # الرؤية 👁️
    # ═══════════════════════════════════════════════════════

    @app.post("/api/vision/analyze")
    async def vision_analyze(r: AnalyzeReq):
        result = vision.analyze(r.filepath)
        return {
            "source": result.source, "type": result.source_type,
            "summary": result.summary, "metadata": result.metadata,
            "objects": [{"label": o.label, "confidence": o.confidence} for o in result.objects],
        }

    @app.get("/api/vision/status")
    async def vision_status():
        return vision.get_status()

    @app.get("/api/vision/cameras")
    async def vision_cameras():
        return {"cameras": vision.camera.cameras, "count": len(vision.camera.cameras)}

    # ═══════════════════════════════════════════════════════
    # التطور الذاتي 🧬
    # ═══════════════════════════════════════════════════════

    @app.get("/api/evolution/status")
    async def evo_status():
        return evolver.get_status()

    @app.get("/api/evolution/health")
    async def evo_health():
        return evolver.check_health()

    @app.get("/api/evolution/improvements")
    async def evo_improvements():
        return {"improvements": evolver.find_improvements()}

    @app.post("/api/evolution/snapshot")
    async def evo_snapshot(r: EvolutionReq):
        return evolver.archive.snapshot(r.reason)

    @app.post("/api/evolution/propose")
    async def evo_propose():
        return evolver.propose_evolution()

    @app.post("/api/evolution/approve")
    async def evo_approve():
        return evolver.approve_evolution()

    @app.post("/api/evolution/reject")
    async def evo_reject(r: EvolutionReq):
        return evolver.reject_evolution(r.reason)

    @app.get("/api/evolution/versions")
    async def evo_versions():
        return {"versions": evolver.archive.list_versions()}

    # ═══════════════════════════════════════════════════════
    # الكشافة 🔍
    # ═══════════════════════════════════════════════════════

    @app.get("/api/scout/status")
    async def scout_status():
        return unified_scout.get_status()

    @app.post("/api/scout/cycle")
    async def scout_cycle():
        return unified_scout.scout_cycle()

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()

    app = create_app()
    if app:
        import uvicorn
        print(f"""
🧠 BI-IDE Brain API v2.0 — http://localhost:{args.port}
{'═' * 50}
/api/chat              /api/council/debate
/api/moe/route         /api/curriculum/status
/api/factory/plan      /api/brain/think
/api/bios/shell        /api/learner/file
/api/capsules/tree     /api/factories
/api/layers            /api/vision/analyze
/api/evolution/status  /api/scout/status
/api/status            /api/memory/stats
{'═' * 50}""")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
