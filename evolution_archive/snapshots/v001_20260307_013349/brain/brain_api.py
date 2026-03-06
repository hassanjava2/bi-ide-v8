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
        }

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
🧠 BI-IDE Brain API v1.0 — http://localhost:{args.port}
{'═' * 40}
/api/chat          /api/council/debate
/api/moe/route     /api/curriculum/status
/api/factory/plan  /api/brain/think
/api/bios/shell    /api/learner/file
/api/status        /api/memory/stats
{'═' * 40}""")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
