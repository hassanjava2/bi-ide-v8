"""
Network Routes - نقاط النهاية للشبكة المتخصصة
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    SpecializationExpandRequest,
    WorkerRegisterRequest,
    WorkerHeartbeatRequest,
    TrainingTaskCreateRequest,
    TrainingTaskClaimRequest,
    TrainingTaskCompleteRequest,
    DualThoughtRequest,
)

router = APIRouter(prefix="/api/v1/network", tags=["network"])

# Service reference
_network_service = None


def set_network_service(service):
    global _network_service
    _network_service = service


def _svc():
    if _network_service is None:
        raise HTTPException(503, "Specialized network not initialized")
    return _network_service


@router.get("/status")
async def network_status():
    return {"status": "ok", "network": _svc().get_status()}


@router.get("/graph")
async def network_graph():
    return {"status": "ok", "graph": _svc().get_graph()}


@router.post("/graph/expand")
async def network_expand(request: SpecializationExpandRequest):
    try:
        node = _svc().expand_specialization(
            parent_id=request.parent_id, name=request.name, description=request.description,
        )
        return {"status": "ok", "node": node}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/workers/register")
async def network_register_worker(request: WorkerRegisterRequest):
    worker = _svc().register_worker(
        worker_id=request.worker_id, hostname=request.hostname, capabilities=request.capabilities,
    )
    return {"status": "ok", "worker": worker}


@router.post("/workers/heartbeat")
async def network_worker_heartbeat(request: WorkerHeartbeatRequest):
    worker = _svc().heartbeat_worker(
        worker_id=request.worker_id, status=request.status, capabilities=request.capabilities,
    )
    return {"status": "ok", "worker": worker}


@router.post("/training/enqueue")
async def network_training_enqueue(request: TrainingTaskCreateRequest):
    try:
        task = _svc().enqueue_training_task(
            topic=request.topic, node_id=request.node_id, priority=request.priority,
        )
        return {"status": "ok", "task": task}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/training/claim")
async def network_training_claim(request: TrainingTaskClaimRequest):
    task = _svc().claim_training_task(request.worker_id)
    return {"status": "ok", "task": task}


@router.post("/training/complete")
async def network_training_complete(request: TrainingTaskCompleteRequest):
    try:
        task = _svc().complete_training_task(
            task_id=request.task_id, worker_id=request.worker_id,
            metrics=request.metrics, artifact_name=request.artifact_name,
            artifact_payload=request.artifact_payload,
        )
        return {"status": "ok", "task": task}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/think/dual")
async def network_dual_thought(request: DualThoughtRequest):
    try:
        result = _svc().dual_thought(node_id=request.node_id, prompt=request.prompt)
        return {"status": "ok", "thought": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
