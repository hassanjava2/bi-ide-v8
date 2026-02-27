"""
RTX 4090 Routes - API endpoints for RTX 4090 management
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
import asyncio
from pydantic import BaseModel

from ai.rtx4090_client import RTX4090Client, RTX4090Pool, get_rtx4090_client
from ai.training.rtx4090_trainer import RTX4090Trainer, get_rtx4090_trainer

router = APIRouter(prefix="/rtx4090", tags=["RTX 4090"])


# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7


class TrainRequest(BaseModel):
    wise_man: str
    training_data: List[Dict[str, str]]
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8


# Global pool instance
_rtx4090_pool: Optional[RTX4090Pool] = None


def get_pool() -> RTX4090Pool:
    """Get or create RTX 4090 pool"""
    global _rtx4090_pool
    if _rtx4090_pool is None:
        _rtx4090_pool = RTX4090Pool()
    return _rtx4090_pool


@router.get("/health")
async def rtx4090_health() -> Dict[str, Any]:
    """Check RTX 4090 server health"""
    async with RTX4090Client() as client:
        try:
            health = await client.health_check()
            status = await client.get_status()
            return {
                "status": "connected",
                "server": health.get("name", "RTX 4090 Server"),
                "version": health.get("version", "unknown"),
                "gpu": health.get("gpu", "unknown"),
                "mode": health.get("mode", "unknown"),
                "details": status
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "error": str(e)
            }


@router.get("/pool/health")
async def pool_health() -> Dict[str, Any]:
    """Check health of all RTX 4090 servers in pool"""
    pool = get_pool()
    health_results = await pool.health_check_all()
    
    healthy_count = sum(1 for h in health_results.values() if h.get("healthy"))
    total_count = len(health_results)
    
    return {
        "total_servers": total_count,
        "healthy_servers": healthy_count,
        "unhealthy_servers": total_count - healthy_count,
        "servers": health_results
    }


@router.post("/generate")
async def generate_text(request: GenerateRequest) -> Dict[str, Any]:
    """Generate text using RTX 4090"""
    async with RTX4090Client() as client:
        try:
            text = await client.generate_text(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream
            )
            return {
                "success": True,
                "text": text,
                "prompt": request.prompt,
                "parameters": {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_completion(request: ChatRequest) -> Dict[str, Any]:
    """Chat completion with message history"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    async with RTX4090Client() as client:
        try:
            response = await client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            return {
                "success": True,
                "response": response,
                "messages_count": len(messages)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints")
async def list_checkpoints() -> Dict[str, Any]:
    """List all checkpoints on RTX 4090 server"""
    async with RTX4090Client() as client:
        try:
            checkpoints = await client.list_checkpoints()
            return {
                "success": True,
                "count": len(checkpoints),
                "checkpoints": checkpoints
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoints/sync")
async def sync_checkpoints(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Sync all checkpoints from RTX 4090 to Windows"""
    async def do_sync():
        async with RTX4090Client() as client:
            return await client.sync_checkpoints_to_windows()
    
    background_tasks.add_task(do_sync)
    return {
        "success": True,
        "message": "Checkpoint sync started in background"
    }


@router.post("/train")
async def train_model(request: TrainRequest) -> Dict[str, Any]:
    """Start training job on RTX 4090"""
    trainer = get_rtx4090_trainer()
    
    result = await trainer.train_wise_man(
        wise_man_name=request.wise_man,
        training_data=request.training_data,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Training failed"))
    
    return result


@router.get("/train/status/{job_id}")
async def training_status(job_id: str) -> Dict[str, Any]:
    """Get status of a training job"""
    trainer = get_rtx4090_trainer()
    status = await trainer.get_training_status(job_id)
    return status


@router.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get loaded model information"""
    async with RTX4090Client() as client:
        try:
            info = await client.get_model_info()
            return {
                "success": True,
                "model": info
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/start")
async def start_learning() -> Dict[str, Any]:
    """Start real API learning on RTX 4090"""
    async with RTX4090Client() as client:
        try:
            result = await client.start_learning()
            return {
                "success": True,
                "status": result.get("status"),
                "message": result.get("message")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/stop")
async def stop_learning() -> Dict[str, Any]:
    """Stop real API learning on RTX 4090"""
    async with RTX4090Client() as client:
        try:
            result = await client.stop_learning()
            return {
                "success": True,
                "status": result.get("status")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/status")
async def learning_status() -> Dict[str, Any]:
    """Get real API learning status"""
    async with RTX4090Client() as client:
        try:
            status = await client.get_status()
            return {
                "success": True,
                "status": status
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get RTX 4090 server statistics"""
    pool = get_pool()
    
    # Get pool health
    health_results = await pool.health_check_all()
    healthy_count = sum(1 for h in health_results.values() if h.get("healthy"))
    
    # Get model info from first healthy server
    model_info_data = None
    checkpoints_count = 0
    
    for host, health in health_results.items():
        if health.get("healthy"):
            async with RTX4090Client(host=host) as client:
                try:
                    model_info_data = await client.get_model_info()
                    checkpoints = await client.list_checkpoints()
                    checkpoints_count = len(checkpoints)
                    break
                except:
                    continue
    
    return {
        "success": True,
        "pool": {
            "total_servers": len(pool.hosts),
            "healthy_servers": healthy_count,
            "servers": list(pool.hosts)
        },
        "model": model_info_data,
        "checkpoints_count": checkpoints_count
    }


# WebSocket endpoint for streaming (if needed)
# @router.websocket("/generate/stream")
# async def generate_stream_ws(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         data = await websocket.receive_json()
#         prompt = data.get("prompt", "")
#         
#         async with RTX4090Client() as client:
#             async for token in client.generate_stream(prompt):
#                 await websocket.send_json({"token": token})
#     except Exception as e:
#         await websocket.send_json({"error": str(e)})
#     finally:
#         await websocket.close()
