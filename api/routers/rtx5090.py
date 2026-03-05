"""
RTX 5090 Router - إدارة والتحكم بـ RTX 5090
Provides endpoints for RTX 5090 GPU management and monitoring.
"""

import os
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rtx5090", tags=["RTX 5090 | GPU Management"])

# RTX Configuration
RTX_HOST = os.getenv("RTX5090_HOST", os.getenv("RTX4090_HOST", "192.168.1.164"))
RTX_PORT = int(os.getenv("RTX5090_PORT", os.getenv("RTX4090_PORT", "8090")))
RTX_BASE_URL = f"http://{RTX_HOST}:{RTX_PORT}"
RTX_TIMEOUT = 30.0


class RTXInferenceRequest(BaseModel):
    """طلب استدلال من RTX"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    model: Optional[str] = None


class RTXTrainingCommand(BaseModel):
    """أمر تدريب RTX"""
    action: str  # "start", "stop", "status", "pause"
    dataset: Optional[str] = None
    epochs: int = 1
    batch_size: int = 4


@router.get(
    "/health",
    summary="فحص حالة RTX 5090 | RTX 5090 Health Check"
)
async def rtx_health():
    """
    فحص سريع لحالة اتصال RTX 5090.
    Quick connectivity check for RTX 5090 server.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{RTX_BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "online",
                    "rtx_url": RTX_BASE_URL,
                    "rtx_response": data,
                    "latency_ms": response.elapsed.total_seconds() * 1000,
                }
    except httpx.ConnectError:
        pass
    except httpx.TimeoutException:
        pass
    except Exception as e:
        logger.warning(f"RTX health check error: {e}")
    
    return {
        "status": "offline",
        "rtx_url": RTX_BASE_URL,
        "message": "RTX 5090 is not reachable. Check network or if the server is running.",
    }


@router.get(
    "/status",
    summary="حالة RTX 5090 التفصيلية | RTX 5090 Detailed Status"
)
async def rtx_status():
    """
    الحصول على حالة تفصيلية لـ RTX 5090 (GPU، ذاكرة، حرارة).
    Get detailed RTX 5090 status (GPU usage, memory, temperature).
    """
    try:
        async with httpx.AsyncClient(timeout=RTX_TIMEOUT) as client:
            # Try multiple status endpoints
            for endpoint in ["/status", "/gpu/status", "/api/status"]:
                try:
                    response = await client.get(f"{RTX_BASE_URL}{endpoint}")
                    if response.status_code == 200:
                        return {
                            "status": "online",
                            "data": response.json(),
                            "endpoint": endpoint,
                        }
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"RTX status error: {e}")
    
    return {
        "status": "offline",
        "data": None,
        "message": "Could not retrieve RTX 5090 status",
    }


@router.post(
    "/inference",
    summary="إرسال طلب استدلال لـ RTX | Send Inference Request to RTX"
)
async def rtx_inference(request: RTXInferenceRequest):
    """
    إرسال طلب استدلال مباشر لـ RTX 5090.
    Send direct inference request to RTX 5090 server.
    """
    try:
        async with httpx.AsyncClient(timeout=RTX_TIMEOUT) as client:
            response = await client.post(
                f"{RTX_BASE_URL}/generate",
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "model": request.model,
                },
            )
            if response.status_code == 200:
                return {
                    "status": "success",
                    "result": response.json(),
                    "latency_ms": response.elapsed.total_seconds() * 1000,
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"RTX returned error: {response.text[:200]}"
                )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="RTX 5090 is offline")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="RTX 5090 request timed out")


@router.post(
    "/training",
    summary="التحكم بالتدريب | Training Control"
)
async def rtx_training(command: RTXTrainingCommand):
    """
    التحكم بعمليات التدريب على RTX 5090.
    Control training operations on RTX 5090.
    """
    try:
        async with httpx.AsyncClient(timeout=RTX_TIMEOUT) as client:
            response = await client.post(
                f"{RTX_BASE_URL}/training/{command.action}",
                json={
                    "dataset": command.dataset,
                    "epochs": command.epochs,
                    "batch_size": command.batch_size,
                },
            )
            return {
                "status": "success" if response.status_code == 200 else "error",
                "action": command.action,
                "result": response.json() if response.status_code == 200 else response.text[:200],
            }
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="RTX 5090 is offline")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="RTX 5090 request timed out")


@router.get(
    "/models",
    summary="قائمة النماذج المتوفرة | Available Models"
)
async def rtx_models():
    """
    الحصول على قائمة النماذج المتوفرة على RTX 5090.
    Get list of available models on RTX 5090.
    """
    try:
        async with httpx.AsyncClient(timeout=RTX_TIMEOUT) as client:
            response = await client.get(f"{RTX_BASE_URL}/models")
            if response.status_code == 200:
                return {"status": "success", "models": response.json()}
    except Exception as e:
        logger.warning(f"RTX models list error: {e}")
    
    return {"status": "offline", "models": [], "message": "RTX 5090 is not reachable"}


@router.get(
    "/config",
    summary="إعدادات الاتصال | Connection Config"
)
async def rtx_config():
    """
    عرض إعدادات الاتصال الحالية بـ RTX 5090.
    Show current RTX 5090 connection configuration.
    """
    return {
        "host": RTX_HOST,
        "port": RTX_PORT,
        "base_url": RTX_BASE_URL,
        "timeout": RTX_TIMEOUT,
        "env_vars": {
            "RTX5090_HOST": os.getenv("RTX5090_HOST", "not set"),
            "RTX5090_PORT": os.getenv("RTX5090_PORT", "not set"),
            "RTX4090_HOST": os.getenv("RTX4090_HOST", "not set (deprecated)"),
        },
    }
