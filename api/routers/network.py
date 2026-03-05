"""
Network Router - إدارة الشبكة والاتصال
Provides endpoints for network status and connectivity monitoring.
"""

import os
import logging
import time
from typing import Dict, Any, List

from fastapi import APIRouter
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/network", tags=["الشبكة | Network"])


# Service endpoints to monitor
SERVICES = {
    "rtx5090": {
        "name": "RTX 5090 GPU Server",
        "url": f"http://{os.getenv('RTX5090_HOST', '192.168.1.164')}:{os.getenv('RTX5090_PORT', '8090')}/health",
        "critical": True,
    },
    "redis": {
        "name": "Redis Cache",
        "url": f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}",
        "critical": True,
    },
    "vps": {
        "name": "VPS (bi-iq.com)",
        "url": "https://bi-iq.com/health",
        "critical": False,
    },
}


async def _check_http_service(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Check HTTP service availability"""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.get(url)
            latency = (time.time() - start) * 1000
            return {
                "status": "up" if response.status_code < 500 else "degraded",
                "status_code": response.status_code,
                "latency_ms": round(latency, 1),
            }
    except httpx.ConnectError:
        return {"status": "down", "error": "Connection refused"}
    except httpx.TimeoutException:
        return {"status": "down", "error": "Timeout"}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


async def _check_redis() -> Dict[str, Any]:
    """Check Redis availability"""
    start = time.time()
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url)
        await r.ping()
        await r.close()
        latency = (time.time() - start) * 1000
        return {"status": "up", "latency_ms": round(latency, 1)}
    except Exception as e:
        return {"status": "down", "error": str(e)[:100]}


@router.get(
    "/status",
    summary="حالة الشبكة الكاملة | Full Network Status"
)
async def network_status():
    """
    فحص حالة جميع الخدمات المتصلة.
    Check connectivity status of all connected services.
    """
    results = {}
    
    for service_id, service in SERVICES.items():
        if service_id == "redis":
            results[service_id] = {
                "name": service["name"],
                "critical": service["critical"],
                **await _check_redis()
            }
        elif service["url"].startswith("http"):
            results[service_id] = {
                "name": service["name"],
                "url": service["url"],
                "critical": service["critical"],
                **await _check_http_service(service["url"])
            }
    
    # Overall status
    critical_services = {k: v for k, v in results.items() if SERVICES[k]["critical"]}
    all_critical_up = all(v.get("status") == "up" for v in critical_services.values())
    any_down = any(v.get("status") == "down" for v in results.values())
    
    overall = "healthy" if all_critical_up and not any_down else (
        "degraded" if all_critical_up else "critical"
    )
    
    return {
        "overall_status": overall,
        "services": results,
        "timestamp": time.time(),
        "total_services": len(results),
        "services_up": sum(1 for v in results.values() if v.get("status") == "up"),
        "services_down": sum(1 for v in results.values() if v.get("status") == "down"),
    }


@router.get(
    "/ping/{service_id}",
    summary="فحص خدمة محددة | Ping Specific Service"
)
async def ping_service(service_id: str):
    """
    فحص سريع لخدمة محددة.
    Quick ping check for a specific service.
    """
    if service_id not in SERVICES:
        return {
            "error": f"Unknown service: {service_id}",
            "available_services": list(SERVICES.keys())
        }
    
    service = SERVICES[service_id]
    
    if service_id == "redis":
        result = await _check_redis()
    else:
        result = await _check_http_service(service["url"])
    
    return {
        "service": service_id,
        "name": service["name"],
        **result
    }


@router.get(
    "/topology",
    summary="طوبولوجيا الشبكة | Network Topology"
)
async def network_topology():
    """
    عرض طوبولوجيا الشبكة وبنية الخدمات.
    Display network topology and service architecture.
    """
    return {
        "architecture": "hybrid",
        "nodes": {
            "mac_dev": {
                "type": "development",
                "role": "API Server + Desktop App",
                "location": "local",
            },
            "rtx5090": {
                "type": "gpu_compute",
                "role": "Model Training + Inference",
                "host": os.getenv("RTX5090_HOST", "192.168.1.164"),
                "port": int(os.getenv("RTX5090_PORT", "8090")),
                "location": "LAN",
            },
            "vps": {
                "type": "cloud",
                "role": "Public API + Monitoring Hub",
                "domain": "bi-iq.com",
                "location": "cloud",
            },
        },
        "connections": [
            {"from": "mac_dev", "to": "rtx5090", "type": "LAN", "protocol": "HTTP"},
            {"from": "mac_dev", "to": "vps", "type": "Internet/Tailscale", "protocol": "HTTPS"},
            {"from": "rtx5090", "to": "vps", "type": "Tailscale", "protocol": "HTTPS"},
        ],
        "failover_chain": ["rtx5090 (LAN)", "vps (Cloud)", "local (Fallback)"],
    }
