"""
Council Routes - مسارات المجلس

Council API endpoints with standardized RTX configuration.
"""


from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time

from hierarchy import ai_hierarchy, RTX_HOST, RTX_PORT, RTX_URL
from api.routers.auth import get_current_active_user

router = APIRouter(prefix="/council", tags=["council"])
RTX4090_URL = RTX_URL
SMART_COUNCIL_AVAILABLE = True


def _check_rtx4090(timeout: float = 2.0) -> bool:
    try:
        import requests
        response = requests.get(f"{RTX4090_URL}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def check_rtx4090_with_retry(timeout: float = 5.0, retries: int = 3, retry_delay: float = 0.5) -> bool:
    for _ in range(max(1, retries)):
        if _check_rtx4090(timeout=timeout):
            return True
        time.sleep(max(0.0, retry_delay))
    return False


def send_rtx4090_request_with_retry(
    endpoint: str,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    retries: int = 3,
    retry_delay: float = 0.5,
):
    try:
        import requests
    except Exception:
        return None

    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{RTX4090_URL}{endpoint}"

    for _ in range(max(1, retries)):
        try:
            response = requests.post(url, json=json_data or {}, timeout=timeout)
            return response
        except Exception:
            time.sleep(max(0.0, retry_delay))

    return None


def get_live_metrics_snapshot() -> Dict[str, Any]:
    return {
        "rtx_connected": _check_rtx4090(timeout=1.5),
        "smart_council_available": SMART_COUNCIL_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


class MessageRequest(BaseModel):
    """نموذج طلب الرسالة"""
    message: str = Field(..., min_length=1, max_length=10000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """نموذج استجابة الرسالة - Standard Schema"""
    response: str
    source: str  # rtx4090, local-fallback, hierarchy
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list = Field(default_factory=list)
    response_source: str  # For backward compatibility
    wise_man: str
    processing_time_ms: int = 0
    timestamp: str


class CouncilStatusResponse(BaseModel):
    """نموذج حالة المجلس"""
    is_meeting: bool
    wise_men_count: int
    active_sages: int
    meeting_status: str
    president_present: bool
    topics_discussed: int
    consensus_rate: float
    rtx_config: Dict[str, Any]


@router.post("/message", response_model=MessageResponse)
async def council_message(
    request: MessageRequest,
    # Auth removed: desktop app sends messages without token
    # current_user = Depends(get_current_active_user)
):
    """
    إرسال رسالة للمجلس والحصول على رد
    
    Pipeline:
    1. Try RTX 4090 server first
    2. Fallback to local hierarchy
    """
    start_time = time.time()
    
    try:
        # Try hierarchy ask() which handles RTX + fallback
        result = ai_hierarchy.ask(request.message)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return MessageResponse(
            response=result.get("response", ""),
            source=result.get("response_source", "unknown"),
            confidence=result.get("confidence", 0.5),
            evidence=result.get("evidence", []),
            response_source=result.get("response_source", "unknown"),
            wise_man=result.get("wise_man", "المجلس"),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=CouncilStatusResponse)
async def get_council_status(
    current_user = Depends(get_current_active_user)
):
    """الحصول على حالة المجلس"""
    status = ai_hierarchy.get_council_status()
    
    return CouncilStatusResponse(
        **status,
        rtx_config={
            "host": RTX_HOST,
            "port": RTX_PORT,
            "url": RTX_URL
        }
    )


@router.get("/members")
async def get_council_members(
    current_user = Depends(get_current_active_user)
):
    """الحصول على قائمة أعضاء المجلس"""
    members = ai_hierarchy.get_all_wise_men()
    return {"members": members}


from datetime import datetime
