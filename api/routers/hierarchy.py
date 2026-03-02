"""
Hierarchy Routes - مسارات النظام الهرمي

API endpoints for the AI Hierarchy system.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hierarchy import ai_hierarchy, AlertLevel
from .auth import get_current_active_user

router = APIRouter(prefix="/hierarchy", tags=["hierarchy"])


class CommandRequest(BaseModel):
    """نموذج طلب تنفيذ أمر"""
    command: str = Field(..., min_length=1, max_length=10000)
    alert_level: str = Field(default="GREEN")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    """نموذج استجابة تنفيذ أمر"""
    command: str
    decision: Dict[str, Any]
    result: Dict[str, Any]
    council_consensus: Optional[float] = None
    balance_score: Optional[float] = None
    expert_recommendation: Optional[str] = None
    timestamp: str


class HierarchyStatusResponse(BaseModel):
    """نموذج حالة النظام الهرمي"""
    president: Dict[str, Any]
    council: Dict[str, Any]
    scouts: Dict[str, Any]
    meta: Dict[str, Any]
    experts: Dict[str, Any]
    execution: Dict[str, Any]
    rtx_config: Dict[str, str]
    is_initialized: bool
    active_mode: str


class LayerStatus(BaseModel):
    """نموذج حالة طبقة"""
    name: str
    specialization: str
    epoch: int
    loss: float
    accuracy: float
    samples: int
    fetches: int
    vram_gb: float


class HierarchyFullStatus(BaseModel):
    """الحالة الكاملة للنظام الهرمي"""
    is_training: bool
    device: str
    mode: str
    layers: Dict[str, LayerStatus]
    gpu: Optional[Dict[str, Any]] = None


class WisdomResponse(BaseModel):
    """نموذج حكمة اليوم"""
    wisdom: str
    source: str
    timestamp: str


class MetaStatusResponse(BaseModel):
    """نموذج حالة الطبقات الفوقية"""
    executive_controller: Dict[str, Any]
    builder_teams: Dict[str, int]
    can_create_layers: bool
    can_destroy_layers: bool
    can_rebuild_hierarchy: bool


@router.post("/execute", response_model=CommandResponse)
async def execute_command(
    request: CommandRequest,
    current_user = Depends(get_current_active_user)
):
    """
    تنفيذ أمر من خلال النظام الهرمي
    
    يمر الأمر بجميع الطبقات: الرئيس → المجلس → التوازن → الكشافة → الخبراء → التنفيذ
    """
    try:
        # Convert alert level string to enum
        alert_level = AlertLevel[request.alert_level.upper()]
        
        # Execute through hierarchy
        result = await ai_hierarchy.execute_command(
            command=request.command,
            alert_level=alert_level,
            context=request.context
        )
        
        return CommandResponse(
            command=result["command"],
            decision=result["decision"],
            result=result["result"],
            council_consensus=result.get("council_consensus"),
            balance_score=result.get("balance_score"),
            expert_recommendation=result.get("expert_recommendation"),
            timestamp=datetime.now().isoformat()
        )
        
    except KeyError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid alert level: {request.alert_level}. Use: GREEN, YELLOW, ORANGE, RED, BLACK"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=HierarchyStatusResponse)
async def get_hierarchy_status(
    current_user = Depends(get_current_active_user)
):
    """الحصول على حالة النظام الهرمي الكاملة"""
    try:
        status = ai_hierarchy.get_full_status()
        return HierarchyStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_hierarchy_metrics(
    current_user = Depends(get_current_active_user)
):
    """الحصول على مقاييس النظام الهرمي"""
    try:
        status = ai_hierarchy.get_full_status()
        
        # Calculate metrics
        metrics = {
            "council_active": status["council"].get("meeting_status") == "in_session",
            "council_consensus": status["council"].get("consensus_rate", 0),
            "scouts_intel_count": status["scouts"].get("intel_buffer_size", 0),
            "experts_count": status["experts"].get("total", 0),
            "execution_active_missions": status["execution"].get("active_missions", 0),
            "rtx_host": status["rtx_config"]["host"],
            "rtx_port": status["rtx_config"]["port"],
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wisdom", response_model=WisdomResponse)
async def get_wisdom(
    horizon: str = "century",
    current_user = Depends(get_current_active_user)
):
    """الحصول على حكمة من النظام"""
    try:
        wisdom = ai_hierarchy.get_wisdom()
        return WisdomResponse(
            wisdom=wisdom,
            source="seventh_dimension",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/guardian")
async def get_guardian_status(
    current_user = Depends(get_current_active_user)
):
    """الحصول على حالة الحارس (Guardian Layer)"""
    try:
        # Get guardian status from hierarchy
        status = ai_hierarchy.get_full_status()
        
        return {
            "guardian_active": True,
            "veto_power": status["president"].get("veto_power", False),
            "security_level": "high",
            "last_check": datetime.now().isoformat(),
            "threats_detected": 0,
            "protected_layers": ["council", "president", "execution"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers")
async def get_layers_status(
    current_user = Depends(get_current_active_user)
):
    """الحصول على حالة جميع الطبقات"""
    try:
        status = ai_hierarchy.get_full_status()
        
        layers = {
            "president": {
                "status": "active" if status["president"].get("in_meeting") else "standby",
                "veto_power": status["president"].get("veto_power")
            },
            "council": status["council"],
            "scouts": {
                "status": "active",
                "intel_buffer": status["scouts"].get("intel_buffer_size", 0)
            },
            "meta": status["meta"],
            "experts": status["experts"],
            "execution": status["execution"]
        }
        
        return layers
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_hierarchy(
    current_user = Depends(get_current_active_user)
):
    """تهيئة النظام الهرمي"""
    try:
        result = await ai_hierarchy.initialize()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta", response_model=MetaStatusResponse)
async def get_meta_status(
    current_user = Depends(get_current_active_user)
):
    """الحصول على حالة الطبقات الفوقية (Meta Layers)"""
    try:
        status = ai_hierarchy.get_meta_status()
        return MetaStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presidential-order")
async def send_presidential_order(
    order: str,
    params: Optional[Dict[str, Any]] = None,
    current_user = Depends(get_current_active_user)
):
    """إرسال أمر رئاسي مباشر للحكيم التنفيذي"""
    try:
        result = await ai_hierarchy.send_presidential_order(order, params or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-layer")
async def create_new_layer(
    name: str,
    layer_type: str = "EXECUTIVE",
    components: Optional[List[str]] = None,
    connections: Optional[List[str]] = None,
    current_user = Depends(get_current_active_user)
):
    """بناء طبقة جديدة ديناميكياً"""
    try:
        result = await ai_hierarchy.create_new_layer(
            name=name,
            layer_type=layer_type,
            components=components,
            connections=connections
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-hierarchy")
async def create_new_hierarchy(
    name: str,
    layers: int = 3,
    current_user = Depends(get_current_active_user)
):
    """إنشاء هيكل هرمي جديد منفصل"""
    try:
        result = await ai_hierarchy.create_new_hierarchy(name, layers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
