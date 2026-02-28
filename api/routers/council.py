"""
روتر المجلس - Council Router

يوفر نقاط النهاية للمجلس الاستشاري والتصويت.
Provides endpoints for council deliberation and voting.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .auth import get_current_active_user, User

router = APIRouter(prefix="/council", tags=["المجلس | Council"])


class DecisionStatus(str, Enum):
    """حالة القرار | Decision status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELIBERATING = "deliberating"


class VoteType(str, Enum):
    """نوع التصويت | Vote type"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class MemberRole(str, Enum):
    """دور العضو | Member role"""
    CHAIR = "chair"
    MEMBER = "member"
    ADVISOR = "advisor"


# نماذج Pydantic - Pydantic Models
class CouncilMember(BaseModel):
    """نموذج عضو المجلس | Council member model"""
    id: int
    name: str
    role: MemberRole
    expertise: List[str]
    is_active: bool
    joined_at: datetime


class CouncilDecision(BaseModel):
    """نموذج قرار المجلس | Council decision model"""
    id: int
    title: str
    description: str
    status: DecisionStatus
    proposed_by: int
    votes: dict
    created_at: datetime
    decided_at: Optional[datetime] = None


class VoteRequest(BaseModel):
    """نموذج طلب التصويت | Vote request model"""
    decision_id: int
    vote: VoteType
    comment: Optional[str] = None


class QueryRequest(BaseModel):
    """نموذج طلب الاستعلام | Query request model"""
    query: str
    context: Optional[dict] = None


class QueryResponse(BaseModel):
    """نموذج استجابة الاستعلام | Query response model"""
    decision_id: int
    recommendation: str
    confidence: float
    reasoning: List[str]
    members_consulted: List[int]


class CouncilStatus(BaseModel):
    """نموذج حالة المجلس | Council status model"""
    is_active: bool
    members_online: int
    active_deliberations: int
    pending_decisions: int
    total_decisions: int


# قاعدة بيانات وهمية - Fake Database
fake_members = {
    1: {
        "id": 1,
        "name": "AI Architect",
        "role": MemberRole.CHAIR,
        "expertise": ["architecture", "design_patterns"],
        "is_active": True,
        "joined_at": datetime.utcnow()
    },
    2: {
        "id": 2,
        "name": "Code Reviewer",
        "role": MemberRole.MEMBER,
        "expertise": ["code_quality", "best_practices"],
        "is_active": True,
        "joined_at": datetime.utcnow()
    },
    3: {
        "id": 3,
        "name": "Security Expert",
        "role": MemberRole.MEMBER,
        "expertise": ["security", "compliance"],
        "is_active": True,
        "joined_at": datetime.utcnow()
    }
}

fake_decisions = {}
fake_decision_counter = 1

# WebSocket connections
websocket_connections: List[WebSocket] = []


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="الاستعلام من المجلس | Query the council"
)
async def query_council(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    إرسال استعلام إلى المجلس للحصول على توصية.
    Send a query to the council for recommendation.
    """
    global fake_decision_counter
    
    # إنشاء قرار جديد | Create new decision
    decision_id = fake_decision_counter
    fake_decision_counter += 1
    
    decision = {
        "id": decision_id,
        "title": f"Query: {request.query[:50]}...",
        "description": request.query,
        "status": DecisionStatus.DELIBERATING,
        "proposed_by": current_user.id,
        "votes": {},
        "context": request.context or {},
        "created_at": datetime.utcnow(),
        "decided_at": None
    }
    
    fake_decisions[decision_id] = decision
    
    # محاكاة التوصية | Simulate recommendation
    members_consulted = [m["id"] for m in fake_members.values() if m["is_active"]]
    
    return QueryResponse(
        decision_id=decision_id,
        recommendation="يوصي المجلس بالموافقة على الطلب مع مراعاة أفضل الممارسات",
        confidence=0.85,
        reasoning=[
            "الطلب يتبع أنماط التصميم المعتمدة",
            "لا توجد مخاطر أمنية واضحة",
            "الأداء متوقع أن يكون ممتاز"
        ],
        members_consulted=members_consulted
    )


@router.get(
    "/status",
    response_model=CouncilStatus,
    status_code=status.HTTP_200_OK,
    summary="حالة المجلس | Council status"
)
async def get_council_status(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على حالة المجلس الحالية.
    Get current council status.
    """
    members_online = sum(1 for m in fake_members.values() if m["is_active"])
    pending = sum(
        1 for d in fake_decisions.values()
        if d["status"] == DecisionStatus.PENDING
    )
    deliberating = sum(
        1 for d in fake_decisions.values()
        if d["status"] == DecisionStatus.DELIBERATING
    )
    
    return CouncilStatus(
        is_active=members_online > 0,
        members_online=members_online,
        active_deliberations=deliberating,
        pending_decisions=pending,
        total_decisions=len(fake_decisions)
    )


@router.get(
    "/decisions",
    response_model=List[CouncilDecision],
    status_code=status.HTTP_200_OK,
    summary="قائمة القرارات | List decisions"
)
async def list_decisions(
    status: Optional[DecisionStatus] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على قائمة قرارات المجلس.
    Get list of council decisions.
    """
    decisions = list(fake_decisions.values())
    
    if status:
        decisions = [d for d in decisions if d["status"] == status]
    
    return [CouncilDecision(**d) for d in sorted(
        decisions,
        key=lambda x: x["created_at"],
        reverse=True
    )]


@router.get(
    "/members",
    response_model=List[CouncilMember],
    status_code=status.HTTP_200_OK,
    summary="قائمة الأعضاء | List members"
)
async def list_members(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على قائمة أعضاء المجلس.
    Get list of council members.
    """
    return [CouncilMember(**m) for m in fake_members.values()]


@router.post(
    "/vote",
    response_model=CouncilDecision,
    status_code=status.HTTP_200_OK,
    summary="تقديم تصويت | Submit vote"
)
async def submit_vote(
    vote: VoteRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    تقديم تصويت على قرار.
    Submit a vote on a decision.
    """
    if vote.decision_id not in fake_decisions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="القرار غير موجود | Decision not found"
        )
    
    decision = fake_decisions[vote.decision_id]
    
    if decision["status"] not in [DecisionStatus.PENDING, DecisionStatus.DELIBERATING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكن التصويت على هذا القرار | Cannot vote on this decision"
        )
    
    # تسجيل التصويت | Record vote
    decision["votes"][current_user.id] = {
        "vote": vote.vote,
        "comment": vote.comment,
        "voted_at": datetime.utcnow()
    }
    
    # التحقق من اكتمال التصويت | Check if voting complete
    active_members = len([m for m in fake_members.values() if m["is_active"]])
    if len(decision["votes"]) >= active_members:
        # احتساب النتيجة | Calculate result
        approve_count = sum(
            1 for v in decision["votes"].values()
            if v["vote"] == VoteType.APPROVE
        )
        reject_count = sum(
            1 for v in decision["votes"].values()
            if v["vote"] == VoteType.REJECT
        )
        
        if approve_count > reject_count:
            decision["status"] = DecisionStatus.APPROVED
        else:
            decision["status"] = DecisionStatus.REJECTED
        
        decision["decided_at"] = datetime.utcnow()
    
    return CouncilDecision(**decision)


# WebSocket endpoint للمناقشة الفورية
@router.websocket("/ws/deliberation")
async def deliberation_websocket(websocket: WebSocket):
    """
    WebSocket للمناقشة الفورية في المجلس.
    WebSocket for real-time council deliberation.
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "متصل بالمجلس | Connected to council",
            "members_online": len(websocket_connections)
        })
        
        while True:
            data = await websocket.receive_json()
            
            # معالجة الرسائل | Process messages
            message_type = data.get("type", "message")
            
            if message_type == "message":
                # بث الرسالة لجميع المتصلين | Broadcast to all
                for conn in websocket_connections:
                    await conn.send_json({
                        "type": "message",
                        "from": data.get("from", "unknown"),
                        "content": data.get("content"),
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            elif message_type == "typing":
                # إشارة الكتابة | Typing indicator
                for conn in websocket_connections:
                    if conn != websocket:
                        await conn.send_json({
                            "type": "typing",
                            "from": data.get("from", "unknown")
                        })
    
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        # إخطار الآخرين بالانفصال | Notify others
        for conn in websocket_connections:
            await conn.send_json({
                "type": "disconnected",
                "members_online": len(websocket_connections)
            })
