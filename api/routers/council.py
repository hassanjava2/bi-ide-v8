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
from services.council_service import council_service, DecisionStatus as ServiceDecisionStatus

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
    id: str
    name: str
    role: MemberRole
    expertise: List[str]
    is_active: bool
    joined_at: datetime


class CouncilDecision(BaseModel):
    """نموذج قرار المجلس | Council decision model"""
    id: str
    title: str
    description: str
    status: DecisionStatus
    proposed_by: str
    votes: dict
    created_at: datetime
    decided_at: Optional[datetime] = None


class VoteRequest(BaseModel):
    """نموذج طلب التصويت | Vote request model"""
    decision_id: str
    vote: VoteType
    comment: Optional[str] = None


class QueryRequest(BaseModel):
    """نموذج طلب الاستعلام | Query request model"""
    query: str
    context: Optional[dict] = None


class QueryResponse(BaseModel):
    """نموذج استجابة الاستعلام | Query response model"""
    decision_id: str
    recommendation: str
    confidence: float
    reasoning: List[str]
    members_consulted: List[str]


class CouncilStatus(BaseModel):
    """نموذج حالة المجلس | Council status model"""
    is_active: bool
    members_online: int
    active_deliberations: int
    pending_decisions: int
    total_decisions: int


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
    decision = await council_service.query_council(
        query=request.query,
        context=request.context or {},
        use_cache=True,
    )

    active_members = await council_service.list_members(active_only=True)

    return QueryResponse(
        decision_id=decision.decision_id,
        recommendation=decision.response,
        confidence=decision.confidence,
        reasoning=decision.evidence or [],
        members_consulted=[m.member_id for m in active_members],
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
    members = await council_service.list_members(active_only=False)
    decisions = await council_service.get_decisions(limit=1000)

    members_online = sum(1 for m in members if m.is_active)
    pending = sum(1 for d in decisions if d.status == ServiceDecisionStatus.PENDING)
    deliberating = sum(1 for d in decisions if d.status == ServiceDecisionStatus.NEEDS_REVIEW)
    
    return CouncilStatus(
        is_active=members_online > 0,
        members_online=members_online,
        active_deliberations=deliberating,
        pending_decisions=pending,
        total_decisions=len(decisions)
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
    service_status = None
    if status:
        status_map = {
            DecisionStatus.PENDING: ServiceDecisionStatus.PENDING,
            DecisionStatus.APPROVED: ServiceDecisionStatus.APPROVED,
            DecisionStatus.REJECTED: ServiceDecisionStatus.REJECTED,
            DecisionStatus.DELIBERATING: ServiceDecisionStatus.NEEDS_REVIEW,
        }
        service_status = status_map.get(status)

    decisions = await council_service.get_decisions(status=service_status, limit=200)

    status_reverse_map = {
        ServiceDecisionStatus.PENDING: DecisionStatus.PENDING,
        ServiceDecisionStatus.APPROVED: DecisionStatus.APPROVED,
        ServiceDecisionStatus.REJECTED: DecisionStatus.REJECTED,
        ServiceDecisionStatus.NEEDS_REVIEW: DecisionStatus.DELIBERATING,
    }

    return [
        CouncilDecision(
            id=d.decision_id,
            title=f"Query: {d.query[:50]}...",
            description=d.query,
            status=status_reverse_map.get(d.status, DecisionStatus.DELIBERATING),
            proposed_by="system",
            votes=d.votes,
            created_at=d.created_at,
            decided_at=d.created_at if d.status in {ServiceDecisionStatus.APPROVED, ServiceDecisionStatus.REJECTED} else None,
        )
        for d in decisions
    ]


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
    members = await council_service.list_members(active_only=False)

    role_map = {
        "system_architect": MemberRole.CHAIR,
        "security_expert": MemberRole.MEMBER,
        "performance_expert": MemberRole.MEMBER,
        "ux_expert": MemberRole.ADVISOR,
    }

    return [
        CouncilMember(
            id=m.member_id,
            name=m.name,
            role=role_map.get(m.role, MemberRole.ADVISOR),
            expertise=m.expertise,
            is_active=m.is_active,
            joined_at=m.joined_at,
        )
        for m in members
    ]


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
    # map vote to service vote format
    vote_map = {
        VoteType.APPROVE: "approve",
        VoteType.REJECT: "reject",
        VoteType.ABSTAIN: "abstain",
    }

    members = await council_service.list_members(active_only=True)
    member_id = members[0].member_id if members else "architect_1"

    success = await council_service.submit_vote(
        decision_id=vote.decision_id,
        member_id=member_id,
        vote=vote_map[vote.vote],
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="القرار غير موجود أو التصويت فشل | Decision not found or vote failed",
        )

    decision = await council_service.get_status(vote.decision_id)
    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="القرار غير موجود | Decision not found",
        )

    status_map = {
        ServiceDecisionStatus.PENDING: DecisionStatus.PENDING,
        ServiceDecisionStatus.APPROVED: DecisionStatus.APPROVED,
        ServiceDecisionStatus.REJECTED: DecisionStatus.REJECTED,
        ServiceDecisionStatus.NEEDS_REVIEW: DecisionStatus.DELIBERATING,
    }

    return CouncilDecision(
        id=decision.decision_id,
        title=f"Query: {decision.query[:50]}...",
        description=decision.query,
        status=status_map.get(decision.status, DecisionStatus.DELIBERATING),
        proposed_by="system",
        votes=decision.votes,
        created_at=decision.created_at,
        decided_at=decision.created_at if decision.status in {ServiceDecisionStatus.APPROVED, ServiceDecisionStatus.REJECTED} else None,
    )


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
