"""
Support Tickets - تذاكر الدعم الفني

إدارة تذاكر الدعم مع:
- دورة حياة التذكرة
- إدارة الأولويات
- تتبع SLA
- ربط بقاعدة المعرفة
"""

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class TicketStatus(Enum):
    """حالات التذكرة"""
    NEW = "new"                       # جديدة
    ASSIGNED = "assigned"             # معينة
    IN_PROGRESS = "in_progress"       # قيد المعالجة
    PENDING = "pending"               # معلقة
    RESOLVED = "resolved"             # محلولة
    CLOSED = "closed"                 # مغلقة
    REOPENED = "reopened"             # معاد فتحها
    ESCALATED = "escalated"           # مصعّدة


class TicketPriority(Enum):
    """أولويات التذكرة"""
    LOW = "low"                       # منخفضة
    MEDIUM = "medium"                 # متوسطة
    HIGH = "high"                     # عالية
    CRITICAL = "critical"             # حرجة
    URGENT = "urgent"                 # عاجلة


class TicketSource(Enum):
    """مصدر التذكرة"""
    EMAIL = "email"
    PHONE = "phone"
    WEB = "web"
    CHAT = "chat"
    PORTAL = "portal"
    SOCIAL_MEDIA = "social_media"
    INTERNAL = "internal"


@dataclass
class TicketComment:
    """تعليق على التذكرة"""
    id: str
    author_id: str
    author_name: str
    content: str
    is_internal: bool = False         # هل هو تعليق داخلي؟
    attachments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "content": self.content,
            "is_internal": self.is_internal,
            "attachments": self.attachments,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TicketActivity:
    """نشاط في التذكرة"""
    id: str
    activity_type: str                # status_change, assignment, comment, etc.
    description: str
    performed_by: str
    old_value: str = ""
    new_value: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "activity_type": self.activity_type,
            "description": self.description,
            "performed_by": self.performed_by,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SLAConfig:
    """إعدادات SLA"""
    priority: TicketPriority
    response_time_hours: int          # وقت الاستجابة
    resolution_time_hours: int        # وقت الحل
    business_hours_only: bool = True  # فقط ساعات العمل


@dataclass
class SupportTicket:
    """تذكرة دعم"""
    id: str
    ticket_number: str                # رقم التذكرة
    
    # Requester
    customer_id: str
    customer_name: str
    
    # Ticket Details (required)
    subject: str
    description: str
    
    # Contact Info
    contact_email: str = ""
    contact_phone: str = ""
    
    # Categorization
    category: str = ""                # الفئة: technical, billing, general, etc.
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.NEW
    source: TicketSource = TicketSource.WEB
    
    # Assignment
    assigned_to: Optional[str] = None # معين لـ
    assigned_team: Optional[str] = None  # فريق
    
    # SLA
    sla_config: SLAConfig = None
    sla_response_deadline: Optional[datetime] = None
    sla_resolution_deadline: Optional[datetime] = None
    first_response_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    sla_breached: bool = False
    
    # Knowledge Base
    kb_article_id: Optional[str] = None  # مقالة قاعدة المعرفة المرتبطة
    related_tickets: List[str] = field(default_factory=list)
    
    # Attachments
    attachments: List[str] = field(default_factory=list)
    
    # Comments & Activities
    comments: List[TicketComment] = field(default_factory=list)
    activities: List[TicketActivity] = field(default_factory=list)
    
    # Satisfaction
    satisfaction_rating: Optional[int] = None  # 1-5
    satisfaction_comment: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    
    @property
    def is_open(self) -> bool:
        """هل التذكرة مفتوحة؟"""
        return self.status not in [TicketStatus.CLOSED, TicketStatus.RESOLVED]
    
    @property
    def age_hours(self) -> float:
        """عمر التذكرة بالساعات"""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 3600
    
    @property
    def response_time_hours(self) -> Optional[float]:
        """وقت الاستجابة بالساعات"""
        if not self.first_response_at:
            return None
        delta = self.first_response_at - self.created_at
        return delta.total_seconds() / 3600
    
    @property
    def resolution_time_hours(self) -> Optional[float]:
        """وقت الحل بالساعات"""
        if not self.resolved_at:
            return None
        delta = self.resolved_at - self.created_at
        return delta.total_seconds() / 3600
    
    @property
    def sla_response_remaining(self) -> Optional[float]:
        """الوقت المتبقي لـ SLA الاستجابة (بالساعات)"""
        if not self.sla_response_deadline:
            return None
        if self.first_response_at:
            return 0
        delta = self.sla_response_deadline - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 3600)
    
    @property
    def sla_resolution_remaining(self) -> Optional[float]:
        """الوقت المتبقي لـ SLA الحل (بالساعات)"""
        if not self.sla_resolution_deadline:
            return None
        if self.resolved_at:
            return 0
        delta = self.sla_resolution_deadline - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 3600)
    
    def assign(self, assignee_id: str, assignee_name: str, team: str = None):
        """تعيين التذكرة"""
        old_assignee = self.assigned_to
        self.assigned_to = assignee_id
        self.assigned_team = team or self.assigned_team
        self.status = TicketStatus.ASSIGNED
        self.updated_at = datetime.now(timezone.utc)
        
        self.activities.append(TicketActivity(
            id=str(uuid.uuid4()),
            activity_type="assignment",
            description=f"Assigned to {assignee_name}",
            performed_by="system",
            old_value=old_assignee or "unassigned",
            new_value=assignee_id
        ))
    
    def add_comment(self, author_id: str, author_name: str, 
                   content: str, is_internal: bool = False) -> TicketComment:
        """إضافة تعليق"""
        comment = TicketComment(
            id=str(uuid.uuid4()),
            author_id=author_id,
            author_name=author_name,
            content=content,
            is_internal=is_internal
        )
        
        self.comments.append(comment)
        self.updated_at = datetime.now(timezone.utc)
        
        # Track first response
        if not self.first_response_at and not is_internal:
            self.first_response_at = datetime.now(timezone.utc)
            # Check SLA breach
            if self.sla_response_deadline and self.first_response_at > self.sla_response_deadline:
                self.sla_breached = True
        
        return comment
    
    def change_status(self, new_status: TicketStatus, changed_by: str, reason: str = ""):
        """تغيير حالة التذكرة"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        
        if new_status == TicketStatus.RESOLVED:
            self.resolved_at = datetime.now(timezone.utc)
            # Check SLA breach
            if self.sla_resolution_deadline and self.resolved_at > self.sla_resolution_deadline:
                self.sla_breached = True
        
        if new_status == TicketStatus.CLOSED:
            self.closed_at = datetime.now(timezone.utc)
        
        self.activities.append(TicketActivity(
            id=str(uuid.uuid4()),
            activity_type="status_change",
            description=f"Status changed from {old_status.value} to {new_status.value}",
            performed_by=changed_by,
            old_value=old_status.value,
            new_value=new_status.value
        ))
    
    def escalate(self, escalated_by: str, reason: str = ""):
        """تصعيد التذكرة"""
        self.status = TicketStatus.ESCALATED
        # Escalate priority to next level
        priority_order = [TicketPriority.LOW, TicketPriority.MEDIUM, TicketPriority.HIGH, TicketPriority.CRITICAL, TicketPriority.URGENT]
        current_idx = priority_order.index(self.priority) if self.priority in priority_order else 0
        next_idx = min(current_idx + 1, len(priority_order) - 1)
        self.priority = priority_order[next_idx]
        self.updated_at = datetime.now(timezone.utc)
        
        self.activities.append(TicketActivity(
            id=str(uuid.uuid4()),
            activity_type="escalation",
            description=f"Escalated: {reason}",
            performed_by=escalated_by
        ))
    
    def link_kb_article(self, article_id: str):
        """ربط مقالة قاعدة معرفة"""
        self.kb_article_id = article_id
        self.updated_at = datetime.now(timezone.utc)
    
    def set_satisfaction_rating(self, rating: int, comment: str = ""):
        """تقييم رضا العميل"""
        self.satisfaction_rating = rating
        self.satisfaction_comment = comment
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ticket_number": self.ticket_number,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "contact_email": self.contact_email,
            "subject": self.subject,
            "description": self.description,
            "category": self.category,
            "priority": self.priority.value,
            "status": self.status.value,
            "source": self.source.value,
            "assigned_to": self.assigned_to,
            "assigned_team": self.assigned_team,
            "age_hours": self.age_hours,
            "response_time_hours": self.response_time_hours,
            "resolution_time_hours": self.resolution_time_hours,
            "sla_response_remaining": self.sla_response_remaining,
            "sla_resolution_remaining": self.sla_resolution_remaining,
            "sla_breached": self.sla_breached,
            "kb_article_id": self.kb_article_id,
            "is_open": self.is_open,
            "tags": self.tags,
            "satisfaction_rating": self.satisfaction_rating,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class TicketManager:
    """
    مدير تذاكر الدعم
    """
    
    # SLA Configurations by priority
    DEFAULT_SLA = {
        TicketPriority.LOW: SLAConfig(TicketPriority.LOW, 24, 72),
        TicketPriority.MEDIUM: SLAConfig(TicketPriority.MEDIUM, 8, 48),
        TicketPriority.HIGH: SLAConfig(TicketPriority.HIGH, 4, 24),
        TicketPriority.CRITICAL: SLAConfig(TicketPriority.CRITICAL, 2, 12),
        TicketPriority.URGENT: SLAConfig(TicketPriority.URGENT, 1, 4)
    }
    
    def __init__(self):
        self.tickets: Dict[str, SupportTicket] = {}
        self._ticket_counter = 0
        self.sla_configs = dict(self.DEFAULT_SLA)
    
    def create_ticket(self, customer_id: str, customer_name: str,
                     subject: str, description: str,
                     priority: TicketPriority = TicketPriority.MEDIUM,
                     category: str = "", source: TicketSource = TicketSource.WEB,
                     contact_email: str = "", contact_phone: str = "") -> SupportTicket:
        """إنشاء تذكرة جديدة"""
        self._ticket_counter += 1
        ticket_number = f"TKT-{datetime.now().strftime('%Y%m')}-{self._ticket_counter:05d}"
        
        # Calculate SLA deadlines
        sla_config = self.sla_configs.get(priority)
        now = datetime.now(timezone.utc)
        sla_response = now + timedelta(hours=sla_config.response_time_hours)
        sla_resolution = now + timedelta(hours=sla_config.resolution_time_hours)
        
        ticket = SupportTicket(
            id=str(uuid.uuid4()),
            ticket_number=ticket_number,
            customer_id=customer_id,
            customer_name=customer_name,
            subject=subject,
            description=description,
            priority=priority,
            category=category,
            source=source,
            contact_email=contact_email,
            contact_phone=contact_phone,
            sla_config=sla_config,
            sla_response_deadline=sla_response,
            sla_resolution_deadline=sla_resolution
        )
        
        self.tickets[ticket.id] = ticket
        return ticket
    
    def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        """الحصول على تذكرة"""
        return self.tickets.get(ticket_id)
    
    def get_ticket_by_number(self, ticket_number: str) -> Optional[SupportTicket]:
        """الحصول على تذكرة برقمها"""
        for ticket in self.tickets.values():
            if ticket.ticket_number == ticket_number:
                return ticket
        return None
    
    def assign_ticket(self, ticket_id: str, assignee_id: str, 
                     assignee_name: str, team: str = None) -> SupportTicket:
        """تعيين تذكرة"""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        ticket.assign(assignee_id, assignee_name, team)
        return ticket
    
    def update_ticket_status(self, ticket_id: str, new_status: TicketStatus,
                            changed_by: str, reason: str = "") -> SupportTicket:
        """تحديث حالة التذكرة"""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        ticket.change_status(new_status, changed_by, reason)
        return ticket
    
    def add_comment(self, ticket_id: str, author_id: str, author_name: str,
                   content: str, is_internal: bool = False) -> TicketComment:
        """إضافة تعليق"""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        return ticket.add_comment(author_id, author_name, content, is_internal)
    
    def escalate_ticket(self, ticket_id: str, escalated_by: str, 
                       reason: str = "") -> SupportTicket:
        """تصعيد تذكرة"""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        ticket.escalate(escalated_by, reason)
        return ticket
    
    def get_tickets_by_status(self, status: TicketStatus) -> List[SupportTicket]:
        """الحصول على التذاكر حسب الحالة"""
        return [t for t in self.tickets.values() if t.status == status]
    
    def get_tickets_by_priority(self, priority: TicketPriority) -> List[SupportTicket]:
        """الحصول على التذاكر حسب الأولوية"""
        return [t for t in self.tickets.values() if t.priority == priority]
    
    def get_tickets_by_assignee(self, assignee_id: str) -> List[SupportTicket]:
        """الحصول على تذاكر معينة لموظف"""
        return [t for t in self.tickets.values() if t.assigned_to == assignee_id]
    
    def get_tickets_by_customer(self, customer_id: str) -> List[SupportTicket]:
        """الحصول على تذاكر عميل"""
        return [t for t in self.tickets.values() if t.customer_id == customer_id]
    
    def get_open_tickets(self) -> List[SupportTicket]:
        """الحصول على التذاكر المفتوحة"""
        return [t for t in self.tickets.values() if t.is_open]
    
    def get_sla_breached_tickets(self) -> List[SupportTicket]:
        """الحصول على التذاكر التي تجاوزت SLA"""
        return [t for t in self.tickets.values() if t.sla_breached]
    
    def get_sla_at_risk_tickets(self, hours: int = 4) -> List[SupportTicket]:
        """الحصول على التذاكر المهددة بتجاوز SLA"""
        at_risk = []
        for ticket in self.get_open_tickets():
            if ticket.sla_resolution_remaining and ticket.sla_resolution_remaining <= hours:
                at_risk.append(ticket)
        return at_risk
    
    def get_ticket_summary(self) -> Dict[str, Any]:
        """ملخص التذاكر"""
        total_tickets = len(self.tickets)
        open_tickets = len(self.get_open_tickets())
        
        by_status = {
            status.value: len(self.get_tickets_by_status(status))
            for status in TicketStatus
        }
        
        by_priority = {
            priority.value: len(self.get_tickets_by_priority(priority))
            for priority in TicketPriority
        }
        
        sla_breached = len(self.get_sla_breached_tickets())
        sla_at_risk = len(self.get_sla_at_risk_tickets())
        
        # Average resolution time
        resolved = [t for t in self.tickets.values() if t.resolved_at]
        avg_resolution = (
            sum(t.resolution_time_hours or 0 for t in resolved) / len(resolved)
            if resolved else 0
        )
        
        # Customer satisfaction
        rated = [t for t in self.tickets.values() if t.satisfaction_rating]
        avg_satisfaction = (
            sum(t.satisfaction_rating for t in rated) / len(rated)
            if rated else 0
        )
        
        return {
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "closed_tickets": total_tickets - open_tickets,
            "by_status": by_status,
            "by_priority": by_priority,
            "sla_breached": sla_breached,
            "sla_at_risk": sla_at_risk,
            "average_resolution_time_hours": avg_resolution,
            "average_satisfaction_rating": avg_satisfaction
        }
    
    def search_tickets(self, query: str) -> List[SupportTicket]:
        """البحث في التذاكر"""
        query = query.lower()
        results = []
        
        for ticket in self.tickets.values():
            if (query in ticket.subject.lower() or
                query in ticket.description.lower() or
                query in ticket.ticket_number.lower() or
                query in ticket.customer_name.lower() or
                query in [t.lower() for t in ticket.tags]):
                results.append(ticket)
        
        return results
    
    def update_sla_config(self, priority: TicketPriority,
                         response_hours: int, resolution_hours: int):
        """تحديث إعدادات SLA"""
        self.sla_configs[priority] = SLAConfig(
            priority, response_hours, resolution_hours
        )
