"""
Sales Pipeline - خط أنابيب المبيعات

إدارة خطوط المبيعات مع:
- مراحل: lead → qualified → proposal → negotiation → closed
- تتبع قيمة الصفقات
- معدل الفوز/الخسارة
- تقارير التوقعات
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class PipelineStage(Enum):
    """مراحل خط الأنابيب"""
    LEAD = "lead"                     # عميل محتمل
    QUALIFIED = "qualified"           # مؤهل
    PROPOSAL = "proposal"             # عرض سعر
    NEGOTIATION = "negotiation"       # تفاوض
    CLOSED_WON = "closed_won"         # مغلق (ناجح)
    CLOSED_LOST = "closed_lost"       # مغلق (خسارة)


class DealPriority(Enum):
    """أولوية الصفقة"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOT = "hot"


@dataclass
class DealActivity:
    """نشاط في الصفقة"""
    id: str
    activity_type: str                # call, email, meeting, note, task
    description: str
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "activity_type": self.activity_type,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class Deal:
    """صفقة مبيعات"""
    id: str
    deal_name: str
    deal_code: str                    # رمز الصفقة
    
    # Customer
    customer_id: str
    customer_name: str
    contact_id: Optional[str] = None  # جهة الاتصال المعنية
    
    # Value
    value: Decimal = field(default_factory=lambda: Decimal('0'))
    currency: str = "SAR"
    
    # Expected close date
    expected_close_date: Optional[date] = None
    actual_close_date: Optional[date] = None
    
    # Pipeline
    stage: PipelineStage = PipelineStage.LEAD
    priority: DealPriority = DealPriority.MEDIUM
    probability: int = 10             # نسبة النجاح المتوقعة (%)
    
    # Assignment
    owner_id: str = ""                # مسؤول الصفقة
    team_members: List[str] = field(default_factory=list)
    
    # Details
    description: str = ""
    source: str = ""                  # مصدر الصفقة
    competitor: str = ""              # منافس
    lost_reason: str = ""             # سبب الخسارة
    
    # Activities
    activities: List[DealActivity] = field(default_factory=list)
    
    # Products/Services
    items: List[Dict] = field(default_factory=list)  # [{product_id, quantity, price, discount}]
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def weighted_value(self) -> Decimal:
        """القيمة المرجحة (value × probability)"""
        return self.value * Decimal(str(self.probability)) / Decimal('100')
    
    @property
    def is_overdue(self) -> bool:
        """هل تاريخ الإغلاق المتوقع قد مر؟"""
        if not self.expected_close_date:
            return False
        if self.stage in [PipelineStage.CLOSED_WON, PipelineStage.CLOSED_LOST]:
            return False
        return date.today() > self.expected_close_date
    
    @property
    def days_in_stage(self) -> int:
        """عدد الأيام في المرحلة الحالية"""
        delta = datetime.now(timezone.utc) - self.updated_at
        return delta.days
    
    @property
    def total_days(self) -> int:
        """إجمالي أيام الصفقة"""
        if self.actual_close_date:
            delta = datetime.combine(self.actual_close_date, datetime.min.time()) - self.created_at
        else:
            delta = datetime.now(timezone.utc) - self.created_at
        return delta.days
    
    def move_to_stage(self, new_stage: PipelineStage, updated_by: str = None):
        """نقل الصفقة لمرحلة جديدة"""
        self.stage = new_stage
        self.updated_at = datetime.now(timezone.utc)
        
        # Update probability based on stage
        stage_probabilities = {
            PipelineStage.LEAD: 10,
            PipelineStage.QUALIFIED: 25,
            PipelineStage.PROPOSAL: 50,
            PipelineStage.NEGOTIATION: 75,
            PipelineStage.CLOSED_WON: 100,
            PipelineStage.CLOSED_LOST: 0
        }
        self.probability = stage_probabilities.get(new_stage, 10)
        
        # Record activity
        self.add_activity(
            "stage_change",
            f"Moved to stage: {new_stage.value}",
            updated_by or "system"
        )
        
        # Set close date if closed
        if new_stage in [PipelineStage.CLOSED_WON, PipelineStage.CLOSED_LOST]:
            self.actual_close_date = date.today()
    
    def close_won(self, updated_by: str = None):
        """إغلاق الصفقة بنجاح"""
        self.move_to_stage(PipelineStage.CLOSED_WON, updated_by)
    
    def close_lost(self, reason: str = "", updated_by: str = None):
        """إغلاق الصفقة بخسارة"""
        self.lost_reason = reason
        self.move_to_stage(PipelineStage.CLOSED_LOST, updated_by)
    
    def add_activity(self, activity_type: str, description: str, 
                    created_by: str, scheduled_for: datetime = None):
        """إضافة نشاط"""
        activity = DealActivity(
            id=str(uuid.uuid4()),
            activity_type=activity_type,
            description=description,
            created_by=created_by,
            scheduled_for=scheduled_for
        )
        
        self.activities.append(activity)
        self.updated_at = datetime.now(timezone.utc)
        return activity
    
    def add_product(self, product_id: str, product_name: str,
                   quantity: int, unit_price: Decimal, discount: Decimal = Decimal('0')):
        """إضافة منتج للصفقة"""
        item = {
            "product_id": product_id,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": float(unit_price),
            "discount": float(discount),
            "total": float((unit_price * Decimal(str(quantity))) - discount)
        }
        
        self.items.append(item)
        
        # Recalculate total value
        self.value = sum(Decimal(str(i["total"])) for i in self.items)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "deal_name": self.deal_name,
            "deal_code": self.deal_code,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "value": float(self.value),
            "weighted_value": float(self.weighted_value),
            "currency": self.currency,
            "stage": self.stage.value,
            "priority": self.priority.value,
            "probability": self.probability,
            "expected_close_date": self.expected_close_date.isoformat() if self.expected_close_date else None,
            "actual_close_date": self.actual_close_date.isoformat() if self.actual_close_date else None,
            "is_overdue": self.is_overdue,
            "days_in_stage": self.days_in_stage,
            "total_days": self.total_days,
            "owner_id": self.owner_id,
            "source": self.source,
            "lost_reason": self.lost_reason,
            "items": self.items,
            "activities": [a.to_dict() for a in self.activities],
            "created_at": self.created_at.isoformat()
        }


class SalesPipeline:
    """
    خط أنابيب المبيعات
    """
    
    def __init__(self):
        self.deals: Dict[str, Deal] = {}
        self._deal_counter = 0
    
    def create_deal(self, deal_name: str, customer_id: str,
                   customer_name: str, value: Decimal = Decimal('0'),
                   owner_id: str = "", **kwargs) -> Deal:
        """إنشاء صفقة جديدة"""
        self._deal_counter += 1
        deal_code = f"DEAL-{datetime.now().strftime('%Y')}-{self._deal_counter:05d}"
        
        deal = Deal(
            id=str(uuid.uuid4()),
            deal_name=deal_name,
            deal_code=deal_code,
            customer_id=customer_id,
            customer_name=customer_name,
            value=Decimal(str(value)),
            owner_id=owner_id,
            **kwargs
        )
        
        self.deals[deal.id] = deal
        return deal
    
    def get_deal(self, deal_id: str) -> Optional[Deal]:
        """الحصول على صفقة"""
        return self.deals.get(deal_id)
    
    def get_deal_by_code(self, code: str) -> Optional[Deal]:
        """الحصول على صفقة بالرمز"""
        for deal in self.deals.values():
            if deal.deal_code == code:
                return deal
        return None
    
    def update_deal(self, deal_id: str, **kwargs) -> Deal:
        """تحديث صفقة"""
        deal = self.deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal {deal_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(deal, key):
                setattr(deal, key, value)
        
        deal.updated_at = datetime.now(timezone.utc)
        return deal
    
    def move_deal(self, deal_id: str, new_stage: PipelineStage, 
                 updated_by: str = None) -> Deal:
        """نقل صفقة لمرحلة جديدة"""
        deal = self.deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal {deal_id} not found")
        
        deal.move_to_stage(new_stage, updated_by)
        return deal
    
    def get_deals_by_stage(self, stage: PipelineStage) -> List[Deal]:
        """الحصول على الصفقات حسب المرحلة"""
        return [d for d in self.deals.values() if d.stage == stage]
    
    def get_deals_by_owner(self, owner_id: str) -> List[Deal]:
        """الحصول على صفقات مسؤول معين"""
        return [d for d in self.deals.values() if d.owner_id == owner_id]
    
    def get_deals_by_customer(self, customer_id: str) -> List[Deal]:
        """الحصول على صفقات عميل معين"""
        return [d for d in self.deals.values() if d.customer_id == customer_id]
    
    def get_open_deals(self) -> List[Deal]:
        """الحصول على الصفقات المفتوحة"""
        return [
            d for d in self.deals.values()
            if d.stage not in [PipelineStage.CLOSED_WON, PipelineStage.CLOSED_LOST]
        ]
    
    def get_overdue_deals(self) -> List[Deal]:
        """الحصول على الصفقات المتأخرة"""
        return [d for d in self.get_open_deals() if d.is_overdue]
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """ملخص خط الأنابيب"""
        by_stage = {}
        total_value = Decimal('0')
        total_weighted_value = Decimal('0')
        
        for stage in PipelineStage:
            deals = self.get_deals_by_stage(stage)
            stage_value = sum(d.value for d in deals)
            stage_weighted = sum(d.weighted_value for d in deals)
            
            by_stage[stage.value] = {
                "count": len(deals),
                "value": float(stage_value),
                "weighted_value": float(stage_weighted)
            }
            
            total_value += stage_value
            total_weighted_value += stage_weighted
        
        # Win/Loss rate
        won_deals = len(self.get_deals_by_stage(PipelineStage.CLOSED_WON))
        lost_deals = len(self.get_deals_by_stage(PipelineStage.CLOSED_LOST))
        total_closed = won_deals + lost_deals
        
        win_rate = (won_deals / total_closed * 100) if total_closed > 0 else 0
        loss_rate = (lost_deals / total_closed * 100) if total_closed > 0 else 0
        
        return {
            "total_deals": len(self.deals),
            "total_value": float(total_value),
            "total_weighted_value": float(total_weighted_value),
            "by_stage": by_stage,
            "open_deals": len(self.get_open_deals()),
            "overdue_deals": len(self.get_overdue_deals()),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "won_deals": won_deals,
            "lost_deals": lost_deals
        }
    
    def get_forecast_report(self, period_months: int = 3) -> Dict[str, Any]:
        """
        تقرير التوقعات
        
        يتوقع المبيعات المستقبلية بناءً على:
        - المرحلة الحالية
        - احتمالية النجاح
        - تاريخ الإغلاق المتوقع
        """
        from collections import defaultdict
        
        monthly_forecast = defaultdict(lambda: {"count": 0, "value": Decimal('0'), "weighted": Decimal('0')})
        
        for deal in self.get_open_deals():
            if deal.expected_close_date:
                month_key = deal.expected_close_date.strftime("%Y-%m")
                monthly_forecast[month_key]["count"] += 1
                monthly_forecast[month_key]["value"] += deal.value
                monthly_forecast[month_key]["weighted"] += deal.weighted_value
        
        # Sort by month
        sorted_forecast = dict(sorted(monthly_forecast.items()))
        
        return {
            "forecast_period_months": period_months,
            "monthly_forecast": {
                k: {
                    "count": v["count"],
                    "value": float(v["value"]),
                    "weighted_value": float(v["weighted"])
                }
                for k, v in list(sorted_forecast.items())[:period_months]
            },
            "total_forecast": float(sum(v["weighted"] for v in monthly_forecast.values()))
        }
    
    def get_sales_performance(self, owner_id: str = None) -> Dict[str, Any]:
        """أداء المبيعات"""
        deals = self.deals.values()
        if owner_id:
            deals = [d for d in deals if d.owner_id == owner_id]
        
        won = [d for d in deals if d.stage == PipelineStage.CLOSED_WON]
        lost = [d for d in deals if d.stage == PipelineStage.CLOSED_LOST]
        open_deals = [d for d in deals if d.stage not in [PipelineStage.CLOSED_WON, PipelineStage.CLOSED_LOST]]
        
        won_value = sum(d.value for d in won)
        lost_value = sum(d.value for d in lost)
        open_value = sum(d.value for d in open_deals)
        
        total_closed = len(won) + len(lost)
        
        return {
            "owner_id": owner_id or "all",
            "won_deals": len(won),
            "won_value": float(won_value),
            "lost_deals": len(lost),
            "lost_value": float(lost_value),
            "open_deals": len(open_deals),
            "open_value": float(open_value),
            "win_rate": (len(won) / total_closed * 100) if total_closed > 0 else 0,
            "average_deal_size": float(won_value / len(won)) if won else 0,
            "average_sales_cycle": sum(d.total_days for d in won) / len(won) if won else 0
        }
