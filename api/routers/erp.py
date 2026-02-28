"""
روتر ERP - ERP Router

يوفر نقاط النهاية لإدارة موارد المؤسسة.
Provides endpoints for enterprise resource planning.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User

router = APIRouter(prefix="/erp", tags=["تخطيط الموارد | ERP"])


class InvoiceStatus(str, Enum):
    """حالة الفاتورة | Invoice status"""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    """نوع التقرير | Report type"""
    SALES = "sales"
    EXPENSES = "expenses"
    PROFIT_LOSS = "profit_loss"
    CASH_FLOW = "cash_flow"
    INVENTORY = "inventory"


class ReportPeriod(str, Enum):
    """فترة التقرير | Report period"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# نماذج Pydantic - Pydantic Models
class InvoiceItem(BaseModel):
    """نموذج عنصر الفاتورة | Invoice item model"""
    id: int
    description: str
    quantity: Decimal = Field(..., gt=0)
    unit_price: Decimal = Field(..., ge=0)
    total: Decimal


class Invoice(BaseModel):
    """نموذج الفاتورة | Invoice model"""
    id: str
    invoice_number: str
    customer_name: str
    customer_email: str
    issue_date: date
    due_date: date
    status: InvoiceStatus
    items: List[InvoiceItem]
    subtotal: Decimal
    tax_rate: Decimal
    tax_amount: Decimal
    total: Decimal
    notes: Optional[str] = None
    created_at: datetime


class InvoiceCreate(BaseModel):
    """نموذج إنشاء الفاتورة | Invoice create model"""
    customer_name: str
    customer_email: str
    due_date: date
    items: List[Dict[str, Any]]
    tax_rate: Decimal = Field(default=0.15, ge=0, le=1)
    notes: Optional[str] = None


class DashboardStats(BaseModel):
    """نموذج إحصائيات لوحة التحكم | Dashboard stats model"""
    total_revenue: Decimal
    pending_invoices: int
    total_invoices: int
    monthly_growth: float
    top_customers: List[Dict[str, Any]]
    recent_transactions: List[Dict[str, Any]]


class Report(BaseModel):
    """نموذج التقرير | Report model"""
    id: str
    type: ReportType
    period: ReportPeriod
    start_date: date
    end_date: date
    generated_at: datetime
    data: Dict[str, Any]
    summary: str


class Insight(BaseModel):
    """نموذج الرؤى المدعومة بالذكاء الاصطناعي | AI-powered insight model"""
    category: str
    severity: str  # high, medium, low
    title: str
    description: str
    recommendation: str
    potential_impact: Optional[str] = None
    confidence_score: float = Field(..., ge=0, le=1)


class InsightsResponse(BaseModel):
    """نموذج استجابة الرؤى | Insights response model"""
    generated_at: datetime
    period: ReportPeriod
    insights: List[Insight]
    overall_health_score: int = Field(..., ge=0, le=100)


# قاعدة بيانات وهمية - Fake Database
fake_invoices = {
    "INV-2024-001": {
        "id": "inv-001",
        "invoice_number": "INV-2024-001",
        "customer_name": "شركة التقنية | Tech Corp",
        "customer_email": "billing@techcorp.com",
        "issue_date": date(2024, 1, 15),
        "due_date": date(2024, 2, 15),
        "status": InvoiceStatus.PAID,
        "items": [
            {
                "id": 1,
                "description": "تطوير برمجي | Software Development",
                "quantity": Decimal("10"),
                "unit_price": Decimal("500"),
                "total": Decimal("5000")
            }
        ],
        "subtotal": Decimal("5000"),
        "tax_rate": Decimal("0.15"),
        "tax_amount": Decimal("750"),
        "total": Decimal("5750"),
        "notes": "شكراً لتعاملكم معنا",
        "created_at": datetime.utcnow()
    },
    "INV-2024-002": {
        "id": "inv-002",
        "invoice_number": "INV-2024-002",
        "customer_name": "مؤسسة الإبداع | Creative Foundation",
        "customer_email": "finance@creative.org",
        "issue_date": date(2024, 2, 1),
        "due_date": date(2024, 3, 1),
        "status": InvoiceStatus.SENT,
        "items": [
            {
                "id": 1,
                "description": "تصميم UI/UX",
                "quantity": Decimal("1"),
                "unit_price": Decimal("3000"),
                "total": Decimal("3000")
            }
        ],
        "subtotal": Decimal("3000"),
        "tax_rate": Decimal("0.15"),
        "tax_amount": Decimal("450"),
        "total": Decimal("3450"),
        "notes": None,
        "created_at": datetime.utcnow()
    }
}

fake_invoice_counter = 3


@router.get(
    "/dashboard",
    response_model=DashboardStats,
    status_code=status.HTTP_200_OK,
    summary="لوحة تحكم ERP | ERP dashboard"
)
async def get_dashboard(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على بيانات لوحة تحكم ERP.
    Get ERP dashboard data.
    """
    total_revenue = sum(
        inv["total"] for inv in fake_invoices.values()
        if inv["status"] == InvoiceStatus.PAID
    )
    
    pending = sum(
        1 for inv in fake_invoices.values()
        if inv["status"] in [InvoiceStatus.SENT, InvoiceStatus.OVERDUE]
    )
    
    return DashboardStats(
        total_revenue=total_revenue,
        pending_invoices=pending,
        total_invoices=len(fake_invoices),
        monthly_growth=12.5,
        top_customers=[
            {"name": "شركة التقنية", "revenue": Decimal("5750")},
            {"name": "مؤسسة الإبداع", "revenue": Decimal("3450")}
        ],
        recent_transactions=[
            {
                "id": "txn-001",
                "type": "payment",
                "amount": Decimal("5750"),
                "date": datetime.utcnow(),
                "description": "دفعة فاتورة INV-2024-001"
            }
        ]
    )


@router.get(
    "/invoices",
    response_model=List[Invoice],
    status_code=status.HTTP_200_OK,
    summary="قائمة الفواتير | List invoices"
)
async def list_invoices(
    status: Optional[InvoiceStatus] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على قائمة الفواتير.
    Get list of invoices.
    """
    invoices = list(fake_invoices.values())
    
    if status:
        invoices = [inv for inv in invoices if inv["status"] == status]
    
    return [Invoice(**inv) for inv in sorted(
        invoices,
        key=lambda x: x["created_at"],
        reverse=True
    )]


@router.post(
    "/invoices",
    response_model=Invoice,
    status_code=status.HTTP_201_CREATED,
    summary="إنشاء فاتورة | Create invoice"
)
async def create_invoice(
    invoice: InvoiceCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    إنشاء فاتورة جديدة.
    Create a new invoice.
    """
    global fake_invoice_counter
    
    invoice_number = f"INV-2024-{fake_invoice_counter:03d}"
    fake_invoice_counter += 1
    
    # حساب المجاميع | Calculate totals
    items = []
    subtotal = Decimal("0")
    
    for idx, item_data in enumerate(invoice.items, 1):
        qty = Decimal(str(item_data.get("quantity", 1)))
        price = Decimal(str(item_data.get("unit_price", 0)))
        item_total = qty * price
        
        items.append({
            "id": idx,
            "description": item_data.get("description", ""),
            "quantity": qty,
            "unit_price": price,
            "total": item_total
        })
        subtotal += item_total
    
    tax_amount = subtotal * invoice.tax_rate
    total = subtotal + tax_amount
    
    new_invoice = {
        "id": f"inv-{fake_invoice_counter-1:03d}",
        "invoice_number": invoice_number,
        "customer_name": invoice.customer_name,
        "customer_email": invoice.customer_email,
        "issue_date": date.today(),
        "due_date": invoice.due_date,
        "status": InvoiceStatus.DRAFT,
        "items": items,
        "subtotal": subtotal,
        "tax_rate": invoice.tax_rate,
        "tax_amount": tax_amount,
        "total": total,
        "notes": invoice.notes,
        "created_at": datetime.utcnow()
    }
    
    fake_invoices[invoice_number] = new_invoice
    
    return Invoice(**new_invoice)


@router.get(
    "/reports",
    response_model=List[Report],
    status_code=status.HTTP_200_OK,
    summary="التقارير | Business reports"
)
async def get_reports(
    report_type: Optional[ReportType] = None,
    period: ReportPeriod = ReportPeriod.MONTHLY,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على التقارير التجارية.
    Get business reports.
    """
    reports = [
        Report(
            id="rep-001",
            type=ReportType.SALES,
            period=period,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            generated_at=datetime.utcnow(),
            data={
                "total_sales": 15000,
                "transactions": 25,
                "average_order": 600
            },
            summary="أداء مبيعات قوي في يناير | Strong sales performance in January"
        ),
        Report(
            id="rep-002",
            type=ReportType.PROFIT_LOSS,
            period=period,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            generated_at=datetime.utcnow(),
            data={
                "revenue": 15000,
                "expenses": 8000,
                "profit": 7000,
                "margin": 0.47
            },
            summary="هامش ربح جيد | Good profit margin"
        )
    ]
    
    if report_type:
        reports = [r for r in reports if r.type == report_type]
    
    return reports


@router.get(
    "/insights",
    response_model=InsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="رؤى مدعومة بالذكاء الاصطناعي | AI-powered insights"
)
async def get_insights(
    period: ReportPeriod = ReportPeriod.MONTHLY,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على رؤى تجارية مدعومة بالذكاء الاصطناعي.
    Get AI-powered business insights.
    """
    insights = [
        Insight(
            category="revenue",
            severity="high",
            title="زيادة الإيرادات المتوقعة | Projected Revenue Increase",
            description="""بناءً على تحليل الاتجاهات، من المتوقع زيادة الإيرادات بنسبة 15% 
            في الربع القادم بناءً على الأداء الحالي.""",
            recommendation="""زيادة الاستثمار في التسويق الرقمي لتحقيق النمو المتوقع.
            | Increase investment in digital marketing.""",
            potential_impact="+15% إيرادات | +15% revenue",
            confidence_score=0.85
        ),
        Insight(
            category="expenses",
            severity="medium",
            title="تحسين التكاليف | Cost Optimization",
            description="""هناك فرصة لتقليل تكاليف البنية التحتية بنسبة 20% 
            من خلال الانتقال إلى السحابة.""",
            recommendation="""قياس فوائد الانتقال إلى البنية التحتية السحابية.
            | Evaluate cloud migration benefits.""",
            potential_impact="-20% تكاليف | -20% costs",
            confidence_score=0.75
        ),
        Insight(
            category="customers",
            severity="low",
            title="retention_rate جيد | Good retention rate",
            description="معدل الاحتفاظ بالعملاء ممتاز عند 85%.",
            recommendation="استمر في برامج ولاء العملاء الحالية.",
            confidence_score=0.92
        )
    ]
    
    return InsightsResponse(
        generated_at=datetime.utcnow(),
        period=period,
        insights=insights,
        overall_health_score=82
    )
