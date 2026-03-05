"""
روتر ERP - ERP Router

⚠️ هذا الروتر يحتاج قاعدة بيانات حقيقية (PostgreSQL).
NO FAKE DATA — per rules: ممنوع أي شي وهمي

الفواتير، التقارير، والرؤى كلها تحتاج بيانات حقيقية.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/erp", tags=["تخطيط الموارد | ERP"])

DB_NOT_READY = "⚠️ خدمة ERP تحتاج قاعدة بيانات حقيقية (PostgreSQL). قيد التطوير — لا بيانات وهمية."


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class ReportType(str, Enum):
    SALES = "sales"
    EXPENSES = "expenses"
    PROFIT_LOSS = "profit_loss"
    CASH_FLOW = "cash_flow"
    INVENTORY = "inventory"


class ReportPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class InvoiceCreate(BaseModel):
    customer_name: str
    customer_email: str
    due_date: date
    items: List[Dict[str, Any]]
    tax_rate: Decimal = Field(default=Decimal("0.15"), ge=0, le=1)
    notes: Optional[str] = None


# ─── Endpoints — all return NOT IMPLEMENTED until real DB ────────

@router.get("/dashboard", summary="لوحة تحكم ERP | ERP dashboard")
async def get_dashboard():
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=DB_NOT_READY)


@router.get("/invoices", summary="قائمة الفواتير | List invoices")
async def list_invoices(invoice_status: Optional[InvoiceStatus] = None):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=DB_NOT_READY)


@router.post("/invoices", summary="إنشاء فاتورة | Create invoice")
async def create_invoice(invoice: InvoiceCreate):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=DB_NOT_READY)


@router.get("/reports", summary="التقارير | Reports")
async def get_reports(report_type: Optional[ReportType] = None, period: ReportPeriod = ReportPeriod.MONTHLY):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=DB_NOT_READY)


@router.get("/insights", summary="رؤى AI | AI insights")
async def get_insights(period: ReportPeriod = ReportPeriod.MONTHLY):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=DB_NOT_READY)


@router.get("/status", summary="حالة ERP | ERP status")
async def erp_status():
    return {
        "status": "not_implemented",
        "message": DB_NOT_READY,
        "requires": "PostgreSQL connection",
        "timestamp": datetime.now().isoformat()
    }
