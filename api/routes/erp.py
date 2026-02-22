"""
ERP Routes - نقاط النهاية لنظام إدارة الموارد
مع تطبيق RBAC + قاعدة بيانات
"""

from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Depends

from api.schemas import InvoiceCreateRequest
from api.rbac import Permission, require_permission

router = APIRouter(prefix="/api/v1/erp", tags=["erp"])

# Service reference – set during startup
_erp_service = None       # In-memory (old)
_erp_db_service = None    # Database-backed (new)


def set_erp_service(service):
    global _erp_service
    _erp_service = service


def set_erp_db_service(service):
    global _erp_db_service
    _erp_db_service = service


def _use_db():
    """Check if we should use DB-backed service."""
    return _erp_db_service is not None


def _svc():
    if _erp_service is None:
        raise HTTPException(500, "ERP not initialized")
    return _erp_service


@router.get(
    "/dashboard",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_erp_dashboard():
    """لوحة تحكم ERP"""
    if _use_db():
        return await _erp_db_service.get_dashboard()
    return _svc().get_dashboard()


@router.get(
    "/invoices",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_invoices(status: Optional[str] = None):
    """الفواتير"""
    if _use_db():
        return await _erp_db_service.get_invoices(status)

    invoices = _svc().accounting.get_invoices(status)
    return [
        {
            "id": inv.id,
            "number": inv.invoice_number,
            "customer": inv.customer_name,
            "amount": inv.amount,
            "total": inv.total,
            "status": inv.status.value,
            "created": inv.created_at.isoformat(),
            "due": inv.due_date.isoformat(),
        }
        for inv in invoices
    ]


@router.post(
    "/invoices",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_CREATE))],
)
async def create_invoice(request: InvoiceCreateRequest):
    """إنشاء فاتورة"""
    if _use_db():
        return await _erp_db_service.create_invoice(request.model_dump())

    invoice = _svc().accounting.create_invoice(request.model_dump())
    return {"id": invoice.id, "number": invoice.invoice_number}


@router.post(
    "/invoices/{invoice_id}/pay",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_EDIT))],
)
async def mark_invoice_paid(invoice_id: str):
    """تحديد فاتورة كمدفوعة"""
    if _use_db():
        success = await _erp_db_service.mark_paid(invoice_id)
        return {"success": success}

    success = _svc().accounting.mark_paid(invoice_id)
    return {"success": success}


@router.get(
    "/inventory",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_READ))],
)
async def get_inventory():
    """المخزون"""
    if _use_db():
        return await _erp_db_service.get_inventory()

    items = _svc().inventory.get_all_items()
    return [
        {
            "id": item.id,
            "sku": item.sku,
            "name": item.name,
            "quantity": item.quantity,
            "reorder_point": item.reorder_point,
            "unit_price": item.unit_price,
            "category": item.category,
        }
        for item in items
    ]


@router.get(
    "/hr/employees",
    dependencies=[Depends(require_permission(Permission.ERP_HR_READ))],
)
async def get_employees():
    """الموظفين"""
    if _use_db():
        return await _erp_db_service.get_employees()

    employees = _svc().hr.get_all_employees()
    return [
        {
            "id": emp.id,
            "employee_id": emp.employee_id,
            "name": emp.name,
            "email": emp.email,
            "department": emp.department,
            "position": emp.position,
            "salary": emp.salary,
            "status": emp.status.value,
        }
        for emp in employees
    ]


@router.get(
    "/hr/payroll",
    dependencies=[Depends(require_permission(Permission.ERP_PAYROLL_READ))],
)
async def get_payroll():
    """الرواتب"""
    if _use_db():
        return await _erp_db_service.get_payroll()
    return _svc().hr.calculate_payroll()


@router.get(
    "/reports/financial",
    dependencies=[Depends(require_permission(Permission.ERP_REPORTS_READ))],
)
async def get_financial_report(period: str = "month"):
    """التقرير المالي"""
    if _use_db():
        return await _erp_db_service.get_financial_report(period)
    return _svc().accounting.get_financial_report(period)


@router.get(
    "/ai-insights",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_erp_ai_insights():
    """رؤى AI للـ ERP"""
    if _use_db():
        return await _erp_db_service.get_ai_insights()
    return await _svc().get_ai_insights()
