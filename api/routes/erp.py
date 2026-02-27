"""
ERP Routes - نقاط النهاية لنظام إدارة الموارد
مع تطبيق RBAC + قاعدة بيانات PostgreSQL
"""

from typing import Optional, Dict

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.schemas import (
    InvoiceCreateRequest, TransactionRequest, StockAdjustmentRequest,
    PayrollRequest, CustomerCreateRequest
)
from api.rbac import Permission, require_permission

router = APIRouter(prefix="/api/v1/erp", tags=["erp"])


class ProductCreateRequest(BaseModel):
    sku: str
    name: str
    description: str = ""
    quantity: int = 0
    unit_price: float = 0
    cost_price: float = 0
    reorder_point: int = 10
    category: str = ""
    location: str = ""
    supplier: str = ""


class StockChangeRequest(BaseModel):
    quantity_change: int
    reason: str
    reference: str = ""

# Service reference – set during startup
_erp_db_service = None


def set_erp_db_service(service):
    """Set the ERP database service during application startup"""
    global _erp_db_service
    _erp_db_service = service


def get_erp_service():
    """Dependency to get ERP service"""
    if _erp_db_service is None:
        raise HTTPException(500, "ERP service not initialized")
    return _erp_db_service


@router.get(
    "/dashboard",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_erp_dashboard(service=Depends(get_erp_service)):
    """لوحة تحكم ERP - Dashboard"""
    return await service.get_dashboard()


@router.get(
    "/invoices",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_invoices(status: Optional[str] = None, service=Depends(get_erp_service)):
    """الحصول على الفواتير - Get all invoices"""
    return await service.get_invoices(status)


@router.post(
    "/invoices",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_CREATE))],
)
async def create_invoice(request: InvoiceCreateRequest, service=Depends(get_erp_service)):
    """إنشاء فاتورة جديدة - Create new invoice"""
    return await service.create_invoice(request.model_dump())


@router.post(
    "/invoices/{invoice_id}/pay",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_EDIT))],
)
async def pay_invoice(invoice_id: str, service=Depends(get_erp_service)):
    """تحديد فاتورة كمدفوعة وإنشاء القيد المحاسبي - Mark invoice as paid with accounting entry"""
    success = await service.mark_invoice_paid(invoice_id)
    if not success:
        raise HTTPException(404, "Invoice not found")
    return {"success": True}


@router.get(
    "/inventory",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_READ))],
)
async def get_inventory(service=Depends(get_erp_service)):
    """الحصول على المخزون - Get inventory"""
    return await service.get_inventory()


@router.post(
    "/products",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_EDIT))],
)
async def create_product(request: ProductCreateRequest, service=Depends(get_erp_service)):
    """إنشاء منتج جديد - Create new product"""
    product = await service.create_product(request.model_dump())
    return product.to_dict() if hasattr(product, "to_dict") else product


@router.post(
    "/products/{product_id}/stock",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_EDIT))],
)
async def change_product_stock(
    product_id: str,
    request: StockChangeRequest,
    service=Depends(get_erp_service),
):
    """تعديل مخزون منتج - Adjust product stock"""
    product = await service.update_stock(
        product_id=product_id,
        quantity_change=request.quantity_change,
        reason=request.reason,
        reference=request.reference,
    )
    if not product:
        raise HTTPException(404, "Product not found")
    return product.to_dict() if hasattr(product, "to_dict") else product


@router.get(
    "/inventory/low-stock",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_READ))],
)
async def get_low_stock(service=Depends(get_erp_service)):
    """الحصول على منتجات المخزون المنخفض - Get low stock items"""
    return await service.get_low_stock()


@router.get(
    "/hr/employees",
    dependencies=[Depends(require_permission(Permission.ERP_HR_READ))],
)
async def get_employees(service=Depends(get_erp_service)):
    """الحصول على الموظفين - Get all employees"""
    return await service.get_employees()


@router.get(
    "/hr/payroll",
    dependencies=[Depends(require_permission(Permission.ERP_PAYROLL_READ))],
)
async def get_payroll(service=Depends(get_erp_service)):
    """الحصول على الرواتب - Get payroll information"""
    return await service.get_payroll()


@router.get(
    "/reports/financial",
    dependencies=[Depends(require_permission(Permission.ERP_REPORTS_READ))],
)
async def get_financial_report(period: str = "month", service=Depends(get_erp_service)):
    """التقرير المالي - Financial report"""
    return await service.get_financial_report(period)


@router.get(
    "/accounting/accounts",
    dependencies=[Depends(require_permission(Permission.ERP_REPORTS_READ))],
)
async def get_accounts(service=Depends(get_erp_service)):
    """الحصول على شجرة الحسابات - Get chart of accounts"""
    return await service.get_chart_of_accounts()


@router.get(
    "/accounting/trial-balance",
    dependencies=[Depends(require_permission(Permission.ERP_REPORTS_READ))],
)
async def get_trial_balance(service=Depends(get_erp_service)):
    """الحصول على ميزان المراجعة - Get trial balance"""
    return await service.get_trial_balance()


@router.get(
    "/accounting/transactions",
    dependencies=[Depends(require_permission(Permission.ERP_REPORTS_READ))],
)
async def get_transactions(
    account_id: Optional[str] = None, 
    limit: int = 100,
    service=Depends(get_erp_service)
):
    """الحصول على القيود المحاسبية - Get accounting transactions"""
    return await service.get_transactions(account_id, limit)


@router.post(
    "/accounting/transactions",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_EDIT))],
)
async def post_transaction(
    data: TransactionRequest,
    service=Depends(get_erp_service)
):
    """تسجيل قيد محاسبي - Post accounting transaction"""
    tx = await service.post_transaction(**data.model_dump())
    return {"id": tx.id, "status": "created"}


@router.get(
    "/customers",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_customers(service=Depends(get_erp_service)):
    """الحصول على العملاء - Get all customers"""
    return await service.get_customers()


@router.post(
    "/customers",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_CREATE))],
)
async def create_customer(
    data: CustomerCreateRequest,
    service=Depends(get_erp_service)
):
    """إنشاء عميل جديد - Create new customer"""
    return await service.create_customer(data.model_dump())


@router.post(
    "/inventory/adjust",
    dependencies=[Depends(require_permission(Permission.ERP_INVENTORY_EDIT))],
)
async def adjust_inventory(
    data: StockAdjustmentRequest,
    service=Depends(get_erp_service)
):
    """تعديل المخزون - Adjust inventory"""
    result = await service.update_stock(**data.model_dump())
    if not result:
        raise HTTPException(404, "Product not found")
    return result


@router.post(
    "/payroll/process",
    dependencies=[Depends(require_permission(Permission.ERP_PAYROLL_EDIT))],
)
async def process_payroll(
    data: PayrollRequest,
    service=Depends(get_erp_service)
):
    """معالجة الرواتب - Process payroll"""
    result = await service.process_payroll(**data.model_dump())
    if not result:
        raise HTTPException(404, "Employee not found")
    return result


@router.get(
    "/ai-insights",
    dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))],
)
async def get_erp_ai_insights(service=Depends(get_erp_service)):
    """رؤى AI للـ ERP - AI insights for ERP"""
    return await service.get_ai_insights()
