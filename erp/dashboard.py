"""
Dashboard Module - لوحة التحكم
Dashboard metrics and analytics
"""
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from erp.invoices import Invoice, InvoiceStatus
from erp.inventory import Product
from erp.hr import Employee, EmployeeStatus, PayrollRecord
from erp.crm import Customer, CustomerStatus


async def get_dashboard_metrics(
    session: AsyncSession,
    as_of_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard metrics / مقاييس لوحة التحكم
    
    Returns key business metrics for the dashboard.
    """
    as_of = as_of_date or date.today()
    month_start = as_of.replace(day=1)
    last_month_start = (month_start - timedelta(days=1)).replace(day=1)
    
    metrics = {
        "as_of_date": as_of.isoformat(),
        "sales": await _get_sales_metrics(session, month_start, as_of, last_month_start),
        "receivables": await _get_receivables_metrics(session),
        "inventory": await _get_inventory_metrics(session),
        "hr": await _get_hr_metrics(session),
        "customers": await _get_customer_metrics(session, month_start, as_of),
        "alerts": await _get_dashboard_alerts(session, as_of)
    }
    
    return metrics


async def _get_sales_metrics(
    session: AsyncSession,
    month_start: date,
    as_of: date,
    last_month_start: date
) -> Dict[str, Any]:
    """Get sales metrics for current and previous month"""
    
    # Current month sales
    result = await session.execute(
        select(
            func.count(Invoice.id).label("invoice_count"),
            func.sum(Invoice.total_amount).label("total_sales")
        )
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= as_of,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
            )
        )
    )
    current = result.one()
    
    # Last month sales
    result = await session.execute(
        select(func.sum(Invoice.total_amount).label("total_sales"))
        .where(
            and_(
                Invoice.issue_date >= last_month_start,
                Invoice.issue_date < month_start,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
            )
        )
    )
    last_month_total = result.scalar() or Decimal("0.00")
    
    current_sales = float(current.total_sales or 0)
    last_sales = float(last_month_total)
    
    # Calculate growth
    growth_pct = 0.0
    if last_sales > 0:
        growth_pct = ((current_sales - last_sales) / last_sales) * 100
    
    return {
        "current_month_sales": current_sales,
        "current_month_invoices": current.invoice_count or 0,
        "last_month_sales": last_sales,
        "month_over_month_growth_pct": round(growth_pct, 1)
    }


async def _get_receivables_metrics(session: AsyncSession) -> Dict[str, Any]:
    """Get accounts receivable metrics"""
    
    result = await session.execute(
        select(
            func.sum(Invoice.total_amount).label("total_invoiced"),
            func.sum(Invoice.paid_amount).label("total_paid"),
            func.sum(Invoice.total_amount - Invoice.paid_amount).label("outstanding")
        )
        .where(
            Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
        )
    )
    row = result.one()
    
    total_invoiced = float(row.total_invoiced or 0)
    total_paid = float(row.total_paid or 0)
    outstanding = float(row.outstanding or 0)
    
    # Calculate days sales outstanding (DSO) approximation
    # Average collection period = (Outstanding / Total Invoiced) * 30
    dso = 0.0
    if total_invoiced > 0:
        dso = (outstanding / total_invoiced) * 30
    
    return {
        "total_invoiced": total_invoiced,
        "total_collected": total_paid,
        "outstanding_receivables": outstanding,
        "collection_rate_pct": round((total_paid / total_invoiced * 100), 1) if total_invoiced > 0 else 0.0,
        "estimated_dso_days": round(dso, 1)
    }


async def _get_inventory_metrics(session: AsyncSession) -> Dict[str, Any]:
    """Get inventory metrics"""
    
    # Total products and value
    result = await session.execute(
        select(
            func.count(Product.id).label("product_count"),
            func.sum(Product.quantity_in_stock).label("total_units"),
            func.sum(Product.quantity_in_stock * Product.cost_price).label("total_value")
        )
        .where(Product.is_active == "true")
    )
    row = result.one()
    
    # Low stock items
    low_stock_result = await session.execute(
        select(func.count(Product.id))
        .where(
            and_(
                Product.is_active == "true",
                Product.quantity_in_stock <= Product.reorder_point
            )
        )
    )
    low_stock_count = low_stock_result.scalar() or 0
    
    return {
        "active_products": row.product_count or 0,
        "total_units_in_stock": row.total_units or 0,
        "inventory_value": float(row.total_value or 0),
        "low_stock_items": low_stock_count
    }


async def _get_hr_metrics(session: AsyncSession) -> Dict[str, Any]:
    """Get HR metrics"""
    
    # Employee counts by status
    result = await session.execute(
        select(
            Employee.status,
            func.count(Employee.id).label("count")
        )
        .where(Employee.is_active == "true")
        .group_by(Employee.status)
    )
    status_counts = {row.status.value: row.count for row in result.all()}
    
    total_employees = sum(status_counts.values())
    active_employees = status_counts.get(EmployeeStatus.ACTIVE.value, 0)
    
    # Monthly payroll estimate
    result = await session.execute(
        select(func.sum(Employee.base_salary))
        .where(
            and_(
                Employee.is_active == "true",
                Employee.status == EmployeeStatus.ACTIVE
            )
        )
    )
    monthly_payroll = float(result.scalar() or 0)
    
    return {
        "total_employees": total_employees,
        "active_employees": active_employees,
        "on_leave": status_counts.get(EmployeeStatus.ON_LEAVE.value, 0),
        "estimated_monthly_payroll": monthly_payroll
    }


async def _get_customer_metrics(
    session: AsyncSession,
    month_start: date,
    as_of: date
) -> Dict[str, Any]:
    """Get customer metrics"""
    
    # Total customers
    result = await session.execute(
        select(func.count(Customer.id))
        .where(Customer.is_active == "true")
    )
    total_customers = result.scalar() or 0
    
    # New customers this month
    result = await session.execute(
        select(func.count(Customer.id))
        .where(
            and_(
                Customer.is_active == "true",
                func.date(Customer.created_at) >= month_start
            )
        )
    )
    new_this_month = result.scalar() or 0
    
    # Active customers (with invoices this month)
    result = await session.execute(
        select(func.count(func.distinct(Invoice.customer_id)))
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= as_of,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
            )
        )
    )
    active_this_month = result.scalar() or 0
    
    return {
        "total_customers": total_customers,
        "new_this_month": new_this_month,
        "active_this_month": active_this_month
    }


async def _get_dashboard_alerts(
    session: AsyncSession,
    as_of: date
) -> List[Dict[str, Any]]:
    """Get dashboard alerts and notifications"""
    alerts = []
    
    # Overdue invoices
    from erp.invoices import get_overdue_invoices
    overdue = await get_overdue_invoices(session, as_of)
    
    if overdue:
        total_overdue = sum(inv["balance_due"] for inv in overdue)
        alerts.append({
            "type": "warning",
            "category": "receivables",
            "message": f"{len(overdue)} overdue invoices totaling {total_overdue:,.2f}",
            "count": len(overdue),
            "amount": total_overdue
        })
    
    # Low stock items
    from erp.inventory import get_low_stock_items
    low_stock = await get_low_stock_items(session, limit=10)
    
    if low_stock:
        alerts.append({
            "type": "warning",
            "category": "inventory",
            "message": f"{len(low_stock)} products below reorder point",
            "count": len(low_stock),
            "items": [item["sku"] for item in low_stock[:5]]
        })
    
    # Upcoming payroll (if near month end)
    days_to_month_end = (as_of.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1) - as_of
    if days_to_month_end.days <= 5:
        alerts.append({
            "type": "info",
            "category": "payroll",
            "message": f"Payroll processing due in {days_to_month_end.days} days",
            "days_remaining": days_to_month_end.days
        })
    
    return alerts


async def get_sales_chart_data(
    session: AsyncSession,
    months: int = 6
) -> List[Dict[str, Any]]:
    """
    Get sales data for charting / بيانات المبيعات للرسم البياني
    
    Returns monthly sales totals for the last N months.
    """
    end_date = date.today()
    data = []
    
    for i in range(months - 1, -1, -1):
        # Calculate month range
        month_end = end_date.replace(day=1) - timedelta(days=1) if i > 0 else end_date
        month_start = month_end.replace(day=1)
        
        # Adjust for current month
        if i == 0:
            month_start = end_date.replace(day=1)
            month_end = end_date
        else:
            for _ in range(i - 1):
                month_end = month_start - timedelta(days=1)
                month_start = month_end.replace(day=1)
        
        result = await session.execute(
            select(func.sum(Invoice.total_amount))
            .where(
                and_(
                    Invoice.issue_date >= month_start,
                    Invoice.issue_date <= month_end,
                    Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
                )
            )
        )
        total = float(result.scalar() or 0)
        
        data.append({
            "month": month_start.strftime("%Y-%m"),
            "month_name": month_start.strftime("%b %Y"),
            "sales": total
        })
    
    return data


async def get_top_customers(
    session: AsyncSession,
    limit: int = 5,
    months: int = 12
) -> List[Dict[str, Any]]:
    """Get top customers by revenue"""
    start_date = date.today() - timedelta(days=30 * months)
    
    result = await session.execute(
        select(
            Invoice.customer_id,
            func.sum(Invoice.total_amount).label("total_revenue"),
            func.count(Invoice.id).label("invoice_count")
        )
        .where(
            and_(
                Invoice.issue_date >= start_date,
                Invoice.status == InvoiceStatus.PAID
            )
        )
        .group_by(Invoice.customer_id)
        .order_by(func.sum(Invoice.total_amount).desc())
        .limit(limit)
    )
    
    top_customers = []
    for row in result.all():
        # Get customer details
        customer_result = await session.execute(
            select(Customer).where(Customer.id == row.customer_id)
        )
        customer = customer_result.scalar_one_or_none()
        
        top_customers.append({
            "customer_id": row.customer_id,
            "company_name": customer.company_name if customer else "Unknown",
            "contact_name": customer.contact_name if customer else "Unknown",
            "total_revenue": float(row.total_revenue or 0),
            "invoice_count": row.invoice_count
        })
    
    return top_customers


async def get_cash_flow_projection(
    session: AsyncSession,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get cash flow projection / توقع التدفق النقدي
    
    Estimates incoming cash from receivables.
    """
    today = date.today()
    projection_end = today + timedelta(days=days)
    
    # Expected collections from existing invoices
    result = await session.execute(
        select(
            func.sum(Invoice.total_amount - Invoice.paid_amount).label("expected"),
            func.sum(func.case(
                (Invoice.due_date <= projection_end, Invoice.total_amount - Invoice.paid_amount),
                else_=0
            )).label("within_period")
        )
        .where(
            and_(
                Invoice.status.in_([InvoiceStatus.SENT, InvoiceStatus.VIEWED, InvoiceStatus.PARTIAL, InvoiceStatus.OVERDUE]),
                Invoice.total_amount > Invoice.paid_amount
            )
        )
    )
    row = result.one()
    
    return {
        "projection_period_days": days,
        "total_outstanding": float(row.expected or 0),
        "expected_within_period": float(row.within_period or 0),
        "projection_end_date": projection_end.isoformat()
    }
