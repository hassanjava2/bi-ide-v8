"""
Reports Module - التقارير
Monthly and periodic business reports
"""
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional

from sqlalchemy import select, and_, func, extract
from sqlalchemy.ext.asyncio import AsyncSession

from erp.invoices import Invoice, InvoiceStatus
from erp.inventory import Product, StockMovement
from erp.hr import Employee, PayrollRecord, EmployeeStatus
from erp.crm import Customer, CustomerStatus
from erp.accounting import Account, Transaction, AccountType


async def generate_monthly_report(
    session: AsyncSession,
    year: int,
    month: int
) -> Dict[str, Any]:
    """
    Generate comprehensive monthly report / تقرير شهرى شامل
    
    Includes sales, expenses, inventory, HR, and customer metrics.
    """
    # Calculate date range
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)
    
    report = {
        "report_period": f"{year}-{month:02d}",
        "month_name": month_start.strftime("%B %Y"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sales": await _generate_sales_report(session, month_start, month_end),
        "collections": await _generate_collections_report(session, month_start, month_end),
        "inventory": await _generate_inventory_report(session, month_start, month_end),
        "hr": await _generate_hr_report(session, year, month),
        "customers": await _generate_customer_report(session, month_start, month_end),
        "financial": await _generate_financial_summary(session, month_start, month_end)
    }
    
    return report


async def _generate_sales_report(
    session: AsyncSession,
    month_start: date,
    month_end: date
) -> Dict[str, Any]:
    """Generate sales report section"""
    
    # Invoice statistics
    result = await session.execute(
        select(
            func.count(Invoice.id).label("invoice_count"),
            func.sum(Invoice.subtotal).label("subtotal"),
            func.sum(Invoice.tax_amount).label("tax"),
            func.sum(Invoice.discount_amount).label("discounts"),
            func.sum(Invoice.total_amount).label("total"),
            func.avg(Invoice.total_amount).label("avg_invoice")
        )
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= month_end,
                Invoice.status.notin_([InvoiceStatus.CANCELLED])
            )
        )
    )
    row = result.one()
    
    # By status breakdown
    status_result = await session.execute(
        select(
            Invoice.status,
            func.count(Invoice.id).label("count"),
            func.sum(Invoice.total_amount).label("total")
        )
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= month_end
            )
        )
        .group_by(Invoice.status)
    )
    by_status = [
        {
            "status": r.status.value,
            "count": r.count,
            "amount": float(r.total or 0)
        }
        for r in status_result.all()
    ]
    
    return {
        "period": f"{month_start} to {month_end}",
        "invoices_issued": row.invoice_count or 0,
        "subtotal": float(row.subtotal or 0),
        "tax_collected": float(row.tax or 0),
        "discounts_given": float(row.discounts or 0),
        "total_sales": float(row.total or 0),
        "average_invoice_value": float(row.avg_invoice or 0),
        "by_status": by_status
    }


async def _generate_collections_report(
    session: AsyncSession,
    month_start: date,
    month_end: date
) -> Dict[str, Any]:
    """Generate collections/cash received report"""
    
    # Payments received this month
    result = await session.execute(
        select(
            func.count(Invoice.id).label("invoice_count"),
            func.sum(Invoice.paid_amount).label("total_collected")
        )
        .where(
            and_(
                func.date(Invoice.paid_at) >= month_start,
                func.date(Invoice.paid_at) <= month_end,
                Invoice.paid_at.isnot(None)
            )
        )
    )
    row = result.one()
    
    # Outstanding at month end
    outstanding_result = await session.execute(
        select(
            func.count(Invoice.id).label("count"),
            func.sum(Invoice.total_amount - Invoice.paid_amount).label("balance")
        )
        .where(
            and_(
                Invoice.issue_date <= month_end,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.PAID])
            )
        )
    )
    outstanding = outstanding_result.one()
    
    return {
        "invoices_paid": row.invoice_count or 0,
        "amount_collected": float(row.total_collected or 0),
        "outstanding_at_month_end": {
            "invoice_count": outstanding.count or 0,
            "balance": float(outstanding.balance or 0)
        }
    }


async def _generate_inventory_report(
    session: AsyncSession,
    month_start: date,
    month_end: date
) -> Dict[str, Any]:
    """Generate inventory report section"""
    
    # Current inventory value
    result = await session.execute(
        select(
            func.count(Product.id).label("product_count"),
            func.sum(Product.quantity_in_stock).label("total_units"),
            func.sum(Product.quantity_in_stock * Product.cost_price).label("cost_value"),
            func.sum(Product.quantity_in_stock * Product.unit_price).label("retail_value")
        )
        .where(Product.is_active == "true")
    )
    row = result.one()
    
    # Stock movements this month
    movement_result = await session.execute(
        select(
            StockMovement.movement_type,
            func.count(StockMovement.id).label("count"),
            func.sum(StockMovement.quantity).label("quantity"),
            func.sum(StockMovement.total_cost).label("value")
        )
        .where(
            and_(
                func.date(StockMovement.created_at) >= month_start,
                func.date(StockMovement.created_at) <= month_end
            )
        )
        .group_by(StockMovement.movement_type)
    )
    movements = [
        {
            "type": m.movement_type.value,
            "count": m.count,
            "quantity": m.quantity or 0,
            "value": float(m.value or 0)
        }
        for m in movement_result.all()
    ]
    
    return {
        "snapshot": {
            "active_products": row.product_count or 0,
            "total_units": row.total_units or 0,
            "cost_value": float(row.cost_value or 0),
            "retail_value": float(row.retail_value or 0),
            "potential_profit": float((row.retail_value or 0) - (row.cost_value or 0))
        },
        "movements_this_month": movements
    }


async def _generate_hr_report(
    session: AsyncSession,
    year: int,
    month: int
) -> Dict[str, Any]:
    """Generate HR report section"""
    
    # Employee counts
    result = await session.execute(
        select(
            Employee.status,
            func.count(Employee.id).label("count")
        )
        .where(Employee.is_active == "true")
        .group_by(Employee.status)
    )
    by_status = {r.status.value: r.count for r in result.all()}
    
    # Department breakdown
    dept_result = await session.execute(
        select(
            Employee.department,
            func.count(Employee.id).label("count"),
            func.sum(Employee.base_salary).label("total_salary")
        )
        .where(Employee.is_active == "true")
        .group_by(Employee.department)
        .order_by(func.count(Employee.id).desc())
    )
    by_department = [
        {
            "department": r.department or "Unassigned",
            "count": r.count,
            "total_salary": float(r.total_salary or 0)
        }
        for r in dept_result.all()
    ]
    
    # Payroll for this month
    payroll_result = await session.execute(
        select(
            func.count(PayrollRecord.id).label("count"),
            func.sum(PayrollRecord.gross_salary).label("gross"),
            func.sum(PayrollRecord.net_salary).label("net"),
            func.sum(PayrollRecord.tax).label("tax"),
            func.sum(PayrollRecord.social_insurance).label("insurance")
        )
        .where(
            and_(
                PayrollRecord.period_year == year,
                PayrollRecord.period_month == month
            )
        )
    )
    payroll = payroll_result.one()
    
    return {
        "employees": {
            "total": sum(by_status.values()),
            "by_status": by_status
        },
        "by_department": by_department,
        "payroll_this_month": {
            "employees_processed": payroll.count or 0,
            "gross_salary": float(payroll.gross or 0),
            "net_salary": float(payroll.net or 0),
            "tax_deducted": float(payroll.tax or 0),
            "social_insurance": float(payroll.insurance or 0)
        }
    }


async def _generate_customer_report(
    session: AsyncSession,
    month_start: date,
    month_end: date
) -> Dict[str, Any]:
    """Generate customer report section"""
    
    # Customer statistics
    result = await session.execute(
        select(
            func.count(Customer.id).label("total"),
            func.sum(func.case((func.date(Customer.created_at) >= month_start, 1), else_=0)).label("new")
        )
        .where(Customer.is_active == "true")
    )
    row = result.one()
    
    # Active customers (made purchases)
    active_result = await session.execute(
        select(func.count(func.distinct(Invoice.customer_id)))
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= month_end,
                Invoice.status.notin_([InvoiceStatus.CANCELLED, InvoiceStatus.DRAFT])
            )
        )
    )
    active_count = active_result.scalar() or 0
    
    # Top customers this month
    top_result = await session.execute(
        select(
            Invoice.customer_id,
            func.sum(Invoice.total_amount).label("total"),
            func.count(Invoice.id).label("count")
        )
        .where(
            and_(
                Invoice.issue_date >= month_start,
                Invoice.issue_date <= month_end,
                Invoice.status.notin_([InvoiceStatus.CANCELLED])
            )
        )
        .group_by(Invoice.customer_id)
        .order_by(func.sum(Invoice.total_amount).desc())
        .limit(5)
    )
    
    top_customers = []
    for r in top_result.all():
        customer_result = await session.execute(
            select(Customer).where(Customer.id == r.customer_id)
        )
        customer = customer_result.scalar_one_or_none()
        
        top_customers.append({
            "customer_id": r.customer_id,
            "name": customer.company_name or customer.contact_name if customer else "Unknown",
            "revenue": float(r.total or 0),
            "invoices": r.count
        })
    
    return {
        "total_customers": row.total or 0,
        "new_customers": row.new or 0,
        "active_customers": active_count,
        "top_customers": top_customers
    }


async def _generate_financial_summary(
    session: AsyncSession,
    month_start: date,
    month_end: date
) -> Dict[str, Any]:
    """Generate financial summary from accounting data"""
    
    # Revenue from income statement accounts
    result = await session.execute(
        select(func.sum(Transaction.amount))
        .where(
            and_(
                Transaction.credit_account_id.in_(
                    select(Account.id).where(Account.type == AccountType.REVENUE)
                ),
                func.date(Transaction.transaction_date) >= month_start,
                func.date(Transaction.transaction_date) <= month_end
            )
        )
    )
    revenue = float(result.scalar() or 0)
    
    # Expenses
    result = await session.execute(
        select(func.sum(Transaction.amount))
        .where(
            and_(
                Transaction.debit_account_id.in_(
                    select(Account.id).where(Account.type == AccountType.EXPENSE)
                ),
                func.date(Transaction.transaction_date) >= month_start,
                func.date(Transaction.transaction_date) <= month_end
            )
        )
    )
    expenses = float(result.scalar() or 0)
    
    return {
        "revenue": revenue,
        "expenses": expenses,
        "net_income": revenue - expenses
    }


async def generate_yearly_comparison(
    session: AsyncSession,
    year: int
) -> List[Dict[str, Any]]:
    """
    Generate month-by-month comparison for a year / مقارنة شهريه
    
    Returns sales, collections, and new customers for each month.
    """
    monthly_data = []
    
    for month in range(1, 13):
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        
        # Sales
        sales_result = await session.execute(
            select(func.sum(Invoice.total_amount))
            .where(
                and_(
                    Invoice.issue_date >= month_start,
                    Invoice.issue_date <= month_end,
                    Invoice.status.notin_([InvoiceStatus.CANCELLED])
                )
            )
        )
        sales = float(sales_result.scalar() or 0)
        
        # Collections
        collections_result = await session.execute(
            select(func.sum(Invoice.paid_amount))
            .where(
                and_(
                    func.date(Invoice.paid_at) >= month_start,
                    func.date(Invoice.paid_at) <= month_end
                )
            )
        )
        collections = float(collections_result.scalar() or 0)
        
        # New customers
        new_customers_result = await session.execute(
            select(func.count(Customer.id))
            .where(
                and_(
                    func.date(Customer.created_at) >= month_start,
                    func.date(Customer.created_at) <= month_end,
                    Customer.is_active == "true"
                )
            )
        )
        new_customers = new_customers_result.scalar() or 0
        
        monthly_data.append({
            "month": month,
            "month_name": month_start.strftime("%B"),
            "sales": sales,
            "collections": collections,
            "new_customers": new_customers
        })
    
    return monthly_data


async def generate_customer_statement(
    session: AsyncSession,
    customer_id: str,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """
    Generate customer statement / كشف حساب العميل
    
    Shows all invoices and payments for a customer in a date range.
    """
    # Get customer info
    customer_result = await session.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    customer = customer_result.scalar_one_or_none()
    
    if not customer:
        return {"error": "Customer not found"}
    
    # Get invoices in period
    result = await session.execute(
        select(Invoice)
        .where(
            and_(
                Invoice.customer_id == customer_id,
                Invoice.issue_date >= start_date,
                Invoice.issue_date <= end_date,
                Invoice.status != InvoiceStatus.CANCELLED
            )
        )
        .order_by(Invoice.issue_date)
    )
    invoices = result.scalars().all()
    
    transactions = []
    running_balance = 0.0
    
    for inv in invoices:
        # Invoice entry
        transactions.append({
            "date": inv.issue_date.isoformat(),
            "type": "invoice",
            "reference": inv.invoice_number,
            "description": f"Invoice #{inv.invoice_number}",
            "debit": float(inv.total_amount),
            "credit": 0.0,
            "balance": 0.0  # Will calculate after
        })
        running_balance += float(inv.total_amount)
        
        # Payment entry if paid
        if inv.paid_amount and inv.paid_amount > 0:
            payment_date = inv.paid_at.date() if inv.paid_at else inv.issue_date
            transactions.append({
                "date": payment_date.isoformat(),
                "type": "payment",
                "reference": inv.invoice_number,
                "description": f"Payment for #{inv.invoice_number}",
                "debit": 0.0,
                "credit": float(inv.paid_amount),
                "balance": 0.0
            })
            running_balance -= float(inv.paid_amount)
    
    # Sort by date and calculate running balance
    transactions.sort(key=lambda x: x["date"])
    balance = 0.0
    for t in transactions:
        balance += t["debit"] - t["credit"]
        t["balance"] = round(balance, 2)
    
    return {
        "customer": {
            "id": customer.id,
            "code": customer.customer_code,
            "name": customer.company_name or customer.contact_name,
            "email": customer.email,
            "phone": customer.phone
        },
        "period": f"{start_date} to {end_date}",
        "opening_balance": 0.0,  # Would need historical data
        "transactions": transactions,
        "closing_balance": round(balance, 2)
    }
