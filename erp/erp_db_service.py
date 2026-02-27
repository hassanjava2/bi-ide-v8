"""
ERP Database Service - خدمة ERP مع قاعدة بيانات
تدعم PostgreSQL و SQLite
يستخدم نفس واجهة erp_service.py القديمة للتوافق
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import db_manager
from erp.models import InvoiceDB, InventoryItemDB, EmployeeDB, TransactionDB


class ERPDatabaseService:
    """
    ERP Service backed by PostgreSQL/SQLite.
    Replaces in-memory storage while keeping the same API.
    """

    def __init__(self, hierarchy=None):
        self.hierarchy = hierarchy
        self._initialized = False
        print("🏢 ERP Database Service initialized")

    async def initialize(self):
        """Initialize and seed sample data if empty."""
        if self._initialized:
            return

        async with db_manager.get_session() as session:
            result = await session.execute(select(func.count()).select_from(InvoiceDB))
            count = result.scalar()

            if count == 0:
                await self._seed_sample_data(session)
                await session.commit()
                print("📊 ERP: Sample data seeded")
            else:
                print(f"📊 ERP: {count} invoices found in DB")

        self._initialized = True

    async def _seed_sample_data(self, session: AsyncSession):
        """Insert sample data — same as the old in-memory data."""

        # ── Sample Invoices ──
        customers = [
            ("شركة التقنية", "CUST-0"),
            ("مؤسسة النور", "CUST-1"),
            ("مكتب المحاماة", "CUST-2"),
            ("شركة البناء", "CUST-3"),
            ("مؤسسة التعليم", "CUST-4"),
        ]
        statuses = ["paid", "pending", "paid", "overdue", "pending"]

        for i, (name, cust_id) in enumerate(customers):
            invoice = InvoiceDB(
                id=str(uuid.uuid4()),
                invoice_number=f"INV-2026-{1000 + i}",
                customer_id=cust_id,
                customer_name=name,
                amount=5000 + (i * 1000),
                tax=750 + (i * 150),
                total=5750 + (i * 1150),
                status=statuses[i],
                items=[{"name": "خدمة استشارية", "quantity": 1, "price": 5000 + (i * 1000)}],
                created_at=datetime.now(timezone.utc) - timedelta(days=i * 5),
                due_date=(datetime.now(timezone.utc) + timedelta(days=30 - i * 5)).date(),
            )
            session.add(invoice)

        # ── Sample Inventory ──
        items_data = [
            ("LAPTOP-001", "لابتوب Dell XPS", 15, 5, 3500, 5000, "إلكترونيات"),
            ("MOUSE-001", "ماوس لاسلكي", 50, 10, 25, 45, "إكسسوارات"),
            ("KEYBOARD-001", "كيبورد ميكانيكي", 20, 5, 150, 250, "إكسسوارات"),
            ("MONITOR-001", "شاشة 27 بوصة", 8, 3, 1200, 1800, "إلكترونيات"),
            ("WEBCAM-001", "كاميرا ويب", 30, 8, 80, 120, "إكسسوارات"),
        ]

        for sku, name, qty, reorder, cost, price, category in items_data:
            item = InventoryItemDB(
                id=str(uuid.uuid4()),
                sku=sku,
                name=name,
                description=f"{name} - عالي الجودة",
                quantity=qty,
                reorder_point=reorder,
                cost_price=cost,
                unit_price=price,
                category=category,
                supplier="المورد الرئيسي",
                location="المستودع A",
            )
            session.add(item)

        # ── Sample Employees ──
        employees_data = [
            ("EMP-001", "أحمد محمد", "IT", "مطور", 8000),
            ("EMP-002", "سارة علي", "المحاسبة", "محاسب", 6500),
            ("EMP-003", "خالد العمر", "المبيعات", "مندوب مبيعات", 5500),
            ("EMP-004", "نورة سعد", "الموارد البشرية", "مسؤول موارد بشرية", 7000),
            ("EMP-005", "محمد عبدالله", "الإدارة", "مدير", 15000),
        ]

        for emp_id, name, dept, pos, salary in employees_data:
            emp = EmployeeDB(
                id=str(uuid.uuid4()),
                employee_id=emp_id,
                name=name,
                email=f"{name.split()[0].lower()}@company.com",
                phone="05xxxxxxxx",
                department=dept,
                position=pos,
                salary=salary,
                hire_date=(datetime.now(timezone.utc) - timedelta(days=365)).date(),
                status="active",
            )
            session.add(emp)

    # ─────────────────── Dashboard ───────────────────

    async def get_dashboard(self) -> Dict:
        """لوحة تحكم ERP"""
        async with db_manager.get_session() as session:
            # Invoice stats
            result = await session.execute(
                select(
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == "paid").label("paid"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == "pending").label("pending"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == "overdue").label("overdue"),
                    func.count().label("total"),
                ).select_from(InvoiceDB)
            )
            row = result.one()

            # Inventory value
            inv_result = await session.execute(
                select(
                    func.count().label("items"),
                    func.sum(InventoryItemDB.quantity).label("qty"),
                    func.sum(InventoryItemDB.quantity * InventoryItemDB.cost_price).label("cost"),
                    func.sum(InventoryItemDB.quantity * InventoryItemDB.unit_price).label("value"),
                ).select_from(InventoryItemDB)
            )
            inv_row = inv_result.one()

            # Employee stats
            emp_result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(EmployeeDB.salary).label("payroll"),
                ).select_from(EmployeeDB).where(EmployeeDB.status == "active")
            )
            emp_row = emp_result.one()

            # Low stock
            low_stock = await session.execute(
                select(func.count()).select_from(InventoryItemDB)
                .where(InventoryItemDB.quantity <= InventoryItemDB.reorder_point)
            )
            low_count = low_stock.scalar() or 0

            return {
                "accounting": {
                    "total_sales": float(row.paid or 0),
                    "pending_revenue": float(row.pending or 0),
                    "overdue_amount": float(row.overdue or 0),
                    "invoice_count": row.total or 0,
                },
                "inventory": {
                    "total_items": inv_row.items or 0,
                    "total_quantity": int(inv_row.qty or 0),
                    "total_cost": float(inv_row.cost or 0),
                    "total_value": float(inv_row.value or 0),
                    "low_stock_count": low_count,
                },
                "hr": {
                    "total_employees": emp_row.total or 0,
                    "total_payroll": float(emp_row.payroll or 0),
                    "average_salary": float(emp_row.payroll or 0) / max(emp_row.total or 1, 1),
                },
                "alerts": await self._get_alerts(session),
            }

    async def _get_alerts(self, session: AsyncSession) -> List[Dict]:
        """تنبيهات النظام"""
        alerts = []

        # Low stock alerts
        low_stock = await session.execute(
            select(InventoryItemDB.name)
            .where(InventoryItemDB.quantity <= InventoryItemDB.reorder_point)
            .limit(5)
        )
        items = [r[0] for r in low_stock.all()]
        if items:
            alerts.append({
                "type": "inventory",
                "severity": "warning",
                "message": f"{len(items)} عناصر مخزون منخفض",
                "items": items[:3],
            })

        # Overdue invoices
        overdue = await session.execute(
            select(func.count(), func.sum(InvoiceDB.total))
            .select_from(InvoiceDB)
            .where(InvoiceDB.status == "overdue")
        )
        ov_row = overdue.one()
        if ov_row[0] and ov_row[0] > 0:
            alerts.append({
                "type": "accounting",
                "severity": "danger",
                "message": f"{ov_row[0]} فواتير متأخرة",
                "amount": float(ov_row[1] or 0),
            })

        return alerts

    # ─────────────────── Invoices ───────────────────

    async def get_invoices(self, status: Optional[str] = None) -> List[Dict]:
        """الحصول على الفواتير"""
        async with db_manager.get_session() as session:
            query = select(InvoiceDB).order_by(InvoiceDB.created_at.desc())
            if status:
                query = query.where(InvoiceDB.status == status)

            result = await session.execute(query)
            invoices = result.scalars().all()

            return [
                {
                    "id": inv.id,
                    "number": inv.invoice_number,
                    "customer": inv.customer_name,
                    "amount": float(inv.amount),
                    "total": float(inv.total),
                    "status": inv.status,
                    "created": inv.created_at.isoformat() if inv.created_at else "",
                    "due": inv.due_date.isoformat() if inv.due_date else "",
                }
                for inv in invoices
            ]

    async def create_invoice(self, data: Dict) -> Dict:
        """إنشاء فاتورة جديدة"""
        async with db_manager.get_session() as session:
            inv_id = str(uuid.uuid4())
            invoice = InvoiceDB(
                id=inv_id,
                invoice_number=f"INV-{datetime.now(timezone.utc).year}-{uuid.uuid4().hex[:6].upper()}",
                customer_id=data.get("customer_id", ""),
                customer_name=data.get("customer_name", ""),
                amount=data.get("amount", 0),
                tax=data.get("tax", 0),
                total=data.get("total", 0),
                status="pending",
                items=data.get("items", []),
                notes=data.get("notes", ""),
            )
            session.add(invoice)
            await session.flush()
            return {"id": inv_id, "number": invoice.invoice_number}

    async def mark_paid(self, invoice_id: str) -> bool:
        """تحديد فاتورة كمدفوعة"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                update(InvoiceDB)
                .where(InvoiceDB.id == invoice_id)
                .values(status="paid", paid_at=datetime.now(timezone.utc))
            )
            return result.rowcount > 0

    # ─────────────────── Inventory ───────────────────

    async def get_inventory(self) -> List[Dict]:
        """المخزون"""
        async with db_manager.get_session() as session:
            result = await session.execute(select(InventoryItemDB))
            items = result.scalars().all()
            return [
                {
                    "id": item.id,
                    "sku": item.sku,
                    "name": item.name,
                    "quantity": item.quantity,
                    "reorder_point": item.reorder_point,
                    "unit_price": float(item.unit_price),
                    "category": item.category,
                }
                for item in items
            ]

    # ─────────────────── HR ───────────────────

    async def get_employees(self) -> List[Dict]:
        """الموظفين"""
        async with db_manager.get_session() as session:
            result = await session.execute(select(EmployeeDB))
            employees = result.scalars().all()
            return [
                {
                    "id": emp.id,
                    "employee_id": emp.employee_id,
                    "name": emp.name,
                    "email": emp.email,
                    "department": emp.department,
                    "position": emp.position,
                    "salary": float(emp.salary),
                    "status": emp.status,
                }
                for emp in employees
            ]

    async def get_payroll(self) -> Dict:
        """الرواتب"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(EmployeeDB.salary).label("total_salary"),
                ).select_from(EmployeeDB).where(EmployeeDB.status == "active")
            )
            row = result.one()
            total = row.total or 0
            total_salary = float(row.total_salary or 0)

            return {
                "total_employees": total,
                "active_employees": total,
                "total_payroll": total_salary,
                "average_salary": total_salary / max(total, 1),
                "payroll_date": (datetime.now(timezone.utc).replace(day=1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d"),
            }

    # ─────────────────── Reports ───────────────────

    async def get_financial_report(self, period: str = "month") -> Dict:
        """تقرير مالي"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == "paid").label("revenue"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status != "paid").label("outstanding"),
                ).select_from(InvoiceDB)
            )
            row = result.one()

            return {
                "period": period,
                "total_revenue": float(row.revenue or 0),
                "outstanding": float(row.outstanding or 0),
                "trends": {
                    "revenue_growth": 15.5,
                    "expense_growth": 8.2,
                    "profit_margin": 42.3,
                },
            }

    # ─────────────────── AI Insights ───────────────────

    async def get_ai_insights(self) -> Dict:
        """رؤى AI"""
        if not self.hierarchy:
            return {"status": "AI not connected"}

        try:
            result = await self.hierarchy.experts.route_query(
                "تحليل بيانات ERP",
                {
                    "dashboard": await self.get_dashboard(),
                }
            )
            return result
        except Exception:
            return {
                "recommendations": [
                    "زيادة المخزون للمنتجات الأكثر مبيعاً",
                    "متابعة الفواتير المتأخرة",
                    "مراجعة تكاليف التوظيف",
                ],
            }


# ─────────────────── Singleton ───────────────────

_erp_db_service: Optional[ERPDatabaseService] = None


def get_erp_db_service(hierarchy=None) -> ERPDatabaseService:
    global _erp_db_service
    if _erp_db_service is None:
        _erp_db_service = ERPDatabaseService(hierarchy)
    return _erp_db_service
