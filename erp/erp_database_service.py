"""
ERP Database Service - خدمة ERP مع قاعدة البيانات
تدعم PostgreSQL و SQLite
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import db_manager
from erp.models.database_models import (
    AccountDB, AccountTypeDB, TransactionDB, InvoiceDB, InvoiceStatusDB, InvoiceItemDB,
    ProductDB, StockMovementDB, EmployeeDB, EmployeeStatusDB, PayrollRecordDB,
    CustomerDB, SupplierDB
)


class ERPDatabaseService:
    """
    ERP Service backed by PostgreSQL/SQLite.
    Provides full CRUD operations for ERP modules.
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
            # Check if we have any accounts
            result = await session.execute(select(func.count()).select_from(AccountDB))
            count = result.scalar()

            if count == 0:
                await self._seed_sample_data(session)
                await session.commit()
                print("📊 ERP: Sample data seeded")
            else:
                print(f"📊 ERP: {count} accounts found in DB")

        self._initialized = True

    async def _seed_sample_data(self, session: AsyncSession):
        """Insert sample data for all ERP modules."""
        
        # ═══════════════════════════════════════════════════════════
        # 1. Chart of Accounts - شجرة الحسابات
        # ═══════════════════════════════════════════════════════════
        accounts_data = [
            # Assets
            ("1000", "Assets", "الأصول", AccountTypeDB.ASSET, 0),
            ("1100", "Cash", "النقد", AccountTypeDB.ASSET, 100000),
            ("1200", "Accounts Receivable", "ذمم مدينة", AccountTypeDB.ASSET, 25000),
            ("1300", "Inventory", "المخزون", AccountTypeDB.ASSET, 50000),
            # Liabilities
            ("2000", "Liabilities", "الخصوم", AccountTypeDB.LIABILITY, 0),
            ("2100", "Accounts Payable", "ذمم دائنة", AccountTypeDB.LIABILITY, 15000),
            ("2200", "Loans", "القروض", AccountTypeDB.LIABILITY, 0),
            # Equity
            ("3000", "Equity", "حقوق الملكية", AccountTypeDB.EQUITY, 160000),
            ("3100", "Capital", "رأس المال", AccountTypeDB.EQUITY, 100000),
            ("3200", "Retained Earnings", "الأرباح المحتجزة", AccountTypeDB.EQUITY, 60000),
            # Revenue
            ("4000", "Revenue", "الإيرادات", AccountTypeDB.REVENUE, 0),
            ("4100", "Sales", "المبيعات", AccountTypeDB.REVENUE, 50000),
            ("4200", "Services", "الخدمات", AccountTypeDB.REVENUE, 20000),
            # Expenses
            ("5000", "Expenses", "المصروفات", AccountTypeDB.EXPENSE, 0),
            ("5100", "Salaries", "الرواتب", AccountTypeDB.EXPENSE, 30000),
            ("5200", "Rent", "الإيجار", AccountTypeDB.EXPENSE, 10000),
            ("5300", "Utilities", "الخدمات", AccountTypeDB.EXPENSE, 5000),
        ]
        
        account_map = {}
        for code, name, name_ar, acc_type, balance in accounts_data:
            account = AccountDB(
                id=str(uuid.uuid4()),
                code=code,
                name=name,
                name_ar=name_ar,
                type=acc_type,
                balance=balance
            )
            session.add(account)
            account_map[code] = account

        # ═══════════════════════════════════════════════════════════
        # 2. Sample Invoices
        # ═══════════════════════════════════════════════════════════
        customers = [
            ("شركة التقنية", "CUST-001", "tech@example.com"),
            ("مؤسسة النور", "CUST-002", "nour@example.com"),
            ("مكتب المحاماة", "CUST-003", "law@example.com"),
            ("شركة البناء", "CUST-004", "build@example.com"),
            ("مؤسسة التعليم", "CUST-005", "edu@example.com"),
        ]
        statuses = [InvoiceStatusDB.PAID, InvoiceStatusDB.SENT, InvoiceStatusDB.PAID, 
                   InvoiceStatusDB.OVERDUE, InvoiceStatusDB.DRAFT]

        for i, (name, cust_id, email) in enumerate(customers):
            invoice = InvoiceDB(
                id=str(uuid.uuid4()),
                invoice_number=f"INV-2026-{1000 + i}",
                customer_id=cust_id,
                customer_name=name,
                customer_email=email,
                amount=5000 + (i * 1000),
                tax_amount=750 + (i * 150),
                total=5750 + (i * 1150),
                status=statuses[i],
                items="خدمة استشارية",
                due_date=(datetime.now(timezone.utc) + timedelta(days=30 - i * 5)).date(),
            )
            session.add(invoice)

        # ═══════════════════════════════════════════════════════════
        # 3. Sample Products
        # ═══════════════════════════════════════════════════════════
        products_data = [
            ("LAPTOP-001", "لابتوب Dell XPS", 15, 5, 3500, 5000, "إلكترونيات"),
            ("MOUSE-001", "ماوس لاسلكي", 50, 10, 25, 45, "إكسسوارات"),
            ("KEYBOARD-001", "كيبورد ميكانيكي", 20, 5, 150, 250, "إكسسوارات"),
            ("MONITOR-001", "شاشة 27 بوصة", 8, 3, 1200, 1800, "إلكترونيات"),
            ("WEBCAM-001", "كاميرا ويب", 30, 8, 80, 120, "إكسسوارات"),
        ]

        for sku, name, qty, reorder, cost, price, category in products_data:
            product = ProductDB(
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
            session.add(product)

        # ═══════════════════════════════════════════════════════════
        # 4. Sample Employees
        # ═══════════════════════════════════════════════════════════
        employees_data = [
            ("EMP-001", "أحمد", "محمد", "ahmed@company.com", "IT", "مطور", 8000),
            ("EMP-002", "سارة", "علي", "sara@company.com", "المحاسبة", "محاسب", 6500),
            ("EMP-003", "خالد", "العمر", "khaled@company.com", "المبيعات", "مندوب مبيعات", 5500),
            ("EMP-004", "نورة", "سعد", "noura@company.com", "الموارد البشرية", "مسؤول موارد بشرية", 7000),
            ("EMP-005", "محمد", "عبدالله", "mohammed@company.com", "الإدارة", "مدير", 15000),
        ]

        for emp_id, first, last, email, dept, pos, salary in employees_data:
            emp = EmployeeDB(
                id=str(uuid.uuid4()),
                employee_number=emp_id,
                first_name=first,
                last_name=last,
                email=email,
                department=dept,
                position=pos,
                salary=salary,
                hire_date=(datetime.now(timezone.utc) - timedelta(days=365)).date(),
                status=EmployeeStatusDB.ACTIVE,
            )
            session.add(emp)

        # ═══════════════════════════════════════════════════════════
        # 5. Sample Customers
        # ═══════════════════════════════════════════════════════════
        for i, (name, cust_id, email) in enumerate(customers):
            customer = CustomerDB(
                id=str(uuid.uuid4()),
                customer_code=cust_id,
                name=name,
                email=email,
                customer_type="regular" if i < 4 else "vip",
            )
            session.add(customer)

    # ═══════════════════════════════════════════════════════════════════
    # ACCOUNTING METHODS - دوال المحاسبة
    # ═══════════════════════════════════════════════════════════════════

    async def get_chart_of_accounts(self) -> List[Dict]:
        """Get full chart of accounts"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(AccountDB).order_by(AccountDB.code)
            )
            accounts = result.scalars().all()
            return [acc.to_dict() for acc in accounts]
    
    async def get_trial_balance(self) -> Dict:
        """Generate trial balance report"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(AccountDB.type, func.sum(AccountDB.balance))
                .group_by(AccountDB.type)
            )
            balances = {row[0].value if row[0] else None: float(row[1] or 0) for row in result.all()}
            
            total_assets = balances.get('asset', 0)
            total_liabilities = balances.get('liability', 0)
            total_equity = balances.get('equity', 0)
            
            return {
                "assets": total_assets,
                "liabilities": total_liabilities,
                "equity": total_equity,
                "revenue": balances.get('revenue', 0),
                "expenses": balances.get('expense', 0),
                "balanced": abs(total_assets - (total_liabilities + total_equity)) < 0.01,
                "accounts": [
                    {"type": k, "balance": v}
                    for k, v in balances.items() if k
                ]
            }

    async def create_account(self, data: dict) -> AccountDB:
        """إنشاء حساب محاسبي جديد"""
        async with db_manager.get_session() as session:
            account = AccountDB(
                id=str(uuid.uuid4()),
                code=data.get("code"),
                name=data.get("name"),
                name_ar=data.get("name_ar"),
                type=AccountTypeDB(data.get("type", "asset")),
                balance=data.get("balance", 0.0),
                is_active=data.get("is_active", True),
            )
            session.add(account)
            await session.flush()
            return account

    async def get_account(self, account_id: str) -> Optional[AccountDB]:
        """الحصول على حساب محاسبي"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(AccountDB).where(AccountDB.id == account_id)
            )
            return result.scalar_one_or_none()

    async def get_account_by_code(self, code: str) -> Optional[AccountDB]:
        """الحصول على حساب برقم الكود"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(AccountDB).where(AccountDB.code == code)
            )
            return result.scalar_one_or_none()

    async def post_transaction(self, debit_account_id: str, credit_account_id: str,
                               amount: float, description: str, 
                               reference: str = "") -> TransactionDB:
        """تسجيل قيد محاسبي"""
        async with db_manager.get_session() as session:
            # Create transaction record
            tx = TransactionDB(
                id=str(uuid.uuid4()),
                debit_account_id=debit_account_id,
                credit_account_id=credit_account_id,
                amount=amount,
                description=description,
                reference=reference,
            )
            session.add(tx)

            # Update account balances
            debit_acc = await session.get(AccountDB, debit_account_id)
            credit_acc = await session.get(AccountDB, credit_account_id)
            
            if debit_acc and credit_acc:
                # Debit increases assets/expenses, decreases liabilities/equity/revenue
                if debit_acc.type in [AccountTypeDB.ASSET, AccountTypeDB.EXPENSE]:
                    debit_acc.balance = float(debit_acc.balance) + amount
                else:
                    debit_acc.balance = float(debit_acc.balance) - amount
                
                # Credit decreases assets/expenses, increases liabilities/equity/revenue
                if credit_acc.type in [AccountTypeDB.ASSET, AccountTypeDB.EXPENSE]:
                    credit_acc.balance = float(credit_acc.balance) - amount
                else:
                    credit_acc.balance = float(credit_acc.balance) + amount

            await session.flush()
            return tx

    async def get_transactions(self, account_id: str = None, limit: int = 100) -> List[Dict]:
        """الحصول على القيود المحاسبية"""
        async with db_manager.get_session() as session:
            query = select(TransactionDB).order_by(TransactionDB.date.desc()).limit(limit)
            if account_id:
                query = query.where(
                    (TransactionDB.debit_account_id == account_id) | 
                    (TransactionDB.credit_account_id == account_id)
                )
            
            result = await session.execute(query)
            txs = result.scalars().all()
            
            return [
                {
                    "id": tx.id,
                    "date": tx.date.isoformat() if tx.date else "",
                    "debit_account_id": tx.debit_account_id,
                    "credit_account_id": tx.credit_account_id,
                    "amount": float(tx.amount),
                    "description": tx.description,
                    "reference": tx.reference,
                }
                for tx in txs
            ]

    async def create_invoice(self, data: Dict) -> Dict:
        """Create invoice with items"""
        async with db_manager.get_session() as session:
            # Create invoice
            invoice_id = str(uuid.uuid4())
            invoice = InvoiceDB(
                id=invoice_id,
                invoice_number=data.get("invoice_number") or f"INV-{datetime.now(timezone.utc).year}-{uuid.uuid4().hex[:6].upper()}",
                customer_id=data.get("customer_id", ""),
                customer_name=data.get("customer_name", ""),
                customer_email=data.get("customer_email"),
                amount=data.get("subtotal", data.get("amount", 0)),
                tax_amount=data.get("tax_amount", 0),
                total=data.get("total", 0),
                status=InvoiceStatusDB.DRAFT,
                notes=data.get("notes", ""),
                due_date=data.get("due_date"),
            )
            session.add(invoice)
            
            # Add line items
            items_data = data.get("items", [])
            if items_data:
                for item_data in items_data:
                    item = InvoiceItemDB(
                        id=str(uuid.uuid4()),
                        invoice_id=invoice_id,
                        product_id=item_data.get("product_id"),
                        description=item_data.get("description", ""),
                        quantity=item_data.get("quantity", 1),
                        unit_price=item_data.get("unit_price", 0),
                        total=item_data.get("quantity", 1) * item_data.get("unit_price", 0)
                    )
                    session.add(item)
            
            await session.commit()
            # Normalize response shape (tests expect `number`)
            base = invoice.to_dict() if hasattr(invoice, "to_dict") else {}
            base.setdefault("id", invoice.id)
            base.setdefault("number", invoice.invoice_number)
            base.setdefault("status", invoice.status.value if invoice.status else "draft")
            base.setdefault("customer", invoice.customer_name)
            base.setdefault("total", float(invoice.total) if invoice.total is not None else 0.0)
            return base
    
    async def mark_invoice_paid(self, invoice_id: str) -> bool:
        """Mark invoice as paid and post accounting entries"""
        async with db_manager.get_session() as session:
            invoice = await session.get(InvoiceDB, invoice_id)
            if not invoice:
                return False
            
            invoice.status = InvoiceStatusDB.PAID
            invoice.paid_at = datetime.now(timezone.utc)
            
            # Get accounts for posting
            cash_account = await session.execute(
                select(AccountDB).where(AccountDB.code == "1100")
            )
            cash_account = cash_account.scalar_one_or_none()
            
            sales_account = await session.execute(
                select(AccountDB).where(AccountDB.code == "4100")
            )
            sales_account = sales_account.scalar_one_or_none()
            
            # Post accounting entry: Dr Cash, Cr Revenue
            if cash_account and sales_account:
                # Update balances
                cash_account.balance = float(cash_account.balance) + float(invoice.total)
                sales_account.balance = float(sales_account.balance) + float(invoice.total)
                
                # Create transaction record
                tx = TransactionDB(
                    id=str(uuid.uuid4()),
                    debit_account_id=cash_account.id,
                    credit_account_id=sales_account.id,
                    amount=float(invoice.total),
                    description=f"Payment for invoice {invoice.invoice_number}",
                    reference=invoice_id
                )
                session.add(tx)
            
            await session.commit()
            return True

    async def get_invoices(self, status: str = None) -> List[Dict]:
        """الحصول على الفواتير"""
        async with db_manager.get_session() as session:
            query = select(InvoiceDB).order_by(InvoiceDB.created_at.desc())
            if status:
                query = query.where(InvoiceDB.status == InvoiceStatusDB(status))

            result = await session.execute(query)
            invoices = result.scalars().all()

            return [
                {
                    "id": inv.id,
                    "number": inv.invoice_number,
                    "customer": inv.customer_name,
                    "amount": float(inv.amount),
                    "total": float(inv.total),
                    "status": inv.status.value if inv.status else "",
                    "created": inv.created_at.isoformat() if inv.created_at else "",
                    "due": inv.due_date.isoformat() if inv.due_date else "",
                }
                for inv in invoices
            ]

    async def mark_paid(self, invoice_id: str) -> bool:
        """تحديد فاتورة كمدفوعة"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                update(InvoiceDB)
                .where(InvoiceDB.id == invoice_id)
                .values(status=InvoiceStatusDB.PAID, paid_at=datetime.now(timezone.utc))
            )
            return result.rowcount > 0

    # ═══════════════════════════════════════════════════════════════════
    # INVENTORY METHODS - دوال المخزون
    # ═══════════════════════════════════════════════════════════════════

    async def create_product(self, data: dict) -> ProductDB:
        """إنشاء منتج جديد"""
        async with db_manager.get_session() as session:
            product = ProductDB(
                id=str(uuid.uuid4()),
                sku=data.get("sku"),
                name=data.get("name"),
                description=data.get("description", ""),
                quantity=data.get("quantity", 0),
                reorder_point=data.get("reorder_point", 10),
                unit_price=data.get("unit_price", 0),
                cost_price=data.get("cost_price", 0),
                category=data.get("category", ""),
                location=data.get("location", ""),
                supplier=data.get("supplier", ""),
            )
            session.add(product)
            await session.flush()
            return product

    async def get_product(self, product_id: str) -> Optional[ProductDB]:
        """الحصول على منتج"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(ProductDB).where(ProductDB.id == product_id)
            )
            return result.scalar_one_or_none()

    async def update_stock(self, product_id: str, quantity_change: int,
                           reason: str, reference: str = "") -> Optional[ProductDB]:
        """تحديث المخزون"""
        async with db_manager.get_session() as session:
            product = await session.get(ProductDB, product_id)
            if not product:
                return None

            # Update quantity
            product.quantity = product.quantity + quantity_change

            # Record movement
            movement = StockMovementDB(
                id=str(uuid.uuid4()),
                product_id=product_id,
                movement_type="in" if quantity_change > 0 else "out",
                quantity=abs(quantity_change),
                reason=reason,
                reference=reference,
            )
            session.add(movement)
            await session.flush()
            return product

    async def get_inventory(self) -> List[Dict]:
        """الحصول على المخزون"""
        async with db_manager.get_session() as session:
            result = await session.execute(select(ProductDB))
            items = result.scalars().all()
            return [item.to_dict() for item in items]

    async def get_low_stock_items(self, threshold: int = 10) -> List[Dict]:
        """Get products below reorder level"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(ProductDB).where(ProductDB.quantity < threshold)
            )
            products = result.scalars().all()
            return [p.to_dict() for p in products]
    
    # Alias for backward compatibility
    async def get_low_stock(self) -> List[Dict]:
        """الحصول على منتجات المخزون المنخفض"""
        return await self.get_low_stock_items()

    # ═══════════════════════════════════════════════════════════════════
    # HR METHODS - دوال الموارد البشرية
    # ═══════════════════════════════════════════════════════════════════

    async def create_employee(self, data: dict) -> EmployeeDB:
        """إنشاء موظف جديد"""
        async with db_manager.get_session() as session:
            emp = EmployeeDB(
                id=str(uuid.uuid4()),
                employee_number=data.get("employee_number"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                email=data.get("email"),
                phone=data.get("phone", ""),
                department=data.get("department"),
                position=data.get("position"),
                salary=data.get("salary", 0),
                hire_date=data.get("hire_date"),
                status=EmployeeStatusDB(data.get("status", "active")),
            )
            session.add(emp)
            await session.flush()
            return emp

    async def get_employee(self, employee_id: str) -> Optional[EmployeeDB]:
        """الحصول على موظف"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(EmployeeDB).where(EmployeeDB.id == employee_id)
            )
            return result.scalar_one_or_none()

    async def get_employees(self) -> List[Dict]:
        """الحصول على جميع الموظفين"""
        async with db_manager.get_session() as session:
            result = await session.execute(select(EmployeeDB))
            employees = result.scalars().all()
            return [emp.to_dict() for emp in employees]
    
    async def get_customers(self) -> List[Dict]:
        """Get all customers"""
        async with db_manager.get_session() as session:
            result = await session.execute(select(CustomerDB))
            customers = result.scalars().all()
            return [c.to_dict() for c in customers]
    
    async def create_customer(self, data: Dict) -> Dict:
        """Create a new customer"""
        async with db_manager.get_session() as session:
            customer = CustomerDB(
                id=str(uuid.uuid4()),
                customer_code=data.get("customer_code"),
                name=data.get("name"),
                email=data.get("email"),
                phone=data.get("phone", ""),
                address=data.get("address", ""),
                customer_type=data.get("customer_type", "regular"),
                credit_limit=data.get("credit_limit", 0),
                balance=data.get("balance", 0),
            )
            session.add(customer)
            await session.flush()
            return customer.to_dict()

    async def process_payroll(
        self,
        employee_id: str,
        month: int,
        year: int,
        overtime_hours: float = 0,
        deductions: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Process monthly payroll with overtime and deductions"""
        async with db_manager.get_session() as session:
            employee = await session.get(EmployeeDB, employee_id)
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")

            # Check if already processed
            result = await session.execute(
                select(PayrollRecordDB).where(
                    (PayrollRecordDB.employee_id == employee_id) &
                    (PayrollRecordDB.month == month) &
                    (PayrollRecordDB.year == year)
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                return existing.to_dict()

            # Calculate payroll
            base_salary = float(employee.salary)
            
            # Calculate overtime (1.5x hourly rate)
            hourly_rate = base_salary / 30 / 8
            overtime_pay = overtime_hours * hourly_rate * 1.5
            
            # Saudi GOSI deduction (10% employee share)
            gosi_deduction = base_salary * 0.10
            
            # Additional deductions
            additional_deductions = sum(deductions.values() if deductions else [])
            total_deductions = gosi_deduction + additional_deductions
            
            net_salary = base_salary + overtime_pay - total_deductions

            payroll = PayrollRecordDB(
                id=str(uuid.uuid4()),
                employee_id=employee_id,
                month=month,
                year=year,
                base_salary=base_salary,
                overtime=overtime_pay,
                allowances=0,
                deductions=total_deductions,
                net_salary=net_salary,
                status="pending",
            )
            session.add(payroll)
            await session.commit()
            return payroll.to_dict()

    async def get_payroll(self) -> Dict:
        """الحصول على إحصائيات الرواتب"""
        async with db_manager.get_session() as session:
            result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(EmployeeDB.salary).label("total_salary"),
                ).select_from(EmployeeDB).where(EmployeeDB.status == EmployeeStatusDB.ACTIVE)
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

    # ═══════════════════════════════════════════════════════════════════
    # DASHBOARD & REPORTS - لوحة التحكم والتقارير
    # ═══════════════════════════════════════════════════════════════════

    async def get_dashboard(self) -> Dict:
        """لوحة تحكم ERP"""
        async with db_manager.get_session() as session:
            # Invoice stats
            result = await session.execute(
                select(
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == InvoiceStatusDB.PAID).label("paid"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == InvoiceStatusDB.SENT).label("pending"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == InvoiceStatusDB.OVERDUE).label("overdue"),
                    func.count().label("total"),
                ).select_from(InvoiceDB)
            )
            row = result.one()

            # Inventory value
            inv_result = await session.execute(
                select(
                    func.count().label("items"),
                    func.sum(ProductDB.quantity).label("qty"),
                    func.sum(ProductDB.quantity * ProductDB.cost_price).label("cost"),
                    func.sum(ProductDB.quantity * ProductDB.unit_price).label("value"),
                ).select_from(ProductDB)
            )
            inv_row = inv_result.one()

            # Employee stats
            emp_result = await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(EmployeeDB.salary).label("payroll"),
                ).select_from(EmployeeDB).where(EmployeeDB.status == EmployeeStatusDB.ACTIVE)
            )
            emp_row = emp_result.one()

            # Low stock count
            low_stock = await session.execute(
                select(func.count()).select_from(ProductDB)
                .where(ProductDB.quantity <= ProductDB.reorder_point)
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
            select(ProductDB.name)
            .where(ProductDB.quantity <= ProductDB.reorder_point)
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
            .where(InvoiceDB.status == InvoiceStatusDB.OVERDUE)
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

    async def get_financial_report(self, period: str = "month") -> Dict:
        """تقرير مالي"""
        async with db_manager.get_session() as session:
            # Revenue
            result = await session.execute(
                select(
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status == InvoiceStatusDB.PAID).label("revenue"),
                    func.sum(InvoiceDB.total).filter(InvoiceDB.status != InvoiceStatusDB.PAID).label("outstanding"),
                ).select_from(InvoiceDB)
            )
            row = result.one()

            # Chart of accounts
            accounts_result = await session.execute(select(AccountDB))
            accounts = accounts_result.scalars().all()
            chart_of_accounts = {
                "assets": [],
                "liabilities": [],
                "equity": [],
                "revenue": [],
                "expenses": [],
            }
            for acc in accounts:
                if acc.type == AccountTypeDB.ASSET:
                    chart_of_accounts["assets"].append({"code": acc.code, "name": acc.name, "balance": float(acc.balance)})
                elif acc.type == AccountTypeDB.LIABILITY:
                    chart_of_accounts["liabilities"].append({"code": acc.code, "name": acc.name, "balance": float(acc.balance)})
                elif acc.type == AccountTypeDB.EQUITY:
                    chart_of_accounts["equity"].append({"code": acc.code, "name": acc.name, "balance": float(acc.balance)})
                elif acc.type == AccountTypeDB.REVENUE:
                    chart_of_accounts["revenue"].append({"code": acc.code, "name": acc.name, "balance": float(acc.balance)})
                elif acc.type == AccountTypeDB.EXPENSE:
                    chart_of_accounts["expenses"].append({"code": acc.code, "name": acc.name, "balance": float(acc.balance)})

            return {
                "period": period,
                "total_revenue": float(row.revenue or 0),
                "outstanding": float(row.outstanding or 0),
                "chart_of_accounts": chart_of_accounts,
                "trends": {
                    "revenue_growth": 15.5,
                    "expense_growth": 8.2,
                    "profit_margin": 42.3,
                },
            }

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


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_erp_db_service: Optional[ERPDatabaseService] = None


def get_erp_db_service(hierarchy=None) -> ERPDatabaseService:
    global _erp_db_service
    if _erp_db_service is None:
        _erp_db_service = ERPDatabaseService(hierarchy)
    return _erp_db_service
