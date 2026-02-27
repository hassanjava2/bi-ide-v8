"""
General Ledger - الدفتر العام
GAAP Compliant Accounting System

المميزات:
- Double-entry accounting (المحاسبة المزدوجة)
- Standard Chart of Accounts (شجرة الحسابات القياسية)
- Trial Balance (ميزان المراجعة)
- Financial Statements (القوائم المالية)
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


class AccountType(Enum):
    """أنواع الحسابات المحاسبية"""
    ASSET = "asset"           # أصول
    LIABILITY = "liability"   # خصوم
    EQUITY = "equity"         # حقوق ملكية
    REVENUE = "revenue"       # إيرادات
    EXPENSE = "expense"       # مصروفات


class AccountCategory(Enum):
    """فئات الحسابات الفرعية"""
    # أصول
    CURRENT_ASSET = "current_asset"
    FIXED_ASSET = "fixed_asset"
    INTANGIBLE_ASSET = "intangible_asset"
    # خصوم
    CURRENT_LIABILITY = "current_liability"
    LONG_TERM_LIABILITY = "long_term_liability"
    # حقوق ملكية
    CAPITAL = "capital"
    RETAINED_EARNINGS = "retained_earnings"
    # إيرادات
    OPERATING_REVENUE = "operating_revenue"
    NON_OPERATING_REVENUE = "non_operating_revenue"
    # مصروفات
    OPERATING_EXPENSE = "operating_expense"
    NON_OPERATING_EXPENSE = "non_operating_expense"


@dataclass
class Account:
    """حساب محاسبي"""
    id: str
    code: str                    # رمز الحساب (مثال: 1100)
    name: str                    # اسم الحساب
    name_ar: str                 # الاسم بالعربية
    type: AccountType            # نوع الحساب
    category: AccountCategory    # الفئة
    parent_code: Optional[str]   # الحساب الأب
    balance: Decimal = field(default_factory=lambda: Decimal('0'))
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "name_ar": self.name_ar,
            "type": self.type.value,
            "category": self.category.value,
            "parent_code": self.parent_code,
            "balance": float(self.balance),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Transaction:
    """معاملة محاسبية"""
    id: str
    date: datetime
    reference: str               # رقم المرجع
    description: str             # الوصف
    debit_account: str           # حساب المدين (code)
    credit_account: str          # حساب الدائن (code)
    amount: Decimal
    currency: str = "SAR"
    exchange_rate: Decimal = field(default_factory=lambda: Decimal('1'))
    created_by: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "reference": self.reference,
            "description": self.description,
            "debit_account": self.debit_account,
            "credit_account": self.credit_account,
            "amount": float(self.amount),
            "currency": self.currency,
            "exchange_rate": float(self.exchange_rate),
            "created_by": self.created_by,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class ChartOfAccounts:
    """
    شجرة الحسابات القياسية
    Chart of Accounts compliant مع GAAP
    """
    
    # القياسية Chart of Accounts
    STANDARD_COA = [
        # الأصول Assets (1000-1999)
        ("1000", "Assets", "الأصول", AccountType.ASSET, AccountCategory.CURRENT_ASSET, None),
        ("1100", "Current Assets", "الأصول المتداولة", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1000"),
        ("1110", "Cash", "النقد", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1100"),
        ("1120", "Bank", "البنك", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1100"),
        ("1130", "Accounts Receivable", "الذمم المدينة", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1100"),
        ("1140", "Inventory", "المخزون", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1100"),
        ("1150", "Prepaid Expenses", "مصروفات مدفوعة مقدماً", AccountType.ASSET, AccountCategory.CURRENT_ASSET, "1100"),
        ("1200", "Fixed Assets", "الأصول الثابتة", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1000"),
        ("1210", "Buildings", "المباني", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1200"),
        ("1220", "Equipment", "المعدات", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1200"),
        ("1230", "Vehicles", "المركبات", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1200"),
        ("1240", "Furniture", "الأثاث", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1200"),
        ("1250", "Accumulated Depreciation", "مجمع الإهلاك", AccountType.ASSET, AccountCategory.FIXED_ASSET, "1200"),
        
        # الخصوم Liabilities (2000-2999)
        ("2000", "Liabilities", "الخصوم", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, None),
        ("2100", "Current Liabilities", "الخصوم المتداولة", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, "2000"),
        ("2110", "Accounts Payable", "الذمم الدائنة", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, "2100"),
        ("2120", "Short-term Loans", "قروض قصيرة الأجل", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, "2100"),
        ("2130", "Accrued Expenses", "مصروفات مستحقة", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, "2100"),
        ("2140", "VAT Payable", "ضريبة القيمة المضافة المستحقة", AccountType.LIABILITY, AccountCategory.CURRENT_LIABILITY, "2100"),
        ("2200", "Long-term Liabilities", "الخصوم طويلة الأجل", AccountType.LIABILITY, AccountCategory.LONG_TERM_LIABILITY, "2000"),
        ("2210", "Long-term Loans", "قروض طويلة الأجل", AccountType.LIABILITY, AccountCategory.LONG_TERM_LIABILITY, "2200"),
        
        # حقوق الملكية Equity (3000-3999)
        ("3000", "Equity", "حقوق الملكية", AccountType.EQUITY, AccountCategory.CAPITAL, None),
        ("3100", "Capital", "رأس المال", AccountType.EQUITY, AccountCategory.CAPITAL, "3000"),
        ("3200", "Retained Earnings", "الأرباح المحتجزة", AccountType.EQUITY, AccountCategory.RETAINED_EARNINGS, "3000"),
        ("3300", "Current Year Earnings", "أرباح السنة الحالية", AccountType.EQUITY, AccountCategory.RETAINED_EARNINGS, "3000"),
        
        # الإيرادات Revenue (4000-4999)
        ("4000", "Revenue", "الإيرادات", AccountType.REVENUE, AccountCategory.OPERATING_REVENUE, None),
        ("4100", "Sales Revenue", "إيرادات المبيعات", AccountType.REVENUE, AccountCategory.OPERATING_REVENUE, "4000"),
        ("4200", "Service Revenue", "إيرادات الخدمات", AccountType.REVENUE, AccountCategory.OPERATING_REVENUE, "4000"),
        ("4300", "Other Revenue", "إيرادات أخرى", AccountType.REVENUE, AccountCategory.NON_OPERATING_REVENUE, "4000"),
        
        # المصروفات Expenses (5000-5999)
        ("5000", "Expenses", "المصروفات", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, None),
        ("5100", "Cost of Goods Sold", "تكلفة البضاعة المباعة", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5200", "Salaries & Wages", "الرواتب والأجور", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5300", "Rent Expense", "مصروف الإيجار", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5400", "Utilities", "المرافق والخدمات", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5500", "Depreciation", "الإهلاك", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5600", "Marketing", "التسويق", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5700", "Administrative Expenses", "المصروفات الإدارية", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
        ("5800", "Tax Expense", "مصروف الضرائب", AccountType.EXPENSE, AccountCategory.OPERATING_EXPENSE, "5000"),
    ]
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self._init_standard_coa()
    
    def _init_standard_coa(self):
        """تهيئة شجرة الحسابات القياسية"""
        for code, name, name_ar, acc_type, category, parent in self.STANDARD_COA:
            account = Account(
                id=str(uuid.uuid4()),
                code=code,
                name=name,
                name_ar=name_ar,
                type=acc_type,
                category=category,
                parent_code=parent
            )
            self.accounts[code] = account
    
    def get_account(self, code: str) -> Optional[Account]:
        """الحصول على حساب برمزه"""
        return self.accounts.get(code)
    
    def get_accounts_by_type(self, acc_type: AccountType) -> List[Account]:
        """الحصول على الحسابات حسب النوع"""
        return [acc for acc in self.accounts.values() if acc.type == acc_type]
    
    def create_account(self, code: str, name: str, name_ar: str, 
                       acc_type: AccountType, category: AccountCategory,
                       parent_code: Optional[str] = None) -> Account:
        """إنشاء حساب جديد"""
        if code in self.accounts:
            raise ValueError(f"Account code {code} already exists")
        
        account = Account(
            id=str(uuid.uuid4()),
            code=code,
            name=name,
            name_ar=name_ar,
            type=acc_type,
            category=category,
            parent_code=parent_code
        )
        self.accounts[code] = account
        return account
    
    def get_account_tree(self) -> Dict[str, Any]:
        """الحصول على شجرة الحسابات"""
        tree = {}
        
        for account in self.accounts.values():
            if account.parent_code is None:
                tree[account.code] = {
                    "account": account.to_dict(),
                    "children": self._get_children(account.code)
                }
        
        return tree
    
    def _get_children(self, parent_code: str) -> Dict[str, Any]:
        """الحصول على الحسابات الفرعية"""
        children = {}
        for account in self.accounts.values():
            if account.parent_code == parent_code:
                children[account.code] = {
                    "account": account.to_dict(),
                    "children": self._get_children(account.code)
                }
        return children


class GeneralLedger:
    """
    الدفتر العام
    GAAP Compliant Double-Entry Accounting System
    """
    
    def __init__(self):
        self.coa = ChartOfAccounts()
        self.transactions: List[Transaction] = []
        self.journal_entries: List[Dict] = []
        self._transaction_counter = 0
    
    def create_account(self, name: str, acc_type: AccountType, 
                       code: str, name_ar: str = "",
                       category: AccountCategory = None,
                       parent_code: Optional[str] = None) -> Account:
        """
        إنشاء حساب جديد في شجرة الحسابات
        
        Args:
            name: اسم الحساب بالإنجليزية
            acc_type: نوع الحساب (Asset, Liability, Equity, Revenue, Expense)
            code: رمز الحساب
            name_ar: اسم الحساب بالعربية
            category: الفئة الفرعية
            parent_code: رمز الحساب الأب
        """
        if category is None:
            # Default category based on type
            category_map = {
                AccountType.ASSET: AccountCategory.CURRENT_ASSET,
                AccountType.LIABILITY: AccountCategory.CURRENT_LIABILITY,
                AccountType.EQUITY: AccountCategory.CAPITAL,
                AccountType.REVENUE: AccountCategory.OPERATING_REVENUE,
                AccountType.EXPENSE: AccountCategory.OPERATING_EXPENSE,
            }
            category = category_map.get(acc_type, AccountCategory.CURRENT_ASSET)
        
        return self.coa.create_account(code, name, name_ar or name, 
                                       acc_type, category, parent_code)
    
    def post_transaction(self, debit_account: str, credit_account: str,
                        amount: Decimal, description: str,
                        reference: Optional[str] = None,
                        currency: str = "SAR",
                        created_by: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Transaction:
        """
        تسجيل معاملة محاسبية (Double Entry)
        
        Args:
            debit_account: رمز حساب المدين
            credit_account: رمز حساب الدائن
            amount: المبلغ
            description: وصف المعاملة
            reference: رقم المرجع
            currency: العملة
            created_by: معرف المستخدم المنشئ
            metadata: بيانات إضافية
        
        Returns:
            Transaction: المعاملة المسجلة
        """
        # التحقق من وجود الحسابات
        debit_acc = self.coa.get_account(debit_account)
        credit_acc = self.coa.get_account(credit_account)
        
        if not debit_acc:
            raise ValueError(f"Debit account {debit_account} not found")
        if not credit_acc:
            raise ValueError(f"Credit account {credit_account} not found")
        
        # Generate reference if not provided
        if reference is None:
            self._transaction_counter += 1
            reference = f"JV-{datetime.now().strftime('%Y%m%d')}-{self._transaction_counter:04d}"
        
        transaction = Transaction(
            id=str(uuid.uuid4()),
            date=datetime.now(timezone.utc),
            reference=reference,
            description=description,
            debit_account=debit_account,
            credit_account=credit_account,
            amount=Decimal(str(amount)),
            currency=currency,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self.transactions.append(transaction)
        
        # Update account balances
        debit_acc.balance += Decimal(str(amount))
        credit_acc.balance -= Decimal(str(amount))
        
        return transaction
    
    def get_balance_sheet(self, as_of: Optional[date] = None) -> Dict[str, Any]:
        """
        قائمة المركز المالي (Balance Sheet)
        Assets = Liabilities + Equity
        
        Returns:
            Dict containing balance sheet data
        """
        if as_of is None:
            as_of = date.today()
        
        assets = self.coa.get_accounts_by_type(AccountType.ASSET)
        liabilities = self.coa.get_accounts_by_type(AccountType.LIABILITY)
        equity = self.coa.get_accounts_by_type(AccountType.EQUITY)
        
        total_assets = sum(acc.balance for acc in assets if acc.balance > 0)
        total_liabilities = sum(abs(acc.balance) for acc in liabilities if acc.balance < 0)
        total_equity = sum(acc.balance for acc in equity)
        
        # Calculate current portions
        current_assets = sum(
            acc.balance for acc in assets 
            if acc.category == AccountCategory.CURRENT_ASSET and acc.balance > 0
        )
        current_liabilities = sum(
            abs(acc.balance) for acc in liabilities 
            if acc.category == AccountCategory.CURRENT_LIABILITY and acc.balance < 0
        )
        
        return {
            "as_of": as_of.isoformat(),
            "assets": {
                "current_assets": {
                    "accounts": [acc.to_dict() for acc in assets 
                                if acc.category == AccountCategory.CURRENT_ASSET],
                    "total": float(current_assets)
                },
                "fixed_assets": {
                    "accounts": [acc.to_dict() for acc in assets 
                                if acc.category == AccountCategory.FIXED_ASSET],
                    "total": float(total_assets - current_assets)
                },
                "total": float(total_assets)
            },
            "liabilities": {
                "current_liabilities": {
                    "accounts": [acc.to_dict() for acc in liabilities 
                                if acc.category == AccountCategory.CURRENT_LIABILITY],
                    "total": float(current_liabilities)
                },
                "long_term_liabilities": {
                    "accounts": [acc.to_dict() for acc in liabilities 
                                if acc.category == AccountCategory.LONG_TERM_LIABILITY],
                    "total": float(total_liabilities - current_liabilities)
                },
                "total": float(total_liabilities)
            },
            "equity": {
                "accounts": [acc.to_dict() for acc in equity],
                "total": float(total_equity)
            },
            "total_liabilities_and_equity": float(total_liabilities + total_equity),
            "balanced": abs(total_assets - (total_liabilities + total_equity)) < Decimal('0.01')
        }
    
    def get_income_statement(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        قائمة الدخل (Income Statement / Profit & Loss)
        Revenue - Expenses = Net Income
        
        Args:
            start_date: تاريخ البداية (default: بداية الشهر الحالي)
            end_date: تاريخ النهاية (default: اليوم)
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date.replace(day=1)
        
        revenue = self.coa.get_accounts_by_type(AccountType.REVENUE)
        expenses = self.coa.get_accounts_by_type(AccountType.EXPENSE)
        
        total_revenue = sum(acc.balance for acc in revenue if acc.balance > 0)
        total_expenses = sum(abs(acc.balance) for acc in expenses if acc.balance < 0)
        
        gross_profit = total_revenue - sum(
            abs(acc.balance) for acc in expenses 
            if acc.code == "5100"  # COGS
        )
        
        operating_expenses = sum(
            abs(acc.balance) for acc in expenses 
            if acc.category == AccountCategory.OPERATING_EXPENSE and acc.code != "5100"
        )
        
        operating_income = gross_profit - operating_expenses
        net_income = total_revenue - total_expenses
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "revenue": {
                "accounts": [acc.to_dict() for acc in revenue],
                "total": float(total_revenue)
            },
            "cost_of_goods_sold": {
                "accounts": [acc.to_dict() for acc in expenses if acc.code == "5100"],
                "total": float(sum(abs(acc.balance) for acc in expenses if acc.code == "5100"))
            },
            "gross_profit": float(gross_profit),
            "operating_expenses": {
                "accounts": [acc.to_dict() for acc in expenses 
                            if acc.category == AccountCategory.OPERATING_EXPENSE and acc.code != "5100"],
                "total": float(operating_expenses)
            },
            "operating_income": float(operating_income),
            "non_operating_items": {
                "accounts": [acc.to_dict() for acc in expenses 
                            if acc.category == AccountCategory.NON_OPERATING_EXPENSE]
            },
            "net_income": float(net_income),
            "profit_margin": float(net_income / total_revenue * 100) if total_revenue > 0 else 0
        }
    
    def get_trial_balance(self) -> Dict[str, Any]:
        """
        ميزان المراجعة (Trial Balance)
        التحقق من توازن المعاملات
        """
        total_debits = Decimal('0')
        total_credits = Decimal('0')
        
        accounts_list = []
        for account in self.coa.accounts.values():
            balance = account.balance
            if balance > 0:
                debit = float(balance)
                credit = 0
                total_debits += balance
            else:
                debit = 0
                credit = float(abs(balance))
                total_credits += abs(balance)
            
            accounts_list.append({
                "code": account.code,
                "name": account.name,
                "name_ar": account.name_ar,
                "debit": debit,
                "credit": credit
            })
        
        return {
            "accounts": accounts_list,
            "total_debits": float(total_debits),
            "total_credits": float(total_credits),
            "balanced": abs(total_debits - total_credits) < Decimal('0.01')
        }
    
    def get_account_transactions(self, account_code: str) -> List[Transaction]:
        """الحصول على معاملات حساب معين"""
        return [
            t for t in self.transactions 
            if t.debit_account == account_code or t.credit_account == account_code
        ]
    
    def get_account_balance(self, account_code: str) -> Decimal:
        """الحصول على رصيد حساب معين"""
        account = self.coa.get_account(account_code)
        return account.balance if account else Decimal('0')
