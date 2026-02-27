"""
Accounting Module - المحاسبة
Double-entry bookkeeping system
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
from sqlalchemy import Column, String, Numeric, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship


class AccountType(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class AccountCategory(str, Enum):
    CURRENT = "current"
    FIXED = "fixed"
    INTANGIBLE = "intangible"


class Account(Base):
    """Chart of accounts / شجرة الحسابات"""
    __tablename__ = "erp_accounts"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    name_ar = Column(String, nullable=True)
    type = Column(SQLEnum(AccountType), nullable=False)
    category = Column(SQLEnum(AccountCategory), nullable=True)
    parent_id = Column(String, ForeignKey("erp_accounts.id"), nullable=True)
    balance = Column(Numeric(15, 2), default=Decimal("0.00"))
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    parent = relationship("Account", remote_side=[id])
    children = relationship("Account", back_populates="parent")
    debit_transactions = relationship("Transaction", foreign_keys="Transaction.debit_account_id")
    credit_transactions = relationship("Transaction", foreign_keys="Transaction.credit_account_id")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "name_ar": self.name_ar,
            "type": self.type.value if self.type else None,
            "category": self.category.value if self.category else None,
            "balance": float(self.balance) if self.balance else 0.0,
            "is_active": self.is_active == "true",
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Transaction(Base):
    """Double-entry transaction / القيد المزدوج"""
    __tablename__ = "erp_transactions"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    debit_account_id = Column(String, ForeignKey("erp_accounts.id"), nullable=False)
    credit_account_id = Column(String, ForeignKey("erp_accounts.id"), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    description = Column(Text)
    reference = Column(String, index=True)
    created_by = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    debit_account = relationship("Account", foreign_keys=[debit_account_id])
    credit_account = relationship("Account", foreign_keys=[credit_account_id])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "debit_account_id": self.debit_account_id,
            "credit_account_id": self.credit_account_id,
            "amount": float(self.amount) if self.amount else 0.0,
            "description": self.description,
            "reference": self.reference,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


async def create_account(
    session: AsyncSession,
    code: str,
    name: str,
    type: str,
    name_ar: Optional[str] = None,
    category: Optional[str] = None,
    parent_id: Optional[str] = None
) -> Account:
    """Create a new account / إنشاء حساب جديد"""
    account = Account(
        code=code,
        name=name,
        name_ar=name_ar,
        type=AccountType(type),
        category=AccountCategory(category) if category else None,
        parent_id=parent_id,
        balance=Decimal("0.00")
    )
    session.add(account)
    await session.flush()
    return account


async def get_account(session: AsyncSession, account_id: str) -> Optional[Account]:
    """Get account by ID"""
    result = await session.execute(
        select(Account).where(Account.id == account_id)
    )
    return result.scalar_one_or_none()


async def get_account_by_code(session: AsyncSession, code: str) -> Optional[Account]:
    """Get account by code"""
    result = await session.execute(
        select(Account).where(Account.code == code)
    )
    return result.scalar_one_or_none()


async def post_transaction(
    session: AsyncSession,
    debit_account_id: str,
    credit_account_id: str,
    amount: float,
    description: str,
    reference: Optional[str] = None,
    transaction_date: Optional[datetime] = None
) -> Transaction:
    """
    Post a double-entry transaction / تسجيل قيد مزدوج
    
    Every transaction affects at least two accounts (debit and credit)
    Total debits must equal total credits
    """
    # Validate accounts exist
    debit_account = await get_account(session, debit_account_id)
    credit_account = await get_account(session, credit_account_id)
    
    if not debit_account:
        raise ValueError(f"Debit account {debit_account_id} not found")
    if not credit_account:
        raise ValueError(f"Credit account {credit_account_id} not found")
    
    decimal_amount = Decimal(str(amount))
    
    # Create transaction
    transaction = Transaction(
        transaction_date=transaction_date or datetime.now(timezone.utc),
        debit_account_id=debit_account_id,
        credit_account_id=credit_account_id,
        amount=decimal_amount,
        description=description,
        reference=reference
    )
    session.add(transaction)
    
    # Update account balances
    # Assets: Debit increases, Credit decreases
    # Liabilities/Equity/Revenue: Credit increases, Debit decreases
    # Expenses: Debit increases, Credit decreases
    
    if debit_account.type in [AccountType.ASSET, AccountType.EXPENSE]:
        debit_account.balance += decimal_amount
    else:
        debit_account.balance -= decimal_amount
    
    if credit_account.type in [AccountType.ASSET, AccountType.EXPENSE]:
        credit_account.balance -= decimal_amount
    else:
        credit_account.balance += decimal_amount
    
    await session.flush()
    return transaction


async def get_trial_balance(
    session: AsyncSession,
    as_of_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Generate trial balance report / ميزان المراجعة
    
    Lists all accounts with their balances.
    Total debits should equal total credits.
    """
    query = select(Account).where(Account.is_active == "true")
    
    result = await session.execute(query)
    accounts = result.scalars().all()
    
    trial_balance = []
    for account in accounts:
        balance = account.balance or Decimal("0.00")
        
        # Determine debit/credit presentation
        if account.type in [AccountType.ASSET, AccountType.EXPENSE]:
            if balance >= 0:
                debit = float(balance)
                credit = 0.0
            else:
                debit = 0.0
                credit = float(abs(balance))
        else:
            if balance >= 0:
                debit = 0.0
                credit = float(balance)
            else:
                debit = float(abs(balance))
                credit = 0.0
        
        trial_balance.append({
            "account_id": account.id,
            "code": account.code,
            "name": account.name,
            "type": account.type.value,
            "debit": debit,
            "credit": credit
        })
    
    return trial_balance


async def get_balance_sheet(
    session: AsyncSession,
    as_of_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Generate balance sheet / الميزانية العمومية
    
    Assets = Liabilities + Equity
    """
    # Assets
    assets_result = await session.execute(
        select(func.sum(Account.balance)).where(
            and_(Account.type == AccountType.ASSET, Account.is_active == "true")
        )
    )
    total_assets = assets_result.scalar() or Decimal("0.00")
    
    # Liabilities
    liabilities_result = await session.execute(
        select(func.sum(Account.balance)).where(
            and_(Account.type == AccountType.LIABILITY, Account.is_active == "true")
        )
    )
    total_liabilities = liabilities_result.scalar() or Decimal("0.00")
    
    # Equity
    equity_result = await session.execute(
        select(func.sum(Account.balance)).where(
            and_(Account.type == AccountType.EQUITY, Account.is_active == "true")
        )
    )
    total_equity = equity_result.scalar() or Decimal("0.00")
    
    return {
        "as_of_date": (as_of_date or datetime.now(timezone.utc)).isoformat(),
        "assets": float(total_assets),
        "liabilities": float(total_liabilities),
        "equity": float(total_equity),
        "balance_check": float(total_assets) == float(total_liabilities + total_equity)
    }


async def get_income_statement(
    session: AsyncSession,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Generate income statement / قائمة الدخل
    
    Revenue - Expenses = Net Income
    """
    # Revenue
    revenue_result = await session.execute(
        select(func.sum(Transaction.amount)).where(
            and_(
                Transaction.credit_account_id.in_(
                    select(Account.id).where(Account.type == AccountType.REVENUE)
                ),
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )
        )
    )
    total_revenue = revenue_result.scalar() or Decimal("0.00")
    
    # Expenses
    expenses_result = await session.execute(
        select(func.sum(Transaction.amount)).where(
            and_(
                Transaction.debit_account_id.in_(
                    select(Account.id).where(Account.type == AccountType.EXPENSE)
                ),
                Transaction.transaction_date >= start_date,
                Transaction.transaction_date <= end_date
            )
        )
    )
    total_expenses = expenses_result.scalar() or Decimal("0.00")
    
    net_income = total_revenue - total_expenses
    
    return {
        "period": f"{start_date.date()} to {end_date.date()}",
        "revenue": float(total_revenue),
        "expenses": float(total_expenses),
        "net_income": float(net_income)
    }
