"""
HR Module - إدارة الموارد البشرية
Human resources and payroll management
"""
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Base
from sqlalchemy import Column, String, Numeric, DateTime, Text, ForeignKey, Enum as SQLEnum, Date, Integer
from sqlalchemy.orm import relationship


class EmployeeStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"


class PayrollStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    PAID = "paid"
    CANCELLED = "cancelled"


class Employee(Base):
    """Employee / الموظف"""
    __tablename__ = "erp_employees"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_number = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(200), nullable=False)
    last_name = Column(String(200), nullable=False)
    first_name_ar = Column(String(200), nullable=True)
    last_name_ar = Column(String(200), nullable=True)
    email = Column(String(300), unique=True, nullable=True)
    phone = Column(String(50), default="")
    department = Column(String(200), index=True)
    position = Column(String(200), nullable=False)
    hire_date = Column(Date, nullable=False)
    termination_date = Column(Date, nullable=True)
    base_salary = Column(Numeric(15, 2), default=Decimal("0.00"))
    currency = Column(String(3), default="SAR")
    status = Column(SQLEnum(EmployeeStatus), default=EmployeeStatus.ACTIVE)
    bank_account = Column(String(100), nullable=True)
    bank_name = Column(String(200), nullable=True)
    id_number = Column(String(50), unique=True, nullable=True)  # National ID
    manager_id = Column(String, ForeignKey("erp_employees.id"), nullable=True)
    notes = Column(Text, nullable=True)
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    manager = relationship("Employee", remote_side=[id])
    subordinates = relationship("Employee", back_populates="manager")
    payroll_records = relationship("PayrollRecord", back_populates="employee")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_number": self.employee_number,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "first_name_ar": self.first_name_ar,
            "last_name_ar": self.last_name_ar,
            "email": self.email,
            "phone": self.phone,
            "department": self.department,
            "position": self.position,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "termination_date": self.termination_date.isoformat() if self.termination_date else None,
            "base_salary": float(self.base_salary) if self.base_salary else 0.0,
            "currency": self.currency,
            "status": self.status.value if self.status else None,
            "bank_account": self.bank_account,
            "bank_name": self.bank_name,
            "id_number": self.id_number,
            "manager_id": self.manager_id,
            "notes": self.notes,
            "is_active": self.is_active == "true",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class PayrollRecord(Base):
    """Payroll record / سجل الرواتب"""
    __tablename__ = "erp_payroll_records"
    __table_args__ = {"extend_existing": True}
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String, ForeignKey("erp_employees.id"), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_month = Column(Integer, nullable=False)
    base_salary = Column(Numeric(15, 2), default=Decimal("0.00"))
    allowances = Column(Numeric(15, 2), default=Decimal("0.00"))  # Housing, transport, etc.
    overtime = Column(Numeric(15, 2), default=Decimal("0.00"))
    bonuses = Column(Numeric(15, 2), default=Decimal("0.00"))
    deductions = Column(Numeric(15, 2), default=Decimal("0.00"))  # Absences, penalties
    tax = Column(Numeric(15, 2), default=Decimal("0.00"))
    social_insurance = Column(Numeric(15, 2), default=Decimal("0.00"))
    gross_salary = Column(Numeric(15, 2), default=Decimal("0.00"))
    net_salary = Column(Numeric(15, 2), default=Decimal("0.00"))
    status = Column(SQLEnum(PayrollStatus), default=PayrollStatus.DRAFT)
    paid_at = Column(DateTime, nullable=True)
    paid_by = Column(String, nullable=True)
    payment_reference = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    employee = relationship("Employee", back_populates="payroll_records")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "period_year": self.period_year,
            "period_month": self.period_month,
            "base_salary": float(self.base_salary) if self.base_salary else 0.0,
            "allowances": float(self.allowances) if self.allowances else 0.0,
            "overtime": float(self.overtime) if self.overtime else 0.0,
            "bonuses": float(self.bonuses) if self.bonuses else 0.0,
            "deductions": float(self.deductions) if self.deductions else 0.0,
            "tax": float(self.tax) if self.tax else 0.0,
            "social_insurance": float(self.social_insurance) if self.social_insurance else 0.0,
            "gross_salary": float(self.gross_salary) if self.gross_salary else 0.0,
            "net_salary": float(self.net_salary) if self.net_salary else 0.0,
            "status": self.status.value if self.status else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "paid_by": self.paid_by,
            "payment_reference": self.payment_reference,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


async def create_employee(
    session: AsyncSession,
    employee_number: str,
    first_name: str,
    last_name: str,
    position: str,
    department: str,
    hire_date: date,
    base_salary: float,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    first_name_ar: Optional[str] = None,
    last_name_ar: Optional[str] = None,
    manager_id: Optional[str] = None,
    bank_account: Optional[str] = None,
    bank_name: Optional[str] = None,
    id_number: Optional[str] = None,
    currency: str = "SAR"
) -> Employee:
    """Create a new employee / إنشاء موظف جديد"""
    employee = Employee(
        employee_number=employee_number,
        first_name=first_name,
        last_name=last_name,
        first_name_ar=first_name_ar,
        last_name_ar=last_name_ar,
        email=email,
        phone=phone or "",
        department=department,
        position=position,
        hire_date=hire_date,
        base_salary=Decimal(str(base_salary)),
        currency=currency,
        manager_id=manager_id,
        bank_account=bank_account,
        bank_name=bank_name,
        id_number=id_number,
        status=EmployeeStatus.ACTIVE
    )
    session.add(employee)
    await session.flush()
    return employee


async def get_employee(session: AsyncSession, employee_id: str) -> Optional[Employee]:
    """Get employee by ID"""
    result = await session.execute(
        select(Employee).where(Employee.id == employee_id)
    )
    return result.scalar_one_or_none()


async def get_employee_by_number(session: AsyncSession, employee_number: str) -> Optional[Employee]:
    """Get employee by employee number"""
    result = await session.execute(
        select(Employee).where(Employee.employee_number == employee_number)
    )
    return result.scalar_one_or_none()


async def update_employee(
    session: AsyncSession,
    employee_id: str,
    **kwargs
) -> Optional[Employee]:
    """Update employee fields"""
    employee = await get_employee(session, employee_id)
    if not employee:
        return None
    
    # Convert Decimal fields
    if 'base_salary' in kwargs:
        kwargs['base_salary'] = Decimal(str(kwargs['base_salary']))
    
    # Convert Enum fields
    if 'status' in kwargs and isinstance(kwargs['status'], str):
        kwargs['status'] = EmployeeStatus(kwargs['status'])
    
    for key, value in kwargs.items():
        if hasattr(employee, key):
            setattr(employee, key, value)
    
    employee.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return employee


async def process_payroll(
    session: AsyncSession,
    employee_id: str,
    period_year: int,
    period_month: int,
    allowances: float = 0.0,
    overtime: float = 0.0,
    bonuses: float = 0.0,
    deductions: float = 0.0,
    tax: float = 0.0,
    social_insurance: float = 0.0,
    notes: Optional[str] = None
) -> PayrollRecord:
    """
    Process payroll for an employee / معالجة الرواتب
    
    Calculates gross and net salary based on components.
    """
    employee = await get_employee(session, employee_id)
    if not employee:
        raise ValueError(f"Employee {employee_id} not found")
    
    base_salary = employee.base_salary or Decimal("0.00")
    
    # Calculate totals
    decimal_allowances = Decimal(str(allowances))
    decimal_overtime = Decimal(str(overtime))
    decimal_bonuses = Decimal(str(bonuses))
    decimal_deductions = Decimal(str(deductions))
    decimal_tax = Decimal(str(tax))
    decimal_social = Decimal(str(social_insurance))
    
    gross_salary = base_salary + decimal_allowances + decimal_overtime + decimal_bonuses
    total_deductions = decimal_deductions + decimal_tax + decimal_social
    net_salary = gross_salary - total_deductions
    
    # Check for existing record
    result = await session.execute(
        select(PayrollRecord).where(
            and_(
                PayrollRecord.employee_id == employee_id,
                PayrollRecord.period_year == period_year,
                PayrollRecord.period_month == period_month
            )
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        # Update existing
        existing.base_salary = base_salary
        existing.allowances = decimal_allowances
        existing.overtime = decimal_overtime
        existing.bonuses = decimal_bonuses
        existing.deductions = decimal_deductions
        existing.tax = decimal_tax
        existing.social_insurance = decimal_social
        existing.gross_salary = gross_salary
        existing.net_salary = net_salary
        existing.notes = notes
        existing.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return existing
    else:
        # Create new record
        record = PayrollRecord(
            employee_id=employee_id,
            period_year=period_year,
            period_month=period_month,
            base_salary=base_salary,
            allowances=decimal_allowances,
            overtime=decimal_overtime,
            bonuses=decimal_bonuses,
            deductions=decimal_deductions,
            tax=decimal_tax,
            social_insurance=decimal_social,
            gross_salary=gross_salary,
            net_salary=net_salary,
            status=PayrollStatus.DRAFT,
            notes=notes
        )
        session.add(record)
        await session.flush()
        return record


async def approve_payroll(
    session: AsyncSession,
    payroll_id: str,
    approved_by: str
) -> Optional[PayrollRecord]:
    """Approve payroll record / اعتماد الراتب"""
    result = await session.execute(
        select(PayrollRecord).where(PayrollRecord.id == payroll_id)
    )
    record = result.scalar_one_or_none()
    
    if not record or record.status != PayrollStatus.DRAFT:
        return None
    
    record.status = PayrollStatus.APPROVED
    record.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return record


async def mark_payroll_paid(
    session: AsyncSession,
    payroll_id: str,
    paid_by: str,
    payment_reference: Optional[str] = None
) -> Optional[PayrollRecord]:
    """Mark payroll as paid / تسجيل دفع الراتب"""
    result = await session.execute(
        select(PayrollRecord).where(PayrollRecord.id == payroll_id)
    )
    record = result.scalar_one_or_none()
    
    if not record or record.status not in [PayrollStatus.DRAFT, PayrollStatus.APPROVED]:
        return None
    
    record.status = PayrollStatus.PAID
    record.paid_at = datetime.now(timezone.utc)
    record.paid_by = paid_by
    record.payment_reference = payment_reference
    record.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return record


async def get_department_summary(session: AsyncSession) -> List[Dict[str, Any]]:
    """Get employee and salary summary by department"""
    result = await session.execute(
        select(
            Employee.department,
            func.count(Employee.id).label("employee_count"),
            func.sum(Employee.base_salary).label("total_salary")
        )
        .where(Employee.is_active == "true")
        .group_by(Employee.department)
        .order_by(Employee.department)
    )
    
    return [
        {
            "department": row.department,
            "employee_count": row.employee_count,
            "total_monthly_salary": float(row.total_salary or 0)
        }
        for row in result.all()
    ]


async def get_payroll_summary(
    session: AsyncSession,
    year: int,
    month: int
) -> Dict[str, Any]:
    """Get payroll summary for a period"""
    result = await session.execute(
        select(
            func.count(PayrollRecord.id).label("employee_count"),
            func.sum(PayrollRecord.gross_salary).label("total_gross"),
            func.sum(PayrollRecord.net_salary).label("total_net"),
            func.sum(PayrollRecord.tax).label("total_tax"),
            func.sum(PayrollRecord.social_insurance).label("total_insurance")
        )
        .where(
            and_(
                PayrollRecord.period_year == year,
                PayrollRecord.period_month == month
            )
        )
    )
    row = result.one()
    
    return {
        "period": f"{year}-{month:02d}",
        "employee_count": row.employee_count or 0,
        "total_gross": float(row.total_gross or 0),
        "total_net": float(row.total_net or 0),
        "total_tax": float(row.total_tax or 0),
        "total_social_insurance": float(row.total_insurance or 0)
    }
