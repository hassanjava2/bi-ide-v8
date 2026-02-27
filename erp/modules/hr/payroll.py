"""
Payroll Processing - معالجة الرواتب

المميزات:
- حساب الرواتب الشهرية
- حساب الاستقطاعات (ضرائب، تأمينات، إلخ)
- حساب البدلات والعلاوات
- توليد كشوف الرواتب
- دعم العملات المتعددة
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class DeductionType(Enum):
    """أنواع الاستقطاعات"""
    TAX = "tax"                       # ضريبة
    SOCIAL_INSURANCE = "social_insurance"  # التأمينات الاجتماعية
    HEALTH_INSURANCE = "health_insurance"  # التأمين الصحي
    PENSION = "pension"               # التقاعد
    LOAN = "loan"                     # قرض
    SALARY_ADVANCE = "salary_advance"  # سلفة راتب
    OTHER = "other"                   # أخرى


class AllowanceType(Enum):
    """أنواع البدلات"""
    HOUSING = "housing"               # بدل سكن
    TRANSPORTATION = "transportation"  # بدل نقل
    MEALS = "meals"                   # بدل طعام
    PHONE = "phone"                   # بدل اتصالات
    OVERTIME = "overtime"             # ساعات إضافية
    BONUS = "bonus"                   # مكافأة
    COMMISSION = "commission"         # عمولة
    OTHER = "other"                   # أخرى


@dataclass
class Deduction:
    """استقطاع"""
    type: DeductionType
    description: str
    amount: Decimal
    is_percentage: bool = False
    percentage: Decimal = field(default_factory=lambda: Decimal('0'))
    reference: str = ""               # رقم مرجعي (مثل رقم البوليصة)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "amount": float(self.amount),
            "is_percentage": self.is_percentage,
            "percentage": float(self.percentage) if self.is_percentage else None,
            "reference": self.reference
        }


@dataclass
class Allowance:
    """بدل/علاوة"""
    type: AllowanceType
    description: str
    amount: Decimal
    is_taxable: bool = True           # خاضع للضريبة؟
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "amount": float(self.amount),
            "is_taxable": self.is_taxable
        }


@dataclass
class SalaryStructure:
    """هيكل الراتب"""
    basic_salary: Decimal = field(default_factory=lambda: Decimal('0'))
    housing_allowance: Decimal = field(default_factory=lambda: Decimal('0'))
    transportation_allowance: Decimal = field(default_factory=lambda: Decimal('0'))
    other_allowances: Decimal = field(default_factory=lambda: Decimal('0'))
    
    @property
    def gross_salary(self) -> Decimal:
        """إجمالي الراتب"""
        return (self.basic_salary + self.housing_allowance + 
                self.transportation_allowance + self.other_allowances)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "basic_salary": float(self.basic_salary),
            "housing_allowance": float(self.housing_allowance),
            "transportation_allowance": float(self.transportation_allowance),
            "other_allowances": float(self.other_allowances),
            "gross_salary": float(self.gross_salary)
        }


@dataclass
class Payslip:
    """كشف راتب"""
    id: str
    employee_id: str
    employee_name: str
    month: int                        # الشهر (1-12)
    year: int                         # السنة
    
    # Earnings
    basic_salary: Decimal = field(default_factory=lambda: Decimal('0'))
    allowances: List[Allowance] = field(default_factory=list)
    overtime_hours: Decimal = field(default_factory=lambda: Decimal('0'))
    overtime_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    bonuses: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # Deductions
    deductions: List[Deduction] = field(default_factory=list)
    
    # Currency
    currency: str = "SAR"
    exchange_rate: Decimal = field(default_factory=lambda: Decimal('1'))
    
    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    notes: str = ""
    
    @property
    def total_allowances(self) -> Decimal:
        """إجمالي البدلات"""
        return sum(allowance.amount for allowance in self.allowances)
    
    @property
    def total_earnings(self) -> Decimal:
        """إجمالي المستحقات"""
        return (self.basic_salary + self.total_allowances + 
                self.overtime_amount + self.bonuses)
    
    @property
    def total_deductions(self) -> Decimal:
        """إجمالي الاستقطاعات"""
        return sum(deduction.amount for deduction in self.deductions)
    
    @property
    def taxable_income(self) -> Decimal:
        """الدخل الخاضع للضريبة"""
        taxable = self.basic_salary
        for allowance in self.allowances:
            if allowance.is_taxable:
                taxable += allowance.amount
        taxable += self.overtime_amount
        return taxable
    
    @property
    def net_salary(self) -> Decimal:
        """صافي الراتب"""
        return self.total_earnings - self.total_deductions
    
    @property
    def net_salary_local(self) -> Decimal:
        """صافي الراتب بالعملة المحلية"""
        return self.net_salary * self.exchange_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "period": f"{self.month:02d}/{self.year}",
            "earnings": {
                "basic_salary": float(self.basic_salary),
                "allowances": [a.to_dict() for a in self.allowances],
                "total_allowances": float(self.total_allowances),
                "overtime_hours": float(self.overtime_hours),
                "overtime_amount": float(self.overtime_amount),
                "bonuses": float(self.bonuses),
                "total_earnings": float(self.total_earnings)
            },
            "deductions": [d.to_dict() for d in self.deductions],
            "total_deductions": float(self.total_deductions),
            "taxable_income": float(self.taxable_income),
            "net_salary": float(self.net_salary),
            "currency": self.currency,
            "exchange_rate": float(self.exchange_rate),
            "generated_at": self.generated_at.isoformat(),
            "approved_by": self.approved_by,
            "notes": self.notes
        }


class PayrollProcessor:
    """
    معالج الرواتب
    
    يدعم:
    - حساب الضرائب حسب قوانين المملكة العربية السعودية
    - التأمينات الاجتماعية
    - العملات المتعددة
    """
    
    # Saudi Tax Brackets (2024)
    TAX_BRACKETS = [
        (0, 0),
        (6000, 5),
        (10000, 10),
        (15000, 15),
        (30000, 20),
        (float('inf'), 22.5)
    ]
    
    # Social Insurance Rates (Saudi Arabia)
    GOSI_EMPLOYEE_RATE = Decimal('0.10')      # 10% للموظف السعودي
    GOSI_EMPLOYER_RATE = Decimal('0.12')      # 12% للمنشأة
    GOSI_NON_SAUDI_RATE = Decimal('0.02')     # 2% للعمالة غير السعودية
    
    def __init__(self):
        self.payslips: Dict[str, Payslip] = {}
        self.exchange_rates: Dict[str, Decimal] = {
            "SAR": Decimal('1'),
            "USD": Decimal('3.75'),
            "EUR": Decimal('4.10'),
            "GBP": Decimal('4.75'),
            "AED": Decimal('1.02'),
        }
    
    def calculate_tax(self, taxable_income: Decimal, 
                     is_saudi: bool = True) -> Decimal:
        """
        حساب الضريبة
        
        للموظفين السعوديين: معفيون من ضريبة الدخل
        للغير سعوديين: تطبق شرائح الضريبة
        """
        if is_saudi:
            return Decimal('0')
        
        # Apply tax brackets
        annual_income = taxable_income * Decimal('12')
        tax = Decimal('0')
        
        # Simple progressive tax calculation
        if annual_income <= 6000 * 12:
            tax_rate = Decimal('0')
        elif annual_income <= 10000 * 12:
            tax_rate = Decimal('0.05')
        elif annual_income <= 15000 * 12:
            tax_rate = Decimal('0.10')
        elif annual_income <= 30000 * 12:
            tax_rate = Decimal('0.15')
        else:
            tax_rate = Decimal('0.20')
        
        monthly_tax = (annual_income * tax_rate) / Decimal('12')
        return monthly_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def calculate_gosi(self, basic_salary: Decimal,
                      is_saudi: bool = True) -> Dict[str, Decimal]:
        """
        حساب التأمينات الاجتماعية (GOSI)
        
        Returns:
            Dict with employee_contribution and employer_contribution
        """
        if not is_saudi:
            # Non-Saudi: 2% occupational hazard only
            occupational_hazard = basic_salary * Decimal('0.02')
            return {
                "employee_contribution": Decimal('0'),
                "employer_contribution": occupational_hazard,
                "occupational_hazard": occupational_hazard,
                "pension": Decimal('0'),
                "unemployment": Decimal('0')
            }
        
        # Saudi employees
        # Pension: 9% employee + 9% employer
        pension_employee = basic_salary * Decimal('0.09')
        pension_employer = basic_salary * Decimal('0.09')
        
        # Unemployment (SANED): 1% employee + 1% employer
        unemployment_employee = basic_salary * Decimal('0.01')
        unemployment_employer = basic_salary * Decimal('0.01')
        
        # Occupational Hazard: 2% employer only
        occupational_hazard = basic_salary * Decimal('0.02')
        
        return {
            "employee_contribution": pension_employee + unemployment_employee,
            "employer_contribution": pension_employer + unemployment_employer + occupational_hazard,
            "pension": pension_employee + pension_employer,
            "unemployment": unemployment_employee + unemployment_employer,
            "occupational_hazard": occupational_hazard
        }
    
    def calculate_salary(self, employee_id: str, employee_name: str,
                        salary_structure: SalaryStructure,
                        month: int, year: int,
                        is_saudi: bool = True,
                        overtime_hours: Decimal = Decimal('0'),
                        hourly_rate: Decimal = None,
                        additional_deductions: List[Deduction] = None,
                        additional_allowances: List[Allowance] = None,
                        currency: str = "SAR",
                        notes: str = "") -> Payslip:
        """
        حساب راتب موظف
        
        Args:
            employee_id: معرف الموظف
            employee_name: اسم الموظف
            salary_structure: هيكل الراتب
            month: الشهر (1-12)
            year: السنة
            is_saudi: هل الموظف سعودي؟
            overtime_hours: ساعات العمل الإضافي
            hourly_rate: سعر الساعة (default: basic_salary / 240)
            additional_deductions: استقطاعات إضافية
            additional_allowances: بدلات إضافية
            currency: العملة
            notes: ملاحظات
        """
        # Calculate overtime
        if hourly_rate is None:
            hourly_rate = salary_structure.basic_salary / Decimal('240')  # 30 days * 8 hours
        
        overtime_amount = overtime_hours * hourly_rate * Decimal('1.5')  # 1.5x for overtime
        
        # Calculate default allowances
        allowances = [
            Allowance(AllowanceType.HOUSING, "بدل سكن", salary_structure.housing_allowance, False),
            Allowance(AllowanceType.TRANSPORTATION, "بدل نقل", salary_structure.transportation_allowance, False),
        ]
        
        if salary_structure.other_allowances > 0:
            allowances.append(Allowance(AllowanceType.OTHER, "بدلات أخرى", salary_structure.other_allowances, True))
        
        # Add additional allowances
        if additional_allowances:
            allowances.extend(additional_allowances)
        
        # Calculate deductions
        deductions = []
        
        # Tax
        taxable_income = (salary_structure.basic_salary + 
                         sum(a.amount for a in allowances if a.is_taxable) +
                         overtime_amount)
        tax_amount = self.calculate_tax(taxable_income, is_saudi)
        if tax_amount > 0:
            deductions.append(Deduction(
                DeductionType.TAX,
                "ضريبة الدخل",
                tax_amount
            ))
        
        # GOSI
        gosi = self.calculate_gosi(salary_structure.basic_salary, is_saudi)
        if gosi["employee_contribution"] > 0:
            deductions.append(Deduction(
                DeductionType.SOCIAL_INSURANCE,
                "التأمينات الاجتماعية",
                gosi["employee_contribution"]
            ))
        
        # Health insurance (example: 500 SAR fixed)
        health_insurance = Decimal('500') if is_saudi else Decimal('1500')
        deductions.append(Deduction(
            DeductionType.HEALTH_INSURANCE,
            "التأمين الصحي",
            health_insurance
        ))
        
        # Add additional deductions
        if additional_deductions:
            deductions.extend(additional_deductions)
        
        # Get exchange rate
        exchange_rate = self.exchange_rates.get(currency, Decimal('1'))
        
        # Create payslip
        payslip = Payslip(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            employee_name=employee_name,
            month=month,
            year=year,
            basic_salary=salary_structure.basic_salary,
            allowances=allowances,
            overtime_hours=overtime_hours,
            overtime_amount=overtime_amount.quantize(Decimal('0.01')),
            deductions=deductions,
            currency=currency,
            exchange_rate=exchange_rate,
            notes=notes
        )
        
        self.payslips[payslip.id] = payslip
        return payslip
    
    def add_bonus(self, payslip_id: str, amount: Decimal, 
                 description: str = "مكافأة"):
        """إضافة مكافأة لكشف الراتب"""
        payslip = self.payslips.get(payslip_id)
        if payslip:
            payslip.bonuses += Decimal(str(amount))
            payslip.allowances.append(
                Allowance(AllowanceType.BONUS, description, Decimal(str(amount)), True)
            )
    
    def add_deduction(self, payslip_id: str, deduction_type: DeductionType,
                     amount: Decimal, description: str, 
                     reference: str = ""):
        """إضافة استقطاع لكشف الراتب"""
        payslip = self.payslips.get(payslip_id)
        if payslip:
            payslip.deductions.append(Deduction(
                deduction_type, description, Decimal(str(amount)),
                reference=reference
            ))
    
    def approve_payslip(self, payslip_id: str, approved_by: str) -> Payslip:
        """اعتماد كشف الراتب"""
        payslip = self.payslips.get(payslip_id)
        if not payslip:
            raise ValueError(f"Payslip {payslip_id} not found")
        
        payslip.approved_by = approved_by
        payslip.approved_at = datetime.now(timezone.utc)
        return payslip
    
    def get_payslip(self, payslip_id: str) -> Optional[Payslip]:
        """الحصول على كشف راتب"""
        return self.payslips.get(payslip_id)
    
    def get_employee_payslips(self, employee_id: str) -> List[Payslip]:
        """الحصول على كشوفات موظف"""
        return [
            p for p in self.payslips.values()
            if p.employee_id == employee_id
        ]
    
    def get_monthly_payroll(self, month: int, year: int) -> Dict[str, Any]:
        """الحصول على كشوفات شهر معين"""
        month_payslips = [
            p for p in self.payslips.values()
            if p.month == month and p.year == year
        ]
        
        total_basic = sum(p.basic_salary for p in month_payslips)
        total_allowances = sum(p.total_allowances for p in month_payslips)
        total_deductions = sum(p.total_deductions for p in month_payslips)
        total_net = sum(p.net_salary for p in month_payslips)
        
        return {
            "period": f"{month:02d}/{year}",
            "total_employees": len(month_payslips),
            "total_basic": float(total_basic),
            "total_allowances": float(total_allowances),
            "total_deductions": float(total_deductions),
            "total_net": float(total_net),
            "payslips": [p.to_dict() for p in month_payslips]
        }
    
    def update_exchange_rate(self, currency: str, rate: Decimal):
        """تحديث سعر صرف العملة"""
        self.exchange_rates[currency] = Decimal(str(rate))
    
    def get_exchange_rate(self, currency: str) -> Decimal:
        """الحصول على سعر الصرف"""
        return self.exchange_rates.get(currency, Decimal('1'))
