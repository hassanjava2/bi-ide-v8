"""
Employee Management - إدارة الموظفين

إدارة شاملة للموظفين مع:
- المعلومات الشخصية
- تفاصيل الوظيفة (المنصب، القسم، الراتب)
- المستندات
- حالة الموظف
- إنشاء هيكل تنظيمي
"""

import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class EmployeeStatus(Enum):
    """حالة الموظف"""
    ACTIVE = "active"                 # نشط
    ON_LEAVE = "on_leave"             # في إجازة
    SUSPENDED = "suspended"           # موقوف
    TERMINATED = "terminated"         # مفصول/منتهي الخدمة
    PROBATION = "probation"           # فترة تجربة


class Gender(Enum):
    """الجنس"""
    MALE = "male"
    FEMALE = "female"


class MaritalStatus(Enum):
    """الحالة الاجتماعية"""
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


class EmploymentType(Enum):
    """نوع التوظيف"""
    FULL_TIME = "full_time"           # دوام كامل
    PART_TIME = "part_time"           # دوام جزئي
    CONTRACT = "contract"             # عقد
    INTERN = "intern"                 # متدرب


@dataclass
class Department:
    """قسم"""
    id: str
    code: str
    name: str
    name_ar: str = ""
    parent_id: Optional[str] = None   # القسم الأب (للهيكل التنظيمي)
    manager_id: Optional[str] = None  # معرف مدير القسم
    cost_center: str = ""
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "name_ar": self.name_ar,
            "parent_id": self.parent_id,
            "manager_id": self.manager_id,
            "cost_center": self.cost_center,
            "is_active": self.is_active
        }


@dataclass
class EmployeeDocument:
    """مستند موظف"""
    id: str
    document_type: str                # نوع المستند: هوية، جواز، عقد، إلخ
    document_number: str
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None
    file_path: str = ""
    notes: str = ""
    
    @property
    def is_expired(self) -> bool:
        """هل المستند منتهي؟"""
        if not self.expiry_date:
            return False
        return date.today() > self.expiry_date
    
    @property
    def days_until_expiry(self) -> Optional[int]:
        """عدد الأيام حتى الانتهاء"""
        if not self.expiry_date:
            return None
        delta = self.expiry_date - date.today()
        return delta.days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_type": self.document_type,
            "document_number": self.document_number,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "is_expired": self.is_expired,
            "days_until_expiry": self.days_until_expiry,
            "file_path": self.file_path,
            "notes": self.notes
        }


@dataclass
class JobDetails:
    """تفاصيل الوظيفة"""
    position: str = ""                # المنصب
    department_id: str = ""           # القسم
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    grade: str = ""                   # الدرجة الوظيفية
    level: int = 1                    # المستوى
    hire_date: Optional[date] = None  # تاريخ التعيين
    termination_date: Optional[date] = None  # تاريخ انتهاء الخدمة
    probation_end_date: Optional[date] = None  # نهاية فترة التجربة
    manager_id: Optional[str] = None  # المدير المباشر
    work_location: str = ""           # موقع العمل
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "department_id": self.department_id,
            "employment_type": self.employment_type.value,
            "grade": self.grade,
            "level": self.level,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "termination_date": self.termination_date.isoformat() if self.termination_date else None,
            "probation_end_date": self.probation_end_date.isoformat() if self.probation_end_date else None,
            "manager_id": self.manager_id,
            "work_location": self.work_location
        }


@dataclass
class Compensation:
    """التعويضات والراتب"""
    basic_salary: Decimal = field(default_factory=lambda: Decimal('0'))
    housing_allowance: Decimal = field(default_factory=lambda: Decimal('0'))
    transportation_allowance: Decimal = field(default_factory=lambda: Decimal('0'))
    other_allowances: Decimal = field(default_factory=lambda: Decimal('0'))
    currency: str = "SAR"
    
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
            "gross_salary": float(self.gross_salary),
            "currency": self.currency
        }


@dataclass
class Employee:
    """موظف"""
    id: str
    employee_id: str                  # الرقم الوظيفي
    
    # Personal Information
    first_name: str = ""
    last_name: str = ""
    full_name_ar: str = ""            # الاسم الكامل بالعربية
    email: str = ""
    phone: str = ""
    mobile: str = ""
    
    # Demographics
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    marital_status: Optional[MaritalStatus] = None
    nationality: str = ""
    id_number: str = ""               # رقم الهوية/الإقامة
    
    # Address
    address: str = ""
    city: str = ""
    country: str = "Saudi Arabia"
    
    # Job Details
    job_details: JobDetails = field(default_factory=JobDetails)
    compensation: Compensation = field(default_factory=Compensation)
    
    # Documents
    documents: List[EmployeeDocument] = field(default_factory=list)
    
    # Status
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    
    # Emergency Contact
    emergency_contact_name: str = ""
    emergency_contact_phone: str = ""
    emergency_contact_relation: str = ""
    
    # Bank Details
    bank_name: str = ""
    bank_account: str = ""
    iban: str = ""
    
    # Metadata
    photo_url: str = ""
    notes: str = ""
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def full_name(self) -> str:
        """الاسم الكامل"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def age(self) -> Optional[int]:
        """العمر"""
        if not self.date_of_birth:
            return None
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def years_of_service(self) -> Optional[int]:
        """سنوات الخدمة"""
        if not self.job_details.hire_date:
            return None
        end_date = self.job_details.termination_date or date.today()
        return end_date.year - self.job_details.hire_date.year - (
            (end_date.month, end_date.day) < (self.job_details.hire_date.month, self.job_details.hire_date.day)
        )
    
    @property
    def is_on_probation(self) -> bool:
        """هل الموظف في فترة التجربة؟"""
        if not self.job_details.probation_end_date:
            return False
        return date.today() <= self.job_details.probation_end_date
    
    def add_document(self, doc_type: str, doc_number: str,
                    issue_date: date = None, expiry_date: date = None,
                    file_path: str = "", notes: str = "") -> EmployeeDocument:
        """إضافة مستند"""
        doc = EmployeeDocument(
            id=str(uuid.uuid4()),
            document_type=doc_type,
            document_number=doc_number,
            issue_date=issue_date,
            expiry_date=expiry_date,
            file_path=file_path,
            notes=notes
        )
        self.documents.append(doc)
        self.updated_at = datetime.now(timezone.utc)
        return doc
    
    def get_expiring_documents(self, days: int = 30) -> List[EmployeeDocument]:
        """الحصول على المستندات التي ستنتهي قريباً"""
        return [
            doc for doc in self.documents
            if doc.days_until_expiry is not None and 0 <= doc.days_until_expiry <= days
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "full_name": self.full_name,
            "full_name_ar": self.full_name_ar,
            "email": self.email,
            "phone": self.phone,
            "mobile": self.mobile,
            "age": self.age,
            "gender": self.gender.value if self.gender else None,
            "nationality": self.nationality,
            "status": self.status.value,
            "years_of_service": self.years_of_service,
            "is_on_probation": self.is_on_probation,
            "job_details": self.job_details.to_dict(),
            "compensation": self.compensation.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "emergency_contact": {
                "name": self.emergency_contact_name,
                "phone": self.emergency_contact_phone,
                "relation": self.emergency_contact_relation
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class EmployeeManager:
    """
    مدير الموظفين
    """
    
    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.departments: Dict[str, Department] = {}
        self._employee_counter = 0
        self._dept_counter = 0
    
    def create_department(self, name: str, code: str = None,
                         name_ar: str = "", parent_id: str = None,
                         manager_id: str = None) -> Department:
        """إنشاء قسم جديد"""
        if not code:
            self._dept_counter += 1
            code = f"DEPT-{self._dept_counter:03d}"
        
        dept = Department(
            id=str(uuid.uuid4()),
            code=code,
            name=name,
            name_ar=name_ar,
            parent_id=parent_id,
            manager_id=manager_id
        )
        
        self.departments[dept.id] = dept
        return dept
    
    def create_employee(self, first_name: str, last_name: str,
                       email: str = "", phone: str = "",
                       **kwargs) -> Employee:
        """إنشاء موظف جديد"""
        self._employee_counter += 1
        employee_id = f"EMP-{datetime.now().strftime('%Y')}-{self._employee_counter:04d}"
        
        employee = Employee(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            **kwargs
        )
        
        self.employees[employee.id] = employee
        return employee
    
    def get_employee(self, employee_id: str) -> Optional[Employee]:
        """الحصول على موظف"""
        return self.employees.get(employee_id)
    
    def get_employee_by_employee_id(self, emp_id: str) -> Optional[Employee]:
        """الحصول على موظف برقمه الوظيفي"""
        for emp in self.employees.values():
            if emp.employee_id == emp_id:
                return emp
        return None
    
    def update_employee(self, employee_id: str, **kwargs) -> Employee:
        """تحديث بيانات موظف"""
        employee = self.employees.get(employee_id)
        if not employee:
            raise ValueError(f"Employee {employee_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(employee, key):
                setattr(employee, key, value)
        
        employee.updated_at = datetime.now(timezone.utc)
        return employee
    
    def change_status(self, employee_id: str, new_status: EmployeeStatus,
                     reason: str = "") -> Employee:
        """تغيير حالة موظف"""
        employee = self.employees.get(employee_id)
        if not employee:
            raise ValueError(f"Employee {employee_id} not found")
        
        employee.status = new_status
        
        if new_status == EmployeeStatus.TERMINATED:
            employee.job_details.termination_date = date.today()
        
        employee.notes += f" | Status changed to {new_status.value}: {reason}"
        employee.updated_at = datetime.now(timezone.utc)
        return employee
    
    def get_employees_by_department(self, department_id: str) -> List[Employee]:
        """الحصول على موظفي قسم معين"""
        return [
            emp for emp in self.employees.values()
            if emp.job_details.department_id == department_id and emp.status == EmployeeStatus.ACTIVE
        ]
    
    def get_employees_by_status(self, status: EmployeeStatus) -> List[Employee]:
        """الحصول على موظفين بحالة معينة"""
        return [emp for emp in self.employees.values() if emp.status == status]
    
    def get_employees_by_manager(self, manager_id: str) -> List[Employee]:
        """الحصول على موظفين تابعين لمدير معين"""
        return [
            emp for emp in self.employees.values()
            if emp.job_details.manager_id == manager_id and emp.status == EmployeeStatus.ACTIVE
        ]
    
    def search_employees(self, query: str) -> List[Employee]:
        """البحث في الموظفين"""
        query = query.lower()
        results = []
        
        for emp in self.employees.values():
            if (query in emp.full_name.lower() or
                query in emp.email.lower() or
                query in emp.employee_id.lower() or
                query in emp.id_number.lower()):
                results.append(emp)
        
        return results
    
    def get_expiring_documents(self, days: int = 30) -> List[Dict]:
        """الحصول على المستندات التي ستنتهي"""
        expiring = []
        
        for emp in self.employees.values():
            for doc in emp.get_expiring_documents(days):
                expiring.append({
                    "employee_id": emp.id,
                    "employee_name": emp.full_name,
                    "document": doc.to_dict()
                })
        
        return expiring
    
    def generate_org_chart(self) -> Dict[str, Any]:
        """
        إنشاء الهيكل التنظيمي
        """
        # Build department tree
        dept_tree = {}
        
        for dept in self.departments.values():
            if dept.parent_id is None:
                dept_tree[dept.id] = self._build_dept_node(dept)
        
        # Build employee hierarchy
        ceo = None
        for emp in self.employees.values():
            if emp.job_details.manager_id is None and emp.status == EmployeeStatus.ACTIVE:
                ceo = emp
                break
        
        employee_tree = None
        if ceo:
            employee_tree = self._build_employee_node(ceo)
        
        return {
            "departments": dept_tree,
            "employee_hierarchy": employee_tree,
            "summary": {
                "total_employees": len(self.employees),
                "active_employees": len([e for e in self.employees.values() if e.status == EmployeeStatus.ACTIVE]),
                "total_departments": len(self.departments)
            }
        }
    
    def _build_dept_node(self, dept: Department) -> Dict[str, Any]:
        """بناء عقدة قسم"""
        node = {
            "department": dept.to_dict(),
            "children": {},
            "employees": [
                emp.to_dict() for emp in self.employees.values()
                if emp.job_details.department_id == dept.id and emp.status == EmployeeStatus.ACTIVE
            ]
        }
        
        # Add children departments
        for child_dept in self.departments.values():
            if child_dept.parent_id == dept.id:
                node["children"][child_dept.id] = self._build_dept_node(child_dept)
        
        return node
    
    def _build_employee_node(self, emp: Employee) -> Dict[str, Any]:
        """بناء عقدة موظف"""
        node = {
            "employee": emp.to_dict(),
            "subordinates": []
        }
        
        # Add subordinates
        for subordinate in self.get_employees_by_manager(emp.id):
            node["subordinates"].append(self._build_employee_node(subordinate))
        
        return node
    
    def get_hr_summary(self) -> Dict[str, Any]:
        """ملخص الموارد البشرية"""
        total_employees = len(self.employees)
        active_employees = len(self.get_employees_by_status(EmployeeStatus.ACTIVE))
        
        # By status
        by_status = {status.value: len(self.get_employees_by_status(status)) 
                    for status in EmployeeStatus}
        
        # By department
        by_department = {}
        for dept in self.departments.values():
            count = len(self.get_employees_by_department(dept.id))
            by_department[dept.name] = count
        
        # Average salary
        salaries = [emp.compensation.gross_salary for emp in self.employees.values()]
        avg_salary = sum(salaries) / len(salaries) if salaries else 0
        
        return {
            "total_employees": total_employees,
            "active_employees": active_employees,
            "by_status": by_status,
            "by_department": by_department,
            "total_departments": len(self.departments),
            "average_salary": float(avg_salary),
            "expiring_documents": len(self.get_expiring_documents(30))
        }
