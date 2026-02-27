"""
Unit Tests for HR Module
اختبارات وحدة الموارد البشرية
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, timedelta

from erp.modules.hr.employees import (
    EmployeeManager, Employee, EmployeeStatus,
    Department, JobDetails, Compensation
)
from erp.modules.hr.payroll import (
    PayrollProcessor, SalaryStructure, Payslip,
    DeductionType, AllowanceType
)
from erp.modules.hr.attendance import (
    AttendanceTracker, LeaveRequest, LeaveType, LeaveStatus,
    AttendanceRecord, AttendanceStatus
)


class TestEmployeeManager:
    """اختبارات مدير الموظفين"""
    
    @pytest.fixture
    def emp_manager(self):
        """إنشاء مدير موظفين للاختبارات"""
        return EmployeeManager()
    
    def test_create_department(self, emp_manager):
        """اختبار إنشاء قسم"""
        dept = emp_manager.create_department(
            name="IT Department",
            code="IT",
            name_ar="قسم تقنية المعلومات"
        )
        
        assert dept.name == "IT Department"
        assert dept.code == "IT"
        assert dept.name_ar == "قسم تقنية المعلومات"
    
    def test_create_employee(self, emp_manager):
        """اختبار إنشاء موظف"""
        emp = emp_manager.create_employee(
            first_name="أحمد",
            last_name="محمد",
            email="ahmed@company.com",
            phone="+966501234567"
        )
        
        assert emp.first_name == "أحمد"
        assert emp.full_name == "أحمد محمد"
        assert emp.employee_id.startswith("EMP-")
        assert emp.status == EmployeeStatus.ACTIVE
    
    def test_employee_age_calculation(self, emp_manager):
        """اختبار حساب عمر الموظف"""
        emp = emp_manager.create_employee(
            "Test", "User",
            date_of_birth=date(1990, 5, 15)
        )
        
        expected_age = date.today().year - 1990
        if (date.today().month, date.today().day) < (5, 15):
            expected_age -= 1
        
        assert emp.age == expected_age
    
    def test_years_of_service(self, emp_manager):
        """اختبار حساب سنوات الخدمة"""
        emp = emp_manager.create_employee("Test", "User")
        emp.job_details.hire_date = date(2020, 1, 1)
        
        expected_years = date.today().year - 2020
        if (date.today().month, date.today().day) < (1, 1):
            expected_years -= 1
        
        assert emp.years_of_service == expected_years
    
    def test_add_document(self, emp_manager):
        """اختبار إضافة مستند"""
        emp = emp_manager.create_employee("Test", "User")
        
        doc = emp.add_document(
            doc_type="Passport",
            doc_number="A12345678",
            issue_date=date(2020, 1, 1),
            expiry_date=date(2030, 1, 1)
        )
        
        assert doc.document_type == "Passport"
        assert doc.document_number == "A12345678"
        assert doc.is_expired is False
        assert len(emp.documents) == 1
    
    def test_expiring_documents(self, emp_manager):
        """اختبار المستندات التي ستنتهي"""
        emp = emp_manager.create_employee("Test", "User")
        
        # Add document expiring soon
        emp.add_document(
            "ID",
            "ID001",
            expiry_date=date.today() + timedelta(days=15)
        )
        
        # Add document not expiring soon
        emp.add_document(
            "Passport",
            "P001",
            expiry_date=date.today() + timedelta(days=100)
        )
        
        expiring = emp.get_expiring_documents(days=30)
        
        assert len(expiring) == 1
        assert expiring[0].document_type == "ID"
    
    def test_change_status(self, emp_manager):
        """اختبار تغيير حالة الموظف"""
        emp = emp_manager.create_employee("Test", "User")
        
        emp_manager.change_status(emp.id, EmployeeStatus.ON_LEAVE, "Annual vacation")
        
        assert emp.status == EmployeeStatus.ON_LEAVE
        assert "Annual vacation" in emp.notes
    
    def test_get_employees_by_department(self, emp_manager):
        """اختبار الحصول على موظفين حسب القسم"""
        dept = emp_manager.create_department("IT", "IT")
        
        emp1 = emp_manager.create_employee("أحمد", "محمد")
        emp1.job_details.department_id = dept.id
        
        emp2 = emp_manager.create_employee("سارة", "علي")
        emp2.job_details.department_id = dept.id
        
        emp3 = emp_manager.create_employee("خالد", "عمر")
        # Different department
        
        dept_emps = emp_manager.get_employees_by_department(dept.id)
        
        assert len(dept_emps) == 2
        assert all(e.job_details.department_id == dept.id for e in dept_emps)
    
    def test_search_employees(self, emp_manager):
        """اختبار البحث في الموظفين"""
        emp_manager.create_employee("أحمد", "محمد", email="ahmed@test.com")
        emp_manager.create_employee("سارة", "علي", email="sara@test.com")
        emp_manager.create_employee("أحمد", "خالد", email="ahmad.k@test.com")
        
        results = emp_manager.search_employees("أحمد")
        
        assert len(results) == 2
    
    def test_generate_org_chart(self, emp_manager):
        """اختبار إنشاء الهيكل التنظيمي"""
        # Create departments
        it_dept = emp_manager.create_department("IT", "IT")
        hr_dept = emp_manager.create_department("HR", "HR")
        
        # Create employees
        ceo = emp_manager.create_employee("المدير", "التنفيذي")
        it_manager = emp_manager.create_employee("مدير", "IT")
        it_manager.job_details.manager_id = ceo.id
        it_manager.job_details.department_id = it_dept.id
        
        org_chart = emp_manager.generate_org_chart()
        
        assert "departments" in org_chart
        assert "employee_hierarchy" in org_chart
        assert org_chart["summary"]["total_employees"] == 2


class TestPayrollProcessor:
    """اختبارات معالج الرواتب"""
    
    @pytest.fixture
    def payroll(self):
        """إنشاء معالج رواتب للاختبارات"""
        return PayrollProcessor()
    
    def test_calculate_tax_saudi(self, payroll):
        """اختبار حساب الضريبة للموظف السعودي (معفى)"""
        tax = payroll.calculate_tax(Decimal("10000"), is_saudi=True)
        assert tax == Decimal("0")
    
    def test_calculate_tax_non_saudi(self, payroll):
        """اختبار حساب الضريبة للموظف غير السعودي"""
        tax = payroll.calculate_tax(Decimal("15000"), is_saudi=False)
        assert tax > Decimal("0")
    
    def test_calculate_gosi_saudi(self, payroll):
        """اختبار حساب التأمينات للموظف السعودي"""
        gosi = payroll.calculate_gosi(Decimal("10000"), is_saudi=True)
        
        assert gosi["employee_contribution"] > Decimal("0")
        assert gosi["employer_contribution"] > Decimal("0")
        assert gosi["pension"] > Decimal("0")
    
    def test_calculate_gosi_non_saudi(self, payroll):
        """اختبار حساب التأمينات للموظف غير السعودي"""
        gosi = payroll.calculate_gosi(Decimal("10000"), is_saudi=False)
        
        assert gosi["employee_contribution"] == Decimal("0")
        assert gosi["employer_contribution"] > Decimal("0")  # 2% occupational hazard
    
    def test_calculate_salary(self, payroll):
        """اختبار حساب راتب"""
        salary_structure = SalaryStructure(
            basic_salary=Decimal("10000"),
            housing_allowance=Decimal("2500"),
            transportation_allowance=Decimal("800"),
            other_allowances=Decimal("0")
        )
        
        payslip = payroll.calculate_salary(
            employee_id="EMP001",
            employee_name="Test Employee",
            salary_structure=salary_structure,
            month=1,
            year=2024,
            is_saudi=True
        )
        
        assert payslip.employee_id == "EMP001"
        assert payslip.basic_salary == Decimal("10000")
        assert payslip.total_earnings == Decimal("13300")  # 10000 + 2500 + 800
        assert payslip.net_salary > Decimal("0")
        assert payslip.net_salary < payslip.total_earnings  # After deductions
    
    def test_payslip_with_overtime(self, payroll):
        """اختبار كشف راتب مع ساعات إضافية"""
        salary_structure = SalaryStructure(
            basic_salary=Decimal("10000"),
            housing_allowance=Decimal("2500"),
            transportation_allowance=Decimal("800")
        )
        
        payslip = payroll.calculate_salary(
            "EMP001", "Test Employee",
            salary_structure, 1, 2024,
            is_saudi=True,
            overtime_hours=Decimal("10"),
            hourly_rate=Decimal("62.50")  # 10000 / 160 hours
        )
        
        expected_overtime = Decimal("10") * Decimal("62.50") * Decimal("1.5")
        assert payslip.overtime_hours == Decimal("10")
        assert payslip.overtime_amount == expected_overtime
    
    def test_add_bonus(self, payroll):
        """اختبار إضافة مكافأة"""
        salary_structure = SalaryStructure(basic_salary=Decimal("10000"))
        
        payslip = payroll.calculate_salary(
            "EMP001", "Test Employee",
            salary_structure, 1, 2024, is_saudi=True
        )
        
        original_net = payslip.net_salary
        payroll.add_bonus(payslip.id, Decimal("5000"), "Performance bonus")
        
        assert payslip.bonuses == Decimal("5000")
        assert payslip.net_salary > original_net
    
    def test_approve_payslip(self, payroll):
        """اختبار اعتماد كشف الراتب"""
        salary_structure = SalaryStructure(basic_salary=Decimal("10000"))
        
        payslip = payroll.calculate_salary(
            "EMP001", "Test Employee",
            salary_structure, 1, 2024, is_saudi=True
        )
        
        approved = payroll.approve_payslip(payslip.id, "MANAGER001")
        
        assert approved.approved_by == "MANAGER001"
        assert approved.approved_at is not None
    
    def test_multi_currency(self, payroll):
        """اختبار دعم العملات المتعددة"""
        salary_structure = SalaryStructure(
            basic_salary=Decimal("5000")
        )
        
        payroll.update_exchange_rate("USD", Decimal("3.75"))
        
        payslip = payroll.calculate_salary(
            "EMP001", "Test Employee",
            salary_structure, 1, 2024, is_saudi=False,
            currency="USD"
        )
        
        assert payslip.currency == "USD"
        assert payslip.exchange_rate == Decimal("3.75")
    
    def test_get_monthly_payroll(self, payroll):
        """اختبار الحصول على كشوفات شهر"""
        salary_structure = SalaryStructure(basic_salary=Decimal("10000"))
        
        payroll.calculate_salary("EMP001", "Employee 1", salary_structure, 1, 2024, True)
        payroll.calculate_salary("EMP002", "Employee 2", salary_structure, 1, 2024, True)
        payroll.calculate_salary("EMP003", "Employee 3", salary_structure, 2, 2024, True)
        
        january_payroll = payroll.get_monthly_payroll(1, 2024)
        
        assert january_payroll["period"] == "01/2024"
        assert january_payroll["total_employees"] == 2


class TestAttendanceTracker:
    """اختبارات متتبع الحضور"""
    
    @pytest.fixture
    def attendance(self):
        """إنشاء متتبع حضور للاختبارات"""
        return AttendanceTracker()
    
    def test_check_in(self, attendance):
        """اختبار تسجيل دخول"""
        record = attendance.check_in(
            employee_id="EMP001",
            method="mobile",
            location="Office"
        )
        
        assert record.employee_id == "EMP001"
        assert record.check_in is not None
        assert record.check_in_method == "mobile"
        assert record.location == "Office"
    
    def test_check_out(self, attendance):
        """اختبار تسجيل خروج"""
        attendance.check_in("EMP001")
        
        record = attendance.check_out(
            employee_id="EMP001",
            method="mobile"
        )
        
        assert record.check_out is not None
        assert record.actual_work_hours >= Decimal("0")
    
    def test_late_check_in(self, attendance):
        """اختبار تأخر في الدخول"""
        from datetime import time
        
        # Check in after 8:15 AM (15 min grace period)
        late_time = datetime.combine(date.today(), time(8, 30))
        
        record = attendance.check_in("EMP001", check_in_time=late_time)
        
        assert record.is_late is True
        assert record.late_minutes > 0
        assert record.status == AttendanceStatus.LATE
    
    def test_request_leave(self, attendance):
        """اختبار طلب إجازة"""
        request = attendance.request_leave(
            employee_id="EMP001",
            leave_type=LeaveType.ANNUAL,
            start_date=date.today() + timedelta(days=7),
            end_date=date.today() + timedelta(days=10),
            reason="Family vacation"
        )
        
        assert request.employee_id == "EMP001"
        assert request.leave_type == LeaveType.ANNUAL
        assert request.days_count == 4
        assert request.status == LeaveStatus.PENDING
    
    def test_approve_leave(self, attendance):
        """اختبار اعتماد إجازة"""
        request = attendance.request_leave(
            "EMP001", LeaveType.ANNUAL,
            date.today() + timedelta(days=7),
            date.today() + timedelta(days=10)
        )
        
        attendance.set_leave_balance("EMP001", LeaveType.ANNUAL, 30)
        
        approved = attendance.approve_leave(request.id, "MANAGER001")
        
        assert approved.status == LeaveStatus.APPROVED
        assert approved.approved_by == "MANAGER001"
    
    def test_leave_balance(self, attendance):
        """اختبار رصيد الإجازات"""
        attendance.set_leave_balance("EMP001", LeaveType.ANNUAL, 30)
        attendance.set_leave_balance("EMP001", LeaveType.SICK, 15)
        
        annual_balance = attendance.get_leave_balance("EMP001", LeaveType.ANNUAL)
        
        assert annual_balance["balance"] == 30
        
        all_balances = attendance.get_leave_balance("EMP001")
        assert "balances" in all_balances
    
    def test_overtime_calculation(self, attendance):
        """اختبار حساب العمل الإضافي"""
        today = date.today()
        
        # Simulate 10 hour work day (2 hours overtime)
        check_in = datetime.combine(today, datetime.min.time().replace(hour=8))
        check_out = datetime.combine(today, datetime.min.time().replace(hour=18))
        
        record = attendance.check_in("EMP001", check_in_time=check_in)
        record.check_out = check_out
        record.calculate_work_hours()
        
        assert record.overtime_hours > Decimal("0")
    
    def test_attendance_report(self, attendance):
        """اختبار تقرير الحضور"""
        today = date.today()
        
        # Create some attendance records
        attendance.check_in("EMP001")
        attendance.check_out("EMP001")
        
        report = attendance.generate_attendance_report(
            employee_id="EMP001",
            start_date=today - timedelta(days=7),
            end_date=today
        )
        
        assert report["employee_id"] == "EMP001"
        assert "summary" in report
        assert "records" in report
    
    def test_work_from_home(self, attendance):
        """اختبار العمل عن بُعد"""
        record = attendance.check_in("EMP001", location="Home")
        record.status = AttendanceStatus.WORK_FROM_HOME
        
        assert record.status == AttendanceStatus.WORK_FROM_HOME
