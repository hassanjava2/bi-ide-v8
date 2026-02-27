# HR Module - وحدة الموارد البشرية
"""
Human Resources Module - وحدة الموارد البشرية

المميزات:
- Employee management
- Payroll processing with tax calculations
- Attendance tracking
"""

from .employees import Employee, EmployeeManager, EmployeeStatus, Department
from .payroll import PayrollProcessor, Payslip, SalaryStructure
from .attendance import AttendanceTracker, LeaveRequest, LeaveType

__all__ = [
    'Employee', 'EmployeeManager', 'EmployeeStatus', 'Department',
    'PayrollProcessor', 'Payslip', 'SalaryStructure',
    'AttendanceTracker', 'LeaveRequest', 'LeaveType',
]
