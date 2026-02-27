"""
Attendance Tracking - تتبع الحضور والانصراف

المميزات:
- تسجيل الدخول/الخروج
- إدارة الإجازات
- حساب العمل الإضافي
- تقارير الحضور
- دمج مع أجهزة البصمة (placeholder)
"""

import uuid
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class LeaveType(Enum):
    """أنواع الإجازات"""
    ANNUAL = "annual"                 # سنوية
    SICK = "sick"                     # مرضية
    EMERGENCY = "emergency"           # طارئة
    UNPAID = "unpaid"                 # بدون راتب
    MATERNITY = "maternity"           # أمومة
    PATERNITY = "paternity"           # أبوة
    HAJJ = "hajj"                     # حج
    BEREAVEMENT = "bereavement"       # وفاة
    STUDY = "study"                   # دراسية
    OTHER = "other"                   # أخرى


class LeaveStatus(Enum):
    """حالات طلب الإجازة"""
    PENDING = "pending"               # معلق
    APPROVED = "approved"             # معتمد
    REJECTED = "rejected"             # مرفوض
    CANCELLED = "cancelled"           # ملغي


class AttendanceStatus(Enum):
    """حالة الحضور"""
    PRESENT = "present"               # حاضر
    ABSENT = "absent"                 # غائب
    LATE = "late"                     # متأخر
    ON_LEAVE = "on_leave"             # في إجازة
    HALF_DAY = "half_day"             # نصف يوم
    WORK_FROM_HOME = "work_from_home"  # عمل عن بُعد


@dataclass
class LeaveRequest:
    """طلب إجازة"""
    id: str
    employee_id: str
    leave_type: LeaveType
    start_date: date
    end_date: date
    
    days_requested: int = 0
    status: LeaveStatus = LeaveStatus.PENDING
    
    reason: str = ""
    attachment: str = ""              # مسار المرفق
    
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: str = ""
    
    @property
    def days_count(self) -> int:
        """عدد أيام الإجازة"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "leave_type": self.leave_type.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "days_count": self.days_count,
            "days_requested": self.days_requested,
            "status": self.status.value,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason
        }


@dataclass
class AttendanceRecord:
    """سجل حضور"""
    id: str
    employee_id: str
    date: date
    
    # Times
    check_in: Optional[datetime] = None
    check_out: Optional[datetime] = None
    
    # Status
    status: AttendanceStatus = AttendanceStatus.PRESENT
    
    # Work hours
    expected_work_hours: Decimal = Decimal('8')
    actual_work_hours: Decimal = Decimal('0')
    
    # Overtime
    overtime_hours: Decimal = Decimal('0')
    is_overtime_approved: bool = False
    
    # Breaks
    break_duration_minutes: int = 60  # استراحة الغداء
    
    # Metadata
    check_in_method: str = "manual"   # manual, biometric, mobile, web
    check_out_method: str = "manual"
    notes: str = ""
    location: str = ""                # موقع الحضور
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_late(self) -> bool:
        """هل الموظف متأخر؟"""
        if not self.check_in:
            return False
        # Assuming work starts at 8:00 AM
        work_start = datetime.combine(self.date, time(8, 0))
        grace_period = timedelta(minutes=15)
        return self.check_in > work_start + grace_period
    
    @property
    def late_minutes(self) -> int:
        """دقائق التأخير"""
        if not self.is_late:
            return 0
        work_start = datetime.combine(self.date, time(8, 0))
        delta = self.check_in - work_start
        return max(0, int(delta.total_seconds() / 60) - 15)  # 15 min grace
    
    @property
    def early_departure_minutes(self) -> int:
        """دقائب الانصراف المبكر"""
        if not self.check_out:
            return 0
        # Assuming work ends at 5:00 PM
        work_end = datetime.combine(self.date, time(17, 0))
        if self.check_out < work_end:
            delta = work_end - self.check_out
            return int(delta.total_seconds() / 60)
        return 0
    
    def calculate_work_hours(self):
        """حساب ساعات العمل"""
        if self.check_in and self.check_out:
            delta = self.check_out - self.check_in
            total_minutes = delta.total_seconds() / 60 - self.break_duration_minutes
            self.actual_work_hours = Decimal(str(max(0, total_minutes / 60)))
            
            # Calculate overtime
            if self.actual_work_hours > self.expected_work_hours:
                self.overtime_hours = self.actual_work_hours - self.expected_work_hours
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "date": self.date.isoformat(),
            "check_in": self.check_in.isoformat() if self.check_in else None,
            "check_out": self.check_out.isoformat() if self.check_out else None,
            "status": self.status.value,
            "is_late": self.is_late,
            "late_minutes": self.late_minutes,
            "early_departure_minutes": self.early_departure_minutes,
            "expected_work_hours": float(self.expected_work_hours),
            "actual_work_hours": float(self.actual_work_hours),
            "overtime_hours": float(self.overtime_hours),
            "is_overtime_approved": self.is_overtime_approved,
            "check_in_method": self.check_in_method,
            "check_out_method": self.check_out_method,
            "location": self.location,
            "notes": self.notes
        }


@dataclass
class BiometricDevice:
    """جهاز بصمة (placeholder للتكامل)"""
    id: str
    name: str
    device_type: str = "fingerprint"
    ip_address: str = ""
    port: int = 4370
    is_active: bool = True
    location: str = ""
    last_sync: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type,
            "ip_address": self.ip_address,
            "is_active": self.is_active,
            "location": self.location,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None
        }


class AttendanceTracker:
    """
    متتبع الحضور والانصراف
    """
    
    def __init__(self):
        self.attendance_records: Dict[str, AttendanceRecord] = {}
        self.leave_requests: Dict[str, LeaveRequest] = {}
        self.biometric_devices: Dict[str, BiometricDevice] = {}
        self.leave_balances: Dict[str, Dict[LeaveType, int]] = {}  # employee_id -> {leave_type: balance}
    
    def check_in(self, employee_id: str, check_in_time: datetime = None,
                method: str = "manual", location: str = "") -> AttendanceRecord:
        """تسجيل دخول"""
        if check_in_time is None:
            check_in_time = datetime.now()
        
        record_date = check_in_time.date()
        record_id = f"{employee_id}_{record_date.isoformat()}"
        
        # Check if already checked in
        if record_id in self.attendance_records:
            record = self.attendance_records[record_id]
            record.check_in = check_in_time
            record.check_in_method = method
            record.location = location
        else:
            record = AttendanceRecord(
                id=record_id,
                employee_id=employee_id,
                date=record_date,
                check_in=check_in_time,
                check_in_method=method,
                location=location
            )
            self.attendance_records[record_id] = record
        
        # Check if late
        if record.is_late:
            record.status = AttendanceStatus.LATE
        
        return record
    
    def check_out(self, employee_id: str, check_out_time: datetime = None,
                 method: str = "manual") -> AttendanceRecord:
        """تسجيل خروج"""
        if check_out_time is None:
            check_out_time = datetime.now()
        
        record_date = check_out_time.date()
        record_id = f"{employee_id}_{record_date.isoformat()}"
        
        record = self.attendance_records.get(record_id)
        if not record:
            # Auto create record if not exists
            record = AttendanceRecord(
                id=record_id,
                employee_id=employee_id,
                date=record_date
            )
            self.attendance_records[record_id] = record
        
        record.check_out = check_out_time
        record.check_out_method = method
        record.calculate_work_hours()
        
        return record
    
    def request_leave(self, employee_id: str, leave_type: LeaveType,
                     start_date: date, end_date: date,
                     reason: str = "", days_requested: int = None) -> LeaveRequest:
        """طلب إجازة"""
        if days_requested is None:
            days_requested = (end_date - start_date).days + 1
        
        request = LeaveRequest(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            days_requested=days_requested,
            reason=reason
        )
        
        self.leave_requests[request.id] = request
        return request
    
    def approve_leave(self, request_id: str, approved_by: str) -> LeaveRequest:
        """اعتماد طلب إجازة"""
        request = self.leave_requests.get(request_id)
        if not request:
            raise ValueError(f"Leave request {request_id} not found")
        
        request.status = LeaveStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now(timezone.utc)
        
        # Deduct from balance
        self._deduct_leave_balance(request.employee_id, request.leave_type, request.days_count)
        
        return request
    
    def reject_leave(self, request_id: str, rejection_reason: str) -> LeaveRequest:
        """رفض طلب إجازة"""
        request = self.leave_requests.get(request_id)
        if not request:
            raise ValueError(f"Leave request {request_id} not found")
        
        request.status = LeaveStatus.REJECTED
        request.rejection_reason = rejection_reason
        return request
    
    def _deduct_leave_balance(self, employee_id: str, leave_type: LeaveType, days: int):
        """خصم من رصيد الإجازات"""
        if employee_id not in self.leave_balances:
            self.leave_balances[employee_id] = {}
        
        current_balance = self.leave_balances[employee_id].get(leave_type, 0)
        self.leave_balances[employee_id][leave_type] = max(0, current_balance - days)
    
    def set_leave_balance(self, employee_id: str, leave_type: LeaveType, days: int):
        """تحديد رصيد إجازات"""
        if employee_id not in self.leave_balances:
            self.leave_balances[employee_id] = {}
        
        self.leave_balances[employee_id][leave_type] = days
    
    def get_leave_balance(self, employee_id: str, leave_type: LeaveType = None) -> Dict:
        """الحصول على رصيد الإجازات"""
        balances = self.leave_balances.get(employee_id, {})
        
        if leave_type:
            return {
                "employee_id": employee_id,
                "leave_type": leave_type.value,
                "balance": balances.get(leave_type, 0)
            }
        
        return {
            "employee_id": employee_id,
            "balances": {k.value: v for k, v in balances.items()}
        }
    
    def get_attendance_record(self, employee_id: str, record_date: date) -> Optional[AttendanceRecord]:
        """الحصول على سجل حضور"""
        record_id = f"{employee_id}_{record_date.isoformat()}"
        return self.attendance_records.get(record_id)
    
    def get_employee_attendance(self, employee_id: str,
                                start_date: date, end_date: date) -> List[AttendanceRecord]:
        """الحصول على سجلات حضور موظف لفترة"""
        records = []
        for record in self.attendance_records.values():
            if (record.employee_id == employee_id and
                start_date <= record.date <= end_date):
                records.append(record)
        
        return sorted(records, key=lambda x: x.date)
    
    def calculate_overtime(self, employee_id: str,
                          start_date: date, end_date: date) -> Dict[str, Any]:
        """حساب ساعات العمل الإضافي"""
        records = self.get_employee_attendance(employee_id, start_date, end_date)
        
        total_overtime = sum(r.overtime_hours for r in records if r.is_overtime_approved)
        unapproved_overtime = sum(r.overtime_hours for r in records if not r.is_overtime_approved)
        
        return {
            "employee_id": employee_id,
            "period": f"{start_date} to {end_date}",
            "total_overtime_hours": float(total_overtime),
            "unapproved_overtime_hours": float(unapproved_overtime),
            "overtime_records": [
                {
                    "date": r.date.isoformat(),
                    "hours": float(r.overtime_hours),
                    "approved": r.is_overtime_approved
                }
                for r in records if r.overtime_hours > 0
            ]
        }
    
    def approve_overtime(self, employee_id: str, record_date: date) -> AttendanceRecord:
        """اعتماد ساعات العمل الإضافي"""
        record = self.get_attendance_record(employee_id, record_date)
        if record:
            record.is_overtime_approved = True
        return record
    
    def generate_attendance_report(self, employee_id: str,
                                  start_date: date, end_date: date) -> Dict[str, Any]:
        """تقرير الحضور"""
        records = self.get_employee_attendance(employee_id, start_date, end_date)
        
        total_days = len(records)
        present_days = len([r for r in records if r.status == AttendanceStatus.PRESENT])
        absent_days = len([r for r in records if r.status == AttendanceStatus.ABSENT])
        late_days = len([r for r in records if r.status == AttendanceStatus.LATE])
        leave_days = len([r for r in records if r.status == AttendanceStatus.ON_LEAVE])
        
        total_late_minutes = sum(r.late_minutes for r in records)
        total_work_hours = sum(r.actual_work_hours for r in records)
        total_overtime = sum(r.overtime_hours for r in records if r.is_overtime_approved)
        
        return {
            "employee_id": employee_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_days": total_days,
                "present_days": present_days,
                "absent_days": absent_days,
                "late_days": late_days,
                "leave_days": leave_days,
                "attendance_rate": (present_days / total_days * 100) if total_days > 0 else 0
            },
            "details": {
                "total_late_minutes": total_late_minutes,
                "total_work_hours": float(total_work_hours),
                "total_overtime_hours": float(total_overtime),
                "average_daily_hours": float(total_work_hours / present_days) if present_days > 0 else 0
            },
            "records": [r.to_dict() for r in records]
        }
    
    def add_biometric_device(self, name: str, ip_address: str,
                            device_type: str = "fingerprint",
                            location: str = "") -> BiometricDevice:
        """إضافة جهاز بصمة"""
        device = BiometricDevice(
            id=str(uuid.uuid4()),
            name=name,
            device_type=device_type,
            ip_address=ip_address,
            location=location
        )
        
        self.biometric_devices[device.id] = device
        return device
    
    def sync_biometric_data(self, device_id: str) -> List[Dict]:
        """مزامنة بيانات البصمة (placeholder)"""
        device = self.biometric_devices.get(device_id)
        if not device:
            raise ValueError(f"Biometric device {device_id} not found")
        
        # Placeholder for actual device integration
        # In real implementation, this would connect to the device and pull data
        device.last_sync = datetime.now(timezone.utc)
        
        return []
    
    def get_pending_leave_requests(self, approver_id: str = None) -> List[LeaveRequest]:
        """الحصول على طلبات الإجازات المعلقة"""
        pending = [
            req for req in self.leave_requests.values()
            if req.status == LeaveStatus.PENDING
        ]
        return pending
    
    def get_leave_summary(self, employee_id: str) -> Dict[str, Any]:
        """ملخص الإجازات"""
        employee_leaves = [
            req for req in self.leave_requests.values()
            if req.employee_id == employee_id and req.status == LeaveStatus.APPROVED
        ]
        
        by_type = {}
        for leave_type in LeaveType:
            count = len([r for r in employee_leaves if r.leave_type == leave_type])
            total_days = sum(r.days_count for r in employee_leaves if r.leave_type == leave_type)
            by_type[leave_type.value] = {
                "count": count,
                "total_days": total_days
            }
        
        return {
            "employee_id": employee_id,
            "total_requests": len(employee_leaves),
            "total_days": sum(r.days_count for r in employee_leaves),
            "by_type": by_type,
            "balances": self.get_leave_balance(employee_id)["balances"]
        }
