#!/usr/bin/env python3
"""
company_camera.py — نظام إدارة شركة بالكاميرات 📹🏢

الوظائف:
  1. حضور وانصراف — التقاط وجوه + تسجيل أوقات
  2. توزيع مهام — من يشتغل على شنو
  3. إدارة الشركة — مراقبة إنتاجية + سلامة
  4. قابل للتوسيع — أي وظيفة جديدة بالمستقبل

البيانات ما تطلع — كلشي محلي 🔒
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("company_camera")

PROJECT_ROOT = Path(__file__).parent.parent
COMPANY_DIR = PROJECT_ROOT / "brain" / "company_data"
EMPLOYEES_FILE = COMPANY_DIR / "employees.json"
ATTENDANCE_FILE = COMPANY_DIR / "attendance.json"
TASKS_FILE = COMPANY_DIR / "tasks.json"
CAMERAS_FILE = COMPANY_DIR / "cameras.json"

COMPANY_DIR.mkdir(parents=True, exist_ok=True)

try:
    from brain.memory_system import memory
except ImportError:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from brain.memory_system import memory


# ═══════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════

class EmployeeStatus(Enum):
    PRESENT = "حاضر"
    ABSENT = "غائب"
    LATE = "متأخر"
    ON_BREAK = "استراحة"
    LEFT_EARLY = "انصرف مبكراً"
    ON_LEAVE = "إجازة"

class TaskStatus(Enum):
    PENDING = "معلّق"
    IN_PROGRESS = "قيد التنفيذ"
    COMPLETED = "مكتمل"
    OVERDUE = "متأخر"
    PAUSED = "متوقف"

class CameraRole(Enum):
    ENTRANCE = "مدخل"
    EXIT = "مخرج"
    PRODUCTION = "إنتاج"
    WAREHOUSE = "مخزن"
    OFFICE = "مكتب"
    SAFETY = "سلامة"
    PARKING = "مواقف"
    CUSTOM = "مخصص"


# ═══════════════════════════════════════════════════════════
# موظف
# ═══════════════════════════════════════════════════════════

@dataclass
class Employee:
    employee_id: str
    name: str
    name_ar: str
    department: str = ""
    role: str = ""
    face_encoding: str = ""       # hash للتعرف بالكاميرا
    status: str = EmployeeStatus.ABSENT.value
    shift_start: str = "08:00"
    shift_end: str = "16:00"
    total_days: int = 0
    late_count: int = 0
    early_leave_count: int = 0
    active_task: str = ""
    skills: List[str] = field(default_factory=list)
    performance_score: float = 0.5


# ═══════════════════════════════════════════════════════════
# كاميرا
# ═══════════════════════════════════════════════════════════

@dataclass
class Camera:
    camera_id: str
    name: str
    url: str                       # rtsp://... أو http://...
    location: str = ""
    role: str = CameraRole.ENTRANCE.value
    active: bool = True
    last_frame_time: str = ""
    detections_today: int = 0
    alerts_today: int = 0


# ═══════════════════════════════════════════════════════════
# مهمة
# ═══════════════════════════════════════════════════════════

@dataclass
class Task:
    task_id: str
    title: str
    description: str = ""
    assigned_to: str = ""          # employee_id
    assigned_by: str = "system"
    department: str = ""
    priority: int = 3              # 1=urgent, 5=low
    status: str = TaskStatus.PENDING.value
    deadline: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    camera_verified: bool = False  # كاميرا أكّدت إنجازه


# ═══════════════════════════════════════════════════════════
# نظام الحضور والانصراف 📋
# ═══════════════════════════════════════════════════════════

@dataclass
class AttendanceRecord:
    employee_id: str
    date: str
    check_in: str = ""
    check_out: str = ""
    status: str = EmployeeStatus.ABSENT.value
    hours_worked: float = 0.0
    camera_id: str = ""            # أي كاميرا سجّلته
    notes: str = ""


# ═══════════════════════════════════════════════════════════
# مدير الشركة 🏢
# ═══════════════════════════════════════════════════════════

class CompanyManager:
    """
    نظام إدارة شركة بالكاميرات 🏢📹

    وظائف:
      - حضور/انصراف أوتوماتيكي من الكاميرا
      - توزيع مهام حسب المهارات + الحضور
      - مراقبة إنتاجية
      - تنبيهات سلامة
      - تقارير يومية/أسبوعية/شهرية

    قابل للتوسيع:
      - أي وظيفة جديدة ← extend_feature()
    """

    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.cameras: Dict[str, Camera] = {}
        self.tasks: Dict[str, Task] = {}
        self.attendance: List[AttendanceRecord] = []
        self.custom_features: Dict[str, Dict] = {}  # وظائف مخصصة
        self._load()

    def _load(self):
        """تحميل البيانات المحفوظة"""
        if EMPLOYEES_FILE.exists():
            try:
                data = json.loads(EMPLOYEES_FILE.read_text())
                for eid, ed in data.items():
                    self.employees[eid] = Employee(**ed)
            except Exception:
                pass

        if CAMERAS_FILE.exists():
            try:
                data = json.loads(CAMERAS_FILE.read_text())
                for cid, cd in data.items():
                    self.cameras[cid] = Camera(**cd)
            except Exception:
                pass

        if TASKS_FILE.exists():
            try:
                data = json.loads(TASKS_FILE.read_text())
                for tid, td in data.items():
                    self.tasks[tid] = Task(**td)
            except Exception:
                pass

        if ATTENDANCE_FILE.exists():
            try:
                data = json.loads(ATTENDANCE_FILE.read_text())
                self.attendance = [AttendanceRecord(**a) for a in data]
            except Exception:
                pass

    def _save(self):
        """حفظ كل البيانات"""
        try:
            EMPLOYEES_FILE.write_text(json.dumps(
                {eid: e.__dict__ for eid, e in self.employees.items()},
                indent=2, ensure_ascii=False))
            CAMERAS_FILE.write_text(json.dumps(
                {cid: c.__dict__ for cid, c in self.cameras.items()},
                indent=2, ensure_ascii=False))
            TASKS_FILE.write_text(json.dumps(
                {tid: t.__dict__ for tid, t in self.tasks.items()},
                indent=2, ensure_ascii=False))
            ATTENDANCE_FILE.write_text(json.dumps(
                [a.__dict__ for a in self.attendance[-5000:]],
                indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Save error: {e}")

    # ═══════════════════════════════════════════════════════
    # إدارة الموظفين
    # ═══════════════════════════════════════════════════════

    def add_employee(self, employee_id: str, name: str, name_ar: str,
                     department: str = "", role: str = "",
                     shift_start: str = "08:00", shift_end: str = "16:00",
                     skills: List[str] = None) -> Employee:
        """إضافة موظف"""
        emp = Employee(
            employee_id=employee_id, name=name, name_ar=name_ar,
            department=department, role=role,
            shift_start=shift_start, shift_end=shift_end,
            skills=skills or [],
        )
        self.employees[employee_id] = emp
        self._save()
        logger.info(f"👤 Employee added: {name_ar} ({department})")
        return emp

    def remove_employee(self, employee_id: str):
        """حذف موظف"""
        if employee_id in self.employees:
            del self.employees[employee_id]
            self._save()

    # ═══════════════════════════════════════════════════════
    # كاميرات
    # ═══════════════════════════════════════════════════════

    def add_camera(self, camera_id: str, name: str, url: str,
                   location: str = "", role: str = CameraRole.ENTRANCE.value) -> Camera:
        """إضافة كاميرا"""
        cam = Camera(
            camera_id=camera_id, name=name, url=url,
            location=location, role=role,
        )
        self.cameras[camera_id] = cam
        self._save()
        logger.info(f"📹 Camera added: {name} ({role}) @ {location}")
        return cam

    # ═══════════════════════════════════════════════════════
    # حضور وانصراف 📋
    # ═══════════════════════════════════════════════════════

    def check_in(self, employee_id: str, camera_id: str = "", time_str: str = None) -> Dict:
        """تسجيل حضور — أوتوماتيكي من الكاميرا أو يدوي"""
        emp = self.employees.get(employee_id)
        if not emp:
            return {"error": f"Employee {employee_id} not found"}

        now = time_str or datetime.now().strftime("%H:%M:%S")
        today = datetime.now().strftime("%Y-%m-%d")

        # فحص إذا مسجل اليوم
        existing = [a for a in self.attendance
                    if a.employee_id == employee_id and a.date == today]
        if existing and existing[-1].check_in:
            return {"status": "already_checked_in", "time": existing[-1].check_in}

        # فحص تأخر
        shift_time = emp.shift_start
        status = EmployeeStatus.PRESENT.value
        if now > shift_time + ":59":
            status = EmployeeStatus.LATE.value
            emp.late_count += 1

        record = AttendanceRecord(
            employee_id=employee_id,
            date=today,
            check_in=now,
            status=status,
            camera_id=camera_id,
        )
        self.attendance.append(record)

        emp.status = status
        emp.total_days += 1
        self._save()

        result = {
            "employee": emp.name_ar,
            "status": status,
            "check_in": now,
            "camera": camera_id,
        }

        if status == EmployeeStatus.LATE.value:
            result["late_by"] = f"متأخر (الدوام {shift_time})"

        memory.save_knowledge(
            topic=f"حضور: {emp.name_ar}",
            content=f"{emp.name_ar} سجّل حضور {now} — {status}",
            source="attendance",
        )

        logger.info(f"📋 Check-in: {emp.name_ar} {now} [{status}]")
        return result

    def check_out(self, employee_id: str, camera_id: str = "", time_str: str = None) -> Dict:
        """تسجيل انصراف"""
        emp = self.employees.get(employee_id)
        if not emp:
            return {"error": f"Employee {employee_id} not found"}

        now = time_str or datetime.now().strftime("%H:%M:%S")
        today = datetime.now().strftime("%Y-%m-%d")

        records = [a for a in self.attendance
                   if a.employee_id == employee_id and a.date == today]
        if not records:
            return {"error": "No check-in record for today"}

        record = records[-1]
        record.check_out = now

        # حساب ساعات العمل
        if record.check_in:
            try:
                cin = datetime.strptime(record.check_in, "%H:%M:%S")
                cout = datetime.strptime(now, "%H:%M:%S")
                hours = (cout - cin).seconds / 3600
                record.hours_worked = round(hours, 2)
            except Exception:
                pass

        # فحص انصراف مبكر
        if now < emp.shift_end + ":00":
            record.status = EmployeeStatus.LEFT_EARLY.value
            emp.early_leave_count += 1

        emp.status = EmployeeStatus.ABSENT.value
        self._save()

        logger.info(f"📋 Check-out: {emp.name_ar} {now} ({record.hours_worked}h)")
        return {
            "employee": emp.name_ar,
            "check_out": now,
            "hours_worked": record.hours_worked,
            "status": record.status,
        }

    # ═══════════════════════════════════════════════════════
    # توزيع مهام 📝
    # ═══════════════════════════════════════════════════════

    def create_task(self, title: str, description: str = "", department: str = "",
                    priority: int = 3, deadline: str = "",
                    assign_to: str = None) -> Task:
        """إنشاء مهمة وتوزيعها"""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"

        task = Task(
            task_id=task_id, title=title, description=description,
            department=department, priority=priority, deadline=deadline,
        )

        if assign_to:
            task.assigned_to = assign_to
            task.status = TaskStatus.IN_PROGRESS.value
            if assign_to in self.employees:
                self.employees[assign_to].active_task = task_id
        else:
            # توزيع أوتوماتيكي حسب المهارات + الحضور
            assigned = self._auto_assign(task)
            if assigned:
                task.assigned_to = assigned
                task.status = TaskStatus.IN_PROGRESS.value

        self.tasks[task_id] = task
        self._save()

        logger.info(f"📝 Task created: {title} → {task.assigned_to or 'unassigned'}")
        return task

    def _auto_assign(self, task: Task) -> Optional[str]:
        """توزيع أوتوماتيكي — يختار أفضل موظف متاح"""
        candidates = []
        for eid, emp in self.employees.items():
            # لازم حاضر + مو مشغول بمهمة + نفس القسم
            if emp.status not in [EmployeeStatus.PRESENT.value, EmployeeStatus.LATE.value]:
                continue
            if emp.active_task:
                continue
            if task.department and emp.department != task.department:
                continue

            # نقاط حسب المهارات + الأداء
            skill_match = len(set(emp.skills).intersection(set(task.title.lower().split())))
            score = emp.performance_score + skill_match * 0.1

            candidates.append((eid, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_id = candidates[0][0]
            self.employees[best_id].active_task = task.task_id
            return best_id
        return None

    def complete_task(self, task_id: str, camera_verified: bool = False) -> Dict:
        """إنهاء مهمة"""
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}

        task.status = TaskStatus.COMPLETED.value
        task.completed_at = datetime.now().isoformat()
        task.camera_verified = camera_verified

        # تحرير الموظف
        if task.assigned_to in self.employees:
            emp = self.employees[task.assigned_to]
            emp.active_task = ""
            emp.performance_score = min(emp.performance_score + 0.02, 1.0)

        self._save()
        return {"task": task_id, "status": "completed", "verified": camera_verified}

    # ═══════════════════════════════════════════════════════
    # كاميرا AI — تحليل مشهد 📹🧠
    # ═══════════════════════════════════════════════════════

    def process_camera_frame(self, camera_id: str, detected_faces: List[str] = None,
                             detected_objects: List[str] = None,
                             alerts: List[str] = None) -> Dict:
        """
        معالجة إطار كاميرا — يتلقى نتائج من vision_layer

        detected_faces: قائمة employee_ids المكتشفة
        detected_objects: قائمة أشياء مكتشفة
        alerts: تنبيهات سلامة
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return {"error": "Camera not found"}

        result = {"camera": camera_id, "actions": []}
        cam.last_frame_time = datetime.now().isoformat()
        cam.detections_today += 1

        # === Auto check-in/out من كاميرات المدخل/المخرج ===
        if detected_faces and cam.role in [CameraRole.ENTRANCE.value, CameraRole.EXIT.value]:
            for face_id in detected_faces:
                if face_id in self.employees:
                    if cam.role == CameraRole.ENTRANCE.value:
                        action = self.check_in(face_id, camera_id)
                        result["actions"].append({"type": "check_in", **action})
                    else:
                        action = self.check_out(face_id, camera_id)
                        result["actions"].append({"type": "check_out", **action})

        # === تنبيهات سلامة ===
        if alerts:
            cam.alerts_today += len(alerts)
            for alert in alerts:
                result["actions"].append({
                    "type": "safety_alert",
                    "alert": alert,
                    "camera": camera_id,
                    "location": cam.location,
                })
                memory.save_knowledge(
                    topic=f"⚠️ تنبيه سلامة: {cam.name}",
                    content=f"كاميرا {cam.name} ({cam.location}): {alert}",
                    source="safety_camera",
                )

        self._save()
        return result

    # ═══════════════════════════════════════════════════════
    # توسيع — أضف أي وظيفة جديدة بالمستقبل
    # ═══════════════════════════════════════════════════════

    def extend_feature(self, feature_name: str, config: Dict) -> Dict:
        """
        إضافة وظيفة جديدة للنظام

        مثال:
          extend_feature("inventory_tracking", {"cameras": ["cam_warehouse"], "items": ["box", "pallet"]})
          extend_feature("delivery_monitoring", {"cameras": ["cam_parking"], "vehicle_types": ["truck", "van"]})
          extend_feature("quality_check_line", {"cameras": ["cam_prod_1"], "defect_threshold": 0.05})
        """
        self.custom_features[feature_name] = {
            "config": config,
            "created_at": datetime.now().isoformat(),
            "active": True,
        }
        self._save()
        logger.info(f"🔌 Feature added: {feature_name}")
        return {"feature": feature_name, "status": "active"}

    # ═══════════════════════════════════════════════════════
    # تقارير 📊
    # ═══════════════════════════════════════════════════════

    def daily_report(self, date: str = None) -> Dict:
        """تقرير يومي"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        records = [a for a in self.attendance if a.date == date]
        present = [r for r in records if r.status in [EmployeeStatus.PRESENT.value, EmployeeStatus.LATE.value]]
        late = [r for r in records if r.status == EmployeeStatus.LATE.value]
        total_hours = sum(r.hours_worked for r in records)

        tasks_today = [t for t in self.tasks.values()
                      if t.created_at.startswith(date)]
        completed = [t for t in tasks_today if t.status == TaskStatus.COMPLETED.value]

        return {
            "date": date,
            "total_employees": len(self.employees),
            "present": len(present),
            "absent": len(self.employees) - len(present),
            "late": len(late),
            "total_hours": round(total_hours, 1),
            "tasks_created": len(tasks_today),
            "tasks_completed": len(completed),
            "camera_alerts": sum(c.alerts_today for c in self.cameras.values()),
        }

    def employee_report(self, employee_id: str) -> Dict:
        """تقرير موظف"""
        emp = self.employees.get(employee_id)
        if not emp:
            return {"error": "Employee not found"}

        records = [a for a in self.attendance if a.employee_id == employee_id]
        total_hours = sum(r.hours_worked for r in records)
        tasks = [t for t in self.tasks.values() if t.assigned_to == employee_id]
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED.value]

        return {
            "employee": emp.name_ar,
            "department": emp.department,
            "role": emp.role,
            "status": emp.status,
            "total_days": emp.total_days,
            "late_count": emp.late_count,
            "early_leaves": emp.early_leave_count,
            "total_hours": round(total_hours, 1),
            "tasks_assigned": len(tasks),
            "tasks_completed": len(completed),
            "performance": round(emp.performance_score, 2),
            "active_task": emp.active_task,
        }

    def get_summary(self) -> Dict:
        """ملخص النظام الكامل"""
        present_count = sum(1 for e in self.employees.values()
                          if e.status in [EmployeeStatus.PRESENT.value, EmployeeStatus.LATE.value])
        return {
            "employees": len(self.employees),
            "present": present_count,
            "cameras": len(self.cameras),
            "active_cameras": sum(1 for c in self.cameras.values() if c.active),
            "total_tasks": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING.value),
            "in_progress": sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS.value),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED.value),
            "custom_features": list(self.custom_features.keys()),
            "today": self.daily_report(),
        }


# Singleton
company = CompanyManager()


if __name__ == "__main__":
    print("🏢📹 Company Camera Management System\n")

    # إضافة موظفين
    company.add_employee("emp_001", "Hassan", "حسن", "هندسة", "مهندس",
                        skills=["welding", "electrical", "plc"])
    company.add_employee("emp_002", "Ali", "علي", "إنتاج", "مشغّل",
                        skills=["machine", "production", "quality"])
    company.add_employee("emp_003", "Zahra", "زهراء", "إدارة", "محاسبة",
                        skills=["accounting", "excel", "report"])
    company.add_employee("emp_004", "Ahmed", "أحمد", "مخزن", "مسؤول مخزون",
                        skills=["inventory", "logistics", "forklift"])

    # كاميرات
    company.add_camera("cam_entrance", "كاميرا المدخل", "rtsp://192.168.1.10:554",
                      location="البوابة الرئيسية", role=CameraRole.ENTRANCE.value)
    company.add_camera("cam_exit", "كاميرا المخرج", "rtsp://192.168.1.11:554",
                      location="البوابة الخلفية", role=CameraRole.EXIT.value)
    company.add_camera("cam_prod", "كاميرا الإنتاج", "rtsp://192.168.1.12:554",
                      location="خط الإنتاج 1", role=CameraRole.PRODUCTION.value)

    print(f"Employees: {len(company.employees)}")
    print(f"Cameras: {len(company.cameras)}")

    # حضور
    print("\n📋 Attendance:")
    for eid in ["emp_001", "emp_002", "emp_003", "emp_004"]:
        result = company.check_in(eid, "cam_entrance")
        print(f"  {result.get('employee', '?')}: {result.get('status', '?')}")

    # مهام
    print("\n📝 Tasks:")
    t1 = company.create_task("صيانة خط إنتاج 1", department="هندسة", priority=1)
    t2 = company.create_task("جرد مخزون شهري", department="مخزن", priority=2)
    t3 = company.create_task("تقرير مالي ربع سنوي", department="إدارة", priority=3)
    for t in [t1, t2, t3]:
        emp_name = company.employees.get(t.assigned_to, Employee("?","?","?")).name_ar
        print(f"  {t.title} → {emp_name} [{t.status}]")

    # تقرير
    print("\n📊 Daily Report:")
    report = company.daily_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ملخص
    print("\n🏢 Summary:")
    summary = company.get_summary()
    print(f"  Employees: {summary['employees']} ({summary['present']} present)")
    print(f"  Cameras: {summary['cameras']} ({summary['active_cameras']} active)")
    print(f"  Tasks: {summary['total_tasks']} ({summary['completed_tasks']} done)")
