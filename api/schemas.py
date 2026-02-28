"""
BI-IDE API Schemas - Pydantic v2 Models
نماذج التحقق من صحة طلبات واستجابات API
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Common Enums - التعدادات المشتركة
# =============================================================================

class UserRole(str, Enum):
    """أدوار المستخدمين"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class VoteType(str, Enum):
    """أنواع التصويت"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class UrgencyLevel(str, Enum):
    """مستويات الأولوية"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkerStatus(str, Enum):
    """حالات العامل"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AlertSeverity(str, Enum):
    """مستويات خطورة التنبيه"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogLevel(str, Enum):
    """مستويات السجلات"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelPreset(str, Enum):
    """إعدادات مسبقة للنماذج"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    ENTERPRISE = "enterprise"


# =============================================================================
# 1. Auth Models - نماذج المصادقة
# =============================================================================

class LoginRequest(BaseModel):
    """طلب تسجيل الدخول"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "username": "developer_01",
            "password": "SecurePass123!"
        }
    })
    
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="اسم المستخدم (3-50 حرف)",
        examples=["developer_01"]
    )
    password: str = Field(
        ...,
        min_length=8,
        description="كلمة المرور (8 أحرف على الأقل)",
        examples=["SecurePass123!"]
    )


class RegisterRequest(BaseModel):
    """طلب التسجيل"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "username": "new_user",
            "email": "user@example.com",
            "password": "SecurePass123!"
        }
    })
    
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="اسم المستخدم (3-50 حرف)",
        examples=["new_user"]
    )
    email: str = Field(
        ...,
        description="عنوان البريد الإلكتروني",
        examples=["user@example.com"]
    )
    password: str = Field(
        ...,
        min_length=8,
        description="كلمة المرور (8 أحرف على الأقل)",
        examples=["SecurePass123!"]
    )


class TokenResponse(BaseModel):
    """استجابة رمز الوصول"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2g...",
            "token_type": "bearer",
            "expires_in": 3600
        }
    })
    
    access_token: str = Field(
        ...,
        description="رمز الوصول JWT",
        examples=["eyJhbGciOiJIUzI1NiIs..."]
    )
    refresh_token: str = Field(
        ...,
        description="رمز التحديث",
        examples=["dGhpcyBpcyBhIHJlZnJlc2g..."]
    )
    token_type: str = Field(
        default="bearer",
        description="نوع الرمز",
        examples=["bearer"]
    )
    expires_in: int = Field(
        ...,
        description="مدة الصلاحية بالثواني",
        examples=[3600]
    )


class UserProfile(BaseModel):
    """ملف المستخدم الشخصي"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "usr_12345",
            "username": "developer_01",
            "role": "developer",
            "created_at": "2024-01-15T10:30:00Z"
        }
    })
    
    id: str = Field(
        ...,
        description="معرف المستخدم الفريد",
        examples=["usr_12345"]
    )
    username: str = Field(
        ...,
        description="اسم المستخدم",
        examples=["developer_01"]
    )
    role: UserRole = Field(
        ...,
        description="دور المستخدم",
        examples=["developer"]
    )
    created_at: datetime = Field(
        ...,
        description="تاريخ إنشاء الحساب",
        examples=["2024-01-15T10:30:00Z"]
    )


# =============================================================================
# 2. Council Models - نماذج المجلس
# =============================================================================

class CouncilQuery(BaseModel):
    """استعلام المجلس"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "أي لغة برمجة تناسب هذا المشروع؟",
            "context": "مشروع ويب متوسط الحجم",
            "urgency": "medium",
            "require_full_council": False
        }
    })
    
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="السؤال المراد طرحه على المجلس",
        examples=["أي لغة برمجة تناسب هذا المشروع؟"]
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="السياق الإضافي للسؤال (اختياري)",
        examples=["مشروع ويب متوسط الحجم"]
    )
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.MEDIUM,
        description="مستوى الأولوية",
        examples=["medium"]
    )
    require_full_council: bool = Field(
        default=False,
        description="هل يتطلب حضور جميع أعضاء المجلس؟",
        examples=[False]
    )


class CouncilVote(BaseModel):
    """تصويت عضو المجلس"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "member_id": "mem_ai_001",
            "vote": "approve"
        }
    })
    
    member_id: str = Field(
        ...,
        description="معرف عضو المجلس",
        examples=["mem_ai_001"]
    )
    vote: VoteType = Field(
        ...,
        description="نوع التصويت: approve/reject/abstain",
        examples=["approve"]
    )


class CouncilDecision(BaseModel):
    """قرار المجلس"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "dec_12345",
            "question": "أي لغة برمجة تناسب هذا المشروع؟",
            "decision": "Python",
            "confidence": 0.85,
            "votes": [
                {"member_id": "mem_ai_001", "vote": "approve"},
                {"member_id": "mem_ai_002", "vote": "approve"}
            ],
            "reasoning": "Python توفر توازنًا مثاليًا بين...",
            "created_at": "2024-01-15T10:30:00Z",
            "execution_time_ms": 1250
        }
    })
    
    id: str = Field(
        ...,
        description="معرف القرار الفريد",
        examples=["dec_12345"]
    )
    question: str = Field(
        ...,
        description="السؤال الأصلي",
        examples=["أي لغة برمجة تناسب هذا المشروع؟"]
    )
    decision: str = Field(
        ...,
        description="القرار النهائي",
        examples=["Python"]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="نسبة الثقة (0-1)",
        examples=[0.85]
    )
    votes: list[CouncilVote] = Field(
        ...,
        description="قائمة التصويتات",
    )
    reasoning: str = Field(
        ...,
        description="التفسير المنطقي للقرار",
        examples=["Python توفر توازنًا مثاليًا بين..."]
    )
    created_at: datetime = Field(
        ...,
        description="وقت اتخاذ القرار",
        examples=["2024-01-15T10:30:00Z"]
    )
    execution_time_ms: int = Field(
        ...,
        description="وقت التنفيذ بالمللي ثانية",
        examples=[1250]
    )


class CouncilMember(BaseModel):
    """عضو المجلس"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "mem_ai_001",
            "name": "CodeExpert",
            "role": "architect",
            "team": "backend",
            "specialization": "Python, Django, FastAPI",
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
    })
    
    id: str = Field(
        ...,
        description="معرف العضو الفريد",
        examples=["mem_ai_001"]
    )
    name: str = Field(
        ...,
        description="اسم العضو",
        examples=["CodeExpert"]
    )
    role: str = Field(
        ...,
        description="دور العضو في المجلس",
        examples=["architect"]
    )
    team: str = Field(
        ...,
        description="الفريق التابع له",
        examples=["backend"]
    )
    specialization: str = Field(
        ...,
        description="التخصصات التقنية",
        examples=["Python, Django, FastAPI"]
    )
    is_active: bool = Field(
        ...,
        description="هل العضو نشط؟",
        examples=[True]
    )
    created_at: datetime = Field(
        ...,
        description="تاريخ إنشاء العضو",
        examples=["2024-01-01T00:00:00Z"]
    )


# =============================================================================
# 3. Training Models - نماذج التدريب
# =============================================================================

class TrainingConfig(BaseModel):
    """إعدادات التدريب"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_preset": "balanced",
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "devices": ["cuda:0", "cuda:1"],
            "distributed": True
        }
    })
    
    model_preset: ModelPreset = Field(
        default=ModelPreset.BALANCED,
        description="الإعداد المسبق للنموذج",
        examples=["balanced"]
    )
    epochs: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="عدد فترات التدريب",
        examples=[10]
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=4096,
        description="حجم الدفعة",
        examples=[32]
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        le=1.0,
        description="معدل التعلم",
        examples=[0.001]
    )
    devices: list[str] = Field(
        default_factory=lambda: ["cuda:0"],
        description="قائمة الأجهزة للتدريب",
        examples=[["cuda:0", "cuda:1"]]
    )
    distributed: bool = Field(
        default=False,
        description="هل التدريب موزع؟",
        examples=[True]
    )
    
    @field_validator("devices")
    @classmethod
    def validate_devices(cls, v: list[str]) -> list[str]:
        """التحقق من صحة الأجهزة"""
        if not v:
            raise ValueError("يجب تحديد جهاز واحد على الأقل")
        return v


class TrainingStatus(BaseModel):
    """حالة التدريب"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "is_active": True,
            "device": "cuda:0",
            "epoch": 5,
            "total_epochs": 10,
            "loss": 0.234,
            "accuracy": 0.89,
            "throughput": 125.5,
            "gpu_utilization": 85.5,
            "estimated_time_remaining": "00:15:30"
        }
    })
    
    is_active: bool = Field(
        ...,
        description="هل التدريب نشط؟",
        examples=[True]
    )
    device: str = Field(
        ...,
        description="الجهاز المستخدم",
        examples=["cuda:0"]
    )
    epoch: int = Field(
        ...,
        ge=0,
        description="الفترة الحالية",
        examples=[5]
    )
    total_epochs: int = Field(
        ...,
        ge=1,
        description="إجمالي الفترات",
        examples=[10]
    )
    loss: float = Field(
        ...,
        ge=0,
        description="قيمة الخسارة",
        examples=[0.234]
    )
    accuracy: float = Field(
        ...,
        ge=0,
        le=1,
        description="الدقة (0-1)",
        examples=[0.89]
    )
    throughput: float = Field(
        ...,
        ge=0,
        description="معدل المعالجة (عينة/ثانية)",
        examples=[125.5]
    )
    gpu_utilization: float = Field(
        ...,
        ge=0,
        le=100,
        description="نسبة استخدام GPU (0-100)",
        examples=[85.5]
    )
    estimated_time_remaining: str = Field(
        ...,
        description="الوقت المتبقي المقدر (HH:MM:SS)",
        examples=["00:15:30"]
    )


class ModelInfo(BaseModel):
    """معلومات النموذج"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "model_v8_001",
            "name": "BI-IDE-Code",
            "version": "8.1.0",
            "params": 7000000000,
            "size_mb": 2800.5,
            "accuracy": 0.92,
            "trained_at": "2024-01-15T10:30:00Z"
        }
    })
    
    id: str = Field(
        ...,
        description="معرف النموذج الفريد",
        examples=["model_v8_001"]
    )
    name: str = Field(
        ...,
        description="اسم النموذج",
        examples=["BI-IDE-Code"]
    )
    version: str = Field(
        ...,
        description="إصدار النموذج",
        examples=["8.1.0"]
    )
    params: int = Field(
        ...,
        description="عدد المعلمات",
        examples=[7000000000]
    )
    size_mb: float = Field(
        ...,
        ge=0,
        description="حجم النموذج بالميجابايت",
        examples=[2800.5]
    )
    accuracy: float = Field(
        ...,
        ge=0,
        le=1,
        description="دقة النموذج",
        examples=[0.92]
    )
    trained_at: datetime = Field(
        ...,
        description="تاريخ التدريب",
        examples=["2024-01-15T10:30:00Z"]
    )


# =============================================================================
# 4. Worker Models - نماذج العمال
# =============================================================================

class WorkerInfo(BaseModel):
    """معلومات العامل"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "worker_id": "worker_node_01",
            "hostname": "gpu-server-01.local",
            "status": "busy",
            "labels": {"gpu": "rtx4090", "region": "us-east"},
            "cpu_cores": 32,
            "ram_gb": 128.0,
            "gpu_name": "NVIDIA RTX 4090",
            "gpu_vram_gb": 24.0,
            "last_seen": "2024-01-15T10:30:00Z"
        }
    })
    
    worker_id: str = Field(
        ...,
        description="معرف العامل الفريد",
        examples=["worker_node_01"]
    )
    hostname: str = Field(
        ...,
        description="اسم المضيف",
        examples=["gpu-server-01.local"]
    )
    status: WorkerStatus = Field(
        ...,
        description="حالة العامل",
        examples=["busy"]
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="التسميات المخصصة",
        examples=[{"gpu": "rtx4090", "region": "us-east"}]
    )
    cpu_cores: int = Field(
        ...,
        ge=1,
        description="عدد أنوية المعالج",
        examples=[32]
    )
    ram_gb: float = Field(
        ...,
        ge=0,
        description="الذاكرة العشوائية بالجيجابايت",
        examples=[128.0]
    )
    gpu_name: Optional[str] = Field(
        default=None,
        description="اسم GPU",
        examples=["NVIDIA RTX 4090"]
    )
    gpu_vram_gb: Optional[float] = Field(
        default=None,
        ge=0,
        description="ذاكرة GPU بالجيجابايت",
        examples=[24.0]
    )
    last_seen: datetime = Field(
        ...,
        description="آخر مرة تم رؤية العامل",
        examples=["2024-01-15T10:30:00Z"]
    )


# =============================================================================
# 5. Monitoring Models - نماذج المراقبة
# =============================================================================

class SystemResources(BaseModel):
    """موارد النظام"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "workers": ["worker_node_01", "worker_node_02"],
            "total_cpu": 64,
            "total_ram": 256.0,
            "total_gpu_vram": 48.0
        }
    })
    
    workers: list[str] = Field(
        ...,
        description="قائمة معرفات العمال",
        examples=[["worker_node_01", "worker_node_02"]]
    )
    total_cpu: int = Field(
        ...,
        ge=0,
        description="إجمالي أنوية المعالج",
        examples=[64]
    )
    total_ram: float = Field(
        ...,
        ge=0,
        description="إجمالي الذاكرة العشوائية بالجيجابايت",
        examples=[256.0]
    )
    total_gpu_vram: Optional[float] = Field(
        default=None,
        ge=0,
        description="إجمالي ذاكرة GPU بالجيجابايت",
        examples=[48.0]
    )


class Alert(BaseModel):
    """التنبيه"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "alert_001",
            "severity": "warning",
            "source": "worker_node_01",
            "message": "استخدام GPU مرتفع (>95%)",
            "timestamp": "2024-01-15T10:30:00Z",
            "resolved": False
        }
    })
    
    id: str = Field(
        ...,
        description="معرف التنبيه الفريد",
        examples=["alert_001"]
    )
    severity: AlertSeverity = Field(
        ...,
        description="مستوى الخطورة",
        examples=["warning"]
    )
    source: str = Field(
        ...,
        description="مصدر التنبيه",
        examples=["worker_node_01"]
    )
    message: str = Field(
        ...,
        description="رسالة التنبيه",
        examples=["استخدام GPU مرتفع (>95%)"]
    )
    timestamp: datetime = Field(
        ...,
        description="وقت التنبيه",
        examples=["2024-01-15T10:30:00Z"]
    )
    resolved: bool = Field(
        default=False,
        description="هل تم حل التنبيه؟",
        examples=[False]
    )


class LogEntry(BaseModel):
    """إدخال السجل"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "timestamp": "2024-01-15T10:30:00Z",
            "level": "info",
            "source": "api_server",
            "message": "تم بدء التدريب بنجاح"
        }
    })
    
    timestamp: datetime = Field(
        ...,
        description="وقت الإدخال",
        examples=["2024-01-15T10:30:00Z"]
    )
    level: LogLevel = Field(
        ...,
        description="مستوى السجل",
        examples=["info"]
    )
    source: str = Field(
        ...,
        description="مصدر السجل",
        examples=["api_server"]
    )
    message: str = Field(
        ...,
        description="رسالة السجل",
        examples=["تم بدء التدريب بنجاح"]
    )


# =============================================================================
# 6. AI Models - نماذج الذكاء الاصطناعي
# =============================================================================

class GenerateRequest(BaseModel):
    """طلب التوليد"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "اكتب دالة لحساب المتوسط المتحرك",
            "context": "Python للتحليل المالي",
            "max_tokens": 500,
            "temperature": 0.7
        }
    })
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="النص المطلوب توليد الكود له",
        examples=["اكتب دالة لحساب المتوسط المتحرك"]
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="السياق الإضافي",
        examples=["Python للتحليل المالي"]
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=8000,
        description="الحد الأقصى للرموز",
        examples=[500]
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="درجة العشوائية (0-2)",
        examples=[0.7]
    )


class GenerateResponse(BaseModel):
    """استجابة التوليد"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "generated_code": "def moving_average(data, window):\\n    ...",
            "tokens_used": 245,
            "model_used": "BI-IDE-Code-v8"
        }
    })
    
    generated_code: str = Field(
        ...,
        description="الكود المُولد",
        examples=["def moving_average(data, window):\\n    ..."]
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="عدد الرموز المستخدمة",
        examples=[245]
    )
    model_used: str = Field(
        ...,
        description="النموذج المستخدم",
        examples=["BI-IDE-Code-v8"]
    )


class CodeCompletionRequest(BaseModel):
    """طلب إكمال الكود"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "code_prefix": "def calculate_sum(",
            "code_suffix": "\\n    return result",
            "language": "python"
        }
    })
    
    code_prefix: str = Field(
        ...,
        description="الكود قبل نقطة الإكمال",
        examples=["def calculate_sum("]
    )
    code_suffix: Optional[str] = Field(
        default=None,
        description="الكود بعد نقطة الإكمال",
        examples=["\\n    return result"]
    )
    language: str = Field(
        ...,
        description="لغة البرمجة",
        examples=["python"]
    )


# =============================================================================
# 7. ERP Models - نماذج تخطيط موارد المؤسسة
# =============================================================================

class Invoice(BaseModel):
    """الفاتورة"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "inv_001",
            "amount": 5000.0,
            "customer": "شركة التقنية المتقدمة",
            "status": "paid",
            "created_at": "2024-01-15T10:30:00Z"
        }
    })
    
    id: str = Field(
        ...,
        description="معرف الفاتورة الفريد",
        examples=["inv_001"]
    )
    amount: float = Field(
        ...,
        gt=0,
        description="المبلغ",
        examples=[5000.0]
    )
    customer: str = Field(
        ...,
        description="اسم العميل",
        examples=["شركة التقنية المتقدمة"]
    )
    status: Literal["pending", "paid", "overdue", "cancelled"] = Field(
        ...,
        description="حالة الفاتورة",
        examples=["paid"]
    )
    created_at: datetime = Field(
        ...,
        description="تاريخ الإنشاء",
        examples=["2024-01-15T10:30:00Z"]
    )


class BusinessReport(BaseModel):
    """التقرير التجاري"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "trends": ["زيادة الطلب على خدمات AI"],
            "optimizations": ["تقليل استهلاك GPU بنسبة 20%"],
            "forecast": "نمو بنسبة 30% للربع القادم",
            "recommendations": ["زيادة عدد العمال", "تحديث النماذج"]
        }
    })
    
    trends: list[str] = Field(
        ...,
        description="الاتجاهات الحالية",
        examples=[["زيادة الطلب على خدمات AI"]]
    )
    optimizations: list[str] = Field(
        ...,
        description="التحسينات المقترحة",
        examples=[["تقليل استهلاك GPU بنسبة 20%"]]
    )
    forecast: str = Field(
        ...,
        description="التوقعات",
        examples=["نمو بنسبة 30% للربع القادم"]
    )
    recommendations: list[str] = Field(
        ...,
        description="التوصيات",
        examples=[["زيادة عدد العمال", "تحديث النماذج"]]
    )


# =============================================================================
# 8. Common Models - النماذج المشتركة
# =============================================================================

class ErrorResponse(BaseModel):
    """استجابة الخطأ"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "validation_error",
            "message": "فشل التحقق من صحة البيانات",
            "details": {"username": "يجب أن يكون 3 أحرف على الأقل"}
        }
    })
    
    error: str = Field(
        ...,
        description="نوع الخطأ",
        examples=["validation_error"]
    )
    message: str = Field(
        ...,
        description="رسالة الخطأ",
        examples=["فشل التحقق من صحة البيانات"]
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="تفاصيل إضافية عن الخطأ",
        examples=[{"username": "يجب أن يكون 3 أحرف على الأقل"}]
    )


class PaginatedResponse(BaseModel):
    """الاستجابة المقسمة"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [],
            "total": 100,
            "page": 1,
            "page_size": 20
        }
    })
    
    items: list[Any] = Field(
        ...,
        description="العناصر",
        examples=[[]]
    )
    total: int = Field(
        ...,
        ge=0,
        description="إجمالي العناصر",
        examples=[100]
    )
    page: int = Field(
        ...,
        ge=1,
        description="الصفحة الحالية",
        examples=[1]
    )
    page_size: int = Field(
        ...,
        ge=1,
        le=1000,
        description="حجم الصفحة",
        examples=[20]
    )


# =============================================================================
# Legacy Schemas — backward compatibility with api/routes/*
# =============================================================================
class CommandRequest(BaseModel):
    command: str
    alert_level: str = "GREEN"
    context: Optional[Dict] = None



class CouncilMessageRequest(BaseModel):
    message: str
    user_id: str = "president"
    alert_level: str = "GREEN"



class CodeSuggestionRequest(BaseModel):
    code: str
    cursor_position: int
    language: str
    file_path: str



class CodeAnalysisRequest(BaseModel):
    code: str
    language: str
    file_path: str



class RefactorSuggestRequest(BaseModel):
    code: str
    language: str
    file_path: str



class TestGenerateRequest(BaseModel):
    code: str
    language: str
    file_path: str



class SymbolDocumentationRequest(BaseModel):
    code: str
    language: str
    file_path: str
    symbol: Optional[str] = None



class TerminalCommandRequest(BaseModel):
    session_id: str
    command: str



class TerminalSessionStartRequest(BaseModel):
    cwd: Optional[str] = None



class GitCommitRequest(BaseModel):
    message: str
    stage_all: bool = True



class GitSyncRequest(BaseModel):
    remote: str = "origin"
    branch: Optional[str] = None



class DebugStartRequest(BaseModel):
    file_path: str
    breakpoints: List[int] = []



class DebugBreakpointRequest(BaseModel):
    session_id: str
    file_path: str
    line: int



class DebugCommandRequest(BaseModel):
    session_id: str
    command: str



class DebugStopRequest(BaseModel):
    session_id: str



class InvoiceItem(BaseModel):
    product_id: Optional[str] = None
    # Accept both legacy keys (name/price) and canonical keys (description/unit_price)
    description: Optional[str] = None
    name: Optional[str] = None
    quantity: int
    unit_price: Optional[float] = None
    price: Optional[float] = None

    @model_validator(mode="after")
    def _normalize_legacy_fields(self):
        if self.description is None and self.name:
            self.description = self.name
        if self.unit_price is None and self.price is not None:
            self.unit_price = self.price

        if not self.description:
            raise ValueError("InvoiceItem requires description or name")
        if self.unit_price is None:
            raise ValueError("InvoiceItem requires unit_price or price")
        return self



class InvoiceCreateRequest(BaseModel):
    customer_name: str
    customer_id: str
    invoice_number: Optional[str] = None
    amount: Optional[float] = 0
    subtotal: Optional[float] = 0
    tax: Optional[float] = 0
    tax_amount: Optional[float] = 0
    total: float
    items: List[InvoiceItem] = []
    notes: Optional[str] = ""
    due_date: Optional[str] = None



class TransactionRequest(BaseModel):
    debit_account_id: str
    credit_account_id: str
    amount: float
    description: str
    reference: Optional[str] = ""



class StockAdjustmentRequest(BaseModel):
    product_id: str
    quantity_change: int
    reason: str
    reference: Optional[str] = ""



class PayrollRequest(BaseModel):
    employee_id: str
    month: int
    year: int
    overtime_hours: Optional[float] = 0
    deductions: Optional[Dict[str, float]] = None



class CustomerCreateRequest(BaseModel):
    customer_code: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = ""
    address: Optional[str] = ""
    customer_type: Optional[str] = "regular"
    credit_limit: Optional[float] = 0



class SpecializationExpandRequest(BaseModel):
    parent_id: str
    name: str
    description: str = ""



class WorkerRegisterRequest(BaseModel):
    worker_id: str
    hostname: str
    capabilities: Dict[str, Any] = {}



class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    status: str = "online"
    capabilities: Dict[str, Any] = {}



class TrainingTaskCreateRequest(BaseModel):
    topic: str
    node_id: Optional[str] = None
    priority: int = 5



class TrainingTaskClaimRequest(BaseModel):
    worker_id: str



class TrainingTaskCompleteRequest(BaseModel):
    task_id: str
    worker_id: str
    metrics: Dict[str, Any] = {}
    artifact_name: Optional[str] = None
    artifact_payload: Optional[Dict[str, Any]] = None



class DualThoughtRequest(BaseModel):
    node_id: str
    prompt: str



class IdeaLedgerUpdateRequest(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    summary: Optional[str] = None
    owner: Optional[str] = None
    priority: Optional[str] = None
    kpi: Optional[str] = None
    status: Optional[str] = None



class RefreshTokenRequest(BaseModel):
    refresh_token: str



class RefreshTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int



# ═══════════════════════════════════════════════════════════════════════════════
# Rebuild all models (required when using 'from __future__ import annotations')
# ═══════════════════════════════════════════════════════════════════════════════
LoginRequest.model_rebuild()
RegisterRequest.model_rebuild()
TokenResponse.model_rebuild()
UserProfile.model_rebuild()
CouncilQuery.model_rebuild()
CouncilVote.model_rebuild()
CouncilDecision.model_rebuild()
CouncilMember.model_rebuild()
TrainingConfig.model_rebuild()
TrainingStatus.model_rebuild()
ModelInfo.model_rebuild()
WorkerInfo.model_rebuild()
SystemResources.model_rebuild()
Alert.model_rebuild()
LogEntry.model_rebuild()
GenerateRequest.model_rebuild()
GenerateResponse.model_rebuild()
CodeCompletionRequest.model_rebuild()
Invoice.model_rebuild()
BusinessReport.model_rebuild()
ErrorResponse.model_rebuild()
PaginatedResponse.model_rebuild()
CommandRequest.model_rebuild()
CouncilMessageRequest.model_rebuild()
CodeSuggestionRequest.model_rebuild()
CodeAnalysisRequest.model_rebuild()
RefactorSuggestRequest.model_rebuild()
TestGenerateRequest.model_rebuild()
SymbolDocumentationRequest.model_rebuild()
TerminalCommandRequest.model_rebuild()
TerminalSessionStartRequest.model_rebuild()
GitCommitRequest.model_rebuild()
GitSyncRequest.model_rebuild()
DebugStartRequest.model_rebuild()
DebugBreakpointRequest.model_rebuild()
DebugCommandRequest.model_rebuild()
DebugStopRequest.model_rebuild()
InvoiceItem.model_rebuild()
InvoiceCreateRequest.model_rebuild()
TransactionRequest.model_rebuild()
StockAdjustmentRequest.model_rebuild()
PayrollRequest.model_rebuild()
CustomerCreateRequest.model_rebuild()
SpecializationExpandRequest.model_rebuild()
WorkerRegisterRequest.model_rebuild()
WorkerHeartbeatRequest.model_rebuild()
TrainingTaskCreateRequest.model_rebuild()
TrainingTaskClaimRequest.model_rebuild()
TrainingTaskCompleteRequest.model_rebuild()
DualThoughtRequest.model_rebuild()
IdeaLedgerUpdateRequest.model_rebuild()
RefreshTokenRequest.model_rebuild()
RefreshTokenResponse.model_rebuild()
