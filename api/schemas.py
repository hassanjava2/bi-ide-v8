"""
API Schemas - Pydantic models for request/response validation
نماذج التحقق من البيانات
"""

from pydantic import BaseModel, model_validator
from typing import Dict, List, Optional, Any


# ========== Council Schemas ==========

class CommandRequest(BaseModel):
    command: str
    alert_level: str = "GREEN"
    context: Optional[Dict] = None


class CouncilMessageRequest(BaseModel):
    message: str
    user_id: str = "president"
    alert_level: str = "GREEN"


# ========== IDE Schemas ==========

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


# ========== ERP Schemas ==========

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


# ========== Network Schemas ==========

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


# ========== Idea Ledger Schemas ==========

class IdeaLedgerUpdateRequest(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    summary: Optional[str] = None
    owner: Optional[str] = None
    priority: Optional[str] = None
    kpi: Optional[str] = None
    status: Optional[str] = None


# ========== Auth Schemas ==========

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
