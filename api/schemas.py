"""
API Schemas - Pydantic models for request/response validation
نماذج التحقق من البيانات
"""

from pydantic import BaseModel
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

class InvoiceCreateRequest(BaseModel):
    customer_name: str
    customer_id: str
    amount: float
    tax: float
    total: float
    items: List[Dict]
    notes: Optional[str] = ""


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
