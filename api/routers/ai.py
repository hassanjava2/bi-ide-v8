"""
روتر الذكاء الاصطناعي - AI Router

يوفر نقاط النهاية لخدمات AI الحقيقية فقط.
NO FAKE RESPONSES — per rules: ممنوع أي شي وهمي

عندما يتصل بالموديل الحقيقي (RTX 5090 LoRA) → يرجع نتائج حقيقية
عندما الموديل مو متوفر → يرجع خطأ واضح
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User

router = APIRouter(
    prefix="/ai",
    tags=["الذكاء الاصطناعي | AI"],
)


# نماذج Pydantic - Pydantic Models
class CodeGenerationRequest(BaseModel):
    """نموذج طلب توليد الكود"""
    prompt: str = Field(..., min_length=10, max_length=5000)
    language: str = Field(default="python")
    context: Optional[str] = None
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)


class CodeCompletionRequest(BaseModel):
    """نموذج طلب إكمال الكود"""
    code: str = Field(..., min_length=1, max_length=10000)
    cursor_position: int = Field(..., ge=0)
    language: str = Field(default="python")
    max_suggestions: int = Field(default=3, ge=1, le=10)


class CodeExplanationRequest(BaseModel):
    """نموذج طلب شرح الكود"""
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="python")
    detail_level: str = Field(default="medium")


class CodeReviewRequest(BaseModel):
    """نموذج طلب مراجعة الكود"""
    code: str = Field(..., min_length=1, max_length=20000)
    language: str = Field(default="python")
    review_focus: List[str] = Field(default=["all"])


# ─── AI Model Connection ─────────────────────────────────────────

async def _call_real_ai(prompt: str, max_tokens: int = 2048) -> Optional[str]:
    """
    محاولة الاتصال بالموديل الحقيقي (RTX 5090 LoRA أو Ollama)
    يرجع None إذا الموديل مو متوفر
    """
    try:
        import httpx
        # Try RTX 5090 first (Tailscale)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://100.104.35.44:8090/generate",
                json={"prompt": prompt, "max_tokens": max_tokens}
            )
            if response.status_code == 200:
                return response.json().get("response")
    except Exception:
        pass
    
    try:
        import httpx
        # Fallback: local Ollama
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return response.json().get("response")
    except Exception:
        pass
    
    return None


AI_UNAVAILABLE_MSG = "⚠️ AI غير متاح حالياً — الموديل المتدرب (RTX 5090) مطفي أو غير متصل. شغّل الجهاز وحاول مرة ثانية."


# ─── Endpoints ────────────────────────────────────────────────────

@router.post("/generate", summary="توليد الكود | Generate code")
async def generate_code(
    request: CodeGenerationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """توليد كود — يستخدم AI حقيقي فقط"""
    result = await _call_real_ai(
        f"Generate {request.language} code for: {request.prompt}",
        max_tokens=request.max_tokens
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=AI_UNAVAILABLE_MSG
        )
    
    return {
        "generated_code": result,
        "language": request.language,
        "source": "real-ai",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/complete", summary="إكمال الكود | Code completion")
async def complete_code(
    request: CodeCompletionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """إكمال الكود — يستخدم AI حقيقي فقط"""
    result = await _call_real_ai(
        f"Complete this {request.language} code at position {request.cursor_position}: {request.code}",
        max_tokens=512
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=AI_UNAVAILABLE_MSG
        )
    
    return {
        "completions": [{"text": result, "description": "AI completion"}],
        "language": request.language,
        "source": "real-ai"
    }


@router.post("/explain", summary="شرح الكود | Explain code")
async def explain_code(
    request: CodeExplanationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """شرح الكود — يستخدم AI حقيقي فقط"""
    result = await _call_real_ai(
        f"Explain this {request.language} code in {request.detail_level} detail: {request.code}",
        max_tokens=2048
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=AI_UNAVAILABLE_MSG
        )
    
    return {
        "explanation": result,
        "language": request.language,
        "detail_level": request.detail_level,
        "source": "real-ai"
    }


@router.post("/review", summary="مراجعة الكود | Review code")
async def review_code(
    request: CodeReviewRequest,
    current_user: User = Depends(get_current_active_user)
):
    """مراجعة الكود — يستخدم AI حقيقي فقط"""
    result = await _call_real_ai(
        f"Review this {request.language} code for {', '.join(request.review_focus)}: {request.code}",
        max_tokens=2048
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=AI_UNAVAILABLE_MSG
        )
    
    return {
        "review": result,
        "language": request.language,
        "source": "real-ai"
    }


@router.get("/status", summary="حالة AI | AI Status")
async def ai_status():
    """فحص حالة AI — هل الموديل متوفر؟"""
    result = await _call_real_ai("ping", max_tokens=10)
    
    return {
        "available": result is not None,
        "message": "AI متصل ✅" if result else AI_UNAVAILABLE_MSG,
        "timestamp": datetime.now().isoformat()
    }
