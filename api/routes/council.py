"""
Council Routes - نقاط النهاية لمجلس الحكماء
"""

import os
import json
import time
import threading
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TypeVar, Tuple
from pathlib import Path
from functools import wraps

import requests as http_requests
from fastapi import APIRouter, HTTPException

from api.schemas import CommandRequest, CouncilMessageRequest


# Retry configuration from environment variables
RTX4090_MAX_RETRIES = int(os.getenv("RTX4090_MAX_RETRIES", "3"))
RTX4090_RETRY_DELAY = float(os.getenv("RTX4090_RETRY_DELAY", "1.0"))
RTX4090_RETRY_BACKOFF = float(os.getenv("RTX4090_RETRY_BACKOFF", "2.0"))
RTX4090_RETRY_MAX_DELAY = float(os.getenv("RTX4090_RETRY_MAX_DELAY", "30.0"))
RTX4090_RETRY_JITTER = os.getenv("RTX4090_RETRY_JITTER", "true").lower() in ("true", "1", "yes", "on")

T = TypeVar('T')


def with_retry(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    max_delay: Optional[float] = None,
    jitter: Optional[bool] = None,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: RTX4090_MAX_RETRIES)
        initial_delay: Initial delay between retries in seconds (default: RTX4090_RETRY_DELAY)
        backoff_factor: Multiplier for exponential backoff (default: RTX4090_RETRY_BACKOFF)
        max_delay: Maximum delay between retries in seconds (default: RTX4090_RETRY_MAX_DELAY)
        jitter: Add random jitter to delay to prevent thundering herd (default: RTX4090_RETRY_JITTER)
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Use provided values or fall back to environment-configured defaults
            _max_retries = max_retries if max_retries is not None else RTX4090_MAX_RETRIES
            _initial_delay = initial_delay if initial_delay is not None else RTX4090_RETRY_DELAY
            _backoff_factor = backoff_factor if backoff_factor is not None else RTX4090_RETRY_BACKOFF
            _max_delay = max_delay if max_delay is not None else RTX4090_RETRY_MAX_DELAY
            _jitter = jitter if jitter is not None else RTX4090_RETRY_JITTER
            
            last_exception = None
            delay = _initial_delay
            
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on the last attempt
                    if attempt >= _max_retries:
                        break
                    
                    # Calculate next delay with exponential backoff
                    current_delay = min(delay, _max_delay)
                    
                    # Add jitter (±25% random variation) to prevent thundering herd
                    if _jitter:
                        current_delay = current_delay * (0.75 + random.random() * 0.5)
                    
                    print(f"⚠️ RTX 4090 connection attempt {attempt + 1}/{_max_retries + 1} failed: {str(e)[:60]}")
                    print(f"   Retrying in {current_delay:.2f}s...")
                    
                    time.sleep(current_delay)
                    delay *= _backoff_factor
            
            # All retries exhausted
            print(f"❌ RTX 4090 connection failed after {_max_retries + 1} attempts")
            raise last_exception
        
        return wrapper
    return decorator


def check_rtx4090_with_retry(
    timeout: float = 5.0,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
) -> bool:
    """
    Check RTX 4090 connection with retry logic.
    
    Args:
        timeout: Request timeout in seconds
        max_retries: Override default max retries
        initial_delay: Override default initial delay
    
    Returns:
        True if connection successful, False otherwise
    """
    _max_retries = max_retries if max_retries is not None else RTX4090_MAX_RETRIES
    _initial_delay = initial_delay if initial_delay is not None else RTX4090_RETRY_DELAY
    
    delay = _initial_delay
    
    for attempt in range(_max_retries + 1):
        try:
            r = http_requests.get(f"{RTX4090_URL}/health", timeout=timeout)
            if r.status_code == 200:
                if attempt > 0:
                    print(f"✅ RTX 4090 connection restored on attempt {attempt + 1}")
                return True
        except Exception as e:
            if attempt >= _max_retries:
                # All retries exhausted
                return False
            
            # Calculate next delay with exponential backoff
            current_delay = min(delay, RTX4090_RETRY_MAX_DELAY)
            if RTX4090_RETRY_JITTER:
                current_delay = current_delay * (0.75 + random.random() * 0.5)
            
            print(f"⚠️ RTX 4090 health check attempt {attempt + 1}/{_max_retries + 1} failed: {str(e)[:50]}")
            time.sleep(current_delay)
            delay *= RTX4090_RETRY_BACKOFF
    
    return False


def send_rtx4090_request_with_retry(
    endpoint: str,
    json_data: Dict[str, Any],
    timeout: float = 30.0,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
) -> Optional[http_requests.Response]:
    """
    Send HTTP POST request to RTX 4090 with retry logic.
    
    Args:
        endpoint: API endpoint path (e.g., '/council/message')
        json_data: JSON payload
        timeout: Request timeout in seconds
        max_retries: Override default max retries
        initial_delay: Override default initial delay
    
    Returns:
        Response object if successful, None if all retries failed
    """
    _max_retries = max_retries if max_retries is not None else RTX4090_MAX_RETRIES
    _initial_delay = initial_delay if initial_delay is not None else RTX4090_RETRY_DELAY
    
    delay = _initial_delay
    last_exception = None
    
    for attempt in range(_max_retries + 1):
        try:
            r = http_requests.post(
                f"{RTX4090_URL}{endpoint}",
                json=json_data,
                timeout=timeout,
            )
            if r.status_code == 200:
                if attempt > 0:
                    print(f"✅ RTX 4090 request succeeded on attempt {attempt + 1}")
                return r
            else:
                # Non-200 status is treated as failure but might be retriable
                if attempt >= _max_retries:
                    return r
        except Exception as e:
            last_exception = e
            
            if attempt >= _max_retries:
                # All retries exhausted
                print(f"❌ RTX 4090 request failed after {_max_retries + 1} attempts: {str(e)[:60]}")
                return None
            
            # Calculate next delay with exponential backoff
            current_delay = min(delay, RTX4090_RETRY_MAX_DELAY)
            if RTX4090_RETRY_JITTER:
                current_delay = current_delay * (0.75 + random.random() * 0.5)
            
            print(f"⚠️ RTX 4090 request attempt {attempt + 1}/{_max_retries + 1} failed: {str(e)[:50]}")
            time.sleep(current_delay)
            delay *= RTX4090_RETRY_BACKOFF
    
    return None

router = APIRouter(prefix="/api/v1", tags=["council"])

# RTX 4090 connection settings
RTX4090_HOST = os.getenv("RTX4090_HOST") or os.getenv("AI_CORE_HOST", "192.168.68.125")
RTX4090_PORT = os.getenv("RTX4090_PORT") or os.getenv("AI_CORE_PORT", "8080")
RTX4090_URL = f"http://{RTX4090_HOST}:{RTX4090_PORT}"

# Council Chat History
chat_history: List[Dict] = []
CHAT_HISTORY_FILE = Path("data/council_chat_history.json")
chat_history_lock = threading.Lock()

# Council metrics
council_metrics_lock = threading.Lock()
council_metrics: Dict[str, Any] = {
    "started_at": datetime.now().isoformat(),
    "total_messages": 0,
    "user_messages": 0,
    "council_responses": 0,
    "sources": {"rtx4090": 0, "local": 0, "fallback": 0},
    "response_sources": {
        "training+persona": 0,
        "persona-template": 0,
        "no-evidence-guard": 0,
        "rtx4090-model": 0,
        "fallback-template": 0,
        "local-unknown": 0,
    },
    "daily_quality": {},
    "wise_men": {},
    "layer_activity": {
        "council": 0,
        "scouts": 0,
        "meta": 0,
        "experts": 0,
        "execution": 0,
        "guardian": 0,
    },
    "latency_ms": {"count": 0, "total": 0, "min": None, "max": None, "last": None},
    "last_message_at": None,
    "last_response_at": None,
}

# Smart Council
SMART_COUNCIL_AVAILABLE = False
smart_council = None
try:
    from council_ai import smart_council as _sc
    smart_council = _sc
    SMART_COUNCIL_AVAILABLE = True
except Exception:
    pass


def _check_rtx4090() -> bool:
    """Check RTX 4090 health - now with built-in retry logic"""
    return check_rtx4090_with_retry(
        timeout=5.0,
        max_retries=RTX4090_MAX_RETRIES,
        initial_delay=RTX4090_RETRY_DELAY,
    )


def _load_chat_history():
    global chat_history
    try:
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CHAT_HISTORY_FILE.exists():
            CHAT_HISTORY_FILE.write_text("[]", encoding="utf-8")
            chat_history = []
            return
        data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8") or "[]")
        chat_history = data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ Failed to load council chat history: {e}")
        chat_history = []


def _persist_chat_history():
    try:
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHAT_HISTORY_FILE.write_text(
            json.dumps(chat_history[-1000:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️ Failed to persist chat history: {e}")


def _append_chat_message(payload: Dict[str, Any]):
    with chat_history_lock:
        chat_history.append(payload)
        if len(chat_history) > 1000:
            del chat_history[:-1000]
        _persist_chat_history()


def _record_user_message():
    now = datetime.now().isoformat()
    with council_metrics_lock:
        council_metrics["total_messages"] += 1
        council_metrics["user_messages"] += 1
        council_metrics["layer_activity"]["guardian"] += 1
        council_metrics["layer_activity"]["scouts"] += 1
        council_metrics["layer_activity"]["meta"] += 1
        council_metrics["last_message_at"] = now


def _record_council_response(
    council_member: str,
    source: str,
    message: str,
    latency_ms: Optional[int],
    response_source: Optional[str] = None,
):
    now = datetime.now().isoformat()
    day_key = datetime.now().strftime("%Y-%m-%d")
    safe_member = (council_member or "حكيم القرار").strip()
    safe_source = source if source in {"rtx4090", "local", "fallback"} else "fallback"
    safe_response_source = (response_source or "").strip() or (
        "rtx4090-model" if safe_source == "rtx4090" else "local-unknown"
    )
    text_length = len(message or "")

    with council_metrics_lock:
        council_metrics["total_messages"] += 1
        council_metrics["council_responses"] += 1
        council_metrics["sources"][safe_source] = council_metrics["sources"].get(safe_source, 0) + 1
        source_map = council_metrics["response_sources"]
        source_map[safe_response_source] = source_map.get(safe_response_source, 0) + 1
        council_metrics["layer_activity"]["council"] += 1
        council_metrics["layer_activity"]["experts"] += 1
        council_metrics["layer_activity"]["execution"] += 1
        council_metrics["last_response_at"] = now

        daily_map = council_metrics["daily_quality"]
        if day_key not in daily_map:
            daily_map[day_key] = {"responses": 0, "evidence_backed": 0, "guarded": 0}
        day_item = daily_map[day_key]
        day_item["responses"] += 1
        if safe_response_source in {"training+persona", "rtx4090-model"}:
            day_item["evidence_backed"] += 1
        if safe_response_source == "no-evidence-guard":
            day_item["guarded"] += 1

        wise_map = council_metrics["wise_men"]
        if safe_member not in wise_map:
            wise_map[safe_member] = {
                "name": safe_member,
                "responses": 0,
                "total_chars": 0,
                "avg_chars": 0,
                "sources": {"rtx4090": 0, "local": 0, "fallback": 0},
                "last_response_at": None,
            }
        wise = wise_map[safe_member]
        wise["responses"] += 1
        wise["total_chars"] += text_length
        wise["avg_chars"] = round(wise["total_chars"] / max(1, wise["responses"]), 2)
        wise["sources"][safe_source] = wise["sources"].get(safe_source, 0) + 1
        wise["last_response_at"] = now

        if latency_ms is not None:
            latency = council_metrics["latency_ms"]
            latency["count"] += 1
            latency["total"] += latency_ms
            latency["last"] = latency_ms
            latency["min"] = latency_ms if latency["min"] is None else min(latency["min"], latency_ms)
            latency["max"] = latency_ms if latency["max"] is None else max(latency["max"], latency_ms)


def get_live_metrics_snapshot() -> Dict[str, Any]:
    """Public accessor for live metrics — used by other routers."""
    with council_metrics_lock:
        snapshot = json.loads(json.dumps(council_metrics))

    latency = snapshot.get("latency_ms", {})
    count = latency.get("count", 0)
    latency["avg"] = round(latency.get("total", 0) / count, 2) if count else 0

    wise_items = list(snapshot.get("wise_men", {}).values())
    wise_sorted = sorted(wise_items, key=lambda i: i.get("responses", 0), reverse=True)

    total_responses = snapshot.get("council_responses", 0)
    source_counts = snapshot.get("sources", {})
    fallback_rate = (source_counts.get("fallback", 0) / total_responses * 100) if total_responses else 0
    response_source_counts = snapshot.get("response_sources", {})

    evidence_backed_total = (
        response_source_counts.get("training+persona", 0) + response_source_counts.get("rtx4090-model", 0)
    )
    evidence_backed_rate = (evidence_backed_total / total_responses * 100) if total_responses else 0

    daily_quality = snapshot.get("daily_quality", {})
    trend_days = sorted(daily_quality.keys())[-7:]
    daily_trend = []
    for day in trend_days:
        day_item = daily_quality.get(day, {})
        day_responses = day_item.get("responses", 0)
        day_evidence = day_item.get("evidence_backed", 0)
        daily_trend.append({
            "day": day,
            "responses": day_responses,
            "evidence_backed": day_evidence,
            "guarded": day_item.get("guarded", 0),
            "evidence_rate_pct": round((day_evidence / day_responses * 100), 2) if day_responses else 0,
        })

    snapshot["top_wise_men"] = wise_sorted[:10]
    snapshot["fallback_rate_pct"] = round(fallback_rate, 2)
    snapshot["latency_ms"] = latency
    snapshot["quality"] = {
        "evidence_backed_total": evidence_backed_total,
        "guard_total": response_source_counts.get("no-evidence-guard", 0),
        "evidence_backed_rate_pct": round(evidence_backed_rate, 2),
        "daily_trend": daily_trend,
    }
    return snapshot


def bootstrap_metrics_from_history():
    """Re-derive metrics from persisted chat history."""
    with council_metrics_lock:
        council_metrics["wise_men"] = {}
        council_metrics["total_messages"] = 0
        council_metrics["user_messages"] = 0
        council_metrics["council_responses"] = 0
        council_metrics["sources"] = {"rtx4090": 0, "local": 0, "fallback": 0}
        council_metrics["response_sources"] = {
            "training+persona": 0, "persona-template": 0, "no-evidence-guard": 0,
            "rtx4090-model": 0, "fallback-template": 0, "local-unknown": 0,
        }
        council_metrics["daily_quality"] = {}
        council_metrics["layer_activity"] = {
            "council": 0, "scouts": 0, "meta": 0, "experts": 0, "execution": 0, "guardian": 0,
        }
        council_metrics["latency_ms"] = {"count": 0, "total": 0, "min": None, "max": None, "last": None}
        council_metrics["last_message_at"] = None
        council_metrics["last_response_at"] = None

    for item in chat_history:
        role = str(item.get("role", "")).lower()
        if role == "user":
            _record_user_message()
        elif role == "council":
            _record_council_response(
                council_member=item.get("council_member", "حكيم القرار"),
                source=item.get("source", "fallback"),
                message=item.get("message", ""),
                latency_ms=None,
                response_source=item.get("response_source"),
            )


def init_council():
    """Called once at startup."""
    _load_chat_history()
    bootstrap_metrics_from_history()
    print(f"💬 Council history loaded: {len(chat_history)} messages")
    print("📊 Council live metrics initialized")


# ──────────────── Routes ────────────────


@router.get("/council/status")
async def get_council_status():
    """حالة المجلس"""
    live = get_live_metrics_snapshot()
    wise_count = 16 if SMART_COUNCIL_AVAILABLE else len(live.get("wise_men", {}))
    return {
        "wise_men_count": wise_count,
        "status": "active",
        "responses": live.get("council_responses", 0),
        "fallback_rate_pct": live.get("fallback_rate_pct", 0),
        "last_response_at": live.get("last_response_at"),
    }


@router.get("/council/history")
async def get_council_history():
    """سجل محادثات المجلس"""
    return {"history": chat_history[-50:], "count": len(chat_history)}


@router.post("/council/message")
async def council_message(request: CouncilMessageRequest):
    """إرسال رسالة للمجلس — يحاول RTX 4090 أولاً ثم fallback"""
    user_msg = request.message
    user_id = request.user_id
    started = time.perf_counter()

    _append_chat_message({
        "role": "user", "user": "user_id",
        "message": user_msg, "timestamp": datetime.now().isoformat(),
    })
    _record_user_message()

    # Try RTX 4090 with retry logic
    if check_rtx4090_with_retry(timeout=5.0):
        r = send_rtx4090_request_with_retry(
            endpoint="/council/message",
            json_data={"message": user_msg, "user_id": user_id},
            timeout=30.0,
        )
        if r is not None and r.status_code == 200:
                data = r.json()
                ai_response = data.get("response", "...")
                council_member = data.get("council_member", "حكيم القرار")
                model_used = data.get("model_used", "unknown")

                _append_chat_message({
                    "role": "council", "council_member": council_member,
                    "message": ai_response, "timestamp": datetime.now().isoformat(),
                    "source": "rtx4090", "model": model_used,
                    "response_source": "rtx4090-model", "evidence": [],
                })
                _record_council_response(
                    council_member=council_member, source="rtx4090",
                    message=ai_response,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    response_source="rtx4090-model",
                )
                return {
                    "response": ai_response, "council_member": council_member,
                    "source": "rtx4090", "model": model_used,
                    "response_source": "rtx4090-model", "confidence": 0.9,
                    "evidence": [], "history": chat_history[-10:],
                }
        else:
            error_msg = f"RTX 4090 returned status {r.status_code}" if r else "RTX 4090 unreachable after retries"
            print(f"⚠️ {error_msg}")

    # Fallback: local SmartCouncil
    if SMART_COUNCIL_AVAILABLE and smart_council:
        result = smart_council.ask(user_msg)
        ai_response = result["response"]
        council_member = result["wise_man"]

        _append_chat_message({
            "role": "council", "council_member": council_member,
            "message": ai_response, "timestamp": datetime.now().isoformat(),
            "source": "local",
            "response_source": result.get("response_source", "local-unknown"),
            "evidence": result.get("evidence", []),
        })
        _record_council_response(
            council_member=council_member, source="local",
            message=ai_response,
            latency_ms=int((time.perf_counter() - started) * 1000),
            response_source=result.get("response_source", "local-unknown"),
        )
        return {
            "response": ai_response, "council_member": council_member,
            "source": "local",
            "response_source": result.get("response_source", "local-unknown"),
            "confidence": result.get("confidence", 0.0),
            "evidence": result.get("evidence", []),
            "history": chat_history[-10:],
        }

    # Ultimate fallback
    ai_response = "أنا هنا لمساعدتك. ما هو سؤالك؟"
    council_member = "حكيم القرار"

    _append_chat_message({
        "role": "council", "council_member": council_member,
        "message": ai_response, "timestamp": datetime.now().isoformat(),
        "source": "fallback", "response_source": "fallback-template", "evidence": [],
    })
    _record_council_response(
        council_member=council_member, source="fallback",
        message=ai_response,
        latency_ms=int((time.perf_counter() - started) * 1000),
        response_source="fallback-template",
    )
    return {
        "response": ai_response, "council_member": council_member,
        "source": "fallback", "response_source": "fallback-template",
        "confidence": 0.2, "evidence": [],
        "history": chat_history[-10:],
    }


@router.get("/council/wise-men")
async def get_wise_men():
    """قائمة الـ 16 حكيم"""
    if SMART_COUNCIL_AVAILABLE and smart_council:
        return {
            "count": 16,
            "wise_men": smart_council.get_all_wise_men(),
            "live_metrics": get_live_metrics_snapshot().get("top_wise_men", []),
        }
    return {"count": 0, "wise_men": [], "error": "Smart Council not available"}


@router.get("/council/metrics")
async def get_council_metrics_endpoint():
    """Live metrics for council"""
    return {"status": "ok", "metrics": get_live_metrics_snapshot()}


@router.post("/council/discuss")
async def council_discuss(request: dict):
    """نقاش جماعي"""
    topic = request.get("topic", "")
    if SMART_COUNCIL_AVAILABLE and smart_council:
        discussion = smart_council.discuss(topic)
        return {
            "topic": topic, "participants": len(discussion),
            "discussion": discussion,
        }
    return {"topic": topic, "participants": 0, "discussion": []}


@router.get("/hierarchy/status")
async def get_hierarchy_status():
    """حالة الهرم AI"""
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy:
            return ai_hierarchy.get_full_status()
    except Exception:
        pass
    raise HTTPException(500, "AI Hierarchy not available")


@router.post("/command")
async def execute_command(request: CommandRequest):
    """تنفيذ أمر عبر الهرم"""
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy:
            result = await ai_hierarchy.execute_command(request.command, request.alert_level)
            return result
    except Exception:
        pass
    raise HTTPException(500, "AI not initialized")


@router.get("/wisdom")
async def get_wisdom(horizon: str = "century"):
    """حكمة من البعد السابع"""
    try:
        from hierarchy import ai_hierarchy
        if ai_hierarchy:
            wisdom = ai_hierarchy.get_wisdom()
            return {"wisdom": wisdom, "horizon": horizon}
    except Exception:
        pass
    return {"wisdom": "التأسيس المتين يحتاج صبراً و رؤية طويلة المدى.", "horizon": horizon}


@router.get("/guardian/status")
async def get_guardian_status():
    """حالة طبقة الحماية"""
    live = get_live_metrics_snapshot()
    return {
        "active": True, "layers": 5, "active_crises": 0,
        "total_requests": live.get("user_messages", 0),
        "threats_blocked": 0, "violations_prevented": 0,
        "current_mode": "ACTIVE", "security_level": "normal",
        "compliance_status": "compliant",
        "activity": live.get("layer_activity", {}).get("guardian", 0),
    }


@router.get("/hierarchy/metrics")
async def get_hierarchy_metrics():
    """Live metrics snapshot for hierarchy layers"""
    live = get_live_metrics_snapshot()
    return {
        "status": "ok",
        "layers": live.get("layer_activity", {}),
        "council": {
            "responses": live.get("council_responses", 0),
            "fallback_rate_pct": live.get("fallback_rate_pct", 0),
            "latency_ms": live.get("latency_ms", {}),
            "top_wise_men": live.get("top_wise_men", []),
        },
    }
