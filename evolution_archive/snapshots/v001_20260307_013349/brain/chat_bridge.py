#!/usr/bin/env python3
"""
chat_bridge.py — جسر الدردشة 🔗

يربط IDE Chat ← Router ← Capsules ← Memory

المسار:
  1. IDE يرسل سؤال
  2. chat_bridge يستقبل
  3. Router يختار الكبسولات
  4. كل كبسولة تجاوب (من الموديل المتدرب)
  5. Router يدمج الأجوبة
  6. Memory يحفظ كلشي للأبد
  7. الجواب يرجع للـ IDE

FastAPI server على port 8400
"""

import json
import logging
import os
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("chat_bridge")

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_ROOT = PROJECT_ROOT / "brain" / "capsules"

# ═══════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════

try:
    from brain.capsule_router import router as capsule_router, CapsuleResponse
    from brain.memory_system import memory
except ImportError:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from brain.capsule_router import router as capsule_router, CapsuleResponse
    from brain.memory_system import memory


class ChatBridge:
    """
    جسر الدردشة — يربط IDE بالكبسولات المتدربة
    """

    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}
        logger.info("🔗 ChatBridge initialized")

    def chat(self, message: str, session_id: str = None,
             mode: str = "fast") -> Dict:
        """
        نقطة الدخول الرئيسية — يستقبل سؤال ويرد

        Returns:
            {
                "response": "...",
                "capsules_used": ["code_python", ...],
                "mode": "fast|deep",
                "confidence": 0.85,
                "research": {...},  # فقط بالوضع العميق
                "needs_training": [...],
                "session_id": "...",
            }
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        # === 1. حفظ سؤال المستخدم ===
        memory.save_conversation(session_id, "user", message, mode=mode)

        # === 2. التوجيه ===
        decision = capsule_router.route(message, mode=mode)
        logger.info(f"🔗 [{mode}] Route: {decision.selected_capsules}")

        # === 3. تشغيل الكبسولات ===
        responses = []
        for capsule_id in decision.selected_capsules:
            resp = self._query_capsule(capsule_id, message, mode)
            if resp:
                responses.append(resp)

        # === 4. دمج الأجوبة ===
        if responses:
            final_response = capsule_router.combine_responses(message, responses)
        else:
            # Fallback: Ollama المحلي
            final_response = self._query_ollama(message)

        # === 5. حفظ الجواب للأبد ===
        elapsed = time.time() - start_time
        memory.save_conversation(
            session_id, "assistant", final_response,
            capsules_used=decision.selected_capsules,
            mode=mode,
            confidence=decision.confidence,
            metadata={"elapsed_ms": int(elapsed * 1000)}
        )

        result = {
            "response": final_response,
            "capsules_used": decision.selected_capsules,
            "mode": mode,
            "confidence": decision.confidence,
            "session_id": session_id,
            "elapsed_ms": int(elapsed * 1000),
        }

        if decision.research_data:
            result["research"] = decision.research_data
        if decision.needs_training:
            result["needs_training"] = decision.needs_training

        return result

    def _query_capsule(self, capsule_id: str, query: str,
                       mode: str = "fast") -> Optional[CapsuleResponse]:
        """تشغيل كبسولة متدربة"""
        model_dir = CAPSULES_ROOT / capsule_id / "model"

        if not model_dir.exists() or not any(model_dir.iterdir() if model_dir.exists() else []):
            logger.debug(f"  ⏭️ {capsule_id}: no trained model")
            return None

        start = time.time()
        try:
            # === تحميل الموديل المتدرب ===
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Prompt
            prompt = capsule_router.get_capsule_prompt(capsule_id, query)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            elapsed = (time.time() - start) * 1000

            return CapsuleResponse(
                capsule_id=capsule_id,
                response=response_text,
                confidence=0.8,
                time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"  ❌ {capsule_id} error: {e}")
            return None

    def _query_ollama(self, query: str) -> str:
        """Fallback: استخدام Ollama المحلي"""
        try:
            import urllib.request
            data = json.dumps({
                "model": "qwen2.5:1.5b",
                "prompt": query,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            return result.get("response", "...")
        except Exception as e:
            logger.warning(f"Ollama fallback failed: {e}")
            return f"الكبسولات لسه ما جاهزة (تدريب مستمر). السؤال: {query}"

    def get_history(self, session_id: str) -> List[Dict]:
        """جلب تاريخ المحادثة"""
        return memory.get_conversation(session_id)

    def search(self, query: str) -> List[Dict]:
        """بحث بكل المحادثات"""
        return memory.search_conversations(query)


# ═══════════════════════════════════════════════════════════
# FastAPI Server (ربط بالـ IDE)
# ═══════════════════════════════════════════════════════════

bridge = ChatBridge()


def create_api():
    """إنشاء FastAPI server"""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not installed — API disabled")
        return None

    app = FastAPI(title="BI-IDE Chat Bridge", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class ChatRequest(BaseModel):
        message: str
        session_id: str = None
        mode: str = "fast"  # fast | deep

    class SearchRequest(BaseModel):
        query: str
        limit: int = 10

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        return bridge.chat(req.message, req.session_id, req.mode)

    @app.get("/api/history/{session_id}")
    async def history(session_id: str):
        return {"messages": bridge.get_history(session_id)}

    @app.post("/api/search")
    async def search(req: SearchRequest):
        return {"results": bridge.search(req.query)}

    @app.get("/api/status")
    async def status():
        from brain.capsule_router import status as router_status
        return {
            "router": router_status(),
            "memory": memory.get_stats(),
        }

    return app


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BI-IDE Chat Bridge")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8400, help="Port")
    parser.add_argument("--chat", type=str, help="Send a message")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "deep"])
    args = parser.parse_args()

    if args.serve:
        app = create_api()
        if app:
            import uvicorn
            print("🔗 Chat Bridge API — http://localhost:8400")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        else:
            print("❌ FastAPI not installed: pip install fastapi uvicorn")
    elif args.chat:
        result = bridge.chat(args.chat, mode=args.mode)
        print(f"\n🧠 Mode: {result['mode']}")
        print(f"📦 Capsules: {result['capsules_used']}")
        print(f"🎯 Confidence: {result['confidence']:.0%}")
        print(f"⏱️ Time: {result['elapsed_ms']}ms")
        print(f"\n{result['response']}")
    else:
        print("🔗 Chat Bridge — BI-IDE")
        print("  --serve     Start API server")
        print("  --chat 'msg'  Send a message")
        print("  --mode fast|deep")
