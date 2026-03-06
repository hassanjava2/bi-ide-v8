"""
capsule_bus.py — ناقل الكبسولات (Inter-Capsule Communication Bus)

يتيح التواصل بين كل الكبسولات:
- كبسولة تسأل كبسولة 
- كبسولة تطلب معرفة → المسؤول يبلّغ الكشاف
- مشروع تعاوني → كل الكبسولات المتخصصة تشتغل مع بعض

البروتوكول:
1. ask(from, to, question) — سؤال مباشر  
2. broadcast(from, question) — سؤال للكل
3. request_knowledge(topic) — أبحث ما أعرف → كشاف يبحث
4. collaborate(project, capsule_ids) — مشروع تعاوني
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from brain.capsule import BrainCapsule
from brain.capsule_registry import CapsuleRegistry, capsule_registry

logger = logging.getLogger("capsule_bus")


class CapsuleBus:
    """
    ناقل التواصل — يربط كل الكبسولات مع بعض
    
    كل رسالة تمر عبر الناقل:
    capsule_a.ask("شنو أفضل تشفير؟") 
        → Bus يبحث بالسجل: "منو يعرف تشفير؟"
        → يوجّه السؤال لـ capsule_crypto
        → يرجع الجواب لـ capsule_a
    """
    
    def __init__(self, registry: Optional[CapsuleRegistry] = None):
        self.registry = registry or capsule_registry
        self.capsules: Dict[str, BrainCapsule] = {}  # loaded capsules
        self.message_log: List[Dict[str, Any]] = []
        self.pending_requests: List[Dict[str, Any]] = []  # knowledge requests for scouts
    
    # ─── Capsule Management ──────────────────────────────────
    
    def add_capsule(self, capsule: BrainCapsule):
        """إضافة كبسولة للناقل"""
        self.capsules[capsule.capsule_id] = capsule
        # Auto-register with registry
        self.registry.register(capsule.get_knowledge_summary())
        logger.info(f"🔗 [{capsule.name}] متصلة بالناقل")
    
    def remove_capsule(self, capsule_id: str):
        """إزالة كبسولة"""
        if capsule_id in self.capsules:
            name = self.capsules[capsule_id].name
            del self.capsules[capsule_id]
            self.registry.unregister(capsule_id)
            logger.info(f"🔌 [{name}] انفصلت عن الناقل")
    
    # ─── Direct Communication ────────────────────────────────
    
    def ask_capsule(
        self,
        from_id: str,
        to_id: str,
        question: str,
        inference_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        سؤال مباشر: كبسولة → كبسولة
        """
        if to_id not in self.capsules:
            return {"status": "error", "message": f"الكبسولة {to_id} غير متصلة"}
        
        target = self.capsules[to_id]
        response = target.ask(question, inference_fn)
        
        self._log_message(from_id, to_id, question, response)
        return response
    
    def ask_expert(
        self,
        from_id: str,
        topic: str,
        question: str,
        inference_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        سؤال ذكي: "أريد أحد يعرف عن [الموضوع]"
        الناقل يبحث بالسجل ويوجّه تلقائياً
        """
        # Find best expert from registry
        experts = self.registry.find_expert(topic)
        
        if not experts:
            # Nobody knows — create knowledge request
            request = {
                "type": "knowledge_request",
                "from_capsule": from_id,
                "topic": topic,
                "question": question,
                "timestamp": datetime.now().isoformat(),
            }
            self.pending_requests.append(request)
            logger.info(f"🔍 ما حد يعرف عن '{topic}' — طلب مرسل للكشاف")
            return {
                "status": "no_expert",
                "message": f"ما حد يعرف عن '{topic}' حالياً — طلب مرسل للكشاف",
                "request": request,
            }
        
        # Ask the best expert
        best_id, best_name, confidence = experts[0]
        
        if best_id in self.capsules:
            response = self.capsules[best_id].ask(question, inference_fn)
            response["routed_by"] = "capsule_bus"
            response["expert_confidence"] = confidence
            self._log_message(from_id, best_id, question, response)
            return response
        else:
            return {
                "status": "expert_offline",
                "expert_id": best_id,
                "expert_name": best_name,
                "confidence": confidence,
                "message": f"{best_name} يعرف بس مو متصل حالياً",
            }
    
    # ─── Broadcast ───────────────────────────────────────────
    
    def broadcast(
        self,
        from_id: str,
        question: str,
        inference_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        سؤال للكل — كل الكبسولات المتصلة تجاوب
        ممتاز لمشاريع تعاونية
        """
        responses = {}
        
        for cid, capsule in self.capsules.items():
            if cid == from_id:
                continue  # Don't ask yourself
            try:
                resp = capsule.ask(question, inference_fn)
                responses[cid] = resp
            except Exception as e:
                responses[cid] = {"status": "error", "error": str(e)}
        
        return {
            "type": "broadcast_response",
            "from": from_id,
            "question": question,
            "responses": responses,
            "respondents": len(responses),
            "answered": sum(1 for r in responses.values() if r.get("status") == "answered"),
            "timestamp": datetime.now().isoformat(),
        }
    
    # ─── Collaborative Project ───────────────────────────────
    
    def collaborate(
        self,
        project_name: str,
        task_description: str,
        capsule_ids: Optional[List[str]] = None,
        inference_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        مشروع تعاوني — كبسولات محددة تشتغل مع بعض
        
        مثال: "سوولي نظام بنكي"
        → architect يصمم + security يفحص + business يحسب
        """
        # If no specific capsules, use all
        participants = capsule_ids or list(self.capsules.keys())
        
        responses = {}
        for cid in participants:
            if cid in self.capsules:
                prompt = f"مشروع: {project_name}\nالمطلوب: {task_description}\nمساهمتك من تخصصك ({self.capsules[cid].specialty}):"
                resp = self.capsules[cid].ask(prompt, inference_fn)
                responses[cid] = {
                    "capsule_name": self.capsules[cid].name,
                    "specialty": self.capsules[cid].specialty,
                    "contribution": resp,
                }
        
        return {
            "project": project_name,
            "task": task_description,
            "participants": len(responses),
            "contributions": responses,
            "timestamp": datetime.now().isoformat(),
        }
    
    # ─── Knowledge Requests ──────────────────────────────────
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """طلبات المعرفة المعلّقة — للكشافة"""
        return list(self.pending_requests)
    
    def fulfill_request(self, topic: str, data: List[Dict], source: str):
        """الكشاف وجد المعلومات — يوزّعها على الكبسولات المحتاجة"""
        fulfilled = []
        remaining = []
        
        for req in self.pending_requests:
            if topic.lower() in req.get("topic", "").lower():
                # Find the requesting capsule and feed it data
                from_id = req.get("from_capsule")
                if from_id in self.capsules:
                    self.capsules[from_id].ingest_data(data, source)
                    self.capsules[from_id].register_knowledge(topic, 0.5)
                    fulfilled.append(req)
                    logger.info(f"✅ [{from_id}] حصل على بيانات '{topic}' من {source}")
            else:
                remaining.append(req)
        
        self.pending_requests = remaining
        return {"fulfilled": len(fulfilled), "remaining": len(remaining)}
    
    # ─── Logging ─────────────────────────────────────────────
    
    def _log_message(self, from_id: str, to_id: str, question: str, response: dict):
        self.message_log.append({
            "from": from_id,
            "to": to_id,
            "question": question[:100],
            "status": response.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
        })
        # Keep last 100 messages
        if len(self.message_log) > 100:
            self.message_log = self.message_log[-100:]
    
    # ─── Status ──────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "connected_capsules": len(self.capsules),
            "capsule_names": {cid: c.name for cid, c in self.capsules.items()},
            "pending_knowledge_requests": len(self.pending_requests),
            "messages_logged": len(self.message_log),
            "registry_stats": self.registry.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }


# ─── Singleton ───────────────────────────────────────────────
capsule_bus = CapsuleBus()
