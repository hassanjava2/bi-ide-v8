"""
Data Flywheel - دولاب البيانات
كل استخدام للنظام = بيانات تدريب جديدة — حلقة لا نهائية

الحلقة:
1. مستخدم يسأل سؤال
2. النظام يجاوب
3. المستخدم يقيّم (👍/👎) أو النظام يقيّم ذاتياً
4. الجواب الجيد → عينة تدريب إيجابية
5. الجواب السيء → يُصلح → عينة تدريب محسّنة
6. التدريب الليلي يستخدم كل العينات الجديدة
7. النموذج يتحسن ← الأجوبة تتحسن ← عينات أحسن ← ...
"""

import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """أنواع التقييم"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    SELF_EVAL_GOOD = "self_eval_good"
    SELF_EVAL_BAD = "self_eval_bad"
    CORRECTION = "correction"  # User provided better answer


class SampleQuality(Enum):
    """جودة العينة"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    REJECT = 1


@dataclass
class TrainingSample:
    """عينة تدريب واحدة"""
    sample_id: str
    query: str
    response: str
    context: Dict[str, Any] = field(default_factory=dict)
    feedback_type: Optional[FeedbackType] = None
    quality_score: SampleQuality = SampleQuality.ACCEPTABLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "response": self.response,
            "context": self.context,
            "feedback_type": self.feedback_type.value if self.feedback_type else None,
            "quality_score": self.quality_score.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        return cls(
            sample_id=data["sample_id"],
            query=data["query"],
            response=data["response"],
            context=data.get("context", {}),
            feedback_type=FeedbackType(data["feedback_type"]) if data.get("feedback_type") else None,
            quality_score=SampleQuality(data.get("quality_score", 3)),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id")
        )


class SelfEvaluator:
    """نظام تقييم ذاتي - يقيّم النظام نفسه"""
    
    def __init__(self):
        self.evaluation_criteria = {
            "completeness": 0.25,  # الجواب كامل؟
            "accuracy": 0.30,      # دقيق علمياً؟
            "relevance": 0.20,     # مرتبط بالسؤال؟
            "clarity": 0.15,       # واضح؟
            "safety": 0.10         # آمن (لا يضر)؟
        }
    
    async def evaluate(self, query: str, response: str, 
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """تقييم ذاتي للجودة"""
        
        scores = {}
        
        # Completeness: Check if answer addresses all parts of query
        scores["completeness"] = self._check_completeness(query, response)
        
        # Accuracy: Check for internal consistency and known facts
        scores["accuracy"] = await self._check_accuracy(response, context)
        
        # Relevance: Semantic similarity between query and response
        scores["relevance"] = self._check_relevance(query, response)
        
        # Clarity: Structure, formatting, readability
        scores["clarity"] = self._check_clarity(response)
        
        # Safety: Check for dangerous advice
        scores["safety"] = self._check_safety(response)
        
        # Calculate weighted score
        total_score = sum(scores[k] * self.evaluation_criteria[k] 
                         for k in scores)
        
        # Determine quality level
        if total_score >= 0.9:
            quality = SampleQuality.EXCELLENT
        elif total_score >= 0.75:
            quality = SampleQuality.GOOD
        elif total_score >= 0.6:
            quality = SampleQuality.ACCEPTABLE
        elif total_score >= 0.4:
            quality = SampleQuality.POOR
        else:
            quality = SampleQuality.REJECT
        
        return {
            "scores": scores,
            "total_score": total_score,
            "quality": quality,
            "feedback_type": FeedbackType.SELF_EVAL_GOOD if total_score >= 0.6 else FeedbackType.SELF_EVAL_BAD,
            "improvement_suggestions": self._generate_suggestions(scores)
        }
    
    def _check_completeness(self, query: str, response: str) -> float:
        """التحقق من اكتمال الجواب"""
        # Simple heuristic: length ratio and keyword coverage
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        coverage = len(query_words & response_words) / len(query_words) if query_words else 0
        length_score = min(1.0, len(response) / (len(query) * 2))
        
        return (coverage * 0.6 + length_score * 0.4)
    
    async def _check_accuracy(self, response: str, context: Dict) -> float:
        """التحقق من الدقة"""
        # Would integrate with knowledge base verification
        # For now, check for contradictions with context
        if not context:
            return 0.8  # Assume good if no context to contradict
        
        # Check for internal consistency
        # (Simplified - would need more sophisticated logic)
        return 0.85
    
    def _check_relevance(self, query: str, response: str) -> float:
        """التحقق من الصلة"""
        # Simple keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & response_words)
        return min(1.0, overlap / len(query_words))
    
    def _check_clarity(self, response: str) -> float:
        """التحقق من الوضوح"""
        # Check for structure (paragraphs, bullet points)
        score = 0.7
        
        if '\n' in response:
            score += 0.1  # Has structure
        if any(marker in response for marker in ['•', '-', '1.', '2.']):
            score += 0.1  # Has lists
        if len(response.split('.')) > 2:
            score += 0.1  # Has sentences
        
        return min(1.0, score)
    
    def _check_safety(self, response: str) -> float:
        """التحقق من الأمان"""
        dangerous_keywords = [
            "poison", "toxic", "explosive", "dangerous without warning",
            "harmful", "deadly", "unsafe"
        ]
        
        response_lower = response.lower()
        
        # Check if dangerous content without warning/safety info
        has_danger = any(kw in response_lower for kw in dangerous_keywords)
        has_safety_warning = any(w in response_lower for w in [
            "warning", "caution", "safety", "protective", "careful"
        ])
        
        if has_danger and not has_safety_warning:
            return 0.3  # Dangerous without warning
        elif has_danger and has_safety_warning:
            return 0.8  # Dangerous but with warnings
        else:
            return 1.0  # Safe content
    
    def _generate_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """توليد اقتراحات للتحسين"""
        suggestions = []
        
        if scores.get("completeness", 1) < 0.7:
            suggestions.append("Provide more comprehensive answer")
        if scores.get("accuracy", 1) < 0.8:
            suggestions.append("Verify technical details")
        if scores.get("relevance", 1) < 0.7:
            suggestions.append("Focus more on the specific question")
        if scores.get("clarity", 1) < 0.7:
            suggestions.append("Improve structure and formatting")
        if scores.get("safety", 1) < 0.8:
            suggestions.append("Add safety warnings where applicable")
        
        return suggestions


class DataFlywheel:
    """
    دولاب البيانات - يحول كل استخدام إلى بيانات تدريب
    """
    
    def __init__(self, storage_path: str = "learning_data/flywheel"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.samples_file = self.storage_path / "training_samples.jsonl"
        self.daily_stats_file = self.storage_path / "daily_stats.json"
        
        self.evaluator = SelfEvaluator()
        self.samples_buffer: List[TrainingSample] = []
        self.buffer_size = 100  # Flush every 100 samples
        
        self._stats = {
            "total_samples": 0,
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0,
            "rejected": 0,
            "by_feedback_type": {ft.value: 0 for ft in FeedbackType}
        }
        
        self._load_stats()
        
        logger.info(f"🔄 Data Flywheel initialized: {storage_path}")
    
    def _load_stats(self):
        """تحميل الإحصائيات"""
        if self.daily_stats_file.exists():
            try:
                with open(self.daily_stats_file, 'r') as f:
                    self._stats = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load stats: {e}")
    
    def _save_stats(self):
        """حفظ الإحصائيات"""
        with open(self.daily_stats_file, 'w') as f:
            json.dump(self._stats, f, indent=2)
    
    async def record_interaction(self, query: str, response: str, 
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                context: Optional[Dict] = None) -> TrainingSample:
        """
        تسجيل تفاعل جديد - يُقيّم ذاتياً فوراً
        """
        # Generate unique ID
        sample_id = hashlib.md5(
            f"{query}:{response}:{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]
        
        # Self-evaluate
        eval_result = await self.evaluator.evaluate(query, response, context or {})
        
        # Create sample
        sample = TrainingSample(
            sample_id=sample_id,
            query=query,
            response=response,
            context=context or {},
            feedback_type=eval_result["feedback_type"],
            quality_score=eval_result["quality"],
            metadata={
                "self_eval_scores": eval_result["scores"],
                "suggestions": eval_result["improvement_suggestions"]
            },
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to buffer
        self.samples_buffer.append(sample)
        
        # Update stats
        self._update_stats(sample)
        
        # Flush if buffer full
        if len(self.samples_buffer) >= self.buffer_size:
            await self._flush_buffer()
        
        logger.debug(f"✅ Recorded interaction: {sample_id} (quality: {sample.quality_score.name})")
        return sample
    
    async def add_user_feedback(self, sample_id: str, feedback: FeedbackType,
                               correction: Optional[str] = None) -> bool:
        """
        إضافة تقييم المستخدم - يُعطى أولوية أعلى من التقييم الذاتي
        """
        # Find sample in buffer or file
        sample = None
        for s in self.samples_buffer:
            if s.sample_id == sample_id:
                sample = s
                break
        
        if not sample:
            # Try to load from file (simplified - would need proper search)
            logger.warning(f"Sample {sample_id} not found in buffer")
            return False
        
        # Update with user feedback (higher priority)
        sample.feedback_type = feedback
        
        # Adjust quality based on user feedback
        if feedback == FeedbackType.THUMBS_UP:
            sample.quality_score = SampleQuality.EXCELLENT
        elif feedback == FeedbackType.THUMBS_DOWN:
            sample.quality_score = SampleQuality.POOR
        elif feedback == FeedbackType.CORRECTION and correction:
            sample.quality_score = SampleQuality.ACCEPTABLE
            sample.metadata["user_correction"] = correction
            sample.response = correction  # Use corrected version for training
        
        logger.info(f"👍 User feedback added: {sample_id} → {feedback.value}")
        return True
    
    def _update_stats(self, sample: TrainingSample):
        """تحديث الإحصائيات"""
        self._stats["total_samples"] += 1
        self._stats[sample.quality_score.name.lower()] += 1
        if sample.feedback_type:
            self._stats["by_feedback_type"][sample.feedback_type.value] += 1
    
    async def _flush_buffer(self):
        """حفظ العينات من الذاكرة للملف"""
        if not self.samples_buffer:
            return
        
        try:
            with open(self.samples_file, 'a', encoding='utf-8') as f:
                for sample in self.samples_buffer:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Flushed {len(self.samples_buffer)} samples to disk")
            self.samples_buffer = []
            self._save_stats()
            
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
    
    async def get_training_batch(self, min_quality: SampleQuality = SampleQuality.ACCEPTABLE,
                                 batch_size: int = 100) -> List[TrainingSample]:
        """
        الحصول على دفعة تدريب - للتدريب الليلي
        """
        # Flush any pending samples first
        await self._flush_buffer()
        
        samples = []
        
        if not self.samples_file.exists():
            return samples
        
        try:
            with open(self.samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        sample = TrainingSample.from_dict(data)
                        
                        # Filter by quality
                        if sample.quality_score.value >= min_quality.value:
                            samples.append(sample)
                        
                        if len(samples) >= batch_size:
                            break
                    except Exception as e:
                        continue
        
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
        
        return samples
    
    async def export_for_training(self, output_path: str, 
                                  format: str = "jsonl") -> int:
        """
        تصدير البيانات للتدريب
        """
        await self._flush_buffer()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        
        try:
            with open(self.samples_file, 'r', encoding='utf-8') as f_in:
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        try:
                            data = json.loads(line.strip())
                            sample = TrainingSample.from_dict(data)
                            
                            # Only export good samples
                            if sample.quality_score.value >= SampleQuality.GOOD.value:
                                # Format for training
                                training_record = {
                                    "instruction": sample.query,
                                    "input": "",
                                    "output": sample.response,
                                    "metadata": {
                                        "quality": sample.quality_score.name,
                                        "timestamp": sample.timestamp.isoformat()
                                    }
                                }
                                f_out.write(json.dumps(training_record, ensure_ascii=False) + '\n')
                                count += 1
                        except:
                            continue
            
            logger.info(f"📤 Exported {count} samples to {output_path}")
            return count
            
        except Exception as e:
            logger.error(f"Error exporting: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات"""
        return {
            **self._stats,
            "buffer_size": len(self.samples_buffer),
            "storage_path": str(self.storage_path),
            "file_size_mb": self.samples_file.stat().st_size / (1024*1024) if self.samples_file.exists() else 0
        }
    
    async def start_continuous_collection(self):
        """بدء جمع مستمر - يعمل في الخلفية"""
        logger.info("🔄 Data Flywheel continuous collection started")
        
        while True:
            try:
                # Flush buffer every 60 seconds
                await asyncio.sleep(60)
                await self._flush_buffer()
                
            except asyncio.CancelledError:
                await self._flush_buffer()
                break
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")


# Global instance
data_flywheel = DataFlywheel()
