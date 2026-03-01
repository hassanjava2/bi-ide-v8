"""
Model Evaluator - مقيّم النماذج

Evaluates models before deployment with threshold checking.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """نوع التقييم"""
    PRE_DEPLOY = "pre_deploy"
    PERIODIC = "periodic"
    BENCHMARK = "benchmark"
    A_B_TEST = "a_b_test"


class EvaluationStatus(Enum):
    """حالة التقييم"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class EvaluationResult:
    """نتيجة تقييم"""
    evaluation_id: str
    model_id: str
    evaluation_type: EvaluationType
    status: EvaluationStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_delta: float = 0.0
    passed_threshold: bool = False
    threshold_value: float = 0.02  # 2% improvement required
    evaluated_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    @property
    def improvement_percentage(self) -> float:
        """نسبة التحسن"""
        return round(self.improvement_delta * 100, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس"""
        return {
            "evaluation_id": self.evaluation_id,
            "model_id": self.model_id,
            "evaluation_type": self.evaluation_type.value,
            "status": self.status.value,
            "metrics": self.metrics,
            "improvement_delta": self.improvement_delta,
            "improvement_percentage": self.improvement_percentage,
            "passed_threshold": self.passed_threshold,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class ModelEvaluator:
    """
    مقيّم النماذج
    
    يقيم النماذج ويتحقق من تحسنها قبل النشر
    """
    
    def __init__(self, min_improvement_delta: float = 0.02):
        self.min_improvement_delta = min_improvement_delta
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.evaluations: Dict[str, EvaluationResult] = {}
        
    def set_baseline(self, model_name: str, metrics: Dict[str, float]):
        """تحديد خط الأساس للمقارنة"""
        self.baseline_metrics[model_name] = metrics.copy()
        logger.info(f"Set baseline for {model_name}: {metrics}")
    
    async def evaluate(
        self,
        model_id: str,
        model_name: str,
        metrics: Dict[str, float],
        evaluation_type: EvaluationType = EvaluationType.PRE_DEPLOY,
        threshold: float = None
    ) -> EvaluationResult:
        """
        تقييم نموذج
        
        Args:
            model_id: معرف النموذج
            model_name: اسم النموذج (للبحث عن خط الأساس)
            metrics: المقاييس الجديدة
            evaluation_type: نوع التقييم
            threshold: عتبة التحسن (اختياري)
            
        Returns:
            EvaluationResult: نتيجة التقييم
        """
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        threshold = threshold or self.min_improvement_delta
        
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_type=evaluation_type,
            status=EvaluationStatus.RUNNING,
            metrics=metrics.copy(),
            threshold_value=threshold
        )
        
        # Get baseline
        baseline = self.baseline_metrics.get(model_name, {})
        result.baseline_metrics = baseline.copy()
        
        # Calculate improvement
        if baseline and metrics:
            result.improvement_delta = self._calculate_improvement(baseline, metrics)
            result.passed_threshold = result.improvement_delta >= threshold
        else:
            # No baseline, assume first deployment
            result.passed_threshold = True
        
        # Determine status
        if result.passed_threshold:
            result.status = EvaluationStatus.PASSED
        else:
            result.status = EvaluationStatus.FAILED
        
        self.evaluations[evaluation_id] = result
        
        logger.info(
            f"Evaluation {evaluation_id}: {model_id} - "
            f"improvement={result.improvement_percentage}%, "
            f"passed={result.passed_threshold}"
        )
        
        return result
    
    def _calculate_improvement(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float]
    ) -> float:
        """حساب نسبة التحسن"""
        improvements = []
        
        # Key metrics to compare (higher is better)
        higher_is_better = ["accuracy", "f1", "precision", "recall", "bleu", "rouge"]
        
        # Key metrics where lower is better
        lower_is_better = ["loss", "perplexity", "error_rate"]
        
        for metric in higher_is_better:
            if metric in baseline and metric in current:
                base_val = baseline[metric]
                curr_val = current[metric]
                if base_val > 0:
                    improvement = (curr_val - base_val) / base_val
                    improvements.append(improvement)
        
        for metric in lower_is_better:
            if metric in baseline and metric in current:
                base_val = baseline[metric]
                curr_val = current[metric]
                if base_val > 0:
                    improvement = (base_val - curr_val) / base_val
                    improvements.append(improvement)
        
        if not improvements:
            return 0.0
        
        # Return average improvement
        return sum(improvements) / len(improvements)
    
    def should_deploy(self, evaluation_id: str) -> bool:
        """التحقق مما إذا كان يجب نشر النموذج"""
        if evaluation_id not in self.evaluations:
            return False
        
        result = self.evaluations[evaluation_id]
        return result.status == EvaluationStatus.PASSED and result.passed_threshold
    
    def get_evaluation(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """الحصول على تقييم محدد"""
        return self.evaluations.get(evaluation_id)
    
    def list_evaluations(
        self,
        model_id: str = None,
        status: EvaluationStatus = None,
        limit: int = 50
    ) -> List[EvaluationResult]:
        """قائمة التقييمات"""
        evaluations = list(self.evaluations.values())
        
        if model_id:
            evaluations = [e for e in evaluations if e.model_id == model_id]
        
        if status:
            evaluations = [e for e in evaluations if e.status == status]
        
        # Sort by date, newest first
        evaluations.sort(key=lambda e: e.evaluated_at, reverse=True)
        
        return evaluations[:limit]
    
    def get_improvement_report(self, model_name: str) -> Dict[str, Any]:
        """تقرير التحسن لنموذج"""
        baseline = self.baseline_metrics.get(model_name, {})
        
        # Find all evaluations for this model
        model_evaluations = [
            e for e in self.evaluations.values()
            if e.model_id.startswith(model_name)
        ]
        
        if not model_evaluations:
            return {
                "model_name": model_name,
                "has_baseline": bool(baseline),
                "evaluations_count": 0,
                "improvement": None
            }
        
        # Get latest evaluation
        latest = max(model_evaluations, key=lambda e: e.evaluated_at)
        
        return {
            "model_name": model_name,
            "has_baseline": bool(baseline),
            "baseline_metrics": baseline,
            "current_metrics": latest.metrics,
            "evaluations_count": len(model_evaluations),
            "improvement_delta": latest.improvement_delta,
            "improvement_percentage": latest.improvement_percentage,
            "passed_threshold": latest.passed_threshold,
        }
    
    def compare_models(
        self,
        model_id_1: str,
        model_id_2: str
    ) -> Dict[str, Any]:
        """مقارنة نموذجين"""
        evals_1 = [e for e in self.evaluations.values() if e.model_id == model_id_1]
        evals_2 = [e for e in self.evaluations.values() if e.model_id == model_id_2]
        
        if not evals_1 or not evals_2:
            return {"error": "Missing evaluations for one or both models"}
        
        latest_1 = max(evals_1, key=lambda e: e.evaluated_at)
        latest_2 = max(evals_2, key=lambda e: e.evaluated_at)
        
        comparison = {
            "model_1": model_id_1,
            "model_2": model_id_2,
            "metrics_comparison": {},
            "winner": None
        }
        
        # Compare each metric
        all_metrics = set(latest_1.metrics.keys()) | set(latest_2.metrics.keys())
        
        wins_1 = 0
        wins_2 = 0
        
        for metric in all_metrics:
            val_1 = latest_1.metrics.get(metric, 0)
            val_2 = latest_2.metrics.get(metric, 0)
            
            comparison["metrics_comparison"][metric] = {
                model_id_1: val_1,
                model_id_2: val_2,
                "diff": round(val_1 - val_2, 4),
                "winner": model_id_1 if val_1 > val_2 else model_id_2 if val_2 > val_1 else "tie"
            }
            
            if val_1 > val_2:
                wins_1 += 1
            elif val_2 > val_1:
                wins_2 += 1
        
        # Determine overall winner
        if wins_1 > wins_2:
            comparison["winner"] = model_id_1
        elif wins_2 > wins_1:
            comparison["winner"] = model_id_2
        else:
            comparison["winner"] = "tie"
        
        comparison["wins"] = {model_id_1: wins_1, model_id_2: wins_2}
        
        return comparison
