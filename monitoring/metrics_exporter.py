"""
مصدّر المقاييس - Metrics Exporter
=================================
تصدير المقاييس لـ Prometheus و Grafana
Export metrics for Prometheus and Grafana
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Counter
from collections import defaultdict
import re

from aiohttp import web

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """أنواع المقاييس - Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """قيمة المقياس - Metric value"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class MetricDefinition:
    """تعريف المقياس - Metric definition"""
    name: str
    type: MetricType
    description: str
    unit: Optional[str] = None
    label_names: List[str] = field(default_factory=list)


class MetricsExporter:
    """
    مصدّر المقاييس
    Metrics exporter
    
    يدعم تنسيق Prometheus للجمع والتصور في Grafana
    Supports Prometheus format for collection and visualization in Grafana
    
    Endpoint: /metrics
    """
    
    def __init__(
        self,
        app_name: str = "bi_ide",
        prefix: str = "bi_ide",
        host: str = "0.0.0.0",
        port: int = 9090
    ):
        """
        تهيئة مصدّر المقاييس
        Initialize metrics exporter
        
        Args:
            app_name: اسم التطبيق
            prefix: بادئة المقاييس
            host: عنوان الاستماع
            port: منفذ الخادم
        """
        self.app_name = app_name
        self.prefix = prefix
        self.host = host
        self.port = port
        
        self._definitions: Dict[str, MetricDefinition] = {}
        self._gauges: Dict[str, Dict[str, MetricValue]] = defaultdict(dict)
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._summaries: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        self._lock = asyncio.Lock()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        
        # تعريف المقاييس الافتراضية
        self._define_default_metrics()
    
    def _define_default_metrics(self) -> None:
        """تعريف المقاييس الافتراضية - Define default metrics"""
        defaults = [
            MetricDefinition(
                name="request_count",
                type=MetricType.COUNTER,
                description="Total number of requests",
                label_names=["method", "endpoint", "status"]
            ),
            MetricDefinition(
                name="request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                label_names=["method", "endpoint"]
            ),
            MetricDefinition(
                name="training_loss",
                type=MetricType.GAUGE,
                description="Current training loss",
                label_names=["run_id", "model"]
            ),
            MetricDefinition(
                name="training_accuracy",
                type=MetricType.GAUGE,
                description="Current training accuracy",
                label_names=["run_id", "model"]
            ),
            MetricDefinition(
                name="gpu_utilization_percent",
                type=MetricType.GAUGE,
                description="GPU utilization percentage",
                label_names=["worker_id", "gpu_index"]
            ),
            MetricDefinition(
                name="gpu_memory_used_bytes",
                type=MetricType.GAUGE,
                description="GPU memory used in bytes",
                label_names=["worker_id", "gpu_index"]
            ),
            MetricDefinition(
                name="gpu_temperature_celsius",
                type=MetricType.GAUGE,
                description="GPU temperature in celsius",
                label_names=["worker_id", "gpu_index"]
            ),
            MetricDefinition(
                name="cpu_utilization_percent",
                type=MetricType.GAUGE,
                description="CPU utilization percentage",
                label_names=["worker_id"]
            ),
            MetricDefinition(
                name="memory_utilization_percent",
                type=MetricType.GAUGE,
                description="Memory utilization percentage",
                label_names=["worker_id"]
            ),
            MetricDefinition(
                name="disk_utilization_percent",
                type=MetricType.GAUGE,
                description="Disk utilization percentage",
                label_names=["worker_id", "mount"]
            ),
            MetricDefinition(
                name="active_training_runs",
                type=MetricType.GAUGE,
                description="Number of active training runs",
                label_names=[]
            ),
            MetricDefinition(
                name="alert_count",
                type=MetricType.COUNTER,
                description="Total number of alerts",
                label_names=["level"]
            ),
            MetricDefinition(
                name="websocket_connections",
                type=MetricType.GAUGE,
                description="Number of active WebSocket connections",
                label_names=[]
            )
        ]
        
        for metric in defaults:
            self._definitions[metric.name] = metric
    
    async def start(self) -> None:
        """بدء خادم المقاييس - Start metrics server"""
        self._app = web.Application()
        self._app.router.add_get('/metrics', self._handle_metrics)
        self._app.router.add_get('/health', self._handle_health)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Metrics exporter started on http://{self.host}:{self.port}/metrics")
    
    async def stop(self) -> None:
        """إيقاف خادم المقاييس - Stop metrics server"""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Metrics exporter stopped")
    
    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """معالج نقطة النهاية /metrics - /metrics endpoint handler"""
        prometheus_format = await self.get_prometheus_format()
        return web.Response(
            text=prometheus_format,
            content_type='text/plain; version=0.0.4'
        )
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """معالج نقطة النهاية /health - /health endpoint handler"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def get_prometheus_format(self) -> str:
        """
        الحصول على المقاييس بتنسيق Prometheus
        Get metrics in Prometheus format
        
        Returns:
            str: المقاييس بتنسيق Prometheus
        """
        lines = []
        
        async with self._lock:
            # معلومات النظام
            lines.append(f"# HELP {self.prefix}_info Information about the BI-IDE application")
            lines.append(f"# TYPE {self.prefix}_info gauge")
            lines.append(f'{self.prefix}_info{{version="1.0.0"}} 1')
            lines.append("")
            
            # Counters
            for name, label_values in self._counters.items():
                if name not in self._definitions:
                    continue
                
                definition = self._definitions[name]
                full_name = f"{self.prefix}_{name}"
                
                lines.append(f"# HELP {full_name} {definition.description}")
                lines.append(f"# TYPE {full_name} {definition.type.value}")
                
                for label_key, value in label_values.items():
                    labels = self._parse_label_key(label_key)
                    label_str = self._format_labels(labels)
                    lines.append(f"{full_name}{label_str} {value}")
                
                lines.append("")
            
            # Gauges
            for name, label_values in self._gauges.items():
                if name not in self._definitions:
                    continue
                
                definition = self._definitions[name]
                full_name = f"{self.prefix}_{name}"
                
                lines.append(f"# HELP {full_name} {definition.description}")
                lines.append(f"# TYPE {full_name} {definition.type.value}")
                
                for label_key, metric in label_values.items():
                    labels = self._parse_label_key(label_key)
                    label_str = self._format_labels(labels)
                    lines.append(f"{full_name}{label_str} {metric.value}")
                
                lines.append("")
            
            # Histograms
            for name, label_values in self._histograms.items():
                if name not in self._definitions:
                    continue
                
                definition = self._definitions[name]
                full_name = f"{self.prefix}_{name}"
                
                lines.append(f"# HELP {full_name} {definition.description}")
                lines.append(f"# TYPE {full_name} histogram")
                
                for label_key, values in label_values.items():
                    labels = self._parse_label_key(label_key)
                    
                    # buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10
                    buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
                    
                    for bucket in buckets:
                        count = sum(1 for v in values if v <= bucket)
                        bucket_labels = {**labels, 'le': str(bucket)}
                        label_str = self._format_labels(bucket_labels)
                        lines.append(f"{full_name}_bucket{label_str} {count}")
                    
                    # +Inf bucket
                    inf_labels = {**labels, 'le': '+Inf'}
                    label_str = self._format_labels(inf_labels)
                    lines.append(f"{full_name}_bucket{label_str} {len(values)}")
                    
                    # sum and count
                    label_str = self._format_labels(labels)
                    lines.append(f"{full_name}_sum{label_str} {sum(values)}")
                    lines.append(f"{full_name}_count{label_str} {len(values)}")
                
                lines.append("")
            
            # Summaries
            for name, label_values in self._summaries.items():
                if name not in self._definitions:
                    continue
                
                definition = self._definitions[name]
                full_name = f"{self.prefix}_{name}"
                
                lines.append(f"# HELP {full_name} {definition.description}")
                lines.append(f"# TYPE {full_name} summary")
                
                for label_key, values in label_values.items():
                    labels = self._parse_label_key(label_key)
                    label_str = self._format_labels(labels)
                    
                    if values:
                        # quantiles: 0.5, 0.9, 0.99
                        sorted_values = sorted(values)
                        n = len(sorted_values)
                        
                        for q in [0.5, 0.9, 0.99]:
                            idx = int(q * n)
                            quantile_labels = {**labels, 'quantile': str(q)}
                            q_label_str = self._format_labels(quantile_labels)
                            lines.append(f"{full_name}{q_label_str} {sorted_values[idx]}")
                        
                        lines.append(f"{full_name}_sum{label_str} {sum(values)}")
                        lines.append(f"{full_name}_count{label_str} {n}")
                
                lines.append("")
        
        return '\n'.join(lines)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """تنسيق التسميات - Format labels"""
        if not labels:
            return ""
        
        pairs = [f'{k}="{self._escape_label(v)}"' for k, v in labels.items()]
        return "{" + ",".join(pairs) + "}"
    
    def _escape_label(self, value: str) -> str:
        """هروب الأحرف الخاصة في التسمية - Escape special characters in label"""
        return value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    
    def _make_label_key(self, labels: Dict[str, str]) -> str:
        """إنشاء مفتاح التسميات - Create label key"""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def _parse_label_key(self, key: str) -> Dict[str, str]:
        """تحليل مفتاح التسميات - Parse label key"""
        labels = {}
        if not key:
            return labels
        
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        
        return labels
    
    async def export_metrics(self) -> Dict[str, Any]:
        """
        تصدير جميع المقاييس
        Export all metrics
        
        Returns:
            Dict[str, Any]: المقاييس
        """
        async with self._lock:
            return {
                'gauges': {
                    name: {k: v.value for k, v in values.items()}
                    for name, values in self._gauges.items()
                },
                'counters': dict(self._counters),
                'histograms': {
                    name: {k: values for k, values in label_values.items()}
                    for name, label_values in self._histograms.items()
                },
                'timestamp': time.time()
            }
    
    async def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        تسجيل قيمة عداد
        Record counter value
        
        Args:
            name: اسم المقياس
            value: القيمة للإضافة
            labels: التسميات
        """
        label_key = self._make_label_key(labels or {})
        
        async with self._lock:
            self._counters[name][label_key] += value
    
    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        تعيين قيمة مقياس
        Set gauge value
        
        Args:
            name: اسم المقياس
            value: القيمة
            labels: التسميات
        """
        label_key = self._make_label_key(labels or {})
        
        async with self._lock:
            self._gauges[name][label_key] = MetricValue(
                name=name,
                value=value,
                labels=labels or {}
            )
    
    async def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        تسجيل قيمة في الهيستوغرام
        Record histogram value
        
        Args:
            name: اسم المقياس
            value: القيمة
            labels: التسميات
        """
        label_key = self._make_label_key(labels or {})
        
        async with self._lock:
            self._histograms[name][label_key].append(value)
            
            # الحفاظ على الحجم المعقول
            if len(self._histograms[name][label_key]) > 10000:
                self._histograms[name][label_key] = self._histograms[name][label_key][-5000:]
    
    async def record_summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        تسجيل قيمة في الملخص
        Record summary value
        
        Args:
            name: اسم المقياس
            value: القيمة
            labels: التسميات
        """
        label_key = self._make_label_key(labels or {})
        
        async with self._lock:
            self._summaries[name][label_key].append(value)
            
            # الحفاظ على الحجم المعقول
            if len(self._summaries[name][label_key]) > 10000:
                self._summaries[name][label_key] = self._summaries[name][label_key][-5000:]
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """
        تسجيل مقياس جديد
        Register new metric
        
        Args:
            definition: تعريف المقياس
        """
        self._definitions[definition.name] = definition
        logger.info(f"Registered metric: {definition.name}")
    
    def unregister_metric(self, name: str) -> bool:
        """
        إلغاء تسجيل مقياس
        Unregister metric
        
        Args:
            name: اسم المقياس
            
        Returns:
            bool: نجاح العملية
        """
        if name in self._definitions:
            del self._definitions[name]
            self._gauges.pop(name, None)
            self._counters.pop(name, None)
            self._histograms.pop(name, None)
            self._summaries.pop(name, None)
            logger.info(f"Unregistered metric: {name}")
            return True
        return False
    
    async def clear_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """
        مسح قيم المقياس
        Clear metric values
        
        Args:
            name: اسم المقياس
            labels: التسميات (None لمسح الكل)
        """
        async with self._lock:
            if labels is None:
                self._gauges.pop(name, None)
                self._counters.pop(name, None)
                self._histograms.pop(name, None)
                self._summaries.pop(name, None)
            else:
                label_key = self._make_label_key(labels)
                if name in self._gauges:
                    self._gauges[name].pop(label_key, None)
                if name in self._counters:
                    self._counters[name].pop(label_key, None)
                if name in self._histograms:
                    self._histograms[name].pop(label_key, None)
                if name in self._summaries:
                    self._summaries[name].pop(label_key, None)
    
    @asynccontextmanager
    async def measure_duration(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        سياق لقياس المدة
        Context manager for measuring duration
        
        Args:
            name: اسم المقياس
            labels: التسميات
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            await self.record_histogram(name, duration, labels)
    
    def get_definition(self, name: str) -> Optional[MetricDefinition]:
        """الحصول على تعريف المقياس - Get metric definition"""
        return self._definitions.get(name)
    
    def list_metrics(self) -> List[str]:
        """الحصول على قائمة المقاييس - List all metrics"""
        return list(self._definitions.keys())
