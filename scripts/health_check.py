#!/usr/bin/env python3
"""
Health Check CLI - فحص الصحة
Command line tool for health checks
أداة سطر أوامر لفحوصات الصحة
"""
import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# إعداد التسجيل - Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


# الألوان - Colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """تلوين النص - Colorize text"""
    return f"{color}{text}{Colors.RESET}"


@dataclass
class HealthCheckResult:
    """نتيجة فحص الصحة - Health check result"""
    component: str
    name_ar: str
    status: str  # "ok", "warning", "error"
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "name_ar": self.name_ar,
            "status": self.status,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "suggestions": self.suggestions
        }


class HealthChecker:
    """
    فاحص الصحة
    Health checker for all components
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results: List[HealthCheckResult] = []
        self.start_time = time.time()
        
    async def check_postgres(self) -> HealthCheckResult:
        """فحص PostgreSQL - Check PostgreSQL"""
        component = "PostgreSQL"
        name_ar = "قاعدة البيانات"
        start = time.time()
        
        try:
            result = subprocess.run(
                ["pg_isready", "-h", "localhost", "-p", "5432"],
                capture_output=True,
                text=True,
                timeout=5
            )
            latency = (time.time() - start) * 1000
            
            if result.returncode == 0:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="ok",
                    message="Connected and accepting connections",
                    latency_ms=latency,
                    details={"version": "PostgreSQL"}
                )
            else:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="error",
                    message=result.stderr or "PostgreSQL not ready",
                    latency_ms=latency,
                    suggestions=[
                        "تأكد من تشغيل PostgreSQL: docker-compose up -d postgres",
                        "تحقق من إعدادات الاتصال في .env"
                    ]
                )
        except FileNotFoundError:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="warning",
                message="pg_isready not found, checking Docker...",
                latency_ms=latency,
                details={"note": "PostgreSQL client not installed"},
                suggestions=["تثبيت PostgreSQL client: brew install libpq"]
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message=str(e),
                latency_ms=latency,
                suggestions=["تحقق من تشغيل خدمة PostgreSQL"]
            )
            
    async def check_redis(self) -> HealthCheckResult:
        """فحص Redis - Check Redis"""
        component = "Redis"
        name_ar = "التخزين المؤقت"
        start = time.time()
        
        try:
            result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            latency = (time.time() - start) * 1000
            
            if "PONG" in result.stdout:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="ok",
                    message="Connected and responding to PING",
                    latency_ms=latency
                )
            else:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="error",
                    message="Redis not responding correctly",
                    latency_ms=latency,
                    suggestions=[
                        "تأكد من تشغيل Redis: docker-compose up -d redis",
                        "تحقق من إعدادات Redis في .env"
                    ]
                )
        except FileNotFoundError:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="warning",
                message="redis-cli not found",
                latency_ms=latency,
                suggestions=["تثبيت Redis client: brew install redis"]
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message=str(e),
                latency_ms=latency
            )
            
    async def check_api(self) -> HealthCheckResult:
        """فحص API Server - Check API server"""
        component = "API Server"
        name_ar = "خادم API"
        start = time.time()
        url = f"{self.api_base_url}/health"
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(url) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            component=component,
                            name_ar=name_ar,
                            status="ok",
                            message="API is healthy",
                            latency_ms=latency,
                            details=data
                        )
                    else:
                        return HealthCheckResult(
                            component=component,
                            name_ar=name_ar,
                            status="error",
                            message=f"HTTP {response.status}",
                            latency_ms=latency,
                            suggestions=[
                                f"تحقق من سجلات API: docker-compose logs api",
                                "تأكد من إعدادات الاتصال بقاعدة البيانات"
                            ]
                        )
        except aiohttp.ClientConnectorError:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message="Cannot connect to API server",
                latency_ms=latency,
                suggestions=[
                    "تشغيل خادم API: python -m uvicorn api.app:app --reload",
                    f"تحقق من أن المنفذ متاح: {self.api_base_url}",
                    "تحقق من جدار الحماية"
                ]
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message=str(e),
                latency_ms=latency
            )
            
    async def check_disk_space(self) -> HealthCheckResult:
        """فحص مساحة القرص - Check disk space"""
        component = "Disk Space"
        name_ar = "مساحة القرص"
        start = time.time()
        
        try:
            result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=5
            )
            latency = (time.time() - start) * 1000
            
            # تحليل الإخراج - Parse output
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                usage_percent = int(parts[4].replace("%", ""))
                available = parts[3]
                
                if usage_percent < 80:
                    status = "ok"
                elif usage_percent < 90:
                    status = "warning"
                else:
                    status = "error"
                    
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status=status,
                    message=f"{usage_percent}% used, {available} available",
                    latency_ms=latency,
                    details={
                        "usage_percent": usage_percent,
                        "available": available,
                        "total": parts[1],
                        "used": parts[2]
                    },
                    suggestions=["تنظيف الملفات غير الضرورية"] if status == "warning" else []
                )
            else:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="warning",
                    message="Could not parse disk info",
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message=str(e),
                latency_ms=latency
            )
            
    async def check_memory(self) -> HealthCheckResult:
        """فحص الذاكرة - Check memory"""
        component = "Memory"
        name_ar = "الذاكرة"
        start = time.time()
        
        try:
            if sys.platform == "darwin":  # macOS
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # vm_stat يعطي إحصائيات مختلفة
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="ok",
                    message="Memory check available via vm_stat",
                    latency_ms=(time.time() - start) * 1000,
                    details={"note": "Use 'vm_stat' or Activity Monitor for details"}
                )
            else:  # Linux
                result = subprocess.run(
                    ["free", "-m"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                latency = (time.time() - start) * 1000
                
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    mem_line = lines[1].split()
                    total = int(mem_line[1])
                    used = int(mem_line[2])
                    usage_percent = int((used / total) * 100)
                    
                    if usage_percent < 80:
                        status = "ok"
                    elif usage_percent < 90:
                        status = "warning"
                    else:
                        status = "error"
                        
                    return HealthCheckResult(
                        component=component,
                        name_ar=name_ar,
                        status=status,
                        message=f"{usage_percent}% used ({used}MB / {total}MB)",
                        latency_ms=latency,
                        details={
                            "total_mb": total,
                            "used_mb": used,
                            "usage_percent": usage_percent
                        }
                    )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="warning",
                message=f"Could not check memory: {e}",
                latency_ms=latency
            )
            
    async def check_docker(self) -> HealthCheckResult:
        """فحص Docker - Check Docker"""
        component = "Docker"
        name_ar = "دوكر"
        start = time.time()
        
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            latency = (time.time() - start) * 1000
            
            if result.returncode == 0:
                # التحقق من الحاويات - Check containers
                ps_result = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                containers = ps_result.stdout.strip().split("\n") if ps_result.stdout else []
                
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="ok",
                    message=f"Docker daemon running ({len(containers)} containers)",
                    latency_ms=latency,
                    details={
                        "running_containers": containers,
                        "container_count": len(containers)
                    }
                )
            else:
                return HealthCheckResult(
                    component=component,
                    name_ar=name_ar,
                    status="error",
                    message="Docker daemon not responding",
                    latency_ms=latency,
                    suggestions=[
                        "تشغيل Docker Desktop",
                        "تحقق من صلاحيات المستخدم: sudo usermod -aG docker $USER"
                    ]
                )
        except FileNotFoundError:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message="Docker not installed",
                latency_ms=latency,
                suggestions=["تثبيت Docker: https://docs.docker.com/get-docker/"]
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                component=component,
                name_ar=name_ar,
                status="error",
                message=str(e),
                latency_ms=latency
            )
            
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """تشغيل جميع الفحوصات - Run all checks"""
        logger.info("🔍 Running health checks...\n")
        
        checks = [
            self.check_docker(),
            self.check_postgres(),
            self.check_redis(),
            self.check_api(),
            self.check_disk_space(),
            self.check_memory(),
        ]
        
        self.results = await asyncio.gather(*checks, return_exceptions=True)
        
        # معالجة الاستثناءات
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                self.results[i] = HealthCheckResult(
                    component="Unknown",
                    name_ar="غير معروف",
                    status="error",
                    message=str(result)
                )
                
        return self.results
        
    def print_table(self):
        """طباعة نتائج الجدول - Print table results"""
        # عنوان الجدول - Table header
        header = f"\n{colorize('BI-IDE Health Check | فحص صحة BI-IDE', Colors.BOLD + Colors.BLUE)}"
        logger.info(header)
        logger.info("=" * 80)
        
        # رأس الأعمدة - Column headers
        logger.info(
            f"{'Component':<20} {'Status':<10} {'Latency':<12} {'Message'}"
        )
        logger.info("-" * 80)
        
        # الصفوف - Rows
        for result in self.results:
            if result.status == "ok":
                status_str = colorize("✅ OK", Colors.GREEN)
            elif result.status == "warning":
                status_str = colorize("⚠️ WARN", Colors.YELLOW)
            else:
                status_str = colorize("❌ FAIL", Colors.RED)
                
            latency_str = f"{result.latency_ms:.1f}ms"
            
            logger.info(
                f"{result.component:<20} {status_str:<20} {latency_str:<12} {result.message}"
            )
            
        # الملخص - Summary
        logger.info("=" * 80)
        ok_count = sum(1 for r in self.results if r.status == "ok")
        warn_count = sum(1 for r in self.results if r.status == "warning")
        error_count = sum(1 for r in self.results if r.status == "error")
        total_time = (time.time() - self.start_time) * 1000
        
        summary = f"Total: {len(self.results)} | "
        summary += colorize(f"OK: {ok_count}", Colors.GREEN) + " | "
        if warn_count > 0:
            summary += colorize(f"Warnings: {warn_count}", Colors.YELLOW) + " | "
        if error_count > 0:
            summary += colorize(f"Errors: {error_count}", Colors.RED) + " | "
        summary += f"Time: {total_time:.1f}ms"
        
        logger.info(summary)
        
        # اقتراحات الإصلاح - Fix suggestions
        errors_with_suggestions = [
            r for r in self.results 
            if r.status == "error" and r.suggestions
        ]
        
        if errors_with_suggestions:
            logger.info(f"\n{colorize('💡 Suggestions | اقتراحات الإصلاح:', Colors.CYAN)}")
            for result in errors_with_suggestions:
                logger.info(f"\n{result.component} ({result.name_ar}):")
                for suggestion in result.suggestions:
                    logger.info(f"  • {suggestion}")
                    
    def print_json(self):
        """طباعة نتائج JSON - Print JSON results"""
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_base_url": self.api_base_url,
            "total_time_ms": round((time.time() - self.start_time) * 1000, 2),
            "summary": {
                "total": len(self.results),
                "ok": sum(1 for r in self.results if r.status == "ok"),
                "warning": sum(1 for r in self.results if r.status == "warning"),
                "error": sum(1 for r in self.results if r.status == "error")
            },
            "checks": [r.to_dict() for r in self.results]
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    def get_exit_code(self) -> int:
        """الحصول على رمز الخروج للـ CI/CD - Get exit code for CI/CD"""
        error_count = sum(1 for r in self.results if r.status == "error")
        return 1 if error_count > 0 else 0


async def main():
    """الدالة الرئيسية - Main function"""
    parser = argparse.ArgumentParser(
        description="BI-IDE Health Check CLI | أداة فحص صحة BI-IDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples | أمثلة:
  python scripts/health_check.py
  python scripts/health_check.py --json
  python scripts/health_check.py --api-url http://api.example.com
  python scripts/health_check.py --component postgres,redis
        """
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format | إخراج بتنسيق JSON"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL for API checks | عنوان URL للفحوصات"
    )
    parser.add_argument(
        "--component",
        help="Check specific components only (comma-separated) | فحص مكونات محددة فقط"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with error code on warnings | الخروج مع خطأ عند التحذيرات"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (only errors) | الوضع الصامت (الأخطاء فقط)"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        
    checker = HealthChecker(api_base_url=args.api_url)
    await checker.run_all_checks()
    
    if args.json:
        checker.print_json()
    else:
        checker.print_table()
        
    sys.exit(checker.get_exit_code())


if __name__ == "__main__":
    asyncio.run(main())
