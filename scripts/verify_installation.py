"""
Installation Verification - التحقق من التثبيت
Verify all components are properly installed and configured
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """نتيجة التحقق - Verification result"""
    component: str
    is_ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """تقرير التحقق - Verification report"""
    success: bool
    timestamp: float
    system_info: Dict[str, Any]
    results: List[VerificationResult]
    errors: List[str]
    warnings: List[str]


class InstallationVerifier:
    """
    متحقق التثبيت
    Installation Verifier
    
    يتحقق من جميع مكونات BI-IDE المطلوبة
    Verifies all required BI-IDE components
    """
    
    # الحد الأدنى لإصدار Python - Minimum Python version
    MIN_PYTHON_VERSION = (3, 9)
    
    # التبعيات المطلوبة - Required dependencies
    REQUIRED_PACKAGES = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'asyncpg', 'aioredis',
        'pydantic', 'aiohttp', 'psutil', 'numpy', 'torch'
    ]
    
    # المنافذ المطلوبة - Required ports
    REQUIRED_PORTS = [8000, 8001, 6379, 5432, 8080]
    
    def __init__(
        self,
        project_root: Optional[str] = None,
        requirements_file: str = "requirements.txt"
    ):
        """
        تهيئة متحقق التثبيت
        Initialize installation verifier
        
        Args:
            project_root: جذر المشروع
            requirements_file: ملف المتطلبات
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.requirements_file = self.project_root / requirements_file
        self._results: List[VerificationResult] = []
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    async def run_verification(self) -> VerificationReport:
        """
        تشغيل التحقق الكامل
        Run full verification
        
        Returns:
            VerificationReport: تقرير التحقق
        """
        logger.info("Starting installation verification...")
        self._results = []
        self._errors = []
        self._warnings = []
        
        import time
        start_time = time.time()
        
        # جمع معلومات النظام
        system_info = self._get_system_info()
        
        # التحقق من إصدار Python
        self._check_python_version()
        
        # التحقق من التبعيات
        await self._check_dependencies()
        
        # التحقق من GPU
        self._check_gpu_drivers()
        
        # التحقق من Docker
        self._check_docker()
        
        # التحقق من صلاحيات الملفات
        self._check_file_permissions()
        
        # اختبار الاتصال بالشبكة
        await self._test_network_connectivity()
        
        # التحقق من ملفات التكوين
        self._check_configuration_files()
        
        success = len(self._errors) == 0
        
        report = VerificationReport(
            success=success,
            timestamp=time.time(),
            system_info=system_info,
            results=self._results,
            errors=self._errors,
            warnings=self._warnings
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Verification completed in {elapsed_ms:.2f}ms")
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النظام - Get system information"""
        return {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'node': platform.node(),
            'cpu_count': os.cpu_count(),
            'memory_gb': self._get_total_memory_gb()
        }
    
    def _get_total_memory_gb(self) -> float:
        """الحصول على إجمالي الذاكرة بالجيجابايت - Get total memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 0.0
    
    def _check_python_version(self) -> None:
        """التحقق من إصدار Python - Check Python version"""
        logger.info("Checking Python version...")
        
        version = sys.version_info[:2]
        version_str = f"{version[0]}.{version[1]}"
        min_version_str = f"{self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}"
        
        if version >= self.MIN_PYTHON_VERSION:
            self._results.append(VerificationResult(
                component="Python Version",
                is_ok=True,
                message=f"Python {version_str} (>= {min_version_str})",
                details={'version': version_str}
            ))
            logger.info(f"Python version check passed: {version_str}")
        else:
            self._results.append(VerificationResult(
                component="Python Version",
                is_ok=False,
                message=f"Python {version_str} is too old (requires >= {min_version_str})",
                details={'version': version_str, 'required': min_version_str}
            ))
            self._errors.append(
                f"Python version {version_str} is not supported. "
                f"Please upgrade to Python {min_version_str}+"
            )
            logger.error(f"Python version check failed: {version_str}")
    
    async def _check_dependencies(self) -> None:
        """التحقق من التبعيات - Check dependencies"""
        logger.info("Checking dependencies...")
        
        missing_packages = []
        outdated_packages = []
        
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package)
                self._results.append(VerificationResult(
                    component=f"Package: {package}",
                    is_ok=True,
                    message=f"{package} is installed",
                    details={}
                ))
            except ImportError:
                missing_packages.append(package)
                self._results.append(VerificationResult(
                    component=f"Package: {package}",
                    is_ok=False,
                    message=f"{package} is not installed",
                    details={}
                ))
        
        if missing_packages:
            self._errors.append(
                f"Missing packages: {', '.join(missing_packages)}. "
                f"Install with: pip install {' '.join(missing_packages)}"
            )
        
        # التحقق من ملف requirements.txt
        if self.requirements_file.exists():
            self._results.append(VerificationResult(
                component="Requirements File",
                is_ok=True,
                message=f"Found {self.requirements_file.name}",
                details={'path': str(self.requirements_file)}
            ))
        else:
            self._warnings.append(
                f"Requirements file not found: {self.requirements_file}"
            )
        
        logger.info(f"Dependency check completed. Missing: {len(missing_packages)}")
    
    def _check_gpu_drivers(self) -> None:
        """التحقق من تعريفات GPU - Check GPU drivers"""
        logger.info("Checking GPU drivers...")
        
        try:
            # محاولة تشغيل nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # استخراج معلومات GPU
                lines = result.stdout.split('\n')
                gpu_info = {}
                
                for line in lines:
                    if 'NVIDIA-SMI' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            gpu_info['driver_version'] = parts[2]
                    if 'CUDA Version' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            gpu_info['cuda_version'] = parts[1].strip()
                
                self._results.append(VerificationResult(
                    component="GPU Drivers",
                    is_ok=True,
                    message=f"NVIDIA drivers installed (Driver: {gpu_info.get('driver_version', 'N/A')})",
                    details=gpu_info
                ))
                logger.info("GPU drivers check passed")
            else:
                self._results.append(VerificationResult(
                    component="GPU Drivers",
                    is_ok=False,
                    message="nvidia-smi returned error",
                    details={'error': result.stderr}
                ))
                self._warnings.append(
                    "NVIDIA GPU drivers not detected. "
                    "GPU acceleration will not be available."
                )
                logger.warning("GPU drivers check failed: nvidia-smi error")
                
        except FileNotFoundError:
            self._results.append(VerificationResult(
                component="GPU Drivers",
                is_ok=False,
                message="nvidia-smi not found",
                details={}
            ))
            self._warnings.append(
                "nvidia-smi not found. NVIDIA GPU drivers may not be installed."
            )
            logger.warning("nvidia-smi not found")
            
        except subprocess.TimeoutExpired:
            self._results.append(VerificationResult(
                component="GPU Drivers",
                is_ok=False,
                message="nvidia-smi timeout",
                details={}
            ))
            self._warnings.append("nvidia-smi timed out")
            logger.warning("nvidia-smi timed out")
            
        except Exception as e:
            self._results.append(VerificationResult(
                component="GPU Drivers",
                is_ok=False,
                message=f"GPU check error: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            logger.error(f"GPU check error: {e}")
    
    def _check_docker(self) -> None:
        """التحقق من Docker - Check Docker installation"""
        logger.info("Checking Docker...")
        
        try:
            # التحقق من وجود Docker
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                
                # التحقق من تشغيل Docker
                ps_result = subprocess.run(
                    ['docker', 'ps'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                is_running = ps_result.returncode == 0
                
                self._results.append(VerificationResult(
                    component="Docker",
                    is_ok=True,
                    message=f"{version} (Running: {is_running})",
                    details={
                        'version': version,
                        'is_running': is_running
                    }
                ))
                
                if not is_running:
                    self._warnings.append(
                        "Docker is installed but not running. "
                        "Start Docker to use containerized services."
                    )
                
                logger.info("Docker check passed")
            else:
                self._results.append(VerificationResult(
                    component="Docker",
                    is_ok=False,
                    message="Docker not properly installed",
                    details={'error': result.stderr}
                ))
                self._warnings.append(
                    "Docker not found. Install Docker for containerization support."
                )
                logger.warning("Docker check failed")
                
        except FileNotFoundError:
            self._results.append(VerificationResult(
                component="Docker",
                is_ok=False,
                message="Docker not found",
                details={}
            ))
            self._warnings.append(
                "Docker is not installed. "
                "Visit https://docs.docker.com/get-docker/ for installation."
            )
            logger.warning("Docker not found")
            
        except subprocess.TimeoutExpired:
            self._results.append(VerificationResult(
                component="Docker",
                is_ok=False,
                message="Docker check timeout",
                details={}
            ))
            self._warnings.append("Docker check timed out")
            logger.warning("Docker check timed out")
            
        except Exception as e:
            self._results.append(VerificationResult(
                component="Docker",
                is_ok=False,
                message=f"Docker check error: {str(e)}",
                details={'error_type': type(e).__name__}
            ))
            logger.error(f"Docker check error: {e}")
    
    def _check_file_permissions(self) -> None:
        """التحقق من صلاحيات الملفات - Check file permissions"""
        logger.info("Checking file permissions...")
        
        critical_paths = [
            self.project_root / 'logs',
            self.project_root / 'data',
            self.project_root / 'alembic',
            self.project_root / 'scripts'
        ]
        
        permission_issues = []
        
        for path in critical_paths:
            if path.exists():
                if os.access(path, os.R_OK | os.W_OK):
                    self._results.append(VerificationResult(
                        component=f"Permissions: {path.name}",
                        is_ok=True,
                        message=f"Read/Write access to {path.name}",
                        details={'path': str(path)}
                    ))
                else:
                    permission_issues.append(str(path))
                    self._results.append(VerificationResult(
                        component=f"Permissions: {path.name}",
                        is_ok=False,
                        message=f"No read/write access to {path.name}",
                        details={'path': str(path)}
                    ))
            else:
                self._results.append(VerificationResult(
                    component=f"Permissions: {path.name}",
                    is_ok=True,
                    message=f"Path {path.name} does not exist yet (will be created)",
                    details={'path': str(path)}
                ))
        
        if permission_issues:
            self._warnings.append(
                f"Permission issues in: {', '.join(permission_issues)}. "
                "Run: chmod -R 755 <path>"
            )
        
        logger.info(f"File permissions check completed. Issues: {len(permission_issues)}")
    
    async def _test_network_connectivity(self) -> None:
        """اختبار الاتصال بالشبكة - Test network connectivity"""
        logger.info("Testing network connectivity...")
        
        test_hosts = [
            ('8.8.8.8', 53, 'Google DNS'),
            ('1.1.1.1', 53, 'Cloudflare DNS'),
            ('pypi.org', 443, 'PyPI'),
            ('github.com', 443, 'GitHub')
        ]
        
        import socket
        
        for host, port, name in test_hosts:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=5
                )
                writer.close()
                await writer.wait_closed()
                
                self._results.append(VerificationResult(
                    component=f"Network: {name}",
                    is_ok=True,
                    message=f"Can reach {name} ({host}:{port})",
                    details={'host': host, 'port': port}
                ))
                
            except asyncio.TimeoutError:
                self._results.append(VerificationResult(
                    component=f"Network: {name}",
                    is_ok=False,
                    message=f"Timeout connecting to {name} ({host}:{port})",
                    details={'host': host, 'port': port}
                ))
                self._warnings.append(f"Cannot reach {name} - check internet connection")
                
            except Exception as e:
                self._results.append(VerificationResult(
                    component=f"Network: {name}",
                    is_ok=False,
                    message=f"Cannot reach {name}: {str(e)}",
                    details={'host': host, 'port': port, 'error': str(e)}
                ))
        
        logger.info("Network connectivity test completed")
    
    def _check_configuration_files(self) -> None:
        """التحقق من ملفات التكوين - Check configuration files"""
        logger.info("Checking configuration files...")
        
        config_files = [
            ('.env', 'Environment variables'),
            ('alembic.ini', 'Alembic configuration'),
            ('docker-compose.yml', 'Docker Compose'),
            ('requirements.txt', 'Python dependencies')
        ]
        
        for filename, description in config_files:
            filepath = self.project_root / filename
            if filepath.exists():
                self._results.append(VerificationResult(
                    component=f"Config: {filename}",
                    is_ok=True,
                    message=f"{description} file exists",
                    details={'path': str(filepath), 'size': filepath.stat().st_size}
                ))
            else:
                self._results.append(VerificationResult(
                    component=f"Config: {filename}",
                    is_ok=False,
                    message=f"{description} file missing",
                    details={'expected_path': str(filepath)}
                ))
                self._warnings.append(f"Missing config file: {filename}")
        
        logger.info("Configuration files check completed")
    
    def print_report(self, report: VerificationReport) -> None:
        """
        طباعة تقرير التحقق
        Print verification report
        
        Args:
            report: تقرير التحقق
        """
        print("\n" + "=" * 70)
        print("🔍 Installation Verification Report - تقرير التحقق من التثبيت")
        print("=" * 70)
        
        # معلومات النظام
        print("\n💻 System Information - معلومات النظام:")
        print("-" * 70)
        for key, value in report.system_info.items():
            print(f"  {key}: {value}")
        
        # الحالة العامة
        status_icon = "✅" if report.success else "❌"
        print(f"\n{status_icon} Overall Status: {'PASSED' if report.success else 'FAILED'}")
        print(f"⏱️  Timestamp: {__import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime(report.timestamp))}")
        
        # النتائج
        print("\n📋 Verification Results:")
        print("-" * 70)
        
        passed = [r for r in report.results if r.is_ok]
        failed = [r for r in report.results if not r.is_ok]
        
        if passed:
            print(f"\n✅ Passed ({len(passed)}):")
            for result in passed:
                print(f"  ✓ {result.component}: {result.message}")
        
        if failed:
            print(f"\n❌ Failed ({len(failed)}):")
            for result in failed:
                print(f"  ✗ {result.component}: {result.message}")
        
        # الأخطاء
        if report.errors:
            print("\n❌ Errors - أخطاء:")
            print("-" * 70)
            for error in report.errors:
                print(f"  • {error}")
        
        # التحذيرات
        if report.warnings:
            print("\n⚠️  Warnings - تحذيرات:")
            print("-" * 70)
            for warning in report.warnings:
                print(f"  • {warning}")
        
        # الملخص
        total = len(report.results)
        passed_count = len(passed)
        failed_count = len(failed)
        
        print("\n" + "=" * 70)
        print(f"📊 Summary: {passed_count}/{total} passed, {failed_count} failed")
        print(f"   Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
        print("=" * 70 + "\n")
    
    def export_report_json(self, report: VerificationReport, filepath: str) -> None:
        """
        تصدير التقرير إلى JSON
        Export report to JSON
        
        Args:
            report: تقرير التحقق
            filepath: مسار ملف الإخراج
        """
        data = {
            'success': report.success,
            'timestamp': report.timestamp,
            'system_info': report.system_info,
            'results': [
                {
                    'component': r.component,
                    'is_ok': r.is_ok,
                    'message': r.message,
                    'details': r.details
                }
                for r in report.results
            ],
            'errors': report.errors,
            'warnings': report.warnings
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report exported to {filepath}")


async def main():
    """الدالة الرئيسية - Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    verifier = InstallationVerifier()
    report = await verifier.run_verification()
    verifier.print_report(report)
    
    # تصدير التقرير إذا طُلب
    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'verification_report.json'
        verifier.export_report_json(report, output_path)
    
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
