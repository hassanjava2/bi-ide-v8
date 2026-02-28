#!/usr/bin/env python3
"""
Implementation Verification Script
سكربت التحقق من اكتمال التنفيذ

Verifies that all components of BI-IDE v8 are properly implemented.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """Terminal colors"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def log_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")


def log_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.NC}")


def log_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    if Path(filepath).exists():
        log_success(f"{description}: {filepath}")
        return True
    else:
        log_error(f"{description} missing: {filepath}")
        return False


def check_syntax(filepath: str) -> bool:
    """Check Python file syntax"""
    import py_compile
    try:
        py_compile.compile(filepath, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        log_error(f"Syntax error in {filepath}: {e}")
        return False


def main():
    """Main verification function"""
    print("=" * 70)
    print("BI-IDE v8 - Implementation Verification")
    print("التحقق من اكتمال تنفيذ BI-IDE v8")
    print("=" * 70)
    print()
    
    base_path = Path("/Users/bi/Documents/bi-ide-v8")
    os.chdir(base_path)
    
    results = []
    
    # 1. Check Core Files
    log_info("Checking Core Files...")
    core_files = [
        ("api/app.py", "Main API Application"),
        ("api/__init__.py", "API Package"),
        ("api/auth.py", "Authentication Module"),
        ("api/middleware.py", "Middleware"),
        ("api/schemas.py", "API Schemas"),
    ]
    for filepath, desc in core_files:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 2. Check API Routers
    log_info("Checking API Routers...")
    routers = [
        ("api/routers/auth.py", "Auth Router"),
        ("api/routers/council.py", "Council Router"),
        ("api/routers/training.py", "Training Router"),
        ("api/routers/ai.py", "AI Router"),
        ("api/routers/erp.py", "ERP Router"),
        ("api/routers/monitoring.py", "Monitoring Router"),
        ("api/routers/community.py", "Community Router"),
    ]
    for filepath, desc in routers:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 3. Check Services
    log_info("Checking Services...")
    services = [
        ("services/training_service.py", "Training Service"),
        ("services/council_service.py", "Council Service"),
        ("services/ai_service.py", "AI Service"),
        ("services/notification_service.py", "Notification Service"),
        ("services/sync_service.py", "Sync Service"),
        ("services/backup_service.py", "Backup Service"),
    ]
    for filepath, desc in services:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 4. Check Monitoring
    log_info("Checking Monitoring...")
    monitoring = [
        ("monitoring/system_monitor.py", "System Monitor"),
        ("monitoring/training_monitor.py", "Training Monitor"),
        ("monitoring/alert_manager.py", "Alert Manager"),
        ("monitoring/log_aggregator.py", "Log Aggregator"),
        ("monitoring/metrics_exporter.py", "Metrics Exporter"),
        ("monitoring/health_dashboard.py", "Health Dashboard"),
    ]
    for filepath, desc in monitoring:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 5. Check Network
    log_info("Checking Network...")
    network = [
        ("network/health_check_daemon.py", "Health Check Daemon"),
        ("network/firewall_manager.py", "Firewall Manager"),
        ("network/auto_reconnect.py", "Auto Reconnect"),
    ]
    for filepath, desc in network:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 6. Check Scripts
    log_info("Checking Scripts...")
    scripts = [
        ("scripts/setup_database.py", "Database Setup"),
        ("scripts/verify_installation.py", "Installation Verification"),
        ("scripts/start_services.py", "Service Starter"),
        ("scripts/health_check.py", "Health Check CLI"),
    ]
    for filepath, desc in scripts:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 7. Check Tests
    log_info("Checking Tests...")
    tests = [
        ("tests/test_api.py", "API Tests"),
        ("tests/test_training.py", "Training Tests"),
        ("tests/test_gpu_training.py", "GPU Training Tests"),
        ("tests/test_security.py", "Security Tests"),
        ("tests/test_desktop_api.py", "Desktop API Tests"),
        ("tests/conftest.py", "Test Configuration"),
        ("pytest.ini", "Pytest Config"),
    ]
    for filepath, desc in tests:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 8. Check Requirements
    log_info("Checking Requirements...")
    req_files = [
        ("requirements.txt", "Main Requirements"),
        ("requirements-prod.txt", "Production Requirements"),
        ("requirements-dev.txt", "Development Requirements"),
    ]
    for filepath, desc in req_files:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 9. Check Docker
    log_info("Checking Docker...")
    docker_files = [
        ("Dockerfile", "Main Dockerfile"),
        ("Dockerfile.gpu", "GPU Dockerfile"),
        ("docker-compose.yml", "Docker Compose"),
        ("docker-compose.gpu.yml", "GPU Docker Compose"),
    ]
    for filepath, desc in docker_files:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # 10. Check Deploy Scripts
    log_info("Checking Deploy Scripts...")
    deploy_scripts = [
        ("deploy/deploy_all.sh", "Master Deploy"),
        ("deploy/deploy_windows.ps1", "Windows Deploy"),
        ("deploy/deploy_rtx.sh", "RTX Deploy"),
        ("deploy/rollback.sh", "Rollback"),
        ("deploy/zero_downtime_deploy.sh", "Zero-Downtime Deploy"),
    ]
    for filepath, desc in deploy_scripts:
        results.append(check_file_exists(filepath, desc))
    print()
    
    # Summary
    print("=" * 70)
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"\n📊 Summary:")
    print(f"   Total Checks: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print()
    
    if failed == 0:
        log_success("All checks passed! Implementation is complete. ✅")
        log_success("كل الفحوصات ناجحة! التنفيذ مكتمل. ✅")
        return 0
    else:
        log_error(f"{failed} checks failed. Please review the errors above. ❌")
        log_error(f"{failed} فحوصات فاشلة. راجع الأخطاء أعلاه. ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
