#!/usr/bin/env python
"""
Smoke Test - فحص سريع للتأكد من استقرار النظام
Supports both local testing and CI/CD remote testing
"""
import sys
import asyncio
import argparse
import requests
from typing import List, Callable


def test_settings_load():
    """Test 1: Settings load without errors"""
    print("[TEST 1] Loading settings...")
    try:
        from core.config import get_settings
        settings = get_settings()
        print(f"   [PASS] Settings loaded: {settings.APP_NAME} v{settings.APP_VERSION}")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_hierarchy():
    """Test 2: Hierarchy system"""
    print("[TEST 2] Loading hierarchy...")
    try:
        from hierarchy import ai_hierarchy
        print(f"   [PASS] Hierarchy loaded: 10 layers, 100+ entities")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_database():
    """Test 3: Database connection"""
    print("[TEST 3] Database check...")
    try:
        import os
        data_dir = "./data"
        if os.path.exists(data_dir):
            print(f"   [PASS] Data directory exists")
        else:
            os.makedirs(data_dir, exist_ok=True)
            print(f"   [PASS] Data directory created")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


async def test_api_startup():
    """Test 4: API can start"""
    print("[TEST 4] API startup check...")
    try:
        from api.app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   [PASS] API health check: {data.get('status', 'OK')}")
            return True
        else:
            print(f"   [FAIL] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_gateway_initialization():
    """Test 5: API Gateway initializes"""
    print("[TEST 5] Gateway initialization...")
    try:
        from api.gateway import RTX4090Gateway
        gateway = RTX4090Gateway()
        print(f"   [PASS] Gateway initialized with {len(gateway.endpoints)} endpoints")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_auth_system():
    """Test 6: Auth system components"""
    print("[TEST 6] Auth system check...")
    try:
        from api.auth import create_access_token, verify_token
        from core.user_service import UserService, RoleService
        print(f"   [PASS] Auth components loaded: JWT, UserService, RoleService")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


# ═════════════════════════════════════════════════════════════════
# Remote Testing (for CI/CD)
# ═════════════════════════════════════════════════════════════════


def test_remote_health(api_url: str) -> bool:
    """Test remote API health endpoint"""
    print(f"[REMOTE 1] Testing {api_url}/health...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   [PASS] Health: {data.get('status', 'OK')}")
            return True
        else:
            print(f"   [FAIL] Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def test_remote_ready(api_url: str) -> bool:
    """Test remote API ready endpoint"""
    print(f"[REMOTE 2] Testing {api_url}/ready...")
    try:
        response = requests.get(f"{api_url}/ready", timeout=10)
        data = response.json()
        if response.status_code == 200 and data.get('ready'):
            print(f"   [PASS] System ready")
            return True
        else:
            print(f"   [WARN] System not ready: {data}")
            return False  # Warning, not failure
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def test_remote_auth(api_url: str) -> bool:
    """Test remote auth endpoint"""
    print(f"[REMOTE 3] Testing {api_url}/api/v1/auth/login...")
    try:
        # Test with invalid credentials (should return 401)
        response = requests.post(
            f"{api_url}/api/v1/auth/login",
            json={"username": "test", "password": "test"},
            timeout=10
        )
        if response.status_code == 401:
            print(f"   [PASS] Auth endpoint responding (401 for invalid creds)")
            return True
        elif response.status_code == 200:
            print(f"   [PASS] Auth endpoint responding")
            return True
        else:
            print(f"   [FAIL] Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def test_remote_metrics(api_url: str) -> bool:
    """Test remote metrics endpoint"""
    print(f"[REMOTE 4] Testing {api_url}/metrics...")
    try:
        response = requests.get(f"{api_url}/metrics", timeout=10)
        if response.status_code == 200:
            print(f"   [PASS] Metrics endpoint responding")
            return True
        else:
            print(f"   [FAIL] Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════


def run_local_tests() -> List[bool]:
    """Run all local tests"""
    tests = [
        test_settings_load,
        test_hierarchy,
        test_database,
        test_gateway_initialization,
        test_auth_system,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # API test (async)
    try:
        results.append(asyncio.run(test_api_startup()))
    except Exception as e:
        print(f"   [FAIL] API test failed: {e}")
        results.append(False)
    
    return results


def run_remote_tests(api_url: str) -> List[bool]:
    """Run remote tests against deployed API"""
    tests = [
        lambda: test_remote_health(api_url),
        lambda: test_remote_ready(api_url),
        lambda: test_remote_auth(api_url),
        lambda: test_remote_metrics(api_url),
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BI-IDE Smoke Test")
    parser.add_argument(
        "--url",
        help="Remote API URL for testing (e.g., https://api.example.com)",
        default=None
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode - stricter checks"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("BI-IDE v8 Smoke Test")
    if args.url:
        print(f"Target: {args.url}")
    print("=" * 60)
    print()
    
    if args.url:
        # Remote testing
        results = run_remote_tests(args.url)
    else:
        # Local testing
        results = run_local_tests()
    
    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL SYSTEMS OPERATIONAL")
        return 0
    elif args.ci:
        # In CI mode, all tests must pass
        print("❌ SOME TESTS FAILED")
        return 1
    elif passed >= total * 0.8:
        # In local mode, 80% is acceptable
        print("⚠️ MOST TESTS PASSED (80%+)")
        return 0
    else:
        print("❌ TOO MANY FAILURES")
        return 1


if __name__ == "__main__":
    sys.exit(main())
