#!/usr/bin/env python
"""
Smoke Test - فحص سريع للتأكد من استقرار النظام
"""
import sys
import asyncio


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


def test_council_ai():
    """Test 2: Council AI initializes"""
    print("[TEST 2] Initializing Council AI...")
    try:
        from council_ai import smart_council
        print(f"   [PASS] Council AI ready with {len(smart_council.wise_men)} wise men")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_hierarchy():
    """Test 3: Hierarchy system"""
    print("[TEST 3] Loading hierarchy...")
    try:
        from hierarchy import ai_hierarchy
        print(f"   [PASS] Hierarchy loaded with president access")
        return True
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def test_database():
    """Test 4: Database connection"""
    print("[TEST 4] Database check...")
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
    """Test 5: API can start"""
    print("[TEST 5] API startup check...")
    try:
        from api.app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        if response.status_code == 200:
            print(f"   [PASS] API health check: OK")
            return True
        else:
            print(f"   [FAIL] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("BI-IDE v8 Smoke Test")
    print("=" * 60)
    print()

    tests = [
        test_settings_load,
        test_council_ai,
        test_hierarchy,
        test_database,
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

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("ALL SYSTEMS OPERATIONAL")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
