"""
Pytest Configuration - إعدادات الاختبارات
"""

import sys
import os

# ═══════════════════════════════════════════════════
# CRITICAL: Prevent encoding_fix from breaking pytest
# encoding_fix wraps sys.stdout/stderr, which conflicts
# with pytest's capture system on Windows.
# ═══════════════════════════════════════════════════
os.environ["PYTEST_RUNNING"] = "1"
os.environ["DEBUG"] = "true"  # Bypass auth in tests

# Set flags that encoding_fix.py checks
sys._called_from_test = True
sys._encoding_fix_applied = True

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch encoding_fix to be a no-op during tests
import types
encoding_fix_module = types.ModuleType("encoding_fix")
encoding_fix_module.safe_print = print  # type: ignore
sys.modules["encoding_fix"] = encoding_fix_module

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create test client for API"""
    from fastapi.testclient import TestClient
    from api.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_code():
    """Sample Python code for testing"""
    return '''
def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    return a + b

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, x, y):
        result = calculate_sum(x, y)
        self.history.append(f"{x} + {y} = {result}")
        return result
'''


@pytest.fixture
def sample_invoice():
    """Sample invoice data for testing"""
    return {
        "customer_name": "Test Customer",
        "customer_id": "CUST-001",
        "amount": 1000.0,
        "tax": 150.0,
        "total": 1150.0,
        "items": [
            {"name": "Item 1", "quantity": 2, "price": 500.0}
        ],
    }


@pytest.fixture
def auth_headers():
    """Get auth headers for protected endpoints"""
    return {"Authorization": "Bearer test-token"}
