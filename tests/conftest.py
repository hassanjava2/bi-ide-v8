"""
Pytest Configuration - Shared Fixtures
======================================
Centralized test configuration and shared fixtures for all tests.

الإعدادات والأدوات المشتركة للاختبارات
"""
import os
import sys
import pytest
import asyncio
from typing import AsyncGenerator, Generator, Dict, Any, List
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

# Set test environment variables BEFORE importing app modules
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTEST_RUNNING"] = "1"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["ADMIN_PASSWORD"] = "president123"
os.environ["DEBUG"] = "false"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["ORCHESTRATOR_TOKEN"] = "test-orchestrator-token"

# Mark that we're running from test
sys._called_from_test = True

# Import after setting environment
from httpx import AsyncClient, ASGITransport

# Import app and database - fail fast if imports fail
from api.app import app
from core.database import db_manager


# =============================================================================
# Event Loop Fixture
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    إنشاء حلقة الأحداث للجلسة
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
async def initialize_test_db():
    """
    Initialize test database once per session.
    تهيئة قاعدة البيانات للاختبار
    """
    await db_manager.initialize()
    
    # Seed default admin user
    try:
        from scripts.create_default_admin import init_roles
        from core.user_service import UserService
        
        await init_roles()
        
        async with db_manager.get_session() as session:
            user_service = UserService(session)
            admin = await user_service.get_user_by_username("president")
            if not admin:
                await user_service.create_user(
                    username="president",
                    email="president@bi-ide.com",
                    password="president123",
                    role_names=["admin"]
                )
    except Exception as e:
        print(f"Warning: Could not seed admin user: {e}")
    
    yield
    
    # Cleanup after all tests
    await db_manager.close()


@pytest.fixture
async def db_session() -> AsyncGenerator:
    """
    Provide a database session for tests.
    جلسة قاعدة بيانات للاختبارات
    """
    async with db_manager.get_session() as session:
        yield session


# =============================================================================
# Application Fixtures
# =============================================================================

@pytest.fixture
def test_client() -> Generator:
    """
    Create a test client for the FastAPI app.
    عميل اختبار لتطبيق FastAPI
    """
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(test_client):
    """Backward-compatible alias for tests expecting `client`."""
    return test_client


@pytest.fixture
async def async_client() -> AsyncGenerator:
    """
    Create an async test client for the FastAPI app.
    عميل اختبار غير متزامن
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
async def auth_token(async_client: AsyncClient) -> str:
    """
    Get authentication token for testing.
    رمز مصادقة للاختبار
    """
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "president", "password": "president123"}
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    return response.json()["access_token"]


@pytest.fixture
async def authorized_client(async_client: AsyncClient, auth_token: str) -> AsyncClient:
    """
    Create an authorized client with authentication headers.
    عميل مصرح له
    """
    async_client.headers["Authorization"] = f"Bearer {auth_token}"
    return async_client


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_redis():
    """
    Mock Redis for testing.
    محاكاة Redis
    """
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_training_service():
    """
    Mock training service for testing.
    محاكاة خدمة التدريب
    """
    mock = MagicMock()
    mock.start_training = AsyncMock(return_value={"status": "started", "job_id": "test-123"})
    mock.get_status = AsyncMock(return_value={"is_training": False, "progress": 0})
    return mock


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_user_data():
    """
    Sample user data for testing.
    بيانات مستخدم نموذجية
    """
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123!",
        "role": "developer"
    }


@pytest.fixture
def sample_training_config():
    """
    Sample training configuration for testing.
    إعدادات تدريب نموذجية
    """
    return {
        "model_preset": "medium",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "devices": ["cuda"],
        "distributed": False
    }


@pytest.fixture
def sample_council_query():
    """
    Sample council query for testing.
    استعلام مجلس نموذجي
    """
    return {
        "question": "How should we improve the system performance?",
        "context": {"current_load": "high", "priority": "critical"},
        "urgency": "high",
        "require_full_council": True
    }
