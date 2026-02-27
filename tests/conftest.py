"""
Pytest Configuration - Shared Fixtures
======================================
Centralized test configuration and shared fixtures for all tests.
"""
import os
import sys
import pytest
import asyncio
from typing import AsyncGenerator, Generator

# Set test environment variables BEFORE importing app modules
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTEST_RUNNING"] = "1"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["ADMIN_PASSWORD"] = "president123"
os.environ["DEBUG"] = "false"

# Mark that we're running from test
sys._called_from_test = True

# Import after setting environment
from httpx import AsyncClient, ASGITransport
from api.app import app
from core.database import db_manager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def initialize_test_db():
    """Initialize test database once per session."""
    await db_manager.initialize()

    # Seed default admin user (the app lifespan skips startup tasks when
    # PYTEST_RUNNING=1, so the president admin is never created)
    try:
        from scripts.create_default_admin import init_roles
        from core.user_service import UserService
        from core.config import settings

        await init_roles()

        async with db_manager.get_session() as session:
            user_service = UserService(session)
            admin = await user_service.get_user_by_username("president")
            if not admin:
                await user_service.create_user(
                    username="president",
                    email="admin@bi-ide.com",
                    password=settings.ADMIN_PASSWORD,
                    first_name="System",
                    last_name="Administrator",
                    is_active=True,
                    is_superuser=True,
                    role_names=["admin"],
                )
    except Exception as e:
        print(f"⚠️ Admin seeding in test: {e}")

    yield
    await db_manager.close()


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an HTTP test client for API testing.
    
    Usage:
        async def test_endpoint(test_client):
            response = await test_client.get("/api/v1/health")
            assert response.status_code == 200
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_db():
    """
    Provide isolated database session for tests.
    
    Usage:
        async def test_db_operation(test_db):
            async with test_db.get_session() as session:
                # Your test code here
                pass
    """
    yield db_manager


@pytest.fixture
async def auth_headers(test_client: AsyncClient) -> dict:
    """
    Get authentication headers for a test user.
    Creates a test user and returns auth headers.
    
    Usage:
        async def test_protected_endpoint(test_client, auth_headers):
            response = await test_client.get("/api/v1/users/me", headers=auth_headers)
            assert response.status_code == 200
    """
    # Create test user
    from core.user_service import UserService
    
    async with db_manager.get_session() as session:
        user_service = UserService(session)
        
        # Check if test user exists
        user = await user_service.get_user_by_username("test_user")
        if not user:
            user = await user_service.create_user(
                username="test_user",
                email="test@example.com",
                password="TestPass123!",
                first_name="Test",
                last_name="User",
                role_names=["admin"]
            )
    
    # Login to get token
    response = await test_client.post("/api/v1/auth/login", json={
        "username": "test_user",
        "password": "TestPass123!"
    })
    
    assert response.status_code == 200, "Failed to login test user"
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(test_client: AsyncClient) -> dict:
    """Get admin authentication headers."""
    return await auth_headers(test_client)


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    from core.config import Settings
    return Settings(
        DEBUG=False,
        SECRET_KEY="test-secret",
        DATABASE_URL="sqlite+aiosqlite:///./test.db"
    )


# pytest configuration
pytest_plugins = []


@pytest.fixture
def client():
    """Provide an authenticated synchronous TestClient for API tests."""
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        # Auto-login as president to get auth token
        resp = c.post("/api/v1/auth/login", json={
            "username": "president",
            "password": "president123",
        })
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            if token:
                c.headers["Authorization"] = f"Bearer {token}"
        yield c


@pytest.fixture
async def cache():
    """Provide a CacheManager instance for cache tests."""
    from core.cache import CacheManager
    cm = CacheManager()
    await cm.initialize()
    yield cm


@pytest.fixture
async def db():
    """Provide database manager for tests (alias for test_db)."""
    yield db_manager


@pytest.fixture
def sample_code():
    """Provide sample Python code for code analysis tests."""
    return '''
def hello_world():
    """Say hello."""
    print("Hello, World!")
    return True
'''


@pytest.fixture
def sample_invoice():
    """Provide sample invoice data for ERP tests."""
    return {
        "invoice_number": "INV-TEST-001",
        "customer_name": "Test Customer",
        "customer_id": "CUST-TEST-001",
        "amount": 1000.0,
        "tax": 150.0,
        "total": 1150.0,
        "items": [{"name": "Item 1", "quantity": 2, "price": 500.0}]
    }


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")

