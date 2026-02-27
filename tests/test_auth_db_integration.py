"""
Test Auth Database Integration
اختبارات تكامل نظام المصادقة مع قاعدة البيانات
"""

import asyncio
import pytest
from datetime import timedelta

# Mark all tests as async
pytestmark = pytest.mark.asyncio


async def test_get_db_dependency():
    """Test that get_db dependency works"""
    from core.database import get_db, db_manager
    
    await db_manager.initialize()
    
    # Test that get_db yields a session
    async for session in get_db():
        assert session is not None
        # Verify it's an async session
        from sqlalchemy.ext.asyncio import AsyncSession
        assert isinstance(session, AsyncSession)
        break
    
    await db_manager.close()


async def test_authenticate_user_with_db():
    """Test authenticate_user function with database"""
    from api.auth import authenticate_user
    from core.database import db_manager
    from core.user_service import UserService
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from core.user_models import UserDB
    
    await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        # Create a test user
        user_service = UserService(session)
        
        # Clean up any existing test user
        existing = await user_service.get_user_by_username("test_auth_user")
        if existing:
            await user_service.delete_user(existing.id)
        
        # Create test user
        user = await user_service.create_user(
            username="test_auth_user",
            email="test_auth@example.com",
            password="TestPass123!",
            first_name="Test",
            last_name="User",
            role_names=["viewer"]
        )
        
        # Eager load roles for the user
        result = await session.execute(
            select(UserDB).options(selectinload(UserDB.roles)).where(UserDB.id == user.id)
        )
        user_with_roles = result.scalar_one()
        
        # Test successful authentication
        result = await authenticate_user("test_auth_user", "TestPass123!", session)
        assert result is not None
        assert result["username"] == "test_auth_user"
        # Role should be loaded via eager loading in authenticate_user
        assert "id" in result
        
        # Test failed authentication - wrong password
        result = await authenticate_user("test_auth_user", "WrongPass", session)
        assert result is None
        
        # Test failed authentication - wrong username
        result = await authenticate_user("nonexistent", "TestPass123!", session)
        assert result is None
        
        # Cleanup
        await user_service.delete_user(user.id)
    
    await db_manager.close()


async def test_verify_token_with_db():
    """Test verify_token function with database"""
    from api.auth import verify_token, create_access_token
    from core.database import db_manager
    from core.user_service import UserService
    
    await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        # Create a test user
        user_service = UserService(session)
        
        # Clean up any existing test user
        existing = await user_service.get_user_by_username("test_token_user")
        if existing:
            await user_service.delete_user(existing.id)
        
        user = await user_service.create_user(
            username="test_token_user",
            email="test_token@example.com",
            password="TestPass123!",
            first_name="Test",
            last_name="User",
            role_names=["admin"]
        )
        
        # Create token
        token = create_access_token({
            "sub": user.id,
            "username": user.username,
            "role": "admin"
        })
        
        # Test verify token - this uses selectinload internally
        payload = await verify_token(token, session)
        assert payload is not None
        assert payload["sub"] == user.id
        assert payload["username"] == "test_token_user"
        assert payload["role"] == "admin"
        
        # Test invalid token
        payload = await verify_token("invalid_token", session)
        assert payload is None
        
        # Cleanup
        await user_service.delete_user(user.id)
    
    await db_manager.close()


async def test_create_access_token():
    """Test create_access_token function"""
    from api.auth import create_access_token, SECRET_KEY, ALGORITHM
    from jose import jwt
    
    data = {"sub": "user123", "username": "testuser", "role": "admin"}
    token = create_access_token(data)
    
    # Verify token can be decoded
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == "user123"
    assert decoded["username"] == "testuser"
    assert decoded["role"] == "admin"
    assert "exp" in decoded
    assert "iat" in decoded


async def test_auth_routes_integration():
    """Test auth routes with database"""
    from httpx import AsyncClient, ASGITransport
    from api.app import app
    from core.database import db_manager
    from core.user_service import UserService
    
    await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        # Create a test user
        user_service = UserService(session)
        
        # Clean up any existing test user
        existing = await user_service.get_user_by_username("test_api_user")
        if existing:
            await user_service.delete_user(existing.id)
        
        await user_service.create_user(
            username="test_api_user",
            email="test_api@example.com",
            password="TestPass123!",
            first_name="Test",
            last_name="User",
            role_names=["viewer"]
        )
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test login with valid credentials
        response = await client.post("/api/v1/auth/login", json={
            "username": "test_api_user",
            "password": "TestPass123!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        
        # Test login with invalid credentials
        response = await client.post("/api/v1/auth/login", json={
            "username": "test_api_user",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
    
    # Cleanup
    async with db_manager.get_session() as session:
        user_service = UserService(session)
        existing = await user_service.get_user_by_username("test_api_user")
        if existing:
            await user_service.delete_user(existing.id)
    
    await db_manager.close()


async def test_default_admin_script():
    """Test create_default_admin script"""
    from scripts.create_default_admin import create_admin, init_roles
    from core.database import db_manager
    from core.user_service import UserService
    
    await db_manager.initialize()
    
    # Run init_roles first
    await init_roles()
    
    # Run create_admin
    await create_admin("president_test", "admin_test@bi-ide.com")
    
    # Verify admin was created
    async with db_manager.get_session() as session:
        user_service = UserService(session)
        admin = await user_service.get_user_by_username("president_test")
        assert admin is not None
        assert admin.is_superuser is True
        assert admin.has_role("admin") is True
        
        # Cleanup
        await user_service.delete_user(admin.id)
    
    await db_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
