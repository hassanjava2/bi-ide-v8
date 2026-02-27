"""
Auth Routes - نقاط النهاية للمصادقة
JWT-based authentication with refresh token rotation
"""

import os
from typing import Optional

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import (
    authenticate_user,
    verify_token,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    security
)
from api.schemas import LoginRequest, TokenResponse, RefreshTokenRequest, RefreshTokenResponse
from core.database import get_db
from core.user_service import RefreshTokenService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Refresh token lifetime (7 days)
REFRESH_TOKEN_EXPIRE_DAYS = 7


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate and get access token.
    تسجيل الدخول والحصول على رمز الوصول.
    """
    if os.getenv("AUTH_DEBUG") == "1":
        try:
            from core.config import get_settings
            s = get_settings()
            print(
                "AUTH_DEBUG login attempt:",
                {
                    "username": request.username,
                    "password_len": len(request.password or ""),
                    "admin_password_match": (request.password == s.ADMIN_PASSWORD),
                },
            )
        except Exception as e:
            print("AUTH_DEBUG login debug failed:", str(e))

    user = await authenticate_user(request.username, request.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user["id"],
            "username": user["username"],
            "role": user["role"],
            "is_superuser": user.get("is_superuser", False)
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login/extended", response_model=RefreshTokenResponse)
async def login_with_refresh(
    request: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate and get both access token and refresh token.
    Allows for token rotation without re-login.
    """
    user = await authenticate_user(request.username, request.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(
        data={
            "sub": user["id"],
            "username": user["username"],
            "role": user["role"],
            "is_superuser": user.get("is_superuser", False)
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    # Create refresh token
    refresh_service = RefreshTokenService(db)
    refresh_token = await refresh_service.create_refresh_token(
        user_id=user["id"],
        expires_days=REFRESH_TOKEN_EXPIRE_DAYS,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent")
    )

    return RefreshTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using a valid refresh token.
    Implements token rotation - old refresh token is revoked and new one is issued.
    """
    refresh_service = RefreshTokenService(db)
    
    # Verify refresh token
    token_record = await refresh_service.verify_refresh_token(request.refresh_token)
    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user data
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from core.user_models import UserDB
    
    result = await db.execute(
        select(UserDB).options(selectinload(UserDB.roles)).where(UserDB.id == token_record.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Revoke old refresh token (token rotation)
    await refresh_service.revoke_refresh_token(request.refresh_token)
    
    # Create new access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "role": user.roles[0].name if user.roles else "viewer",
            "is_superuser": user.is_superuser
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
async def logout(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user by revoking the refresh token.
    Access token will expire naturally.
    """
    refresh_service = RefreshTokenService(db)
    revoked = await refresh_service.revoke_refresh_token(request.refresh_token)
    
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token",
        )
    
    return {"status": "success", "message": "Logged out successfully"}


@router.post("/logout-all")
async def logout_all(
    credentials: Optional[object] = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user from all devices by revoking all refresh tokens.
    Requires valid access token.
    """
    from api.auth import get_current_user
    
    current_user = await get_current_user(credentials, db)
    user_id = current_user.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )
    
    refresh_service = RefreshTokenService(db)
    revoked_count = await refresh_service.revoke_all_user_tokens(user_id)
    
    return {
        "status": "success",
        "message": "Logged out from all devices",
        "revoked_tokens": revoked_count
    }
