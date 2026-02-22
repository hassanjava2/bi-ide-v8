"""
Auth Routes - نقاط النهاية للمصادقة
"""

import os

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status

from api.auth import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from api.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
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

    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh")
async def refresh_token():
    """Refresh access token (placeholder)"""
    # TODO: Implement refresh token logic
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Token refresh not yet implemented",
    )
