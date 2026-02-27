"""
User Management Routes - نقاط النهاية لإدارة المستخدمين
API endpoints for user management, authentication, and authorization
"""

from typing import List, Optional
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

from core.database import db_manager, get_db
from core.user_models import UserDB, RoleDB
from core.user_service import UserService, RoleService, RefreshTokenService
from api.auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    require_admin
)

router = APIRouter(prefix="/api/v1/users", tags=["users"])
security = HTTPBearer()


# ========== Pydantic Schemas ==========

class UserRegisterRequest(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)


class UserCreateRequest(BaseModel):
    """User creation request (admin only)"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    is_active: bool = True
    is_superuser: bool = False
    role_names: Optional[List[str]] = []


class UserUpdateRequest(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    full_name: str
    is_active: bool
    is_superuser: bool
    email_verified: bool
    created_at: str
    last_login: Optional[str]
    roles: List[dict]
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Optional[UserResponse] = None


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


class ProfileUpdateRequest(BaseModel):
    """Profile update request"""
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=100)
    job_title: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = None
    timezone: Optional[str] = Field(None, max_length=50)
    language: Optional[str] = Field(None, max_length=10)


class ProfileResponse(BaseModel):
    """Profile response"""
    id: str
    user_id: str
    phone: Optional[str]
    department: Optional[str]
    job_title: Optional[str]
    avatar_url: Optional[str]
    timezone: str
    language: str


class RoleCreateRequest(BaseModel):
    """Role creation request"""
    name: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = None
    permissions: List[str] = []


class RoleResponse(BaseModel):
    """Role response"""
    id: str
    name: str
    description: Optional[str]
    permissions: List[str]
    created_at: str


# ========== Auth Routes ==========

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user
    تسجيل مستخدم جديد
    """
    user_service = UserService(db)
    
    # Check if username exists
    existing_user = await user_service.get_user_by_username(request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Check if email exists
    existing_email = await user_service.get_user_by_email(request.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = await user_service.create_user(
        username=request.username,
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name
    )
    
    return user.to_dict()


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate and get access token
    تسجيل الدخول والحصول على رمز الوصول
    """
    user_service = UserService(db)
    token_service = RefreshTokenService(db)
    
    # Authenticate user
    user = await user_service.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Create tokens
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "is_superuser": user.is_superuser
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    
    # Create refresh token
    refresh_token = await token_service.create_refresh_token(
        user_id=user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent")
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user.to_dict()
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    تجديد رمز الوصول باستخدام رمز التجديد
    """
    token_service = RefreshTokenService(db)
    user_service = UserService(db)
    
    # Verify refresh token
    refresh_token_record = await token_service.verify_refresh_token(credentials.credentials)
    if not refresh_token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user
    user = await user_service.get_user_by_id(refresh_token_record.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "is_superuser": user.is_superuser
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout and revoke refresh token
    تسجيل الخروج وإلغاء رمز التجديد
    """
    token_service = RefreshTokenService(db)
    success = await token_service.revoke_refresh_token(credentials.credentials)
    
    return {"message": "Logged out successfully", "revoked": success}


@router.post("/password-reset-request")
async def request_password_reset(
    request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset
    طلب إعادة تعيين كلمة المرور
    
    Security: Always returns success even if email not found to prevent email enumeration.
    The token is NEVER returned in the response.
    """
    user_service = UserService(db)
    
    # Create reset token (returns None if email not found)
    token = await user_service.create_password_reset_token(request.email)
    
    # Send email if user exists
    if token:
        try:
            from core.email_service import send_password_reset_email
            await send_password_reset_email(request.email, token)
        except Exception as e:
            # Log error but don't expose to user (security)
            print(f"⚠️ Failed to send password reset email: {e}")
            # Still continue - don't let attacker know if email exists
    
    # ✅ SECURITY FIX: Never return token in response!
    return {
        "message": "If the email exists, a password reset link has been sent"
    }


@router.post("/password-reset")
async def confirm_password_reset(
    request: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm password reset with token
    تأكيد إعادة تعيين كلمة المرور بالرمز
    """
    user_service = UserService(db)
    
    success = await user_service.reset_password(request.token, request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )
    
    return {"message": "Password reset successfully"}


# ========== Current User Routes ==========

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user information
    الحصول على معلومات المستخدم الحالي
    """
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from core.user_models import UserDB
    
    # Load user with roles and profile eagerly
    result = await db.execute(
        select(UserDB).options(
            selectinload(UserDB.roles),
            selectinload(UserDB.profile)
        ).where(UserDB.id == current_user["sub"])
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user.to_dict(include_profile=True)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    request: UserUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user information
    تحديث معلومات المستخدم الحالي
    """
    user_service = UserService(db)
    
    update_data = request.model_dump(exclude_unset=True)
    user = await user_service.update_user(current_user["sub"], **update_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user.to_dict()


@router.post("/me/change-password")
async def change_current_user_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user password
    تغيير كلمة مرور المستخدم الحالي
    """
    user_service = UserService(db)
    
    # Verify current password
    if not await user_service.verify_password(current_user["sub"], request.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Change password
    success = await user_service.change_password(current_user["sub"], request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )
    
    # Revoke all refresh tokens for security
    token_service = RefreshTokenService(db)
    await token_service.revoke_all_user_tokens(current_user["sub"])
    
    return {"message": "Password changed successfully. Please login again."}


# ========== Profile Routes ==========

@router.get("/me/profile", response_model=ProfileResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user profile
    الحصول على الملف الشخصي للمستخدم
    """
    user_service = UserService(db)
    profile = await user_service.get_profile(current_user["sub"])
    
    if not profile:
        # Create empty profile if doesn't exist
        profile = await user_service.update_profile(current_user["sub"])
    
    return profile.to_dict()


@router.put("/me/profile", response_model=ProfileResponse)
async def update_current_user_profile(
    request: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user profile
    تحديث الملف الشخصي للمستخدم
    """
    user_service = UserService(db)
    
    update_data = request.model_dump(exclude_unset=True)
    profile = await user_service.update_profile(current_user["sub"], **update_data)
    
    return profile.to_dict()


# ========== Admin User Routes ==========

@router.get("", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users (admin only)
    قائمة جميع المستخدمين (للمدير فقط)
    """
    user_service = UserService(db)
    users = await user_service.get_users(skip=skip, limit=limit, is_active=is_active)
    
    return [user.to_dict(include_profile=True) for user in users]


@router.post("/create", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    request: UserCreateRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user (admin only)
    إنشاء مستخدم جديد (للمدير فقط)
    """
    user_service = UserService(db)
    
    # Check if username exists
    existing_user = await user_service.get_user_by_username(request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Check if email exists
    existing_email = await user_service.get_user_by_email(request.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = await user_service.create_user(
        username=request.username,
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name,
        is_active=request.is_active,
        is_superuser=request.is_superuser,
        role_names=request.role_names or []
    )
    
    return user.to_dict(include_profile=True)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user by ID (admin only)
    الحصول على مستخدم بالمعرف (للمدير فقط)
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user.to_dict(include_profile=True)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user (admin only)
    تحديث مستخدم (للمدير فقط)
    """
    user_service = UserService(db)
    
    update_data = request.model_dump(exclude_unset=True)
    user = await user_service.update_user(user_id, **update_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user.to_dict(include_profile=True)


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user (admin only)
    حذف مستخدم (للمدير فقط)
    """
    user_service = UserService(db)
    
    # Prevent self-deletion
    if user_id == current_user.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    success = await user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deleted successfully"}


@router.post("/{user_id}/roles/{role_name}")
async def assign_role(
    user_id: str,
    role_name: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Assign role to user (admin only)
    تعيين دور لمستخدم (للمدير فقط)
    """
    user_service = UserService(db)
    
    success = await user_service.assign_role_to_user(user_id, role_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User or role not found"
        )
    
    return {"message": f"Role '{role_name}' assigned to user"}


@router.delete("/{user_id}/roles/{role_name}")
async def remove_role(
    user_id: str,
    role_name: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove role from user (admin only)
    إزالة دور من مستخدم (للمدير فقط)
    """
    user_service = UserService(db)
    
    success = await user_service.remove_role_from_user(user_id, role_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User or role not found"
        )
    
    return {"message": f"Role '{role_name}' removed from user"}


# ========== Role Management Routes ==========

@router.get("/roles/list", response_model=List[RoleResponse])
async def list_roles(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all roles (admin only)
    قائمة جميع الأدوار (للمدير فقط)
    """
    role_service = RoleService(db)
    roles = await role_service.get_roles(skip=skip, limit=limit)
    
    return [role.to_dict() for role in roles]


@router.post("/roles/create", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    request: RoleCreateRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new role (admin only)
    إنشاء دور جديد (للمدير فقط)
    """
    role_service = RoleService(db)
    
    # Check if role name exists
    existing = await role_service.get_role_by_name(request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role name already exists"
        )
    
    role = await role_service.create_role(
        name=request.name,
        description=request.description,
        permissions=request.permissions
    )
    
    return role.to_dict()


@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: str,
    request: RoleCreateRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update role (admin only)
    تحديث دور (للمدير فقط)
    """
    role_service = RoleService(db)
    
    role = await role_service.update_role(
        role_id=role_id,
        description=request.description,
        permissions=request.permissions
    )
    
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    return role.to_dict()


@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete role (admin only)
    حذف دور (للمدير فقط)
    """
    role_service = RoleService(db)
    
    success = await role_service.delete_role(role_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    return {"message": "Role deleted successfully"}
