"""
Admin Routes - نقاط النهاية لإدارة النظام
إدارة المستخدمين والصلاحيات
"""

from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.rbac import (
    Role, Permission,
    require_role, require_permission,
    get_all_roles, get_all_permissions,
    get_role_permissions,
)
from core.database import get_db
from core.user_service import UserService, RoleService

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ─────────────── Request Models ───────────────

class CreateUserRequest(BaseModel):
    username: str
    password: str
    full_name: str = ""
    email: Optional[str] = None
    role: str = "viewer"
    permissions: List[str] = []


class UpdateUserRoleRequest(BaseModel):
    role: str
    permissions: Optional[List[str]] = None


# ─────────────── Role & Permission Info ───────────────

@router.get(
    "/roles",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_READ))],
)
async def list_roles():
    """قائمة الأدوار والصلاحيات"""
    return {
        "roles": get_all_roles(),
        "total": len(Role),
    }


@router.get(
    "/permissions",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_READ))],
)
async def list_permissions():
    """كل الصلاحيات المتوفرة"""
    return {
        "permissions": get_all_permissions(),
        "total": len(Permission),
    }


# ─────────────── User Management ───────────────


@router.get(
    "/users",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_READ))],
)
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """قائمة المستخدمين"""
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    from core.user_models import UserDB
    
    # Load users with roles eagerly
    result = await db.execute(
        select(UserDB).options(selectinload(UserDB.roles)).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    
    return {
        "users": [user.to_dict(include_profile=False) for user in users],
        "total": len(users)
    }


@router.post(
    "/users",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def create_user(request: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """إنشاء مستخدم جديد"""
    user_service = UserService(db)
    
    # Check if username exists
    existing = await user_service.get_user_by_username(request.username)
    if existing:
        raise HTTPException(400, f"User '{request.username}' already exists")
    
    # Check if email exists
    if request.email:
        existing_email = await user_service.get_user_by_email(request.email)
        if existing_email:
            raise HTTPException(400, f"Email '{request.email}' already exists")
    
    # Validate role
    try:
        Role(request.role)
    except ValueError:
        valid_roles = [r.value for r in Role]
        raise HTTPException(400, f"Invalid role '{request.role}'. Valid: {valid_roles}")
    
    # Create user
    email = request.email or f"{request.username}@bi-ide.local"
    names = request.full_name.split(" ", 1) if request.full_name else ["", ""]
    first_name = names[0]
    last_name = names[1] if len(names) > 1 else ""
    
    user = await user_service.create_user(
        username=request.username,
        email=email,
        password=request.password,
        first_name=first_name,
        last_name=last_name,
        role_names=[request.role]
    )

    return {
        "message": f"User '{request.username}' created",
        "username": request.username,
        "role": request.role,
        "id": user.id,
    }


@router.patch(
    "/users/{username}/role",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def update_user_role(username: str, request: UpdateUserRoleRequest, db: AsyncSession = Depends(get_db)):
    """تحديث دور المستخدم"""
    user_service = UserService(db)
    
    # Get user by username
    user = await user_service.get_user_by_username(username)
    if not user:
        raise HTTPException(404, f"User '{username}' not found")

    # Validate role
    try:
        Role(request.role)
    except ValueError:
        valid_roles = [r.value for r in Role]
        raise HTTPException(400, f"Invalid role '{request.role}'. Valid: {valid_roles}")

    # Assign role
    await user_service.assign_role_to_user(user.id, request.role)

    return {
        "message": f"User '{username}' updated",
        "role": request.role,
    }


@router.delete(
    "/users/{username}",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def delete_user(username: str, db: AsyncSession = Depends(get_db)):
    """حذف مستخدم"""
    user_service = UserService(db)
    
    # Get user by username
    user = await user_service.get_user_by_username(username)
    if not user:
        raise HTTPException(404, f"User '{username}' not found")

    if username == "president":
        raise HTTPException(403, "Cannot delete the president account")

    await user_service.delete_user(user.id)
    return {"message": f"User '{username}' deleted"}


@router.get("/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """معلومات المستخدم الحالي"""
    role = current_user.get("role", "viewer")
    perms = get_role_permissions(role)
    extra = current_user.get("permissions", [])

    return {
        "username": current_user.get("username", "unknown"),
        "role": role,
        "role_permissions": sorted([p.value for p in perms]),
        "extra_permissions": extra,
        "is_admin": role in ("admin", "president") or current_user.get("is_superuser", False),
    }
