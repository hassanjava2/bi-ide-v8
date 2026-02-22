"""
Admin Routes - نقاط النهاية لإدارة النظام
إدارة المستخدمين والصلاحيات
"""

from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user, _hash_password, DEFAULT_USERS
from api.rbac import (
    Role, Permission,
    require_role, require_permission,
    get_all_roles, get_all_permissions,
    get_role_permissions,
)

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
async def list_users():
    """قائمة المستخدمين"""
    users = []
    for username, user_data in DEFAULT_USERS.items():
        users.append({
            "username": username,
            "full_name": user_data.get("full_name", ""),
            "role": user_data.get("role", "viewer"),
            "permissions": user_data.get("permissions", []),
        })
    return {"users": users, "total": len(users)}


@router.post(
    "/users",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def create_user(request: CreateUserRequest):
    """إنشاء مستخدم جديد"""
    if request.username in DEFAULT_USERS:
        raise HTTPException(400, f"User '{request.username}' already exists")

    # Validate role
    try:
        Role(request.role)
    except ValueError:
        valid_roles = [r.value for r in Role]
        raise HTTPException(400, f"Invalid role '{request.role}'. Valid: {valid_roles}")

    DEFAULT_USERS[request.username] = {
        "username": request.username,
        "hashed_password": _hash_password(request.password),
        "full_name": request.full_name,
        "email": request.email,
        "role": request.role,
        "permissions": request.permissions,
        "created_at": datetime.utcnow().isoformat(),
    }

    return {
        "message": f"User '{request.username}' created",
        "username": request.username,
        "role": request.role,
    }


@router.patch(
    "/users/{username}/role",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def update_user_role(username: str, request: UpdateUserRoleRequest):
    """تحديث دور المستخدم"""
    if username not in DEFAULT_USERS:
        raise HTTPException(404, f"User '{username}' not found")

    # Validate role
    try:
        Role(request.role)
    except ValueError:
        valid_roles = [r.value for r in Role]
        raise HTTPException(400, f"Invalid role '{request.role}'. Valid: {valid_roles}")

    DEFAULT_USERS[username]["role"] = request.role
    if request.permissions is not None:
        DEFAULT_USERS[username]["permissions"] = request.permissions

    return {
        "message": f"User '{username}' updated",
        "role": request.role,
        "permissions": DEFAULT_USERS[username].get("permissions", []),
    }


@router.delete(
    "/users/{username}",
    dependencies=[Depends(require_permission(Permission.SYSTEM_USERS_MANAGE))],
)
async def delete_user(username: str):
    """حذف مستخدم"""
    if username not in DEFAULT_USERS:
        raise HTTPException(404, f"User '{username}' not found")

    if username == "president":
        raise HTTPException(403, "Cannot delete the president account")

    del DEFAULT_USERS[username]
    return {"message": f"User '{username}' deleted"}


@router.get("/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """معلومات المستخدم الحالي"""
    role = current_user.get("role", "viewer")
    perms = get_role_permissions(role)
    extra = current_user.get("permissions", [])

    return {
        "username": current_user.get("sub", "unknown"),
        "role": role,
        "role_permissions": sorted([p.value for p in perms]),
        "extra_permissions": extra,
        "is_admin": role in ("admin", "president"),
    }
