"""
RBAC Module - Role-Based Access Control
نظام التحكم بالصلاحيات حسب الأدوار
"""

from enum import Enum
from typing import Dict, List, Optional, Set

from fastapi import Depends, HTTPException, status


# ─────────────────── Roles ───────────────────


class Role(str, Enum):
    """System roles — ordered from least to most privileged"""
    VIEWER = "viewer"         # قراءة فقط
    EMPLOYEE = "employee"     # موظف — وصول محدود
    ACCOUNTANT = "accountant" # محاسب — وصول مالي
    MANAGER = "manager"       # مدير — وصول كامل للقسم
    ADMIN = "admin"           # مدير النظام — وصول كامل
    PRESIDENT = "president"   # الرئيس — صلاحيات كاملة + veto


# ─────────────────── Permissions ───────────────────


class Permission(str, Enum):
    """Fine-grained permissions"""
    # IDE
    IDE_READ = "ide:read"
    IDE_WRITE = "ide:write"
    IDE_EXECUTE = "ide:execute"
    IDE_DEBUG = "ide:debug"
    IDE_GIT = "ide:git"

    # ERP — Accounting
    ERP_INVOICES_READ = "erp:invoices:read"
    ERP_INVOICES_CREATE = "erp:invoices:create"
    ERP_INVOICES_EDIT = "erp:invoices:edit"
    ERP_INVOICES_DELETE = "erp:invoices:delete"
    ERP_REPORTS_READ = "erp:reports:read"

    # ERP — Inventory
    ERP_INVENTORY_READ = "erp:inventory:read"
    ERP_INVENTORY_EDIT = "erp:inventory:edit"

    # ERP — HR
    ERP_HR_READ = "erp:hr:read"
    ERP_HR_EDIT = "erp:hr:edit"
    ERP_PAYROLL_READ = "erp:payroll:read"
    ERP_PAYROLL_MANAGE = "erp:payroll:manage"

    # Council & AI
    COUNCIL_READ = "council:read"
    COUNCIL_MESSAGE = "council:message"
    COUNCIL_DISCUSS = "council:discuss"
    HIERARCHY_READ = "hierarchy:read"
    HIERARCHY_COMMAND = "hierarchy:command"

    # Network
    NETWORK_READ = "network:read"
    NETWORK_MANAGE = "network:manage"

    # System
    SYSTEM_CONFIG_READ = "system:config:read"
    SYSTEM_CONFIG_EDIT = "system:config:edit"
    SYSTEM_USERS_READ = "system:users:read"
    SYSTEM_USERS_MANAGE = "system:users:manage"


# ─────────────────── Role → Permissions Map ───────────────────

ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.IDE_READ,
        Permission.ERP_INVOICES_READ,
        Permission.ERP_INVENTORY_READ,
        Permission.ERP_HR_READ,
        Permission.COUNCIL_READ,
        Permission.HIERARCHY_READ,
        Permission.NETWORK_READ,
    },

    Role.EMPLOYEE: {
        # Everything VIEWER has + write access
        Permission.IDE_READ,
        Permission.IDE_WRITE,
        Permission.ERP_INVOICES_READ,
        Permission.ERP_INVOICES_CREATE,
        Permission.ERP_INVENTORY_READ,
        Permission.ERP_HR_READ,
        Permission.COUNCIL_READ,
        Permission.COUNCIL_MESSAGE,
        Permission.HIERARCHY_READ,
        Permission.NETWORK_READ,
    },

    Role.ACCOUNTANT: {
        Permission.IDE_READ,
        Permission.ERP_INVOICES_READ,
        Permission.ERP_INVOICES_CREATE,
        Permission.ERP_INVOICES_EDIT,
        Permission.ERP_REPORTS_READ,
        Permission.ERP_INVENTORY_READ,
        Permission.ERP_INVENTORY_EDIT,
        Permission.ERP_HR_READ,
        Permission.ERP_PAYROLL_READ,
        Permission.COUNCIL_READ,
        Permission.COUNCIL_MESSAGE,
        Permission.HIERARCHY_READ,
        Permission.NETWORK_READ,
    },

    Role.MANAGER: {
        # Almost everything
        Permission.IDE_READ,
        Permission.IDE_WRITE,
        Permission.IDE_EXECUTE,
        Permission.IDE_GIT,
        Permission.ERP_INVOICES_READ,
        Permission.ERP_INVOICES_CREATE,
        Permission.ERP_INVOICES_EDIT,
        Permission.ERP_INVOICES_DELETE,
        Permission.ERP_REPORTS_READ,
        Permission.ERP_INVENTORY_READ,
        Permission.ERP_INVENTORY_EDIT,
        Permission.ERP_HR_READ,
        Permission.ERP_HR_EDIT,
        Permission.ERP_PAYROLL_READ,
        Permission.ERP_PAYROLL_MANAGE,
        Permission.COUNCIL_READ,
        Permission.COUNCIL_MESSAGE,
        Permission.COUNCIL_DISCUSS,
        Permission.HIERARCHY_READ,
        Permission.HIERARCHY_COMMAND,
        Permission.NETWORK_READ,
        Permission.NETWORK_MANAGE,
        Permission.SYSTEM_CONFIG_READ,
        Permission.SYSTEM_USERS_READ,
    },

    Role.ADMIN: {
        perm for perm in Permission  # All permissions
    },

    Role.PRESIDENT: {
        perm for perm in Permission  # All permissions (same as admin, but with veto power)
    },
}


def get_role_permissions(role: str) -> Set[Permission]:
    """Get all permissions for a role."""
    try:
        role_enum = Role(role)
    except ValueError:
        return set()
    return ROLE_PERMISSIONS.get(role_enum, set())


def user_has_permission(user: Dict, permission: Permission) -> bool:
    """Check if a user has a specific permission."""
    role = user.get("role", "viewer")
    role_perms = get_role_permissions(role)

    # Check role-level permissions
    if permission in role_perms:
        return True

    # Check additional per-user permissions
    extra_perms = user.get("permissions", [])
    if isinstance(extra_perms, list) and permission.value in extra_perms:
        return True

    return False


# ─────────────────── FastAPI Dependencies ───────────────────

def require_permission(*permissions: Permission):
    """
    Factory that creates a FastAPI dependency to require specific permissions.

    Usage:
        @router.get("/invoices", dependencies=[Depends(require_permission(Permission.ERP_INVOICES_READ))])
        async def get_invoices(): ...
    """
    from api.auth import get_current_user

    async def _check(current_user: Dict = Depends(get_current_user)) -> Dict:
        for perm in permissions:
            if not user_has_permission(current_user, perm):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {perm.value} required",
                )
        return current_user

    return _check


def require_role(*roles: Role):
    """
    Factory that creates a FastAPI dependency to require specific roles.

    Usage:
        @router.post("/users", dependencies=[Depends(require_role(Role.ADMIN))])
        async def create_user(): ...
    """
    from api.auth import get_current_user

    async def _check(current_user: Dict = Depends(get_current_user)) -> Dict:
        user_role = current_user.get("role", "viewer")
        role_values = [r.value for r in roles]
        if user_role not in role_values:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {', '.join(role_values)} required. You have: {user_role}",
            )
        return current_user

    return _check


# ─────────────────── Helper functions ───────────────────


def get_all_roles() -> List[Dict]:
    """Get all roles with their permissions (for admin UI)."""
    return [
        {
            "role": role.value,
            "permissions": sorted([p.value for p in perms]),
            "permission_count": len(perms),
        }
        for role, perms in ROLE_PERMISSIONS.items()
    ]


def get_all_permissions() -> List[Dict]:
    """Get all available permissions (for admin UI)."""
    grouped = {}
    for perm in Permission:
        parts = perm.value.split(":")
        group = parts[0]
        if group not in grouped:
            grouped[group] = []
        grouped[group].append({
            "permission": perm.value,
            "description": perm.name.replace("_", " ").title(),
        })
    return [
        {"group": group, "permissions": perms}
        for group, perms in grouped.items()
    ]
