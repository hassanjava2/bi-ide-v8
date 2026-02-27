"""
User Service - خدمة المستخدمين
Business logic for user management, authentication, and authorization
"""

import uuid
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import bcrypt

from core.user_models import UserDB, RoleDB, UserProfileDB, PasswordResetToken, RefreshToken


def _hash_password(password: str) -> str:
    """Hash password with bcrypt (max 72 bytes)"""
    # bcrypt has a 72 byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    password_bytes = password.encode('utf-8')[:72]
    return bcrypt.checkpw(password_bytes, hashed.encode('utf-8'))


class UserService:
    """User management service"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    # ========== User CRUD Operations ==========
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str = None,
        last_name: str = None,
        is_active: bool = True,
        is_superuser: bool = False,
        role_names: List[str] = None
    ) -> UserDB:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password (will be hashed)
            first_name: Optional first name
            last_name: Optional last name
            is_active: Whether the account is active
            is_superuser: Whether the user has superuser privileges
            role_names: List of role names to assign
            
        Returns:
            Created UserDB instance
        """
        from core.user_models import user_roles
        
        # Hash password
        hashed = _hash_password(password)
        
        # Create user
        user = UserDB(
            username=username,
            email=email,
            hashed_password=hashed,
            first_name=first_name,
            last_name=last_name,
            is_active=is_active,
            is_superuser=is_superuser,
        )
        self.db.add(user)
        await self.db.flush()  # Flush to get the user ID
        
        # Create default profile
        profile = UserProfileDB(user_id=user.id)
        self.db.add(profile)
        
        # Assign roles if specified
        if role_names:
            for role_name in role_names:
                result = await self.db.execute(
                    select(RoleDB).where(RoleDB.name == role_name)
                )
                role = result.scalar_one_or_none()
                if role:
                    await self.db.execute(
                        user_roles.insert().values(user_id=user.id, role_id=role.id)
                    )
        
        await self.db.commit()
        await self.db.refresh(user)
        # Eager-load relationships to avoid MissingGreenlet on attribute access
        try:
            await self.db.refresh(user, attribute_names=["roles", "profile"])
        except Exception:
            pass
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserDB]:
        """Get user by ID"""
        result = await self.db.execute(
            select(UserDB)
            .options(selectinload(UserDB.roles), selectinload(UserDB.profile))
            .where(UserDB.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[UserDB]:
        """Get user by username"""
        result = await self.db.execute(
            select(UserDB)
            .options(selectinload(UserDB.roles), selectinload(UserDB.profile))
            .where(UserDB.username == username)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[UserDB]:
        """Get user by email"""
        result = await self.db.execute(
            select(UserDB)
            .options(selectinload(UserDB.roles), selectinload(UserDB.profile))
            .where(UserDB.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_users(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[UserDB]:
        """Get list of users with optional filtering"""
        query = select(UserDB).options(selectinload(UserDB.roles), selectinload(UserDB.profile))
        
        if is_active is not None:
            query = query.where(UserDB.is_active == is_active)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def update_user(
        self,
        user_id: str,
        **kwargs
    ) -> Optional[UserDB]:
        """
        Update user fields
        
        Args:
            user_id: User ID to update
            **kwargs: Fields to update (email, first_name, last_name, is_active, etc.)
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ['email', 'first_name', 'last_name', 'is_active', 'is_superuser', 'email_verified']
        for field, value in kwargs.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)
        
        user.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(user)
        return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        await self.db.delete(user)
        await self.db.commit()
        return True
    
    async def change_password(self, user_id: str, new_password: str) -> bool:
        """Change user password"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.hashed_password = _hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        return True
    
    async def verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        return _verify_password(password, user.hashed_password)
    
    # ========== Authentication ==========
    
    async def authenticate(
        self,
        username_or_email: str,
        password: str
    ) -> Optional[UserDB]:
        """
        Authenticate user by username/email and password
        
        Args:
            username_or_email: Username or email
            password: Plain text password
            
        Returns:
            UserDB instance if authenticated, None otherwise
        """
        result = await self.db.execute(
            select(UserDB).where(
                or_(
                    UserDB.username == username_or_email,
                    UserDB.email == username_or_email
                )
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not _verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        await self.db.commit()

        # Ensure relationships are loaded for downstream serialization
        try:
            await self.db.refresh(user, attribute_names=["roles", "profile"])
        except Exception:
            pass
        
        return user
    
    # ========== Role Management ==========
    
    async def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user"""
        from core.user_models import user_roles
        
        # Check if user exists
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Get role
        result = await self.db.execute(
            select(RoleDB).where(RoleDB.name == role_name)
        )
        role = result.scalar_one_or_none()
        
        if not role:
            return False
        
        # Check if association already exists
        result = await self.db.execute(
            select(user_roles).where(
                and_(
                    user_roles.c.user_id == user_id,
                    user_roles.c.role_id == role.id
                )
            )
        )
        existing = result.fetchone()
        
        if not existing:
            # Insert association directly
            await self.db.execute(
                user_roles.insert().values(user_id=user_id, role_id=role.id)
            )
            await self.db.commit()
        
        return True
    
    async def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Remove a role from a user"""
        from core.user_models import user_roles
        
        # Get role
        result = await self.db.execute(
            select(RoleDB).where(RoleDB.name == role_name)
        )
        role = result.scalar_one_or_none()
        
        if not role:
            return False
        
        # Delete association directly
        await self.db.execute(
            user_roles.delete().where(
                and_(
                    user_roles.c.user_id == user_id,
                    user_roles.c.role_id == role.id
                )
            )
        )
        await self.db.commit()
        
        return True
    
    # ========== Profile Management ==========
    
    async def update_profile(
        self,
        user_id: str,
        **kwargs
    ) -> Optional[UserProfileDB]:
        """Update user profile"""
        result = await self.db.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
        
        if not profile:
            # Create profile if doesn't exist
            profile = UserProfileDB(user_id=user_id)
            self.db.add(profile)
        
        allowed_fields = ['phone', 'department', 'job_title', 'avatar_url', 'timezone', 'language']
        for field, value in kwargs.items():
            if field in allowed_fields and hasattr(profile, field):
                setattr(profile, field, value)
        
        profile.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(profile)
        return profile
    
    async def get_profile(self, user_id: str) -> Optional[UserProfileDB]:
        """Get user profile"""
        result = await self.db.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    # ========== Password Reset ==========
    
    async def create_password_reset_token(
        self,
        email: str,
        expires_hours: int = 24
    ) -> Optional[str]:
        """
        Create a password reset token for user
        
        Returns:
            Reset token string if user exists, None otherwise
        """
        user = await self.get_user_by_email(email)
        if not user:
            return None
        
        # Generate secure token
        token = secrets.token_urlsafe(32)
        
        # Create reset token record
        reset_token = PasswordResetToken(
            user_id=user.id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        )
        self.db.add(reset_token)
        await self.db.commit()
        
        return token
    
    async def verify_password_reset_token(self, token: str) -> Optional[UserDB]:
        """Verify a password reset token and return user if valid"""
        result = await self.db.execute(
            select(PasswordResetToken).where(
                and_(
                    PasswordResetToken.token == token,
                    PasswordResetToken.used == False,
                    PasswordResetToken.expires_at > datetime.now(timezone.utc)
                )
            )
        )
        reset_token = result.scalar_one_or_none()
        
        if not reset_token:
            return None
        
        return await self.get_user_by_id(reset_token.user_id)
    
    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset password using a valid reset token
        
        Returns:
            True if password was reset successfully
        """
        result = await self.db.execute(
            select(PasswordResetToken).where(
                and_(
                    PasswordResetToken.token == token,
                    PasswordResetToken.used == False,
                    PasswordResetToken.expires_at > datetime.now(timezone.utc)
                )
            )
        )
        reset_token = result.scalar_one_or_none()
        
        if not reset_token:
            return False
        
        # Update password
        user = await self.get_user_by_id(reset_token.user_id)
        if not user:
            return False
        
        user.hashed_password = _hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        
        # Mark token as used
        reset_token.used = True
        reset_token.used_at = datetime.now(timezone.utc)
        
        await self.db.commit()
        return True


class RoleService:
    """Role management service"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_role(
        self,
        name: str,
        description: str = None,
        permissions: List[str] = None
    ) -> RoleDB:
        """
        Create a new role
        
        Args:
            name: Unique role name
            description: Role description
            permissions: List of permission strings
        """
        role = RoleDB(
            name=name,
            description=description,
            permissions=json.dumps(permissions or [])
        )
        self.db.add(role)
        await self.db.commit()
        await self.db.refresh(role)
        return role
    
    async def get_role_by_id(self, role_id: str) -> Optional[RoleDB]:
        """Get role by ID"""
        result = await self.db.execute(
            select(RoleDB).where(RoleDB.id == role_id)
        )
        return result.scalar_one_or_none()
    
    async def get_role_by_name(self, name: str) -> Optional[RoleDB]:
        """Get role by name"""
        result = await self.db.execute(
            select(RoleDB).where(RoleDB.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_roles(self, skip: int = 0, limit: int = 100) -> List[RoleDB]:
        """Get all roles"""
        result = await self.db.execute(
            select(RoleDB).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def update_role(
        self,
        role_id: str,
        description: str = None,
        permissions: List[str] = None
    ) -> Optional[RoleDB]:
        """Update role fields"""
        role = await self.get_role_by_id(role_id)
        if not role:
            return None
        
        if description is not None:
            role.description = description
        
        if permissions is not None:
            role.set_permissions(permissions)
        
        await self.db.commit()
        await self.db.refresh(role)
        return role
    
    async def delete_role(self, role_id: str) -> bool:
        """Delete a role"""
        role = await self.get_role_by_id(role_id)
        if not role:
            return False
        
        await self.db.delete(role)
        await self.db.commit()
        return True
    
    async def add_permission(self, role_id: str, permission: str) -> bool:
        """Add a permission to a role"""
        role = await self.get_role_by_id(role_id)
        if not role:
            return False
        
        permissions = role.get_permissions()
        if permission not in permissions:
            permissions.append(permission)
            role.set_permissions(permissions)
            await self.db.commit()
        
        return True
    
    async def remove_permission(self, role_id: str, permission: str) -> bool:
        """Remove a permission from a role"""
        role = await self.get_role_by_id(role_id)
        if not role:
            return False
        
        permissions = role.get_permissions()
        if permission in permissions:
            permissions.remove(permission)
            role.set_permissions(permissions)
            await self.db.commit()
        
        return True


class RefreshTokenService:
    """Refresh token management service for token rotation"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create_refresh_token(
        self,
        user_id: str,
        expires_days: int = 7,
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """Create a new refresh token"""
        token = secrets.token_urlsafe(32)
        
        refresh_token = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days),
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.db.add(refresh_token)
        await self.db.commit()
        
        return token
    
    async def verify_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Verify a refresh token is valid"""
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.token == token,
                    RefreshToken.revoked == False,
                    RefreshToken.expires_at > datetime.now(timezone.utc)
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def revoke_refresh_token(self, token: str) -> bool:
        """Revoke a refresh token"""
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token == token)
        )
        refresh_token = result.scalar_one_or_none()
        
        if not refresh_token:
            return False
        
        refresh_token.revoke()
        await self.db.commit()
        return True
    
    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """Revoke all refresh tokens for a user"""
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.user_id == user_id,
                    RefreshToken.revoked == False
                )
            )
        )
        tokens = result.scalars().all()
        
        count = 0
        for token in tokens:
            token.revoke()
            count += 1
        
        if count > 0:
            await self.db.commit()
        
        return count
    
    async def cleanup_expired_tokens(self) -> int:
        """Delete expired and revoked tokens (cleanup task)"""
        result = await self.db.execute(
            select(RefreshToken).where(
                or_(
                    RefreshToken.expires_at < datetime.now(timezone.utc),
                    RefreshToken.revoked == True
                )
            )
        )
        tokens = result.scalars().all()
        
        count = 0
        for token in tokens:
            await self.db.delete(token)
            count += 1
        
        if count > 0:
            await self.db.commit()
        
        return count


# ========== Default Roles Definition ==========

DEFAULT_ROLES = {
    "admin": {
        "description": "مدير النظام - Full system access",
        "permissions": ["*"]  # All permissions
    },
    "accountant": {
        "description": "محاسب - Access to financial modules",
        "permissions": [
            "erp.invoices.read",
            "erp.invoices.write",
            "erp.payments.read",
            "erp.payments.write",
            "erp.reports.read",
            "erp.financial.read"
        ]
    },
    "hr_manager": {
        "description": "مدير الموارد البشرية - HR management access",
        "permissions": [
            "erp.employees.read",
            "erp.employees.write",
            "erp.attendance.read",
            "erp.attendance.write",
            "erp.payroll.read",
            "erp.payroll.write",
            "erp.hr_reports.read"
        ]
    },
    "salesperson": {
        "description": "مندوب مبيعات - Sales and customer access",
        "permissions": [
            "erp.customers.read",
            "erp.customers.write",
            "erp.orders.read",
            "erp.orders.write",
            "erp.products.read",
            "erp.sales_reports.read"
        ]
    },
    "viewer": {
        "description": "مشاهد - Read-only access",
        "permissions": [
            "erp.invoices.read",
            "erp.customers.read",
            "erp.products.read",
            "erp.reports.read"
        ]
    },
    "developer": {
        "description": "مطور - IDE and code access",
        "permissions": [
            "ide.code.read",
            "ide.code.write",
            "ide.debug",
            "ide.git",
            "ide.terminal",
            "ai.suggest"
        ]
    }
}
