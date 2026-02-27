"""
User Models - نماذج المستخدمين
SQLAlchemy models for user management system
"""

import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Table, ForeignKey, Text
from sqlalchemy.orm import relationship

from core.database import Base


# Association table for user roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('role_id', String, ForeignKey('roles.id'), primary_key=True)
)


class UserDB(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("RoleDB", secondary=user_roles, back_populates="users")
    profile = relationship("UserProfileDB", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<UserDB(id={self.id}, username={self.username}, email={self.email})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name or self.username
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission"""
        if self.is_superuser:
            return True
        for role in self.roles:
            if role.has_permission(permission):
                return True
        return False
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role"""
        if self.is_superuser and role_name == "admin":
            return True
        return any(role.name == role_name for role in self.roles)
    
    def to_dict(self, include_profile: bool = False) -> dict:
        """Convert user to dictionary"""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "roles": [role.to_dict() for role in self.roles],
        }
        if include_profile and self.profile:
            data["profile"] = self.profile.to_dict()
        return data


class RoleDB(Base):
    """Role database model"""
    __tablename__ = "roles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    permissions = Column(Text, default="[]")  # JSON string of permissions
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    users = relationship("UserDB", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        return f"<RoleDB(id={self.id}, name={self.name})>"
    
    def get_permissions(self) -> List[str]:
        """Get list of permissions"""
        try:
            return json.loads(self.permissions or "[]")
        except json.JSONDecodeError:
            return []
    
    def set_permissions(self, permissions: List[str]):
        """Set permissions list"""
        self.permissions = json.dumps(permissions)
    
    def has_permission(self, permission: str) -> bool:
        """Check if role has a specific permission"""
        perms = self.get_permissions()
        return permission in perms or "*" in perms
    
    def to_dict(self) -> dict:
        """Convert role to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": self.get_permissions(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserProfileDB(Base):
    """User profile database model - additional user information"""
    __tablename__ = "user_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    phone = Column(String, nullable=True)
    department = Column(String, nullable=True)
    job_title = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    timezone = Column(String, default="Asia/Riyadh")
    language = Column(String, default="ar")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("UserDB", back_populates="profile")
    
    def __repr__(self):
        return f"<UserProfileDB(id={self.id}, user_id={self.user_id})>"
    
    def to_dict(self) -> dict:
        """Convert profile to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "phone": self.phone,
            "department": self.department,
            "job_title": self.job_title,
            "avatar_url": self.avatar_url,
            "timezone": self.timezone,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PasswordResetToken(Base):
    """Password reset token database model"""
    __tablename__ = "password_reset_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    used_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id}, used={self.used})>"
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid (not used and not expired)"""
        return not self.used and not self.is_expired()
    
    def to_dict(self) -> dict:
        """Convert token to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "token": self.token,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "used": self.used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "used_at": self.used_at.isoformat() if self.used_at else None,
            "is_valid": self.is_valid(),
        }


class RefreshToken(Base):
    """Refresh token database model for token rotation"""
    __tablename__ = "refresh_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid (not revoked and not expired)"""
        return not self.revoked and not self.is_expired()
    
    def revoke(self):
        """Revoke the token"""
        self.revoked = True
        self.revoked_at = datetime.now(timezone.utc)
