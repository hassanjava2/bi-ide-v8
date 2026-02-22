"""
Authentication Module - JWT-based authentication
نظام المصادقة بـ JWT
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import get_settings

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("⚠️ python-jose not installed. Auth will use fallback mode.")

# Use bcrypt directly (passlib has incompatibility with bcrypt >= 4.1)
try:
    import bcrypt as _bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    print("⚠️ bcrypt not installed. Password hashing will use SHA-256 fallback.")


# Configuration
_settings = get_settings()
SECRET_KEY = _settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = _settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Security scheme
security = HTTPBearer(auto_error=False)

# Default admin user (should be stored in DB in production)
DEFAULT_USERS: Dict[str, Dict] = {
    "president": {
        "username": "president",
        "hashed_password": None,  # Set on first startup
        "role": "admin",
        "full_name": "الرئيس",
    }
}


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt directly (bypasses passlib)."""
    if BCRYPT_AVAILABLE:
        # bcrypt limit = 72 bytes, truncate to be safe
        pw_bytes = password.encode("utf-8")[:72]
        salt = _bcrypt.gensalt()
        return _bcrypt.hashpw(pw_bytes, salt).decode("utf-8")
    # Fallback: SHA-256 (NOT for production)
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if BCRYPT_AVAILABLE and hashed_password.startswith("$2"):
        pw_bytes = plain_password.encode("utf-8")[:72]
        return _bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    # Fallback: SHA-256
    import hashlib
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def _init_default_users():
    """Initialize default users with hashed passwords"""
    default_password = _settings.ADMIN_PASSWORD
    for username, user_data in DEFAULT_USERS.items():
        if user_data["hashed_password"] is None:
            user_data["hashed_password"] = _hash_password(default_password)


# Initialize on module load
_init_default_users()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    if not JWT_AVAILABLE:
        # Fallback: simple token
        import hashlib
        import json
        token_data = json.dumps({**data, "exp": str(datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)))})
        return hashlib.sha256(token_data.encode()).hexdigest()

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode a JWT token"""
    if not JWT_AVAILABLE:
        return {"sub": "president", "role": "admin"}  # Fallback: always valid

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user with username and password"""
    user = DEFAULT_USERS.get(username)
    if not user:
        return None
    if not _verify_password(password, user["hashed_password"]):
        return None
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict:
    """
    Dependency to get the current authenticated user.
    If no token is provided, allows access in development mode.
    """
    # Development mode: allow access without token
    debug_mode = _settings.DEBUG

    if credentials is None:
        if debug_mode:
            return {"sub": "president", "role": "admin", "mode": "debug"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Dependency to require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
