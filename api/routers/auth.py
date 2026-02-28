"""
روتر المصادقة - Authentication Router

يوفر نقاط النهاية للمصادقة وإدارة المستخدمين.
Provides endpoints for authentication and user management.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# إعدادات JWT - JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# إعدادات bcrypt - Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

router = APIRouter(prefix="/auth", tags=["المصادقة | Authentication"])


# نماذج Pydantic - Pydantic Models
class UserBase(BaseModel):
    """نموذج المستخدم الأساسي | Base user model"""
    email: EmailStr
    username: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """نموذج إنشاء المستخدم | User creation model"""
    password: str


class User(UserBase):
    """نموذج المستخدم | User model"""
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """نموذج الرمز | Token model"""
    access_token: str
    refresh_token: str
    token_type: str


class TokenRefresh(BaseModel):
    """نموذج تحديث الرمز | Token refresh model"""
    refresh_token: str


class TokenPayload(BaseModel):
    """نموذج حمولة الرمز | Token payload model"""
    sub: Optional[int] = None
    exp: Optional[datetime] = None
    type: Optional[str] = None


class LoginResponse(BaseModel):
    """نموذج استجابة تسجيل الدخول | Login response model"""
    access_token: str
    refresh_token: str
    token_type: str
    user: User


# قاعدة بيانات وهمية - Fake Database (استبدل بقاعدة بيانات حقيقية)
fake_users_db = {}
fake_user_id_counter = 1


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """التحقق من كلمة المرور | Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """تجزئة كلمة المرور | Hash password"""
    return pwd_context.hash(password)


def get_user(user_id: int) -> Optional[User]:
    """الحصول على مستخدم | Get user by ID"""
    user_data = fake_users_db.get(user_id)
    if user_data:
        return User(**user_data)
    return None


def get_user_by_email(email: str) -> Optional[User]:
    """الحصول على مستخدم بالبريد | Get user by email"""
    for user_data in fake_users_db.values():
        if user_data["email"] == email:
            return User(**user_data)
    return None


def get_user_by_username(username: str) -> Optional[User]:
    """الحصول على مستخدم باسم المستخدم | Get user by username"""
    for user_data in fake_users_db.values():
        if user_data["username"] == username:
            return User(**user_data)
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """مصادقة المستخدم | Authenticate user"""
    user = get_user_by_username(username)
    if not user:
        return None
    user_data = fake_users_db.get(user.id)
    if not verify_password(password, user_data["hashed_password"]):
        return None
    return user


def create_token(subject: int, token_type: str = "access", expires_delta: Optional[timedelta] = None) -> str:
    """إنشاء رمز JWT | Create JWT token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        if token_type == "access":
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "type": token_type
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """الحصول على المستخدم الحالي | Get current user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="بيانات الاعتماد غير صالحة | Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "access":
            raise credentials_exception
        
        token_data = TokenPayload(sub=int(user_id), type=token_type)
    except JWTError:
        raise credentials_exception
    
    user = get_user(token_data.sub)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """الحصول على المستخدم النشط | Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="المستخدم غير نشط | User is inactive"
        )
    return current_user


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="تسجيل الدخول | User login"
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    تسجيل الدخول والحصول على رمز JWT.
    Login and obtain JWT token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="اسم المستخدم أو كلمة المرور غير صحيحة | Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_token(user.id, token_type="access")
    refresh_token = create_token(user.id, token_type="refresh")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user
    }


@router.post(
    "/register",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
    summary="تسجيل مستخدم جديد | User registration"
)
async def register(user_data: UserCreate):
    """
    تسجيل مستخدم جديد.
    Register a new user.
    """
    global fake_user_id_counter
    
    # التحقق من وجود البريد الإلكتروني | Check if email exists
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="البريد الإلكتروني مسجل مسبقاً | Email already registered"
        )
    
    # التحقق من وجود اسم المستخدم | Check if username exists
    if get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="اسم المستخدم موجود مسبقاً | Username already taken"
        )
    
    # إنشاء المستخدم | Create user
    user_id = fake_user_id_counter
    fake_user_id_counter += 1
    
    hashed_password = get_password_hash(user_data.password)
    
    user_dict = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    fake_users_db[user_id] = user_dict
    
    return User(**user_dict)


@router.post(
    "/refresh",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="تحديث الرمز | Refresh token"
)
async def refresh(token_data: TokenRefresh):
    """
    تحديث رمز الوصول باستخدام رمز التحديث.
    Refresh access token using refresh token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="رمز التحديث غير صالح | Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token_data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "refresh":
            raise credentials_exception
        
        user_id = int(user_id)
    except JWTError:
        raise credentials_exception
    
    user = get_user(user_id)
    if user is None:
        raise credentials_exception
    
    access_token = create_token(user.id, token_type="access")
    refresh_token = create_token(user.id, token_type="refresh")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="تسجيل الخروج | User logout"
)
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    تسجيل خروج المستخدم (إلغاء صلاحية الرمز).
    Logout user (invalidate token).
    
    ملاحظة: في الإنتاج، أضف الرمز إلى قائمة سوداء.
    Note: In production, add token to a blacklist.
    """
    # في الإنتاج: أضف الرمز إلى قائمة سوداء في Redis
    # In production: Add token to Redis blacklist
    return {
        "message": "تم تسجيل الخروج بنجاح | Successfully logged out",
        "user_id": current_user.id
    }


@router.get(
    "/me",
    response_model=User,
    status_code=status.HTTP_200_OK,
    summary="معلومات المستخدم الحالي | Current user info"
)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على معلومات المستخدم الحالي.
    Get current user information.
    """
    return current_user
