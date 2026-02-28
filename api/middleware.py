"""
BI-IDE API Middleware
البرامج الوسيطة لواجهة برمجة التطبيقات

This module contains all custom middleware for the BI-IDE API:
- LoggingMiddleware: تسجيل جميع الطلبات مع التوقيت
- RateLimitMiddleware: تقييد معدل الطلبات باستخدام Redis
- AuthMiddleware: مصادقة JWT
- ErrorHandlerMiddleware: معالجة الأخطاء العامة
"""

import time
import logging
import json
from typing import Optional, Callable
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis

# إعداد التسجيل
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests with timing
    برنامج وسيط لتسجيل جميع الطلبات مع التوقيت
    
    Logs request method, path, status code, and processing time.
    يسجل طريقة الطلب والمسار ورمز الحالة ووقت المعالجة.
    """
    
    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware"""
        super().__init__(app)
        self.logger = logging.getLogger("api.request")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details
        معالجة الطلب وتسجيل التفاصيل
        
        Args:
            request: The incoming request
            call_next: The next middleware/endpoint to call
            
        Returns:
            Response: The HTTP response
        """
        # تسجيل وقت البدء
        start_time = time.time()
        
        # استخراج معلومات الطلب
        method = request.method
        url = str(request.url)
        client_host = request.client.host if request.client else "unknown"
        
        try:
            # معالجة الطلب
            response = await call_next(request)
            
            # حساب وقت المعالجة
            process_time = time.time() - start_time
            
            # تسجيل المعلومات
            self.logger.info(
                f"{method} {request.url.path} - {response.status_code} - "
                f"{process_time:.3f}s - {client_host}"
            )
            
            # إضافة رأس الوقت
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # تسجيل الخطأ
            process_time = time.time() - start_time
            self.logger.error(
                f"{method} {request.url.path} - ERROR - "
                f"{process_time:.3f}s - {client_host} - {str(e)}"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for token bucket rate limiting using Redis
    برنامج وسيط للحد من معدل الطلبات باستخدام Redis
    
    Implements a token bucket algorithm for rate limiting.
    ينفذ خوارزمية دلو الرموز للحد من معدل الطلبات.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 100,
        window: int = 60,
        redis_url: str = "redis://localhost:6379"
    ) -> None:
        """
        Initialize the rate limit middleware
        
        Args:
            app: The ASGI application
            max_requests: Maximum requests allowed per window
            window: Time window in seconds
            redis_url: Redis connection URL
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger("api.rate_limit")
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(self.redis_url)
        return self.redis_client
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting
        الحصول على معرف العميل للحد من معدل الطلبات
        """
        # استخدم عنوان IP أو رمز API إذا كان موجوداً
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _is_rate_limited(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit
        التحقق مما إذا كان العميل قد تجاوز الحد المسموح
        
        Args:
            client_id: The client identifier
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        try:
            redis_conn = await self._get_redis()
            key = f"rate_limit:{client_id}"
            
            # الحصول على العدد الحالي
            current = await redis_conn.get(key)
            
            if current is None:
                # أول طلب - تعيين العداد
                await redis_conn.setex(key, self.window, 1)
                return False
            
            current_count = int(current)
            
            if current_count >= self.max_requests:
                return True
            
            # زيادة العداد
            await redis_conn.incr(key)
            return False
            
        except Exception as e:
            # في حالة فشل Redis، اسمح بالطلب
            self.logger.warning(f"Rate limit check failed: {e}")
            return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limit and process request
        التحقق من الحد والمعالجة
        """
        # تخطي الحد للمسارات المحددة
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        if await self._is_rate_limited(client_id):
            self.logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "لقد تجاوزت الحد المسموح من الطلبات. يرجى المحاولة لاحقاً.",
                    "retry_after": self.window
                }
            )
        
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT authentication
    برنامج وسيط لمصادقة JWT
    
    Validates JWT tokens on protected routes.
    يتحقق من رموز JWT على المسارات المحمية.
    """
    
    # المسارات المستثناة من المصادقة
    EXCLUDED_PATHS = [
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v1/auth/forgot-password",
    ]
    
    def __init__(self, app: ASGIApp, secret_key: Optional[str] = None) -> None:
        """
        Initialize the auth middleware
        
        Args:
            app: The ASGI application
            secret_key: JWT secret key
        """
        super().__init__(app)
        self.secret_key = secret_key or "your-secret-key-change-in-production"
        self.logger = logging.getLogger("api.auth")
    
    def _is_excluded(self, path: str) -> bool:
        """
        Check if path is excluded from authentication
        التحقق مما إذا كان المسار مستثنى من المصادقة
        """
        for excluded in self.EXCLUDED_PATHS:
            if path.startswith(excluded):
                return True
        return False
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from request headers
        استخراج رمز JWT من رؤوس الطلب
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]
    
    def _validate_token(self, token: str) -> Optional[dict]:
        """
        Validate JWT token
        التحقق من صحة رمز JWT
        
        Args:
            token: The JWT token
            
        Returns:
            dict: Token payload if valid, None otherwise
        """
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Authenticate request and process
        مصادقة الطلب ومعالجته
        """
        path = request.url.path
        
        # تخطي المصادقة للمسارات المستثناة
        if self._is_excluded(path):
            return await call_next(request)
        
        # استخراج والتحقق من الرمز
        token = self._extract_token(request)
        
        if not token:
            self.logger.warning(f"Missing token for {path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "يرجى تسجيل الدخول للوصول إلى هذا المورد"
                }
            )
        
        payload = self._validate_token(token)
        
        if not payload:
            self.logger.warning(f"Invalid token for {path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "رمز المصادقة غير صالح أو منتهي الصلاحية"
                }
            )
        
        # إضافة معلومات المستخدم للطلب
        request.state.user = payload
        
        return await call_next(request)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global error handling
    برنامج وسيط لمعالجة الأخطاء العامة
    
    Catches and handles all unhandled exceptions.
    يلتقط ويعالج جميع الاستثناءات غير المعالجة.
    """
    
    def __init__(self, app: ASGIApp, debug: bool = False) -> None:
        """
        Initialize the error handler middleware
        
        Args:
            app: The ASGI application
            debug: Enable debug mode with detailed error messages
        """
        super().__init__(app)
        self.debug = debug
        self.logger = logging.getLogger("api.error")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle request with error catching
        معالجة الطلب مع التقاط الأخطاء
        """
        try:
            return await call_next(request)
            
        except Exception as exc:
            # تسجيل الخطأ
            self.logger.exception(f"Unhandled error: {exc}")
            
            # إنشاء رد الخطأ
            error_response = {
                "error": "Internal Server Error",
                "message": "حدث خطأ داخلي في الخادم",
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
            
            # إضافة تفاصيل في وضع التصحيح
            if self.debug:
                import traceback
                error_response["detail"] = str(exc)
                error_response["traceback"] = traceback.format_exc()
            
            return JSONResponse(
                status_code=500,
                content=error_response
            )
