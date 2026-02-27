"""
Redis-Backed Rate Limiting - تحديد معدل الطلبات باستخدام Redis
Supports multi-instance deployments with shared rate limiting state.
"""

import time
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from api.rate_limit import RateLimitConfig


class RedisRateLimiter:
    """
    Distributed rate limiter using Redis.
    Shares rate limit state across all instances.
    Uses sliding window algorithm for accuracy.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
            except Exception as e:
                print(f"⚠️ Redis rate limiter unavailable: {e}")
        
        if not self.redis_client:
            print("⚠️ Falling back to InMemoryRateLimiter")
            from api.rate_limit import InMemoryRateLimiter
            self.fallback = InMemoryRateLimiter(config)
    
    def _get_client_key(self, request: Request) -> str:
        """Get a unique key for the client."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return ip
    
    def _get_window_key(self, client_key: str, category: str, window: int) -> str:
        """Generate Redis key for sliding window."""
        return f"rate_limit:{client_key}:{category}:{window}"
    
    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict]:
        """
        Check if request is allowed using sliding window.
        Returns (allowed, headers_dict).
        """
        # Fallback if Redis unavailable
        if not self.redis_client:
            return self.fallback.check_rate_limit(request)
        
        path = request.url.path
        rpm, category = self.config.get_limit_for_path(path)
        
        if rpm == 0:
            return True, {}
        
        client_key = self._get_client_key(request)
        now = int(time.time())
        window_size = 60  # 1 minute windows
        current_window = now // window_size
        previous_window = current_window - 1
        
        current_key = self._get_window_key(client_key, category, current_window)
        previous_key = self._get_window_key(client_key, category, previous_window)
        
        # Pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        pipe.get(previous_key)
        pipe.incr(current_key)
        pipe.expire(current_key, 120)  # 2 minute TTL
        results = pipe.execute()
        
        previous_count = int(results[0] or 0)
        current_count = results[1]
        
        # Sliding window calculation
        window_progress = (now % window_size) / window_size
        estimated_count = previous_count * (1 - window_progress) + current_count
        
        max_requests = int(rpm * self.config.burst_multiplier)
        allowed = estimated_count <= max_requests
        
        remaining = max(0, max_requests - int(estimated_count))
        
        headers = {
            "X-RateLimit-Limit": str(rpm),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Category": category,
            "X-RateLimit-Window": "sliding",
        }
        
        if not allowed:
            retry_after = window_size - (now % window_size)
            headers["Retry-After"] = str(retry_after)
        
        return allowed, headers
    
    def reset_limit(self, client_key: str, category: str = None):
        """Reset rate limit for a client (admin use)."""
        if not self.redis_client:
            return
        
        pattern = f"rate_limit:{client_key}:{category or '*'}:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for distributed rate limiting with Redis.
    Automatically falls back to in-memory if Redis unavailable.
    """
    
    def __init__(self, app, redis_url: str = None, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        if redis_url:
            self.limiter = RedisRateLimiter(redis_url, config)
        else:
            # Try to get from environment
            import os
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.limiter = RedisRateLimiter(redis_url, config)
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        # Skip rate limiting during tests
        import os
        if os.getenv("PYTEST_RUNNING") == "1":
            return await call_next(request)
        
        allowed, headers = self.limiter.check_rate_limit(request)
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after": int(headers.get("Retry-After", 1)),
                },
                headers=headers,
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        for key, value in headers.items():
            response.headers[key] = str(value)
        
        return response


# Factory function for easy switching
def create_rate_limit_middleware(app, use_redis: bool = True, redis_url: str = None):
    """
    Create appropriate rate limit middleware.
    
    Args:
        app: FastAPI app
        use_redis: Whether to try Redis first
        redis_url: Redis connection URL
    
    Returns:
        Middleware instance
    """
    if use_redis:
        try:
            return RedisRateLimitMiddleware(app, redis_url)
        except Exception as e:
            print(f"⚠️ Redis rate limiter failed, using in-memory: {e}")
    
    # Fallback to in-memory
    from api.rate_limit import RateLimitMiddleware
    return RateLimitMiddleware(app)
