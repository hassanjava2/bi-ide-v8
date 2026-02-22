"""
Rate Limiting Middleware - تحديد معدل الطلبات
In-memory rate limiter that works without Redis
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


@dataclass
class RateBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_refill: float
    max_tokens: int
    refill_rate: float  # tokens per second

    def can_consume(self, now: float) -> Tuple[bool, float]:
        """Try to consume a token. Returns (allowed, wait_time)."""
        # Refill tokens
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True, 0
        else:
            wait_time = (1 - self.tokens) / self.refill_rate
            return False, wait_time


class RateLimitConfig:
    """Rate limiting configuration."""

    def __init__(
        self,
        # General API rate limits
        default_rpm: int = 120,          # 120 requests per minute (general)
        auth_rpm: int = 10,              # 10 login attempts per minute
        admin_rpm: int = 60,             # 60 admin requests per minute
        heavy_rpm: int = 30,             # 30 heavy requests per minute (council, AI)
        # Burst allowance
        burst_multiplier: float = 1.5,   # Allow 1.5x burst
        # Cleanup
        cleanup_interval: int = 300,     # Clean stale buckets every 5 min
        bucket_ttl: int = 600,           # Remove bucket after 10 min of inactivity
    ):
        self.default_rpm = default_rpm
        self.auth_rpm = auth_rpm
        self.admin_rpm = admin_rpm
        self.heavy_rpm = heavy_rpm
        self.burst_multiplier = burst_multiplier
        self.cleanup_interval = cleanup_interval
        self.bucket_ttl = bucket_ttl

    def get_limit_for_path(self, path: str) -> Tuple[int, str]:
        """Determine rate limit based on request path."""
        if path.startswith("/api/v1/auth/"):
            return self.auth_rpm, "auth"
        elif path.startswith("/api/v1/admin/"):
            return self.admin_rpm, "admin"
        elif any(path.startswith(p) for p in [
            "/api/v1/council/", "/api/v1/command",
            "/api/v1/hierarchy/", "/api/v1/wisdom",
            "/api/v1/network/", "/api/v1/checkpoints/sync",
        ]):
            return self.heavy_rpm, "heavy"
        elif path.startswith("/api/"):
            return self.default_rpm, "default"
        else:
            return 0, "static"  # no limit for static files


class InMemoryRateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.
    Works without Redis — suitable for single-instance deployments.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.buckets: Dict[str, RateBucket] = {}
        self.last_cleanup = time.time()

    def _get_client_key(self, request: Request) -> str:
        """Get a unique key for the client."""
        # Use X-Forwarded-For if behind a reverse proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return ip

    def _cleanup_stale_buckets(self, now: float):
        """Remove buckets that haven't been used recently."""
        if now - self.last_cleanup < self.config.cleanup_interval:
            return

        stale_keys = [
            key for key, bucket in self.buckets.items()
            if now - bucket.last_refill > self.config.bucket_ttl
        ]
        for key in stale_keys:
            del self.buckets[key]

        self.last_cleanup = now

    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict]:
        """
        Check if the request is allowed under rate limits.
        Returns (allowed, headers_dict).
        """
        path = request.url.path
        rpm, category = self.config.get_limit_for_path(path)

        # No limit for static files
        if rpm == 0:
            return True, {}

        now = time.time()
        self._cleanup_stale_buckets(now)

        client_key = self._get_client_key(request)
        bucket_key = f"{client_key}:{category}"

        max_tokens = int(rpm * self.config.burst_multiplier)
        refill_rate = rpm / 60.0  # tokens per second

        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = RateBucket(
                tokens=max_tokens,
                last_refill=now,
                max_tokens=max_tokens,
                refill_rate=refill_rate,
            )

        bucket = self.buckets[bucket_key]
        allowed, wait_time = bucket.can_consume(now)

        headers = {
            "X-RateLimit-Limit": str(rpm),
            "X-RateLimit-Remaining": str(max(0, int(bucket.tokens))),
            "X-RateLimit-Category": category,
        }

        if not allowed:
            headers["Retry-After"] = str(int(wait_time) + 1)

        return allowed, headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    """

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(config)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready", "/metrics"):
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

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response
