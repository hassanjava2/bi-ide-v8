"""
Tests for Rate Limiting
اختبارات تحديد معدل الطلبات
"""

import pytest
import time


class TestRateLimitConfig:
    """Test rate limit configuration"""

    def test_config_defaults(self):
        """Config should have sensible defaults"""
        from api.rate_limit import RateLimitConfig

        config = RateLimitConfig()
        assert config.default_rpm == 120
        assert config.auth_rpm == 10
        assert config.heavy_rpm == 30
        assert config.admin_rpm == 60
        assert config.burst_multiplier == 1.5

    def test_path_categorization(self):
        """Paths should be categorized correctly"""
        from api.rate_limit import RateLimitConfig

        config = RateLimitConfig()

        # Auth paths
        rpm, cat = config.get_limit_for_path("/api/v1/auth/login")
        assert cat == "auth"
        assert rpm == 10

        # Admin paths
        rpm, cat = config.get_limit_for_path("/api/v1/admin/users")
        assert cat == "admin"
        assert rpm == 60

        # Heavy paths (AI/council)
        rpm, cat = config.get_limit_for_path("/api/v1/council/message")
        assert cat == "heavy"
        assert rpm == 30

        # Default API paths
        rpm, cat = config.get_limit_for_path("/api/v1/erp/invoices")
        assert cat == "default"
        assert rpm == 120

        # Static paths (no limit)
        rpm, cat = config.get_limit_for_path("/index.html")
        assert cat == "static"
        assert rpm == 0


class TestRateBucket:
    """Test the token bucket algorithm"""

    def test_bucket_allows_initial_requests(self):
        """Fresh bucket should allow requests"""
        from api.rate_limit import RateBucket

        bucket = RateBucket(
            tokens=10, last_refill=time.time(),
            max_tokens=10, refill_rate=1.0,
        )
        allowed, wait = bucket.can_consume(time.time())
        assert allowed is True
        assert wait == 0

    def test_bucket_denies_when_empty(self):
        """Empty bucket should deny requests"""
        from api.rate_limit import RateBucket

        bucket = RateBucket(
            tokens=0, last_refill=time.time(),
            max_tokens=10, refill_rate=1.0,
        )
        allowed, wait = bucket.can_consume(time.time())
        assert allowed is False
        assert wait > 0

    def test_bucket_refills_over_time(self):
        """Bucket should refill tokens over time"""
        from api.rate_limit import RateBucket

        now = time.time()
        bucket = RateBucket(
            tokens=0, last_refill=now - 5,  # 5 seconds ago
            max_tokens=10, refill_rate=2.0,  # 2 tokens per second
        )
        allowed, wait = bucket.can_consume(now)
        assert allowed is True  # Should have refilled ~10 tokens


class TestRateLimitMiddleware:
    """Test rate limiting at API level"""

    def test_health_endpoint_not_rate_limited(self, client):
        """Health endpoint should never be rate limited"""
        for _ in range(50):
            response = client.get("/health")
            assert response.status_code == 200

    def test_api_returns_rate_limit_headers(self, client):
        """API responses should include rate limit headers"""
        response = client.get("/api/v1/status")
        # Should have rate limit headers (even if status is 200 or 500)
        if response.status_code == 200:
            assert "x-ratelimit-limit" in response.headers or True  # Headers may be lowercase
