"""
BI-IDE v8 - DDoS Protection Module
Rate limiting, request filtering, and CloudFlare integration
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

import aiohttp
from aiohttp import web
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    """Types of challenges"""
    JS_CHALLENGE = "js_challenge"
    CAPTCHA = "captcha"
    MANAGED = "managed"
    BLOCK = "block"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 100
    burst_size: int = 20
    window_seconds: int = 60
    block_duration: int = 300  # 5 minutes
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)


@dataclass
class RequestMetrics:
    """Request metrics for an IP"""
    ip: str
    request_count: int = 0
    first_request: float = 0.0
    last_request: float = 0.0
    blocked_until: float = 0.0
    challenges_passed: int = 0
    challenges_failed: int = 0


class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._redis: Optional[aioredis.Redis] = None
        self._redis_url = redis_url
        self._local_cache: Dict[str, RequestMetrics] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self._redis = await aioredis.from_url(self._redis_url)
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def close(self):
        """Cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._redis:
            await self._redis.close()
    
    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for identifier"""
        return f"ratelimit:{identifier}"
    
    async def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed
        
        Returns: (allowed, metadata)
        """
        # Check whitelist
        if self._is_whitelisted(identifier):
            return True, {'reason': 'whitelisted'}
        
        # Check blacklist
        if self._is_blacklisted(identifier):
            return False, {'reason': 'blacklisted', 'block_remaining': -1}
        
        # Check if currently blocked
        is_blocked, block_remaining = await self._is_blocked(identifier)
        if is_blocked:
            return False, {'reason': 'blocked', 'block_remaining': block_remaining}
        
        # Token bucket algorithm
        key = self._get_key(identifier)
        now = time.time()
        
        # Get current bucket state
        pipe = self._redis.pipeline()
        pipe.hgetall(key)
        pipe.ttl(key)
        results = await pipe.execute()
        
        bucket = results[0] or {}
        ttl = results[1]
        
        tokens = float(bucket.get(b'tokens', self.config.burst_size))
        last_update = float(bucket.get(b'last_update', now))
        
        # Add tokens based on time passed
        time_passed = now - last_update
        tokens = min(
            self.config.burst_size,
            tokens + time_passed * (self.config.requests_per_minute / 60)
        )
        
        # Check if request can be allowed
        if tokens >= 1:
            tokens -= 1
            allowed = True
            
            # Update bucket
            await self._redis.hmset(key, {
                'tokens': tokens,
                'last_update': now
            })
            await self._redis.expire(key, self.config.window_seconds)
        else:
            allowed = False
            # Block the identifier
            await self._block(identifier)
        
        return allowed, {
            'tokens_remaining': int(tokens),
            'reset_time': int(now + self.config.window_seconds)
        }
    
    async def _is_blocked(self, identifier: str) -> Tuple[bool, int]:
        """Check if identifier is blocked"""
        block_key = f"block:{identifier}"
        ttl = await self._redis.ttl(block_key)
        return ttl > 0, ttl if ttl > 0 else 0
    
    async def _block(self, identifier: str):
        """Block an identifier"""
        block_key = f"block:{identifier}"
        await self._redis.setex(block_key, self.config.block_duration, '1')
        logger.warning(f"Blocked {identifier} for {self.config.block_duration}s")
    
    def _is_whitelisted(self, identifier: str) -> bool:
        """Check if identifier is in whitelist"""
        for entry in self.config.whitelist:
            if self._match_ip(identifier, entry):
                return True
        return False
    
    def _is_blacklisted(self, identifier: str) -> bool:
        """Check if identifier is in blacklist"""
        for entry in self.config.blacklist:
            if self._match_ip(identifier, entry):
                return True
        return False
    
    def _match_ip(self, ip: str, pattern: str) -> bool:
        """Match IP against pattern (CIDR or exact)"""
        try:
            if '/' in pattern:
                return ipaddress.ip_address(ip) in ipaddress.ip_network(pattern)
            return ip == pattern
        except ValueError:
            return ip == pattern
    
    async def unblock(self, identifier: str):
        """Manually unblock an identifier"""
        block_key = f"block:{identifier}"
        await self._redis.delete(block_key)
        logger.info(f"Unblocked {identifier}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                # Redis handles expiration automatically
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


class RequestFilter:
    """Request filtering based on patterns"""
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'\.(php|asp|jsp|cgi)$',  # Forbidden extensions
        r'(\.{2,}|%2e%2e)',  # Path traversal
        r'(<script|javascript:|onerror=|onload=)',  # XSS attempts
        r'(union\s+select|insert\s+into|delete\s+from)',  # SQL injection
        r'(etc/passwd|win\.ini|boot\.ini)',  # File access
        r'(wget|curl|nc\s+-|bash\s+-i)',  # Command injection
    ]
    
    # Known bad user agents
    BAD_USER_AGENTS = [
        'sqlmap',
        'nikto',
        'nmap',
        'masscan',
        'zgrab',
        'gobuster',
        'dirbuster',
        'wfuzz',
        'burpsuite',
    ]
    
    def __init__(self):
        import re
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS]
    
    def analyze_request(self, request: web.Request) -> Tuple[bool, List[str], float]:
        """
        Analyze request for threats
        
        Returns: (is_threat, threat_types, threat_score)
        """
        threats = []
        score = 0.0
        
        # Check user agent
        user_agent = request.headers.get('User-Agent', '').lower()
        for bad_ua in self.BAD_USER_AGENTS:
            if bad_ua in user_agent:
                threats.append(f'bad_user_agent:{bad_ua}')
                score += 0.3
        
        # Check for missing user agent
        if not user_agent:
            threats.append('missing_user_agent')
            score += 0.1
        
        # Check URL path
        path = request.path
        for pattern in self._patterns:
            if pattern.search(path):
                threats.append(f'path_pattern:{pattern.pattern[:30]}')
                score += 0.4
        
        # Check query string
        query = request.query_string
        for pattern in self._patterns:
            if pattern.search(query):
                threats.append(f'query_pattern:{pattern.pattern[:30]}')
                score += 0.4
        
        # Check headers for anomalies
        headers = dict(request.headers)
        if len(str(headers)) > 8192:  # Suspiciously large headers
            threats.append('large_headers')
            score += 0.2
        
        # Check for required headers
        if request.method == 'POST' and 'Content-Length' not in headers:
            threats.append('missing_content_length')
            score += 0.1
        
        return len(threats) > 0 and score > 0.5, threats, min(1.0, score)


class ChallengeResponse:
    """Challenge-response mechanism for bot detection"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self._redis = await aioredis.from_url(self._redis_url)
    
    async def close(self):
        """Cleanup resources"""
        if self._redis:
            await self._redis.close()
    
    def generate_js_challenge(self, client_id: str) -> Tuple[str, str]:
        """
        Generate JavaScript challenge
        
        Returns: (challenge_html, challenge_token)
        """
        token = hashlib.sha256(f"{client_id}:{time.time()}".encode()).hexdigest()[:32]
        
        # Simple proof-of-work challenge
        challenge_code = f"""
        <html>
        <head>
            <script>
                (function() {{
                    var start = Date.now();
                    var iterations = 0;
                    var target = '{token}';
                    
                    while (Date.now() - start < 100) {{
                        iterations++;
                    }}
                    
                    fetch('/.well-known/challenge-verify', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            token: target,
                            iterations: iterations,
                            timestamp: Date.now()
                        }})
                    }}).then(() => {{
                        window.location.reload();
                    }});
                }})();
            </script>
        </head>
        <body>
            <p>Verifying your browser...</p>
        </body>
        </html>
        """
        
        return challenge_code, token
    
    async def verify_js_challenge(self, token: str, response: Dict) -> bool:
        """Verify JavaScript challenge response"""
        # Check token exists
        key = f"challenge:{token}"
        exists = await self._redis.exists(key)
        
        if not exists:
            return False
        
        # Verify response parameters
        iterations = response.get('iterations', 0)
        timestamp = response.get('timestamp', 0)
        
        # Basic validation
        if iterations < 1000:  # Too few iterations
            return False
        
        if time.time() * 1000 - timestamp > 30000:  # Took too long
            return False
        
        # Mark challenge as passed
        await self._redis.setex(f"challenge_passed:{token}", 3600, '1')
        await self._redis.delete(key)
        
        return True
    
    async def has_passed_challenge(self, token: str) -> bool:
        """Check if client has passed challenge"""
        return await self._redis.exists(f"challenge_passed:{token}") > 0


class CloudFlareIntegration:
    """CloudFlare API integration for DDoS protection"""
    
    def __init__(self, api_token: str, zone_id: str):
        self.api_token = api_token
        self.zone_id = zone_id
        self._base_url = "https://api.cloudflare.com/client/v4"
    
    async def _make_request(self, method: str, endpoint: str, 
                            data: Optional[Dict] = None) -> Dict:
        """Make CloudFlare API request"""
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, 
                                       json=data) as response:
                return await response.json()
    
    async def get_security_level(self) -> str:
        """Get current security level"""
        result = await self._make_request(
            "GET", 
            f"/zones/{self.zone_id}/settings/security_level"
        )
        return result.get('result', {}).get('value', 'medium')
    
    async def set_security_level(self, level: str) -> bool:
        """Set security level (off/essentially_off/low/medium/high/under_attack)"""
        result = await self._make_request(
            "PATCH",
            f"/zones/{self.zone_id}/settings/security_level",
            {"value": level}
        )
        return result.get('success', False)
    
    async def create_rate_limit_rule(self, threshold: int, period: int,
                                      action: str = "challenge") -> bool:
        """Create rate limiting rule"""
        data = {
            "threshold": threshold,
            "period": period,
            "action": {"mode": action},
            "match": {"request": {"url": "*"}}
        }
        
        result = await self._make_request(
            "POST",
            f"/zones/{self.zone_id}/rate_limits",
            data
        )
        return result.get('success', False)
    
    async def ip_geolocation(self, ip: str) -> Dict:
        """Get geolocation for IP"""
        result = await self._make_request(
            "GET",
            f"/zones/{self.zone_id}/firewall/access_rules/rules",
            {"configuration.value": ip}
        )
        return result
    
    async def block_ip(self, ip: str, notes: str = "") -> bool:
        """Block an IP address"""
        data = {
            "mode": "block",
            "configuration": {
                "target": "ip",
                "value": ip
            },
            "notes": notes
        }
        
        result = await self._make_request(
            "POST",
            f"/zones/{self.zone_id}/firewall/access_rules/rules",
            data
        )
        return result.get('success', False)
    
    async def unblock_ip(self, rule_id: str) -> bool:
        """Unblock an IP by rule ID"""
        result = await self._make_request(
            "DELETE",
            f"/zones/{self.zone_id}/firewall/access_rules/rules/{rule_id}"
        )
        return result.get('success', False)
    
    async def enable_under_attack_mode(self) -> bool:
        """Enable Under Attack mode"""
        return await self.set_security_level("under_attack")
    
    async def disable_under_attack_mode(self) -> bool:
        """Disable Under Attack mode"""
        return await self.set_security_level("medium")
    
    async def get_analytics(self, since_minutes: int = 60) -> Dict:
        """Get security analytics"""
        result = await self._make_request(
            "GET",
            f"/zones/{self.zone_id}/analytics/dashboard",
            {"since": f"{since_minutes}m"}
        )
        return result.get('result', {})


class DDoSProtection:
    """Main DDoS protection coordinator"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 rate_config: Optional[RateLimitConfig] = None,
                 cloudflare_config: Optional[Dict] = None):
        self.rate_limiter = RateLimiter(redis_url, rate_config)
        self.request_filter = RequestFilter()
        self.challenge_response = ChallengeResponse(redis_url)
        self.cloudflare = None
        
        if cloudflare_config:
            self.cloudflare = CloudFlareIntegration(
                cloudflare_config['api_token'],
                cloudflare_config['zone_id']
            )
        
        self._attack_mode_threshold = 0.8  # Enable attack mode if threat ratio > 80%
        self._request_history: List[Tuple[float, bool]] = []
    
    async def initialize(self):
        """Initialize all components"""
        await self.rate_limiter.initialize()
        await self.challenge_response.initialize()
    
    async def close(self):
        """Cleanup resources"""
        await self.rate_limiter.close()
        await self.challenge_response.close()
    
    async def process_request(self, request: web.Request) -> Tuple[bool, Optional[web.Response]]:
        """
        Process incoming request through all protection layers
        
        Returns: (allowed, response if blocked)
        """
        client_ip = self._get_client_ip(request)
        
        # Layer 1: Request filtering
        is_threat, threats, threat_score = self.request_filter.analyze_request(request)
        
        if threat_score > 0.9:
            logger.warning(f"High threat request blocked: {client_ip} - {threats}")
            return False, web.Response(status=403, text="Forbidden")
        
        # Layer 2: Rate limiting
        allowed, rate_metadata = await self.rate_limiter.is_allowed(client_ip)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded: {client_ip}")
            
            # Return challenge for borderline cases
            if rate_metadata.get('reason') != 'blacklisted':
                challenge_html, token = self.challenge_response.generate_js_challenge(client_ip)
                return False, web.Response(
                    status=429,
                    text=challenge_html,
                    content_type='text/html',
                    headers={'X-Challenge-Token': token}
                )
            
            return False, web.Response(status=429, text="Too Many Requests")
        
        # Layer 3: Challenge-response for suspicious requests
        if is_threat and threat_score > 0.5:
            challenge_token = request.headers.get('X-Challenge-Passed')
            if not challenge_token or not await self.challenge_response.has_passed_challenge(challenge_token):
                challenge_html, token = self.challenge_response.generate_js_challenge(client_ip)
                return False, web.Response(
                    status=403,
                    text=challenge_html,
                    content_type='text/html',
                    headers={'X-Challenge-Token': token}
                )
        
        # Update attack detection
        await self._update_attack_detection(is_threat)
        
        return True, None
    
    def _get_client_ip(self, request: web.Request) -> str:
        """Extract client IP from request"""
        # Check CloudFlare header first
        cf_ip = request.headers.get('CF-Connecting-IP')
        if cf_ip:
            return cf_ip
        
        # Check X-Forwarded-For
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        
        # Fallback to peer name
        return request.remote or 'unknown'
    
    async def _update_attack_detection(self, is_threat: bool):
        """Update attack detection metrics"""
        now = time.time()
        self._request_history.append((now, is_threat))
        
        # Keep only last 5 minutes
        cutoff = now - 300
        self._request_history = [(t, s) for t, s in self._request_history if t > cutoff]
        
        # Check if under attack
        if len(self._request_history) > 100:
            threat_ratio = sum(1 for _, s in self._request_history if s) / len(self._request_history)
            
            if threat_ratio > self._attack_mode_threshold:
                if self.cloudflare:
                    logger.critical(f"Attack detected! Threat ratio: {threat_ratio:.2%}")
                    await self.cloudflare.enable_under_attack_mode()
    
    async def get_statistics(self) -> Dict:
        """Get protection statistics"""
        now = time.time()
        recent_requests = [(t, s) for t, s in self._request_history if t > now - 60]
        
        return {
            'requests_per_minute': len(recent_requests),
            'threat_ratio': sum(1 for _, s in recent_requests if s) / len(recent_requests) if recent_requests else 0,
            'blocked_count': len([s for _, s in recent_requests if s]),
            'attack_mode_active': len(recent_requests) > 100 and 
                sum(1 for _, s in recent_requests if s) / len(recent_requests) > self._attack_mode_threshold
        }


# Middleware for aiohttp
@web.middleware
async def ddos_protection_middleware(request: web.Request, handler: Callable) -> web.Response:
    """DDoS protection middleware"""
    protection = request.app.get('ddos_protection')
    
    if protection:
        allowed, response = await protection.process_request(request)
        if not allowed:
            return response
    
    return await handler(request)


# Global protection instance
_protection: Optional[DDoSProtection] = None


def init_protection(redis_url: str = "redis://localhost:6379/0",
                   rate_config: Optional[RateLimitConfig] = None,
                   cloudflare_config: Optional[Dict] = None) -> DDoSProtection:
    """Initialize global DDoS protection"""
    global _protection
    _protection = DDoSProtection(redis_url, rate_config, cloudflare_config)
    return _protection


def get_protection() -> DDoSProtection:
    """Get global DDoS protection"""
    if _protection is None:
        raise RuntimeError("DDoS protection not initialized")
    return _protection
