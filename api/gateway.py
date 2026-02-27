"""
API Gateway - بوابة الـ API
============================
Pattern: Gateway with load balancing, circuit breaker, and failover
Supports routing between Windows (main API) and Ubuntu (RTX 4090 workers)
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
from functools import wraps
import random

import httpx
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings
from core.logging_config import logger

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    weight: int = 1
    healthy: bool = True
    last_check: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def mark_healthy(self):
        """Mark endpoint as healthy"""
        self.healthy = True
        self.failure_count = 0
        self.last_check = datetime.now(timezone.utc)
        self.success_count += 1
    
    def mark_unhealthy(self):
        """Mark endpoint as unhealthy"""
        self.failure_count += 1
        self.last_check = datetime.now(timezone.utc)
        if self.failure_count >= 3:
            self.healthy = False


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    half_open_calls: int = field(default=0)
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._reset()
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    burst_size: int = 10
    
    def __post_init__(self):
        self._requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit"""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > window_start
            ]
        else:
            self._requests[key] = []
        
        # Check burst limit
        recent_requests = len(self._requests[key])
        if recent_requests >= self.burst_size:
            # Check if within rate limit
            if recent_requests >= self.requests_per_minute:
                retry_after = int(60 - (now - self._requests[key][0]))
                return False, {
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "retry_after": max(1, retry_after)
                }
        
        # Allow request
        self._requests[key].append(now)
        
        return True, {
            "limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - len(self._requests[key])),
            "reset_time": int(now + 60)
        }


class RTX4090Gateway:
    """
    Gateway for RTX 4090 inference servers
    Features:
    - Load balancing across multiple workers
    - Circuit breaker for fault tolerance
    - Automatic failover
    - Health checking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.endpoints: List[ServiceEndpoint] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiter = RateLimitConfig()
        self._health_check_task: Optional[asyncio.Task] = None
        self._initialized = False
        
        # Initialize endpoints
        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """Setup RTX 4090 endpoints from configuration"""
        # Primary RTX 4090 server
        primary = ServiceEndpoint(
            name="rtx4090-primary",
            host=self.settings.RTX4090_HOST,
            port=self.settings.RTX4090_PORT,
            weight=10
        )
        self.endpoints.append(primary)
        self.circuit_breakers[primary.name] = CircuitBreaker()
        
        # Add AI Core endpoints if configured
        for i, url in enumerate(self.settings.ai_core_urls):
            # Parse URL to extract host and port
            try:
                host_port = url.replace("http://", "").split(":")
                host = host_port[0]
                port = int(host_port[1]) if len(host_port) > 1 else 8080
                
                endpoint = ServiceEndpoint(
                    name=f"ai-core-{i}",
                    host=host,
                    port=port,
                    weight=5
                )
                self.endpoints.append(endpoint)
                self.circuit_breakers[endpoint.name] = CircuitBreaker()
            except Exception as e:
                logger.warning(f"Failed to parse AI Core URL {url}: {e}")
    
    async def initialize(self):
        """Initialize gateway and start health checks"""
        if self._initialized:
            return
        
        logger.info("Initializing RTX 4090 Gateway...")
        
        # Initial health check
        await self._health_check_all()
        
        # Start background health checks
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
        
        self._initialized = True
        logger.info(f"Gateway initialized with {len(self.endpoints)} endpoints")
    
    async def shutdown(self):
        """Shutdown gateway"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Gateway shutdown complete")
    
    def _get_healthy_endpoints(self) -> List[ServiceEndpoint]:
        """Get list of healthy endpoints"""
        return [ep for ep in self.endpoints if ep.healthy]
    
    def _select_endpoint(self) -> Optional[ServiceEndpoint]:
        """Select endpoint using weighted round-robin"""
        healthy = self._get_healthy_endpoints()
        
        if not healthy:
            # Fall back to all endpoints if none are healthy
            healthy = self.endpoints
        
        if not healthy:
            return None
        
        # Weighted random selection
        total_weight = sum(ep.weight for ep in healthy)
        if total_weight == 0:
            return random.choice(healthy)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for ep in healthy:
            cumulative += ep.weight
            if r <= cumulative:
                return ep
        
        return healthy[-1]
    
    async def forward_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: float = 30.0,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Forward request to RTX 4090 server
        
        Args:
            endpoint: API endpoint path
            data: Request data
            timeout: Request timeout
            retries: Number of retries
            
        Returns:
            Response data
            
        Raises:
            HTTPException: If all endpoints fail
        """
        if not self._initialized:
            await self.initialize()
        
        last_error = None
        attempted_endpoints: Set[str] = set()
        
        for attempt in range(retries):
            # Select endpoint
            selected = self._select_endpoint()
            
            if not selected:
                raise HTTPException(
                    status_code=503,
                    detail="No RTX 4090 endpoints available"
                )
            
            # Skip if already attempted
            if selected.name in attempted_endpoints:
                continue
            attempted_endpoints.add(selected.name)
            
            # Check circuit breaker
            circuit = self.circuit_breakers.get(selected.name)
            if circuit and not circuit.can_execute():
                logger.warning(f"Circuit breaker open for {selected.name}")
                continue
            
            # Attempt request
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{selected.url}/{endpoint}",
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                
                # Record success
                elapsed = time.time() - start_time
                selected.mark_healthy()
                selected.avg_response_time = (
                    selected.avg_response_time * 0.9 + elapsed * 0.1
                )
                
                if circuit:
                    circuit.record_success()
                
                logger.debug(
                    f"Request to {selected.name} succeeded in {elapsed:.2f}s"
                )
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                last_error = e
                
                # Record failure
                selected.mark_unhealthy()
                if circuit:
                    circuit.record_failure()
                
                logger.warning(
                    f"Request to {selected.name} failed: {str(e)[:100]}"
                )
                
                # Continue to next endpoint
                continue
        
        # All endpoints failed
        raise HTTPException(
            status_code=503,
            detail=f"All RTX 4090 endpoints failed. Last error: {last_error}"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all endpoints"""
        results = []
        
        for endpoint in self.endpoints:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{endpoint.url}/health")
                    healthy = response.status_code == 200
                    
                    if healthy:
                        endpoint.mark_healthy()
                    else:
                        endpoint.mark_unhealthy()
                    
                    results.append({
                        "name": endpoint.name,
                        "url": endpoint.url,
                        "healthy": healthy,
                        "status_code": response.status_code,
                        "avg_response_time": endpoint.avg_response_time,
                        "circuit_state": self.circuit_breakers[endpoint.name].state.value
                    })
            except Exception as e:
                endpoint.mark_unhealthy()
                results.append({
                    "name": endpoint.name,
                    "url": endpoint.url,
                    "healthy": False,
                    "error": str(e)[:100],
                    "circuit_state": self.circuit_breakers[endpoint.name].state.value
                })
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoints": results,
            "healthy_count": sum(1 for r in results if r["healthy"]),
            "total_count": len(results)
        }
    
    async def _health_check_all(self):
        """Run health check on all endpoints once"""
        for endpoint in self.endpoints:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{endpoint.url}/health")
                    if response.status_code == 200:
                        endpoint.mark_healthy()
                    else:
                        endpoint.mark_unhealthy()
            except Exception:
                endpoint.mark_unhealthy()
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._health_check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")


class GatewayMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for gateway functionality
    - Rate limiting
    - Request logging
    - Error handling
    """
    
    def __init__(self, app, gateway: RTX4090Gateway):
        super().__init__(app)
        self.gateway = gateway
        self.rate_limiter = RateLimitConfig()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through gateway"""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, headers = self.rate_limiter.is_allowed(client_id)
        if not allowed:
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "retry_after": headers["retry_after"]
                }),
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(headers["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(headers["retry_after"]),
                    "Content-Type": "application/json"
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(headers["limit"])
        response.headers["X-RateLimit-Remaining"] = str(headers["remaining"])
        response.headers["X-RateLimit-Reset"] = str(headers["reset_time"])
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique identifier for client"""
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"
        
        # Use authenticated user if available
        auth = request.headers.get("Authorization")
        if auth:
            return f"auth:{hashlib.sha256(auth.encode()).hexdigest()[:16]}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"


# Singleton instance
gateway = RTX4090Gateway()


async def get_gateway() -> RTX4090Gateway:
    """Get initialized gateway instance"""
    if not gateway._initialized:
        await gateway.initialize()
    return gateway
