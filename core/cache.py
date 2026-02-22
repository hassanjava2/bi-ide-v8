"""
Caching Layer - طبقة التخزين المؤقت
Supports Redis and in-memory fallback
"""
import asyncio
import sys
sys.path.insert(0, '.')
import encoding_fix

import os
import json
import pickle
import hashlib
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
from functools import wraps

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available, using in-memory cache")

from .logging_config import logger


class CacheManager:
    """Cache manager with Redis and in-memory fallback"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self.memory_cache = {}  # Fallback
        self.memory_expiry = {}
        self.enabled = True
        
    async def initialize(self):
        """Initialize cache connection"""
        if not REDIS_AVAILABLE:
            logger.info("Using in-memory cache (Redis not available)")
            return
        
        try:
            redis_connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", "2"))
            redis_socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2"))

            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_connect_timeout=redis_connect_timeout,
                socket_timeout=redis_socket_timeout,
            )
            await asyncio.to_thread(self.redis_client.ping)
            logger.info(f"Redis cache connected: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {e}")
            self.redis_client = None
    
    def _make_key(self, key: str, prefix: str = "") -> str:
        """Create cache key with prefix"""
        if prefix:
            return f"bi_ide:{prefix}:{key}"
        return f"bi_ide:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes"""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value"""
        return pickle.loads(data)
    
    async def get(self, key: str, prefix: str = "") -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        full_key = self._make_key(key, prefix)
        
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(full_key)
                if data:
                    return self._deserialize(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to memory
        if full_key in self.memory_cache:
            if datetime.now() < self.memory_expiry.get(full_key, datetime.now()):
                return self.memory_cache[full_key]
            else:
                # Expired
                del self.memory_cache[full_key]
                del self.memory_expiry[full_key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, prefix: str = ""):
        """Set value in cache with TTL (seconds)"""
        if not self.enabled:
            return
        
        full_key = self._make_key(key, prefix)
        serialized = self._serialize(value)
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(full_key, ttl, serialized)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory
        self.memory_cache[full_key] = value
        self.memory_expiry[full_key] = datetime.now() + timedelta(seconds=ttl)
        
        # Cleanup expired entries if memory cache is too large
        if len(self.memory_cache) > 10000:
            self._cleanup_memory_cache()
    
    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache"""
        now = datetime.now()
        expired = [k for k, v in self.memory_expiry.items() if v < now]
        for k in expired:
            del self.memory_cache[k]
            del self.memory_expiry[k]
        logger.debug(f"Cleaned up {len(expired)} expired cache entries")
    
    async def delete(self, key: str, prefix: str = ""):
        """Delete value from cache"""
        full_key = self._make_key(key, prefix)
        
        if self.redis_client:
            try:
                self.redis_client.delete(full_key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if full_key in self.memory_cache:
            del self.memory_cache[full_key]
            del self.memory_expiry[full_key]
    
    async def clear_prefix(self, prefix: str):
        """Clear all keys with prefix"""
        pattern = f"bi_ide:{prefix}:*"
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        # Clear from memory
        keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern[:-1])]
        for k in keys_to_delete:
            del self.memory_cache[k]
            if k in self.memory_expiry:
                del self.memory_expiry[k]
    
    async def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            "memory_entries": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_memory_used"] = info.get("used_memory_human", "N/A")
                stats["redis_keys"] = self.redis_client.dbsize()
            except Exception as e:
                stats["redis_error"] = str(e)
        
        return stats


def cached(ttl: int = 3600, prefix: str = "", key_func=None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash of function name and arguments
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_value = await cache_manager.get(cache_key, prefix)
            if cached_value is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set(cache_key, result, ttl, prefix)
            
            return result
        return wrapper
    return decorator


# Global instance
cache_manager = CacheManager()
