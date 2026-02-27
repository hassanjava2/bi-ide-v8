"""
BI-IDE v8 - Multi-Level Caching System
L1: In-Memory (LRU Cache)
L2: Redis
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
import logging

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

logger = logging.getLogger(__name__)
T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"                    # Time-based expiration
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_ASIDE = "cache_aside"


@dataclass
class CacheConfig:
    """Cache configuration"""
    ttl: int = 3600
    max_size: int = 10000
    strategy: CacheStrategy = CacheStrategy.TTL
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 50
    memory_cache_size: int = 1000
    compression: bool = True
    compression_threshold: int = 1024  # bytes


class CacheEntry:
    """Cache entry with metadata"""
    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 1
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


class BaseCache(ABC):
    """Abstract base cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        pass


class MemoryCache(BaseCache):
    """L1: In-Memory LRU Cache"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.touch()
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = CacheEntry(value, ttl)
            
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return False
            return True
    
    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        async with self._lock:
            return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_usage_bytes": self._estimate_memory()
            }
    
    def _estimate_memory(self) -> int:
        """Estimate memory usage"""
        try:
            total = 0
            for entry in self._cache.values():
                total += len(pickle.dumps(entry))
            return total
        except:
            return 0


class RedisCache(BaseCache):
    """L2: Redis Cache"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", pool_size: int = 50):
        self._redis_url = redis_url
        self._pool_size = pool_size
        self._redis: Optional[Redis] = None
        self._lock = asyncio.Lock()
    
    async def connect(self):
        """Connect to Redis"""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url,
                max_connections=self._pool_size,
                decode_responses=False
            )
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes"""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value"""
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        await self.connect()
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        await self.connect()
        try:
            data = self._serialize(value)
            if ttl:
                await self._redis.setex(key, ttl, data)
            else:
                await self._redis.set(key, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        await self.connect()
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        await self.connect()
        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        await self.connect()
        try:
            await self._redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        await self.connect()
        try:
            keys = await self._redis.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        await self.connect()
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis delete_pattern error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        await self.connect()
        try:
            info = await self._redis.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_keys": sum(db.get("keys", 0) for db in info.get("keyspace", {}).values())
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}


class RedisCacheManager(BaseCache):
    """Multi-Level Cache Manager (L1 + L2)"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._l1 = MemoryCache(self.config.memory_cache_size)
        self._l2 = RedisCache(self.config.redis_url, self.config.redis_pool_size)
        self._invalidation_callbacks: List[Callable[[str], None]] = []
        self._warming_tasks: Set[asyncio.Task] = set()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from L1 first, then L2"""
        # Try L1
        value = await self._l1.get(key)
        if value is not None:
            logger.debug(f"L1 cache hit: {key}")
            return value
        
        # Try L2
        value = await self._l2.get(key)
        if value is not None:
            logger.debug(f"L2 cache hit: {key}")
            # Populate L1
            await self._l1.set(key, value, self.config.ttl)
            return value
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in both L1 and L2"""
        ttl = ttl or self.config.ttl
        
        # Set L1
        await self._l1.set(key, value, ttl)
        
        # Set L2
        await self._l2.set(key, value, ttl)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete from both caches"""
        await self._l1.delete(key)
        await self._l2.delete(key)
        
        # Trigger invalidation callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(key)
            except Exception as e:
                logger.error(f"Invalidation callback error: {e}")
        
        return True
    
    async def exists(self, key: str) -> bool:
        if await self._l1.exists(key):
            return True
        return await self._l2.exists(key)
    
    async def clear(self) -> bool:
        await self._l1.clear()
        await self._l2.clear()
        return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys from L2 (source of truth)"""
        return await self._l2.keys(pattern)
    
    # Cache Invalidation Strategies
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        l1_keys = await self._l1.keys(pattern)
        for key in l1_keys:
            await self._l1.delete(key)
        
        return await self._l2.delete_pattern(pattern)
    
    async def invalidate_tag(self, tag: str) -> int:
        """Invalidate by tag"""
        tag_key = f"__tag__:{tag}"
        tagged_keys = await self._l2.get(tag_key)
        if tagged_keys:
            count = 0
            for key in tagged_keys:
                if await self.delete(key):
                    count += 1
            await self._l2.delete(tag_key)
            return count
        return 0
    
    async def set_with_tags(self, key: str, value: Any, tags: List[str], ttl: Optional[int] = None):
        """Set with tags for grouped invalidation"""
        await self.set(key, value, ttl)
        
        # Update tag index
        for tag in tags:
            tag_key = f"__tag__:{tag}"
            existing = await self._l2.get(tag_key) or []
            if key not in existing:
                existing.append(key)
                await self._l2.set(tag_key, existing, ttl)
    
    # Cache Warming
    
    def register_invalidation_callback(self, callback: Callable[[str], None]):
        """Register callback for cache invalidation events"""
        self._invalidation_callbacks.append(callback)
    
    async def warm_cache(self, keys_values: Dict[str, Any], ttl: Optional[int] = None):
        """Pre-populate cache with data"""
        for key, value in keys_values.items():
            await self.set(key, value, ttl)
        logger.info(f"Cache warmed with {len(keys_values)} entries")
    
    async def warm_cache_async(self, fetcher: Callable[[List[str]], Dict[str, Any]], 
                               keys: List[str], ttl: Optional[int] = None):
        """Async cache warming with data fetcher"""
        data = await asyncio.get_event_loop().run_in_executor(None, fetcher, keys)
        await self.warm_cache(data, ttl)
    
    async def schedule_warming(self, interval: int, fetcher: Callable, keys: List[str]):
        """Schedule periodic cache warming"""
        async def warming_task():
            while True:
                try:
                    await self.warm_cache_async(fetcher, keys)
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Cache warming error: {e}")
                    await asyncio.sleep(60)  # Retry in 1 minute
        
        task = asyncio.create_task(warming_task())
        self._warming_tasks.add(task)
        task.add_done_callback(self._warming_tasks.discard)
    
    # Decorator for caching
    
    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """Decorator to cache function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()
                
                # Try cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    # Statistics
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "l1_memory": await self._l1.get_stats(),
            "l2_redis": await self._l2.get_stats()
        }
    
    async def close(self):
        """Cleanup resources"""
        for task in self._warming_tasks:
            task.cancel()
        await self._l2.disconnect()


# Utility functions

def generate_cache_key(*args, **kwargs) -> str:
    """Generate consistent cache key"""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


# Global cache manager instance
_cache_manager: Optional[RedisCacheManager] = None


def get_cache_manager() -> RedisCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
    return _cache_manager


async def init_cache(config: CacheConfig):
    """Initialize global cache with config"""
    global _cache_manager
    _cache_manager = RedisCacheManager(config)


async def close_cache():
    """Close global cache"""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None
