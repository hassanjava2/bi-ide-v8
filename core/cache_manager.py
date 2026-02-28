"""
Cache Manager - مدير التخزين المؤقت
Manage Redis and in-memory caching
إدارة التخزين المؤقت في Redis والذاكرة
"""
import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aioredis

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """إدخال التخزين المؤقت - Cache entry"""
    value: Any
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_expired(self) -> bool:
        """التحقق من انتهاء الصلاحية - Check if expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class LRUCache:
    """
    ذاكرة التخزين المؤقت LRU في الذاكرة
    In-memory LRU cache with TTL support
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 300):
        """
        Initialize LRU cache
        
        Args:
            max_size: الحد الأقصى لعدد العناصر
            default_ttl: مدة الصلاحية الافتراضية بالثواني
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """الحصول على قيمة من الذاكرة - Get value from cache"""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
                
            if entry.is_expired():
                # حذف العنصر منتهي الصلاحية
                del self._cache[key]
                self._misses += 1
                return None
                
            # تحديث ترتيب LRU - Update LRU order
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._hits += 1
            
            return entry.value
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        تخزين قيمة في الذاكرة
        Store value in cache
        
        Args:
            key: مفتاح التخزين
            value: القيمة
            ttl: مدة الصلاحية بالثواني (None = لا تنتهي)
        """
        async with self._lock:
            # حساب وقت الانتهاء - Calculate expiry
            ttl = ttl if ttl is not None else self.default_ttl
            expires_at = None
            if ttl is not None and ttl > 0:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                
            # إنشاء إدخال جديد - Create new entry
            entry = CacheEntry(
                value=value,
                expires_at=expires_at
            )
            
            # إزالة العنصر القديم إذا موجود - Remove old entry if exists
            if key in self._cache:
                self._cache.move_to_end(key)
                
            # إضافة العنصر الجديد - Add new entry
            self._cache[key] = entry
            
            # إزالة العناصر الزائدة - Evict excess entries
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
                
    async def delete(self, key: str) -> bool:
        """حذف عنصر من الذاكرة - Delete item from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
    async def clear(self) -> None:
        """مسح جميع العناصر - Clear all items"""
        async with self._lock:
            self._cache.clear()
            
    async def keys(self, pattern: str = "*") -> List[str]:
        """الحصول على المفاتيح - Get keys (simple filter)"""
        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            # تصفية بسيطة - Simple filtering
            return [k for k in self._cache.keys() if pattern.replace("*", "") in k]
            
    async def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات - Get cache statistics"""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            # حساب العناصر منتهية الصلاحية
            expired_count = sum(1 for e in self._cache.values() if e.is_expired())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate_percent": round(hit_rate, 2),
                "expired_items": expired_count
            }
            
    async def cleanup_expired(self) -> int:
        """تنظيف العناصر منتهية الصلاحية - Clean up expired items"""
        async with self._lock:
            expired_keys = [
                k for k, e in self._cache.items() if e.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


class CacheManager:
    """
    مدير التخزين المؤقت الرئيسي
    Main cache manager with Redis and in-memory support
    """
    
    def __init__(
        self,
        redis_url: str = None,
        lru_max_size: int = 10000,
        lru_default_ttl: int = 300,
        fallback_to_memory: bool = True
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_to_memory = fallback_to_memory
        
        self.redis: Optional[aioredis.Redis] = None
        self.lru = LRUCache(max_size=lru_max_size, default_ttl=lru_default_ttl)
        self._redis_available = False
        self._lock = asyncio.Lock()
        
        # الإحصائيات - Statistics
        self._stats = {
            "redis_hits": 0,
            "redis_misses": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "fallback_uses": 0,
            "errors": 0
        }
        
    async def initialize(self) -> None:
        """Initialize cache manager"""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                decode_responses=False,  # نحتاج bytes للـ pickle
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            await self.redis.ping()
            self._redis_available = True
            logger.info("✅ Cache manager connected to Redis")
        except Exception as e:
            self._redis_available = False
            logger.warning(
                f"⚠️ Redis unavailable, using memory cache only: {e}"
            )
            
    def _serialize(self, value: Any) -> bytes:
        """تسلسل القيمة - Serialize value"""
        return pickle.dumps(value)
        
    def _deserialize(self, data: bytes) -> Any:
        """إلغاء تسلسل القيمة - Deserialize value"""
        return pickle.loads(data)
        
    def _make_key(self, *args, **kwargs) -> str:
        """إنشاء مفتاح من الوسائط - Create key from arguments"""
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
    async def get(self, key: str) -> Optional[Any]:
        """
        الحصول على قيمة من التخزين المؤقت
        Get value from cache (Redis first, then memory)
        """
        # محاولة Redis أولاً - Try Redis first
        if self._redis_available and self.redis:
            try:
                data = await self.redis.get(key)
                if data:
                    self._stats["redis_hits"] += 1
                    return self._deserialize(data)
                self._stats["redis_misses"] += 1
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self._stats["errors"] += 1
                
        # محاولة الذاكرة - Try memory cache
        value = await self.lru.get(key)
        if value is not None:
            self._stats["memory_hits"] += 1
            return value
        self._stats["memory_misses"] += 1
        
        return None
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        memory_only: bool = False
    ) -> bool:
        """
        تخزين قيمة في التخزين المؤقت
        Store value in cache
        
        Args:
            key: مفتاح التخزين
            value: القيمة
            ttl: مدة الصلاحية بالثواني
            memory_only: تخزين في الذاكرة فقط (بدون Redis)
        """
        success = True
        
        # تخزين في Redis - Store in Redis
        if not memory_only and self._redis_available and self.redis:
            try:
                serialized = self._serialize(value)
                if ttl:
                    await self.redis.setex(key, ttl, serialized)
                else:
                    await self.redis.set(key, serialized)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self._stats["errors"] += 1
                success = False
                
        # تخزين في الذاكرة - Store in memory
        await self.lru.set(key, value, ttl=ttl)
        
        return success
        
    async def delete(self, key: str) -> bool:
        """حذف عنصر من التخزين المؤقت - Delete from cache"""
        success = True
        
        # حذف من Redis
        if self._redis_available and self.redis:
            try:
                await self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                success = False
                
        # حذف من الذاكرة
        await self.lru.delete(key)
        
        return success
        
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        إلغاء صلاحية جميع المفاتيح المطابقة للنمط
        Invalidate all keys matching pattern
        """
        count = 0
        
        # إلغاء من Redis
        if self._redis_available and self.redis:
            try:
                keys = []
                async for key in self.redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    await self.redis.delete(*keys)
                    count += len(keys)
            except Exception as e:
                logger.warning(f"Redis pattern delete error: {e}")
                
        # إلغاء من الذاكرة (تصفية بسيطة)
        memory_keys = await self.lru.keys(pattern)
        for key in memory_keys:
            await self.lru.delete(key)
            count += 1
            
        logger.info(f"Invalidated {count} keys matching '{pattern}'")
        return count
        
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        الحصول على قيمة أو إنشاؤها إذا غير موجودة
        Get value or create if not exists
        """
        # محاولة الحصول على القيمة
        value = await self.get(key)
        if value is not None:
            return value
            
        # إنشاء القيمة
        try:
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
        except Exception as e:
            logger.error(f"Factory error for key {key}: {e}")
            raise
            
        # تخزين القيمة
        await self.set(key, value, ttl)
        
        return value
        
    def cached(
        self,
        ttl: Optional[int] = 300,
        key_prefix: str = "",
        memory_only: bool = False
    ):
        """
        مزخرف للتخزين المؤقت للدوال
        Decorator for caching function results
        
        Args:
            ttl: مدة الصلاحية بالثواني
            key_prefix: بادئة المفتاح
            memory_only: تخزين في الذاكرة فقط
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # إنشاء مفتاح - Create key
                cache_key = f"{key_prefix}:{func.__name__}:"
                cache_key += self._make_key(*args, **kwargs)
                
                # محاولة الحصول من الذاكرة
                result = await self.get(cache_key)
                if result is not None:
                    return result
                    
                # تنفيذ الدالة
                result = await func(*args, **kwargs)
                
                # تخزين النتيجة
                await self.set(cache_key, result, ttl, memory_only)
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # إنشاء مفتاح
                cache_key = f"{key_prefix}:{func.__name__}:"
                cache_key += self._make_key(*args, **kwargs)
                
                # للدوال المتزامنة نستخدم الذاكرة فقط
                # For sync functions, use a simple sync wrapper
                import threading
                local_result = [None]
                event = threading.Event()
                
                async def _async_wrapper():
                    result = await self.get(cache_key)
                    if result is None:
                        result = func(*args, **kwargs)
                        await self.set(cache_key, result, ttl, memory_only=True)
                    local_result[0] = result
                    event.set()
                    
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_async_wrapper())
                except RuntimeError:
                    asyncio.run(_async_wrapper())
                    
                event.wait(timeout=5)
                return local_result[0]
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    async def clear(self) -> None:
        """مسح جميع العناصر - Clear all cache"""
        if self._redis_available and self.redis:
            try:
                await self.redis.flushdb()
            except Exception as e:
                logger.warning(f"Redis flush error: {e}")
                
        await self.lru.clear()
        logger.info("Cache cleared")
        
    async def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات - Get cache statistics"""
        redis_stats = None
        
        if self._redis_available and self.redis:
            try:
                info = await self.redis.info()
                redis_stats = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "keys": await self.redis.dbsize(),
                    "connected_clients": info.get("connected_clients", 0),
                    "uptime_seconds": info.get("uptime_in_seconds", 0)
                }
            except Exception as e:
                logger.warning(f"Redis stats error: {e}")
                
        return {
            "redis_available": self._redis_available,
            "redis_stats": redis_stats,
            "memory_stats": await self.lru.get_stats(),
            "operation_stats": self._stats.copy()
        }
        
    async def cleanup(self) -> None:
        """تنظيف التخزين المؤقت - Clean up cache"""
        expired = await self.lru.cleanup_expired()
        if expired > 0:
            logger.debug(f"Cleaned up {expired} expired cache entries")
            
    async def close(self) -> None:
        """إغلاق مدير التخزين المؤقت - Close cache manager"""
        if self.redis:
            await self.redis.close()
            
        logger.info("Cache manager closed")


# نسخة عامة - Global instance
cache_manager = CacheManager()


# دوال مساعدة - Helper functions
async def get_cache() -> CacheManager:
    """الحصول على مدير التخزين المؤقت - Get cache manager"""
    return cache_manager


cached = cache_manager.cached


async def invalidate_cache_pattern(pattern: str) -> int:
    """إلغاء صلاحية نمط - Invalidate pattern"""
    return await cache_manager.invalidate_pattern(pattern)
