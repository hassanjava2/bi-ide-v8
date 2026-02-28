"""
Auto Reconnect - إعادة الاتصال التلقائي
Automatic reconnection when connection drops
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, 
    Optional, TypeVar, Union, Set
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """حالات Circuit Breaker - Circuit Breaker states"""
    CLOSED = "closed"      # العادي - Normal operation
    OPEN = "open"          # مفتوح - Failing
    HALF_OPEN = "half_open"  # نصف مفتوح - Testing


@dataclass
class ConnectionPoolConfig:
    """إعدادات تجمع الاتصالات - Connection pool configuration"""
    max_connections: int = 10
    min_connections: int = 2
    max_idle_time: float = 300.0  # 5 دقائق
    connection_timeout: float = 30.0
    health_check_interval: float = 60.0


@dataclass
class RetryConfig:
    """إعدادات إعادة المحاولة - Retry configuration"""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """إعدادات Circuit Breaker - Circuit Breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit Breaker Pattern
    نمط قاطع الدائرة
    
    يمنع المحاولات المتكررة للاتصال الفاشل
    Prevents repeated connection attempts to failing services
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """
        تهيئة Circuit Breaker
        Initialize circuit breaker
        
        Args:
            name: اسم الخدمة
            config: الإعدادات
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs) -> T:
        """
        استدعاء دالة مع Circuit Breaker
        Call function with circuit breaker
        
        Args:
            func: الدالة المستهدفة
            *args: معاملات الدالة
            **kwargs: معاملات الدالة
            
        Returns:
            T: نتيجة الدالة
            
        Raises:
            Exception: إذا كان Circuit Breaker مفتوحاً
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if await self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(
                        f"Circuit breaker {self.name} is OPEN"
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(
                        f"Circuit breaker {self.name} HALF_OPEN limit reached"
                    )
                self._half_open_calls += 1
        
        # تنفيذ الدالة خارج القفل
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        """معالجة النجاح - Handle success"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit breaker {self.name} CLOSED")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self) -> None:
        """معالجة الفشل - Handle failure"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPEN (failure in HALF_OPEN)")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} OPEN "
                    f"({self.failure_count} failures)"
                )
    
    async def _should_attempt_reset(self) -> bool:
        """التحقق مما إذا كان يجب محاولة إعادة التعيين"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout
    
    def _reset(self) -> None:
        """إعادة تعيين الحالة - Reset state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._half_open_calls = 0
        self.last_failure_time = None
    
    def get_state(self) -> Dict[str, Any]:
        """الحصول على الحالة - Get state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class ConnectionPool:
    """
    تجمع الاتصالات
    Connection Pool
    
    يدير مجموعة من الاتصالات لإعادة استخدامها
    Manages a pool of connections for reuse
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, Any]],
        config: Optional[ConnectionPoolConfig] = None
    ):
        """
        تهيئة تجمع الاتصالات
        Initialize connection pool
        
        Args:
            name: اسم التجمع
            factory: دالة إنشاء الاتصال
            config: الإعدادات
        """
        self.name = name
        self.factory = factory
        self.config = config or ConnectionPoolConfig()
        
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: Set[Any] = set()
        self._total_connections = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """تهيئة التجمع - Initialize pool"""
        # إنشاء الاتصالات الدنيا
        for _ in range(self.config.min_connections):
            conn = await self._create_connection()
            if conn:
                await self._available.put(conn)
        
        # بدء فحص الصحة
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(
            f"Connection pool {self.name} initialized "
            f"({self.config.min_connections} connections)"
        )
    
    async def acquire(self) -> Optional[Any]:
        """
        الحصول على اتصال من التجمع
        Acquire connection from pool
        
        Returns:
            Optional[Any]: الاتصال أو None
        """
        if self._closed:
            return None
        
        try:
            # انتظار اتصال متاح مع مهلة
            conn = await asyncio.wait_for(
                self._available.get(),
                timeout=self.config.connection_timeout
            )
            
            async with self._lock:
                self._in_use.add(conn)
            
            return conn
            
        except asyncio.TimeoutError:
            # إنشاء اتصال جديد إذا لم نصل للحد الأقصى
            async with self._lock:
                if self._total_connections < self.config.max_connections:
                    conn = await self._create_connection()
                    if conn:
                        self._in_use.add(conn)
                        return conn
            
            logger.warning(f"Connection pool {self.name} exhausted")
            return None
    
    async def release(self, conn: Any) -> None:
        """
        إرجاع الاتصال إلى التجمع
        Release connection back to pool
        
        Args:
            conn: الاتصال
        """
        if conn is None:
            return
        
        async with self._lock:
            self._in_use.discard(conn)
        
        if not self._closed:
            await self._available.put(conn)
    
    async def close(self) -> None:
        """إغلاق التجمع - Close pool"""
        self._closed = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # إغلاق جميع الاتصالات
        async with self._lock:
            for conn in list(self._in_use):
                await self._close_connection(conn)
            self._in_use.clear()
        
        while not self._available.empty():
            conn = await self._available.get()
            await self._close_connection(conn)
        
        logger.info(f"Connection pool {self.name} closed")
    
    async def _create_connection(self) -> Optional[Any]:
        """إنشاء اتصال جديد - Create new connection"""
        try:
            conn = await self.factory()
            async with self._lock:
                self._total_connections += 1
            return conn
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            return None
    
    async def _close_connection(self, conn: Any) -> None:
        """إغلاق اتصال - Close connection"""
        try:
            if hasattr(conn, 'close'):
                await conn.close()
            async with self._lock:
                self._total_connections -= 1
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _health_check_loop(self) -> None:
        """حلقة فحص صحة الاتصالات - Health check loop"""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")
    
    async def _health_check(self) -> None:
        """فحص صحة الاتصالات - Check connection health"""
        # فحص الاتصالات المتاحة
        temp_queue = asyncio.Queue()
        
        while not self._available.empty():
            try:
                conn = self._available.get_nowait()
                
                # التحقق من صحة الاتصال (افتراضياً نعتبرها صحية)
                is_healthy = True
                
                if is_healthy:
                    await temp_queue.put(conn)
                else:
                    await self._close_connection(conn)
            except asyncio.QueueEmpty:
                break
        
        # إعادة الاتصالات الصحية
        while not temp_queue.empty():
            conn = await temp_queue.get()
            await self._available.put(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التجمع - Get pool stats"""
        return {
            'name': self.name,
            'total': self._total_connections,
            'available': self._available.qsize(),
            'in_use': len(self._in_use),
            'max': self.config.max_connections
        }


class AutoReconnect:
    """
    ديكوراتور إعادة الاتصال التلقائي
    Auto-reconnect decorator
    
    يضيف إعادة المحاولة والـ Circuit Breaker للدوال
    Adds retry and circuit breaker to functions
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[Exception], None]] = None
    ):
        """
        تهيئة AutoReconnect
        Initialize AutoReconnect
        
        Args:
            retry_config: إعدادات إعادة المحاولة
            circuit_breaker_config: إعدادات Circuit Breaker
            on_connect: دالة استدعاء عند الاتصال
            on_disconnect: دالة استدعاء عند فقدان الاتصال
        """
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def __call__(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """
        تطبيق الديكوراتور
        Apply decorator
        
        Args:
            func: الدالة المستهدفة
            
        Returns:
            Callable: الدالة المغلفة
        """
        func_name = func.__name__
        
        # إنشاء Circuit Breaker إذا لزم الأمر
        if self.circuit_breaker_config:
            if func_name not in self._circuit_breakers:
                self._circuit_breakers[func_name] = CircuitBreaker(
                    func_name,
                    self.circuit_breaker_config
                )
            circuit_breaker = self._circuit_breakers[func_name]
        else:
            circuit_breaker = None
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, self.retry_config.max_attempts + 1):
                try:
                    # استخدام Circuit Breaker إذا متاح
                    if circuit_breaker:
                        result = await circuit_breaker.call(func, *args, **kwargs)
                    else:
                        result = await func(*args, **kwargs)
                    
                    # نجح الاتصال
                    if attempt > 1 and self.on_connect:
                        try:
                            self.on_connect()
                        except Exception as e:
                            logger.error(f"Error in on_connect callback: {e}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # فشل الاتصال
                    if self.on_disconnect:
                        try:
                            self.on_disconnect(e)
                        except Exception as cb_e:
                            logger.error(f"Error in on_disconnect callback: {cb_e}")
                    
                    # تسجيل الفشل
                    logger.warning(
                        f"Function {func_name} failed (attempt {attempt}): {e}"
                    )
                    
                    # التحقق مما إذا كانت هذه المحاولة الأخيرة
                    if attempt >= self.retry_config.max_attempts:
                        break
                    
                    # حساب وقت الانتظار
                    delay = self._calculate_delay(attempt)
                    
                    logger.info(f"Retrying {func_name} in {delay:.2f}s...")
                    await asyncio.sleep(delay)
            
            # جميع المحاولات فشلت
            raise last_exception or Exception(f"All retry attempts failed for {func_name}")
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        حساب وقت الانتظار للمحاولة
        Calculate delay for attempt
        
        Args:
            attempt: رقم المحاولة
            
        Returns:
            float: وقت الانتظار بالثواني
        """
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay
        )
        
        # إضافة عشوائية (jitter) لتجنب thundering herd
        if self.retry_config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


# ديكوراتور مبسط للاستخدام السريع
def reconnect(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    on_connect: Optional[Callable[[], None]] = None,
    on_disconnect: Optional[Callable[[Exception], None]] = None,
    use_circuit_breaker: bool = True
):
    """
    ديكوراتور إعادة الاتصال السريع
    Quick reconnect decorator
    
    Args:
        max_attempts: الحد الأقصى للمحاولات
        base_delay: التأخير الأساسي
        on_connect: دالة استدعاء عند الاتصال
        on_disconnect: دالة استدعاء عند فقدان الاتصال
        use_circuit_breaker: استخدام Circuit Breaker
        
    Returns:
        Callable: الديكوراتور
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay
    )
    
    circuit_config = CircuitBreakerConfig() if use_circuit_breaker else None
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        reconnector = AutoReconnect(
            retry_config=retry_config,
            circuit_breaker_config=circuit_config,
            on_connect=on_connect,
            on_disconnect=on_disconnect
        )
        return reconnector(func)
    
    return decorator


class ReconnectingClient:
    """
    عميل مع إعادة اتصال تلقائي
    Client with auto-reconnect
    
    فئة مساعدة لإدارة الاتصالات مع إعادة الاتصال
    Helper class for managing connections with reconnect
    """
    
    def __init__(
        self,
        connect_func: Callable[[], Coroutine[Any, Any, Any]],
        disconnect_func: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        تهيئة العميل
        Initialize client
        
        Args:
            connect_func: دالة الاتصال
            disconnect_func: دالة فصل الاتصال
            retry_config: إعدادات إعادة المحاولة
        """
        self.connect_func = connect_func
        self.disconnect_func = disconnect_func
        self.retry_config = retry_config or RetryConfig()
        
        self.connection: Optional[Any] = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': []
        }
    
    async def connect(self) -> Any:
        """
        الاتصال مع إعادة المحاولة
        Connect with retry
        
        Returns:
            Any: الاتصال
        """
        async with self._lock:
            if self._connected and self.connection:
                return self.connection
            
            last_exception = None
            
            for attempt in range(1, self.retry_config.max_attempts + 1):
                try:
                    self.connection = await self.connect_func()
                    self._connected = True
                    
                    # استدعاء الدوال المسجلة
                    for callback in self._callbacks['on_connect']:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Error in on_connect callback: {e}")
                    
                    logger.info("Connected successfully")
                    return self.connection
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Connection attempt {attempt} failed: {e}")
                    
                    if attempt < self.retry_config.max_attempts:
                        delay = min(
                            self.retry_config.base_delay * (2 ** (attempt - 1)),
                            self.retry_config.max_delay
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception or Exception("All connection attempts failed")
    
    async def disconnect(self) -> None:
        """قطع الاتصال - Disconnect"""
        async with self._lock:
            if self.connection and self.disconnect_func:
                try:
                    await self.disconnect_func(self.connection)
                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")
            
            self.connection = None
            self._connected = False
            
            # استدعاء الدوال المسجلة
            for callback in self._callbacks['on_disconnect']:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in on_disconnect callback: {e}")
            
            logger.info("Disconnected")
    
    def on_connect(self, callback: Callable[[], None]) -> None:
        """تسجيل دالة استدعاء عند الاتصال - Register connect callback"""
        self._callbacks['on_connect'].append(callback)
    
    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """تسجيل دالة استدعاء عند فقدان الاتصال - Register disconnect callback"""
        self._callbacks['on_disconnect'].append(callback)
    
    def is_connected(self) -> bool:
        """التحقق من حالة الاتصال - Check connection status"""
        return self._connected
    
    async def execute(self, func: Callable[[Any], Coroutine[Any, Any, T]]) -> T:
        """
        تنفيذ دالة على الاتصال مع إعادة الاتصال إذا لزم
        Execute function on connection with reconnect if needed
        
        Args:
            func: الدالة المنفذة
            
        Returns:
            T: نتيجة الدالة
        """
        if not self._connected or not self.connection:
            await self.connect()
        
        try:
            return await func(self.connection)
        except Exception as e:
            # إعادة الاتصال والمحاولة مرة أخرى
            logger.warning(f"Execution failed, reconnecting: {e}")
            await self.disconnect()
            await self.connect()
            return await func(self.connection)


async def main():
    """الدالة الرئيسية - Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # اختبار CircuitBreaker
    print("\n=== Circuit Breaker Test ===")
    cb = CircuitBreaker("test_service", CircuitBreakerConfig(failure_threshold=3))
    
    async def failing_func():
        raise Exception("Simulated failure")
    
    async def success_func():
        return "Success!"
    
    # محاولات فاشلة
    for i in range(4):
        try:
            await cb.call(failing_func)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
    
    print(f"Circuit state: {cb.state.value}")
    
    # اختبار AutoReconnect
    print("\n=== AutoReconnect Test ===")
    
    attempt_count = 0
    
    @reconnect(max_attempts=3, base_delay=0.5)
    async def test_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    result = await test_function()
    print(f"Result: {result}")
    
    # اختبار ConnectionPool
    print("\n=== Connection Pool Test ===")
    
    connection_counter = 0
    
    async def create_conn():
        nonlocal connection_counter
        connection_counter += 1
        return {"id": connection_counter, "active": True}
    
    pool = ConnectionPool("test_pool", create_conn, ConnectionPoolConfig(max_connections=5))
    await pool.initialize()
    
    print(f"Pool stats: {pool.get_stats()}")
    
    # الحصول على اتصال
    conn = await pool.acquire()
    print(f"Acquired connection: {conn}")
    print(f"Pool stats after acquire: {pool.get_stats()}")
    
    # إرجاع الاتصال
    await pool.release(conn)
    print(f"Pool stats after release: {pool.get_stats()}")
    
    await pool.close()
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
