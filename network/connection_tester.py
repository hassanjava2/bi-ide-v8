"""
Network Connection Tester - BI-IDE v8
اختبار الاتصال بين Windows و Ubuntu
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not available, using fallback")

from core.config import settings
from core.logging_config import logger


@dataclass
class ConnectionResult:
    """نتيجة اختبار الاتصال"""
    success: bool
    target: str
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "target": self.target,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat() + "Z",
            "error": self.error,
            "details": self.details,
        }


class ConnectionTester:
    """فاحص الاتصالات بين Windows و Ubuntu"""
    
    def __init__(self):
        self.timeout = 10.0
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        if AIOHTTP_AVAILABLE:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self, 
        url: str, 
        method: str = "GET",
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> ConnectionResult:
        """إجراء طلب HTTP غير متزامن"""
        start_time = time.perf_counter()
        
        if not AIOHTTP_AVAILABLE or not self.session:
            # Fallback using asyncio and httpx if available
            try:
                import httpx
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=headers)
                    elif method.upper() == "POST":
                        response = await client.post(url, headers=headers, json=data)
                    else:
                        response = await client.request(method, url, headers=headers, json=data)
                    
                    latency = (time.perf_counter() - start_time) * 1000
                    return ConnectionResult(
                        success=response.status_code < 400,
                        target=url,
                        latency_ms=latency,
                        details={
                            "status_code": response.status_code,
                            "content_length": len(response.content),
                        }
                    )
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000
                return ConnectionResult(
                    success=False,
                    target=url,
                    latency_ms=latency,
                    error=str(e)
                )
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data
            ) as response:
                latency = (time.perf_counter() - start_time) * 1000
                content = await response.text()
                
                return ConnectionResult(
                    success=response.status < 400,
                    target=url,
                    latency_ms=latency,
                    details={
                        "status_code": response.status,
                        "content_length": len(content),
                        "content_type": response.headers.get("Content-Type"),
                    }
                )
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            return ConnectionResult(
                success=False,
                target=url,
                latency_ms=latency,
                error="Connection timeout"
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ConnectionResult(
                success=False,
                target=url,
                latency_ms=latency,
                error=str(e)
            )
    
    async def test_rtx4090_connection(self, endpoint: str = "/") -> ConnectionResult:
        """
        اختبار الاتصال بـ RTX 4090
        
        Args:
            endpoint: نقطة النهاية للاختبار (افتراضي: /)
            
        Returns:
            ConnectionResult: نتيجة الاختبار
        """
        url = f"{settings.rtx4090_url}{endpoint}"
        logger.info(f"Testing RTX 4090 connection: {url}")
        
        result = await self._make_request(url)
        
        if result.success:
            logger.info(f"✅ RTX 4090 connection successful: {result.latency_ms:.2f}ms")
        else:
            logger.error(f"❌ RTX 4090 connection failed: {result.error}")
        
        return result
    
    async def test_windows_api_connection(
        self, 
        port: int = 8000, 
        endpoint: str = "/api/v1/system/status"
    ) -> ConnectionResult:
        """
        اختبار API محلي على Windows
        
        Args:
            port: رقم المنفذ
            endpoint: نقطة النهاية للاختبار
            
        Returns:
            ConnectionResult: نتيجة الاختبار
        """
        url = f"http://localhost:{port}{endpoint}"
        logger.info(f"Testing Windows API connection: {url}")
        
        result = await self._make_request(url)
        
        if result.success:
            logger.info(f"✅ Windows API connection successful: {result.latency_ms:.2f}ms")
        else:
            logger.error(f"❌ Windows API connection failed: {result.error}")
        
        return result
    
    async def test_end_to_end(
        self,
        test_inference: bool = True
    ) -> Dict[str, Any]:
        """
        اختبار التدفق الكامل من Windows إلى Ubuntu (RTX 4090)
        
        Args:
            test_inference: اختبار inference أيضاً
            
        Returns:
            Dict: نتائج جميع الاختبارات
        """
        logger.info("=" * 60)
        logger.info("Running End-to-End Connection Test")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "tests": {},
            "overall_success": True,
        }
        
        # Test 1: RTX 4090 basic connection
        rtx_result = await self.test_rtx4090_connection("/")
        results["tests"]["rtx4090_basic"] = rtx_result.to_dict()
        
        # Test 2: RTX 4090 status endpoint
        rtx_status = await self.test_rtx4090_connection("/status")
        results["tests"]["rtx4090_status"] = rtx_status.to_dict()
        
        # Test 3: Windows API
        windows_result = await self.test_windows_api_connection()
        results["tests"]["windows_api"] = windows_result.to_dict()
        
        # Test 4: Inference endpoint (if requested)
        if test_inference:
            inference_result = await self.test_rtx4090_connection("/inference")
            results["tests"]["rtx4090_inference"] = inference_result.to_dict()
        
        # Calculate overall success
        all_success = all(
            t.get("success", False) 
            for t in results["tests"].values()
        )
        results["overall_success"] = all_success
        
        # Summary
        total_latency = sum(
            t.get("latency_ms", 0) 
            for t in results["tests"].values()
        )
        results["summary"] = {
            "total_tests": len(results["tests"]),
            "passed": sum(1 for t in results["tests"].values() if t.get("success")),
            "failed": sum(1 for t in results["tests"].values() if not t.get("success")),
            "average_latency_ms": round(total_latency / len(results["tests"]), 2) if results["tests"] else 0,
        }
        
        logger.info("=" * 60)
        logger.info(f"End-to-End Test Complete: {'✅ PASSED' if all_success else '❌ FAILED'}")
        logger.info(f"Tests: {results['summary']['passed']}/{results['summary']['total_tests']} passed")
        logger.info("=" * 60)
        
        return results
    
    async def run_parallel_tests(
        self,
        urls: List[str],
        method: str = "GET"
    ) -> List[ConnectionResult]:
        """
        تشغيل اختبارات متوازية على عدة URLs
        
        Args:
            urls: قائمة URLs للاختبار
            method: HTTP method
            
        Returns:
            List[ConnectionResult]: نتائج جميع الاختبارات
        """
        tasks = [self._make_request(url, method) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)


# Convenience functions for direct use
async def test_rtx4090_connection(endpoint: str = "/") -> ConnectionResult:
    """اختبار الاتصال بـ RTX 4090 (دالة مساعدة)"""
    async with ConnectionTester() as tester:
        return await tester.test_rtx4090_connection(endpoint)


async def test_windows_api_connection(
    port: int = 8000, 
    endpoint: str = "/api/v1/system/status"
) -> ConnectionResult:
    """اختبار API محلي (دالة مساعدة)"""
    async with ConnectionTester() as tester:
        return await tester.test_windows_api_connection(port, endpoint)


async def test_end_to_end(test_inference: bool = True) -> Dict[str, Any]:
    """اختبار التدفق الكامل (دالة مساعدة)"""
    async with ConnectionTester() as tester:
        return await tester.test_end_to_end(test_inference)


# Sync wrappers for convenience
def test_rtx4090_connection_sync(endpoint: str = "/") -> ConnectionResult:
    """اختبار الاتصال بـ RTX 4090 (متزامن)"""
    return asyncio.run(test_rtx4090_connection(endpoint))


def test_windows_api_connection_sync(
    port: int = 8000, 
    endpoint: str = "/api/v1/system/status"
) -> ConnectionResult:
    """اختبار API محلي (متزامن)"""
    return asyncio.run(test_windows_api_connection(port, endpoint))


def test_end_to_end_sync(test_inference: bool = True) -> Dict[str, Any]:
    """اختبار التدفق الكامل (متزامن)"""
    return asyncio.run(test_end_to_end(test_inference))


if __name__ == "__main__":
    # Test when run directly
    result = test_end_to_end_sync(test_inference=False)
    print("\n" + "=" * 60)
    print("Connection Test Results:")
    print("=" * 60)
    print(f"Overall Success: {result['overall_success']}")
    print(f"Tests Passed: {result['summary']['passed']}/{result['summary']['total_tests']}")
    print(f"Average Latency: {result['summary']['average_latency_ms']}ms")
    print("\nDetailed Results:")
    for test_name, test_result in result['tests'].items():
        status = "✅" if test_result['success'] else "❌"
        print(f"  {status} {test_name}: {test_result['latency_ms']}ms")
        if test_result['error']:
            print(f"      Error: {test_result['error']}")
