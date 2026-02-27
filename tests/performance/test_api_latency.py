"""
API Latency Performance Test
اختبار زمن استجابة API
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

pytestmark = pytest.mark.performance


class TestAPILatency:
    """اختبارات زمن استجابة API"""
    
    @pytest.fixture
    def base_url(self):
        """URL الأساسي للـ API"""
        from core.config import settings
        return f"http://localhost:{settings.PORT}"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_latency(self, base_url):
        """اختبار زمن استجابة نقطة نهاية Health"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        num_requests = 20
        
        print(f"\nHealth Endpoint Latency ({num_requests} requests):")
        
        latencies = []
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                except (aiohttp.ClientConnectorError, aiohttp.ClientError, OSError):
                    if i == 0:
                        pytest.skip("Server not running at " + base_url)
                    latencies.append(5000)  # 5s as timeout
                except Exception:
                    latencies.append(5000)  # 5s as timeout
                await asyncio.sleep(0.05)
        
        if latencies:
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
            print(f"  Avg: {statistics.mean(latencies):.2f}ms")
            print(f"  P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
            print(f"  P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
            
            avg_latency = statistics.mean(latencies)
            assert avg_latency < 1000, f"Average latency {avg_latency:.2f}ms exceeds 1s threshold"
    
    @pytest.mark.asyncio
    async def test_gateway_endpoint_latency(self, base_url):
        """اختبار زمن استجابة Gateway"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/gateway/status"
        
        print("\nGateway Endpoint Latency:")
        
        latencies = []
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                start = time.perf_counter()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        
        if latencies:
            print(f"  Avg: {statistics.mean(latencies):.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, base_url):
        """اختبار طلبات API متزامنة"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        num_requests = 50
        concurrent = 10
        
        print(f"\nConcurrent API Requests ({num_requests} requests, {concurrent} concurrent):")
        
        async def make_request(session):
            start = time.perf_counter()
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    await response.text()
                    return (time.perf_counter() - start) * 1000
            except Exception:
                return None
        
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_request():
                async with semaphore:
                    return await make_request(session)
            
            results = await asyncio.gather(*[bounded_request() for _ in range(num_requests)])
        
        total_time = time.perf_counter() - start_time
        successful = [r for r in results if r is not None]
        
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Successful: {len(successful)}/{num_requests}")
        
        if successful:
            print(f"  Throughput: {len(successful)/total_time:.2f} req/s")
            print(f"  Avg Latency: {statistics.mean(successful):.2f}ms")
            print(f"  P95: {sorted(successful)[int(len(successful)*0.95)]:.2f}ms")


class TestEndpointComparison:
    """مقارنة زمن استجابة النقاط المختلفة"""
    
    @pytest.mark.asyncio
    async def test_endpoints_comparison(self):
        """مقارنة زمن استجابة مختلف النقاط"""
        from core.config import settings
        import aiohttp
        
        base_url = f"http://localhost:{settings.PORT}"
        
        endpoints = [
            ("System Status", f"{base_url}/api/v1/system/status"),
            ("Health Check", f"{base_url}/api/v1/system/health"),
            ("Gateway Status", f"{base_url}/gateway/status"),
        ]
        
        print("\nEndpoint Comparison:")
        print(f"  {'Endpoint':<20} {'Avg Latency':<15} {'Min':<10} {'Max':<10}")
        print("  " + "-" * 60)
        
        async with aiohttp.ClientSession() as session:
            for name, url in endpoints:
                latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            await response.text()
                            latency = (time.perf_counter() - start) * 1000
                            latencies.append(latency)
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                
                if latencies:
                    print(f"  {name:<20} {statistics.mean(latencies):>10.2f}ms {min(latencies):>10.2f}ms {max(latencies):>10.2f}ms")
                else:
                    print(f"  {name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")


class TestLatencyDistribution:
    """اختبارات توزيع زمن الاستجابة"""
    
    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        """اختبار percentiles للـ latency"""
        from core.config import settings
        import aiohttp
        
        url = f"http://localhost:{settings.PORT}/api/v1/system/status"
        num_requests = 100
        
        print(f"\nLatency Percentiles ({num_requests} requests):")
        
        latencies = []
        async with aiohttp.ClientSession() as session:
            for _ in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                except Exception:
                    pass
        
        if latencies:
            sorted_latencies = sorted(latencies)
            percentiles = [50, 90, 95, 99]
            
            for p in percentiles:
                idx = int(len(sorted_latencies) * p / 100) - 1
                if 0 <= idx < len(sorted_latencies):
                    print(f"  P{p}: {sorted_latencies[idx]:.2f}ms")
            
            print(f"\n  Mean: {statistics.mean(latencies):.2f}ms")
            print(f"  StdDev: {statistics.stdev(latencies):.2f}ms" if len(latencies) > 1 else "  StdDev: N/A")


class TestWarmupAndCaching:
    """اختبارات Warmup والكاش"""
    
    @pytest.mark.asyncio
    async def test_cold_vs_warm_latency(self):
        """مقارنة latency البارد والدافئ"""
        from core.config import settings
        import aiohttp

        url = f"http://localhost:{settings.PORT}/api/v1/system/status"

        print("\nCold vs Warm Latency:")

        try:
            async with aiohttp.ClientSession() as session:
                # Cold request
                start = time.perf_counter()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.text()
                    cold_latency = (time.perf_counter() - start) * 1000
                print(f"  Cold: {cold_latency:.2f}ms")

                # Warm requests
                warm_latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        warm_latencies.append((time.perf_counter() - start) * 1000)
                    await asyncio.sleep(0.05)

                if warm_latencies:
                    print(f"  Warm (avg): {statistics.mean(warm_latencies):.2f}ms")
                    print(f"  Warm (min): {min(warm_latencies):.2f}ms")
        except (aiohttp.ClientConnectorError, aiohttp.ClientError, OSError):
            pytest.skip("Server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
