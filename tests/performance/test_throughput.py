"""
Throughput Performance Test
اختبار الإنتاجية (Throughput)
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

pytestmark = pytest.mark.performance


class TestThroughput:
    """اختبارات الإنتاجية"""
    
    @pytest.fixture
    def base_url(self):
        """URL الأساسي"""
        from core.config import settings
        return f"http://localhost:{settings.PORT}"
    
    @pytest.mark.asyncio
    async def test_api_throughput_baseline(self, base_url):
        """اختبار خط أساس الإنتاجية للـ API"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        duration = 10  # seconds
        
        print(f"\nAPI Throughput Baseline ({duration}s):")
        
        request_count = 0
        success_count = 0
        error_count = 0
        
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            while time.perf_counter() - start_time < duration:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        await response.text()
                        if response.status < 400:
                            success_count += 1
                        else:
                            error_count += 1
                except (aiohttp.ClientConnectorError, OSError):
                    if request_count == 0:
                        pytest.skip("Server not running at " + base_url)
                    error_count += 1
                except Exception:
                    error_count += 1
                request_count += 1
        
        actual_duration = time.perf_counter() - start_time
        throughput = success_count / actual_duration
        
        print(f"  Total Requests: {request_count}")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        
        assert throughput > 1, f"Throughput {throughput:.2f} req/s too low"
    
    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, base_url):
        """اختبار الإنتاجية المتزامنة"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        concurrent = 20
        total_requests = 200
        
        print(f"\nConcurrent Throughput ({concurrent} concurrent, {total_requests} total):")
        
        results = {"success": 0, "error": 0}
        
        async def make_request(session):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                    await response.text()
                    if response.status < 400:
                        results["success"] += 1
                    else:
                        results["error"] += 1
            except Exception:
                results["error"] += 1
        
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_request():
                async with semaphore:
                    return await make_request(session)
            
            await asyncio.gather(*[bounded_request() for _ in range(total_requests)])
        
        duration = time.perf_counter() - start_time
        throughput = results["success"] / duration
        
        print(f"  Successful: {results['success']}")
        print(f"  Errors: {results['error']}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
    
    @pytest.mark.asyncio
    async def test_sustained_throughput(self, base_url):
        """اختبار الإنتاجية المستدامة"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        duration = 30  # 30 seconds
        interval = 0.1  # 100ms between requests
        
        print(f"\nSustained Throughput ({duration}s with {interval}s interval):")
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            while time.perf_counter() - start_time < duration:
                req_start = time.perf_counter()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                        await response.text()
                        latency = (time.perf_counter() - req_start) * 1000
                        results.append({"success": True, "latency": latency})
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
                
                await asyncio.sleep(interval)
        
        actual_duration = time.perf_counter() - start_time
        successful = sum(1 for r in results if r.get("success"))
        throughput = successful / actual_duration
        
        print(f"  Total Requests: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
    
    def test_multiprocess_throughput(self, base_url):
        """اختبار الإنتاجية مع multiprocessing"""
        import requests as req_lib

        url = f"{base_url}/api/v1/system/status"

        # Pre-check: server must be running
        try:
            req_lib.get(url, timeout=2)
        except Exception:
            pytest.skip("Server not running at " + base_url)

        num_processes = 4
        requests_per_process = 25

        print(f"\nMultiprocess Throughput ({num_processes} processes, {requests_per_process} req each):")

        # Use ThreadPoolExecutor instead of mp.Pool to avoid pickle errors
        from concurrent.futures import ThreadPoolExecutor

        def worker(process_id):
            results = {"success": 0, "error": 0}
            for _ in range(requests_per_process):
                try:
                    response = req_lib.get(url, timeout=3)
                    if response.status_code < 400:
                        results["success"] += 1
                    else:
                        results["error"] += 1
                except Exception:
                    results["error"] += 1
            return results

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            worker_results = list(executor.map(worker, range(num_processes)))

        duration = time.perf_counter() - start_time
        total_success = sum(r["success"] for r in worker_results)
        total_error = sum(r["error"] for r in worker_results)
        throughput = total_success / duration

        print(f"  Successful: {total_success}")
        print(f"  Errors: {total_error}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")


class TestRTX4090Throughput:
    """اختبارات إنتاجية RTX 4090"""
    
    @pytest.fixture
    def rtx4090_url(self):
        """URL RTX 4090"""
        from core.config import settings
        return settings.rtx4090_url
    
    @pytest.mark.asyncio
    async def test_rtx4090_status_throughput(self, rtx4090_url):
        """اختبار إنتاجية RTX 4090 status"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{rtx4090_url}/status"
        duration = 10
        
        print(f"\nRTX 4090 Status Throughput ({duration}s):")
        
        success_count = 0
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            while time.perf_counter() - start_time < duration:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        await response.text()
                        if response.status < 400:
                            success_count += 1
                except Exception:
                    pass
        
        actual_duration = time.perf_counter() - start_time
        throughput = success_count / actual_duration
        
        print(f"  Successful: {success_count}")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")


class TestLoadPatterns:
    """أنماط الحمل المختلفة"""

    @pytest.fixture
    def base_url(self):
        from core.config import settings
        return f"http://localhost:{settings.PORT}"

    @pytest.mark.asyncio
    async def test_spike_load(self, base_url):
        """اختبار حمل مفاجئ (spike)"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{base_url}/api/v1/system/status"
        
        print("\nSpike Load Test:")
        
        # Baseline
        print("  Baseline (10 req/s for 5s)...")
        baseline_success = 0
        start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            while time.perf_counter() - start < 5:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        await response.text()
                        baseline_success += 1
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        
        print(f"    Success: {baseline_success}")
        
        # Spike
        print("  Spike (100 concurrent)...")
        spike_success = 0
        async with aiohttp.ClientSession() as session:
            async def req():
                nonlocal spike_success
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        await response.text()
                        spike_success += 1
                except Exception:
                    pass
            
            await asyncio.gather(*[req() for _ in range(100)])
        
        print(f"    Success: {spike_success}/100")
        
        # Recovery
        print("  Recovery (10 req/s for 5s)...")
        recovery_success = 0
        start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            while time.perf_counter() - start < 5:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        await response.text()
                        recovery_success += 1
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        
        print(f"    Success: {recovery_success}")
        print(f"  Recovery Rate: {recovery_success/max(baseline_success, 1)*100:.1f}%")


class TestResourceUtilization:
    """اختبارات استخدام الموارد"""

    @pytest.fixture
    def base_url(self):
        from core.config import settings
        return f"http://localhost:{settings.PORT}"

    def test_memory_usage_under_load(self, base_url):
        """اختبار استخدام الذاكرة تحت الحمل"""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")
        import requests
        
        print("\nMemory Usage Under Load:")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  Initial: {initial_memory:.2f} MB")
        
        url = f"{base_url}/api/v1/system/status"
        for _ in range(100):
            try:
                requests.get(url, timeout=2)
            except Exception:
                pass
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        print(f"  Peak: {peak_memory:.2f} MB")
        print(f"  Increase: {peak_memory - initial_memory:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
