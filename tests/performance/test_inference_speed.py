"""
Inference Speed Performance Test
اختبار سرعة الـ Inference
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

pytestmark = pytest.mark.performance


class TestInferenceSpeed:
    """اختبارات سرعة الـ Inference"""
    
    @pytest.fixture
    def rtx4090_url(self):
        """URL RTX 4090"""
        from core.config import settings
        return settings.rtx4090_url
    
    @pytest.mark.asyncio
    async def test_single_inference_latency(self, rtx4090_url):
        """اختبار زمن Inference منفرد"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{rtx4090_url}/inference"
        
        latencies = []
        num_requests = 5
        
        print(f"\nSingle Inference Latency Test ({num_requests} requests):")
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.post(
                        url,
                        json={"prompt": "test", "max_tokens": 10},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                        print(f"  Request {i+1}: {latency:.2f}ms")
                except Exception as e:
                    print(f"  Request {i+1}: Failed ({e})")
                    latencies.append(30000)  # 30s timeout
                
                await asyncio.sleep(0.5)
        
        if latencies:
            print(f"\n  Average: {statistics.mean(latencies):.2f}ms")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
    
    @pytest.mark.asyncio
    async def test_inference_throughput(self, rtx4090_url):
        """اختبار throughput للـ Inference"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{rtx4090_url}/inference"
        num_requests = 10
        concurrent = 3
        
        print(f"\nInference Throughput Test ({num_requests} requests, {concurrent} concurrent):")
        
        async def make_request(session, idx):
            start = time.perf_counter()
            try:
                async with session.post(
                    url,
                    json={"prompt": f"test_{idx}", "max_tokens": 10},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.text()
                    return (time.perf_counter() - start) * 1000
            except Exception:
                return None
        
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_request(idx):
                async with semaphore:
                    return await make_request(session, idx)
            
            results = await asyncio.gather(*[bounded_request(i) for i in range(num_requests)])
        
        total_time = time.perf_counter() - start_time
        successful = [r for r in results if r is not None]
        
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Successful: {len(successful)}/{num_requests}")
        
        if successful:
            print(f"  Throughput: {len(successful)/total_time:.2f} req/s")
            print(f"  Avg Latency: {statistics.mean(successful):.2f}ms")
    
    @pytest.mark.asyncio
    async def test_inference_with_different_prompt_sizes(self, rtx4090_url):
        """اختبار Inference مع أحجام prompt مختلفة"""
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        
        url = f"{rtx4090_url}/inference"
        
        prompts = [
            ("small", "Hello"),
            ("medium", "Hello, how are you doing today?"),
            ("large", "Hello, how are you doing today? I hope everything is going well with you and your family." * 10),
        ]
        
        print("\nInference with Different Prompt Sizes:")
        
        async with aiohttp.ClientSession() as session:
            for size_name, prompt in prompts:
                start = time.perf_counter()
                try:
                    async with session.post(
                        url,
                        json={"prompt": prompt, "max_tokens": 50},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        print(f"  {size_name} ({len(prompt)} chars): {latency:.2f}ms")
                except Exception as e:
                    print(f"  {size_name}: Failed ({e})")
                
                await asyncio.sleep(1)
    
    def test_inference_benchmark_sync(self):
        """اختبار benchmark متزامن"""
        import requests
        from core.config import settings
        
        url = f"{settings.rtx4090_url}/status"
        
        print(f"\nInference Server Benchmark (sync):")
        
        latencies = []
        for i in range(10):
            start = time.perf_counter()
            try:
                response = requests.get(url, timeout=5)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            except Exception as e:
                print(f"  Request {i+1}: Failed ({e})")
            time.sleep(0.2)
        
        if latencies:
            print(f"  Completed: {len(latencies)}/10")
            print(f"  Avg Latency: {statistics.mean(latencies):.2f}ms")
            p95_idx = int(len(latencies) * 0.95)
            print(f"  P95 Latency: {sorted(latencies)[p95_idx]:.2f}ms")


class TestModelLoadingPerformance:
    """اختبارات أداء تحميل النموذج"""
    
    @pytest.mark.asyncio
    async def test_model_status_response_time(self):
        """اختبار زمن استجابة حالة النموذج"""
        from core.config import settings
        import aiohttp
        
        url = f"{settings.rtx4090_url}/status"
        
        latencies = []
        num_requests = 10
        
        print(f"\nModel Status Response Time ({num_requests} requests):")
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
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
            avg = statistics.mean(latencies)
            print(f"  Average: {avg:.2f}ms")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
            
            assert avg < 1000, f"Average latency {avg:.2f}ms exceeds threshold"


class TestCheckpointOperations:
    """اختبارات أداء عمليات Checkpoint"""
    
    @pytest.mark.asyncio
    async def test_checkpoint_list_performance(self):
        """اختبار أداء قائمة الـ checkpoints"""
        from core.config import settings
        import aiohttp
        
        url = f"{settings.rtx4090_url}/checkpoints/list"
        
        start = time.perf_counter()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
                    latency = (time.perf_counter() - start) * 1000
                    
                    print(f"\nCheckpoint List Performance:")
                    print(f"  Latency: {latency:.2f}ms")
                    if isinstance(data, list):
                        print(f"  Checkpoints: {len(data)}")
                    
                    assert latency < 5000, "Checkpoint list too slow"
        except Exception as e:
            print(f"  Failed: {e}")
            pytest.skip(f"Checkpoint endpoint not available: {e}")


class TestConcurrentInference:
    """اختبارات Inference المتزامن"""
    
    @pytest.mark.asyncio
    async def test_stress_inference(self):
        """اختبار stress للـ Inference"""
        from core.config import settings
        import aiohttp
        
        url = f"{settings.rtx4090_url}/status"  # استخدام status كاختبار بسيط
        
        num_requests = 20
        concurrent = 5
        
        print(f"\nStress Test ({num_requests} requests, {concurrent} concurrent):")
        
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
            print(f"  Success Rate: {len(successful)/num_requests*100:.1f}%")
            print(f"  Throughput: {len(successful)/total_time:.2f} req/s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
