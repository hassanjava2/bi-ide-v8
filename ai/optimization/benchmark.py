"""
Benchmark - قياس الأداء
=====================
Performance benchmarking for models and API
"""
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from contextlib import contextmanager
import json

import torch
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float
    memory_mb: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    Performance benchmarking utility
    
    Features:
    - Latency measurement (mean, p50, p95, p99)
    - Throughput calculation
    - Memory tracking
    - GPU utilization
    - Result storage and comparison
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        fn: Callable,
        name: str,
        num_runs: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a function
        
        Args:
            fn: Function to benchmark
            name: Benchmark name
            num_runs: Number of runs
            warmup: Warmup runs
            *args, **kwargs: Arguments to pass to fn
            
        Returns:
            BenchmarkResult
        """
        print(f"Benchmarking '{name}' ({num_runs} runs)...")
        
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        memory_before = self._get_memory_usage()
        
        for _ in range(num_runs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        memory_after = self._get_memory_usage()
        
        # Calculate statistics
        times_sorted = sorted(times)
        mean_time = statistics.mean(times)
        
        benchmark_result = BenchmarkResult(
            name=name,
            mean_ms=mean_time,
            min_ms=min(times),
            max_ms=max(times),
            p50_ms=times_sorted[len(times) // 2],
            p95_ms=times_sorted[int(len(times) * 0.95)],
            p99_ms=times_sorted[int(len(times) * 0.99)],
            throughput=1000 / mean_time if mean_time > 0 else 0,
            memory_mb=memory_after - memory_before
        )
        
        self.results.append(benchmark_result)
        self._print_result(benchmark_result)
        
        return benchmark_result
    
    async def run_async(
        self,
        fn: Callable,
        name: str,
        num_runs: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark an async function"""
        print(f"Benchmarking async '{name}' ({num_runs} runs)...")
        
        # Warmup
        for _ in range(warmup):
            await fn(*args, **kwargs)
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            await fn(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        times_sorted = sorted(times)
        mean_time = statistics.mean(times)
        
        benchmark_result = BenchmarkResult(
            name=name,
            mean_ms=mean_time,
            min_ms=min(times),
            max_ms=max(times),
            p50_ms=times_sorted[len(times) // 2],
            p95_ms=times_sorted[int(len(times) * 0.95)],
            p99_ms=times_sorted[int(len(times) * 0.99)],
            throughput=1000 / mean_time if mean_time > 0 else 0,
            memory_mb=0
        )
        
        self.results.append(benchmark_result)
        self._print_result(benchmark_result)
        
        return benchmark_result
    
    def compare(
        self,
        name1: str,
        name2: str
    ) -> Dict[str, Any]:
        """Compare two benchmark results"""
        result1 = next((r for r in self.results if r.name == name1), None)
        result2 = next((r for r in self.results if r.name == name2), None)
        
        if not result1 or not result2:
            raise ValueError(f"Results not found: {name1}, {name2}")
        
        speedup = result1.mean_ms / result2.mean_ms if result2.mean_ms > 0 else 1
        
        return {
            "baseline": name1,
            "optimized": name2,
            "speedup": speedup,
            "speedup_pct": (speedup - 1) * 100,
            "latency_reduction_ms": result1.mean_ms - result2.mean_ms,
            "latency_reduction_pct": ((result1.mean_ms - result2.mean_ms) / result1.mean_ms * 100)
            if result1.mean_ms > 0 else 0
        }
    
    def save_results(self, path: str):
        """Save benchmark results to JSON"""
        data = {
            "benchmarks": [
                {
                    "name": r.name,
                    "mean_ms": r.mean_ms,
                    "p50_ms": r.p50_ms,
                    "p95_ms": r.p95_ms,
                    "throughput": r.throughput,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0
        process = psutil.Process()  # type: ignore[union-attr]
        return process.memory_info().rss / (1024 * 1024)
    
    def _get_gpu_usage(self) -> Optional[Dict]:
        """Get GPU utilization"""
        if not GPUTIL_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "temperature": gpu.temperature
                }
        except Exception:
            pass
        return None
    
    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result"""
        print(f"  {result.name}:")
        print(f"    Mean: {result.mean_ms:.2f}ms")
        print(f"    P50: {result.p50_ms:.2f}ms")
        print(f"    P95: {result.p95_ms:.2f}ms")
        print(f"    P99: {result.p99_ms:.2f}ms")
        print(f"    Throughput: {result.throughput:.1f} items/sec")
        print(f"    Memory: {result.memory_mb:.1f} MB")


@contextmanager
def benchmark_context(name: str):
    """Context manager for quick benchmarking"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"{name}: {elapsed_ms:.2f}ms")


# Quick benchmark decorator
def benchmark_fn(name: str, num_runs: int = 100):
    """Decorator to benchmark a function"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            bench = Benchmark()
            return bench.run(fn, name, num_runs, 10, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test
    bench = Benchmark()
    
    def test_function(n: int = 1000000):
        """Test function: sum of squares"""
        return sum(i * i for i in range(n))
    
    # Run benchmark
    result = bench.run(test_function, "sum_of_squares", 50, 5)
    
    print("\n" + "=" * 50)
    print("Benchmark Complete")
    print("=" * 50)
