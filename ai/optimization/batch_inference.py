"""
Batch Inference Module
Efficient batch processing for inference with dynamic batching
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from queue import Queue, PriorityQueue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from pathlib import Path
import json


@dataclass
class InferenceRequest:
    """Single inference request."""
    id: str
    inputs: Any
    callback: Optional[Callable] = None
    priority: int = 5
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __lt__(self, other):
        # Higher priority (lower number) and earlier timestamp first
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class InferenceResult:
    """Inference result."""
    request_id: str
    outputs: Any
    processing_time: float
    batch_size: int = 1


class BatchProcessor:
    """
    Dynamic batching processor for efficient inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01,
        device: str = 'cuda',
        dynamic_padding: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            model: Model for inference
            max_batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch to fill (seconds)
            device: Device for inference
            dynamic_padding: Whether to use dynamic padding
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        self.dynamic_padding = dynamic_padding
        
        # Request queue (priority queue)
        self.request_queue = PriorityQueue()
        
        # Results storage
        self.results: Dict[str, InferenceResult] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_tokens': 0,
            'avg_batch_size': 0,
            'avg_latency': 0
        }
        
        # Processing thread
        self._stop_event = threading.Event()
        self._processing_thread = None
        self._lock = threading.Lock()
    
    def submit(
        self, 
        inputs: Any, 
        request_id: Optional[str] = None,
        priority: int = 5,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit inference request.
        
        Args:
            inputs: Model inputs
            request_id: Unique request ID (auto-generated if None)
            priority: Priority (1-10, lower is higher priority)
            callback: Optional callback function
            
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}_{np.random.randint(1000)}"
        
        request = InferenceRequest(
            id=request_id,
            inputs=inputs,
            callback=callback,
            priority=priority
        )
        
        self.request_queue.put(request)
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        return request_id
    
    def submit_batch(
        self,
        inputs_list: List[Any],
        priorities: Optional[List[int]] = None
    ) -> List[str]:
        """Submit multiple requests as a batch."""
        if priorities is None:
            priorities = [5] * len(inputs_list)
        
        request_ids = []
        for inputs, priority in zip(inputs_list, priorities):
            req_id = self.submit(inputs, priority=priority)
            request_ids.append(req_id)
        
        return request_ids
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        """
        Get result for a request.
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            InferenceResult or None if timeout
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self._lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
            
            time.sleep(0.001)
        
        return None
    
    def process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of results
        """
        if not requests:
            return []
        
        start_time = time.time()
        
        # Collate inputs
        batch_inputs = self._collate_inputs([r.inputs for r in requests])
        
        # Move to device
        if isinstance(batch_inputs, torch.Tensor):
            batch_inputs = batch_inputs.to(self.device)
        elif isinstance(batch_inputs, dict):
            batch_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_inputs.items()
            }
        elif isinstance(batch_inputs, list):
            batch_inputs = [
                v.to(self.device) if isinstance(v, torch.Tensor) else v
                for v in batch_inputs
            ]
        
        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                outputs = self.model(batch_inputs)
        
        # Split outputs
        individual_outputs = self._split_outputs(outputs, len(requests))
        
        # Create results
        processing_time = time.time() - start_time
        results = []
        
        for request, output in zip(requests, individual_outputs):
            result = InferenceResult(
                request_id=request.id,
                outputs=output,
                processing_time=processing_time,
                batch_size=len(requests)
            )
            results.append(result)
            
            # Store result
            with self._lock:
                self.results[request.id] = result
            
            # Call callback if provided
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    print(f"Callback error for {request.id}: {e}")
        
        # Update stats
        with self._lock:
            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = (
                (self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + len(requests))
                / self.stats['total_batches']
            )
            self.stats['avg_latency'] = (
                (self.stats['avg_latency'] * (self.stats['total_batches'] - 1) + processing_time)
                / self.stats['total_batches']
            )
        
        return results
    
    def _collate_inputs(self, inputs_list: List[Any]) -> Any:
        """Collate individual inputs into batch."""
        if not inputs_list:
            return None
        
        first_input = inputs_list[0]
        
        if isinstance(first_input, torch.Tensor):
            # Pad tensors if needed
            if self.dynamic_padding:
                return self._pad_and_stack(inputs_list)
            else:
                return torch.stack(inputs_list)
        
        elif isinstance(first_input, dict):
            # Collate dict inputs
            result = {}
            for key in first_input.keys():
                values = [inp[key] for inp in inputs_list]
                if isinstance(values[0], torch.Tensor):
                    result[key] = self._pad_and_stack(values) if self.dynamic_padding else torch.stack(values)
                else:
                    result[key] = values
            return result
        
        elif isinstance(first_input, (list, tuple)):
            return inputs_list
        
        else:
            return inputs_list
    
    def _pad_and_stack(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad and stack tensors of different lengths."""
        max_len = max(t.shape[0] for t in tensors)
        
        padded = []
        for tensor in tensors:
            if tensor.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - tensor.shape[0],
                    *tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device
                )
                tensor = torch.cat([tensor, padding], dim=0)
            padded.append(tensor)
        
        return torch.stack(padded)
    
    def _split_outputs(self, outputs: Any, num_items: int) -> List[Any]:
        """Split batched outputs into individual results."""
        if isinstance(outputs, torch.Tensor):
            return [outputs[i] for i in range(num_items)]
        
        elif isinstance(outputs, dict):
            return [
                {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                for i in range(num_items)
            ]
        
        elif isinstance(outputs, (list, tuple)):
            return outputs[:num_items]
        
        else:
            return [outputs] * num_items
    
    def start(self):
        """Start the batch processing loop."""
        if self._processing_thread is not None:
            return
        
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        print("Batch processor started")
    
    def stop(self):
        """Stop the batch processing loop."""
        self._stop_event.set()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None
        
        print("Batch processor stopped")
    
    def _processing_loop(self):
        """Main processing loop."""
        while not self._stop_event.is_set():
            requests = []
            start_wait = time.time()
            
            # Collect requests until batch is full or timeout
            while len(requests) < self.max_batch_size:
                try:
                    # Wait for request with timeout
                    timeout = max(0, self.max_wait_time - (time.time() - start_wait))
                    
                    if timeout <= 0 and requests:
                        break
                    
                    request = self.request_queue.get(timeout=timeout if requests else None)
                    requests.append(request)
                    
                except:
                    if requests:
                        break
            
            # Process batch if we have requests
            if requests:
                self.process_batch(requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self.stats = {
                'total_requests': 0,
                'total_batches': 0,
                'total_tokens': 0,
                'avg_batch_size': 0,
                'avg_latency': 0
            }


class InferenceServer:
    """
    High-level inference server with batch processing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01,
        device: str = 'cuda',
        num_workers: int = 1
    ):
        self.processor = BatchProcessor(
            model=model,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            device=device
        )
        self.num_workers = num_workers
        
    def start(self):
        """Start the inference server."""
        self.processor.start()
    
    def stop(self):
        """Stop the inference server."""
        self.processor.stop()
    
    def predict(
        self,
        inputs: Any,
        timeout: float = 30.0
    ) -> Any:
        """
        Synchronous prediction.
        
        Args:
            inputs: Model inputs
            timeout: Timeout in seconds
            
        Returns:
            Model outputs
        """
        request_id = self.processor.submit(inputs)
        result = self.processor.get_result(request_id, timeout=timeout)
        
        if result is None:
            raise TimeoutError("Inference timeout")
        
        return result.outputs
    
    def predict_batch(
        self,
        inputs_list: List[Any],
        timeout: float = 60.0
    ) -> List[Any]:
        """
        Batch prediction.
        
        Args:
            inputs_list: List of inputs
            timeout: Timeout per request
            
        Returns:
            List of outputs
        """
        request_ids = self.processor.submit_batch(inputs_list)
        
        outputs = []
        for req_id in request_ids:
            result = self.processor.get_result(req_id, timeout=timeout)
            if result is None:
                raise TimeoutError(f"Inference timeout for request {req_id}")
            outputs.append(result.outputs)
        
        return outputs


class DynamicBatcher:
    """
    Dynamic batching with bucketing for variable-length sequences.
    """
    
    def __init__(
        self,
        model: nn.Module,
        bucket_boundaries: List[int] = None,
        max_batch_size: int = 32,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        
        if bucket_boundaries is None:
            bucket_boundaries = [32, 64, 128, 256, 512, 1024]
        
        self.bucket_boundaries = bucket_boundaries
        self.max_batch_size = max_batch_size
        self.device = device
        
        # Buckets
        self.buckets: Dict[int, List] = {b: [] for b in bucket_boundaries}
        self.buckets['overflow'] = []
    
    def get_bucket(self, length: int) -> int:
        """Get appropriate bucket for sequence length."""
        for boundary in self.bucket_boundaries:
            if length <= boundary:
                return boundary
        return 'overflow'
    
    def add_request(self, inputs: Any, length: int) -> Optional[List[Any]]:
        """
        Add request to bucket.
        
        Returns:
            Full batch if ready, None otherwise
        """
        bucket = self.get_bucket(length)
        self.buckets[bucket].append(inputs)
        
        if len(self.buckets[bucket]) >= self.max_batch_size:
            batch = self.buckets[bucket][:self.max_batch_size]
            self.buckets[bucket] = self.buckets[bucket][self.max_batch_size:]
            return batch
        
        return None
    
    def process_bucket(self, bucket: int) -> Optional[Any]:
        """Process all requests in a bucket."""
        if not self.buckets[bucket]:
            return None
        
        batch = self.buckets[bucket]
        self.buckets[bucket] = []
        
        # Collate and process
        batch_inputs = self._collate(batch)
        
        with torch.no_grad():
            outputs = self.model(batch_inputs.to(self.device))
        
        return outputs
    
    def _collate(self, inputs: List[Any]) -> torch.Tensor:
        """Collate inputs with padding."""
        max_len = max(inp.shape[0] for inp in inputs)
        
        padded = []
        for inp in inputs:
            if inp.shape[0] < max_len:
                padding = torch.zeros(max_len - inp.shape[0], *inp.shape[1:])
                inp = torch.cat([inp, padding], dim=0)
            padded.append(inp)
        
        return torch.stack(padded)


if __name__ == '__main__':
    print("Batch Inference Module Demo")
    print("="*50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    
    # Create inference server
    server = InferenceServer(
        model=model,
        max_batch_size=8,
        device='cpu'
    )
    
    # Start server
    server.start()
    
    # Test inference
    test_inputs = torch.randn(5, 100)
    result = server.predict(test_inputs)
    print(f"Single inference result shape: {result.shape}")
    
    # Test batch inference
    batch_inputs = [torch.randn(100) for _ in range(20)]
    results = server.predict_batch(batch_inputs)
    print(f"Batch inference results: {len(results)} items")
    
    # Print stats
    stats = server.processor.get_stats()
    print(f"\nStatistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
    
    # Stop server
    server.stop()
