"""
RTX 4090 Client - اتصال حقيقي بـ RTX 4090 Inference Server
"""
import aiohttp
import asyncio
from typing import Dict, Optional, AsyncGenerator, List, Any
import json
import os
from pathlib import Path


class RTX4090Client:
    """Client for RTX 4090 inference server"""
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        port: int = 8080,
        timeout: int = 300
    ):
        self.host = host or os.getenv("RTX4090_HOST", "192.168.68.125")
        self.port = port or int(os.getenv("RTX4090_PORT", "8080"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if RTX 4090 server is healthy"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.get(f"{self.base_url}/") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "unhealthy", "code": resp.status}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed status from RTX 4090 server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.get(f"{self.base_url}/status") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "error", "code": resp.status}
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text using RTX 4090"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "stream": stream,
            "stop_sequences": stop_sequences or []
        }
        
        async with self.session.post(
            f"{self.base_url}/generate",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"RTX 4090 error {resp.status}: {error_text}")
            
            result = await resp.json()
            return result.get("text", result.get("generated_text", ""))
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream generation results token by token"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        async with self.session.post(
            f"{self.base_url}/generate",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"RTX 4090 streaming error {resp.status}: {error_text}")
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        token = data.get("token", data.get("text", ""))
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Chat completion with message history"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            f"{self.base_url}/chat",
            json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"RTX 4090 chat error {resp.status}: {error_text}")
            
            result = await resp.json()
            return result.get("response", result.get("text", ""))
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.get(f"{self.base_url}/") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get model info: {resp.status}"}
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints on RTX 4090 server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.get(f"{self.base_url}/checkpoints/list") as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get("checkpoints", [])
            return []
    
    async def start_learning(self) -> Dict[str, Any]:
        """Start real API learning on RTX 4090"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.post(f"{self.base_url}/start") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "error", "code": resp.status}
    
    async def stop_learning(self) -> Dict[str, Any]:
        """Stop real API learning on RTX 4090"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.post(f"{self.base_url}/stop") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "error", "code": resp.status}
    
    async def sync_checkpoints_to_windows(self) -> Dict[str, Any]:
        """Sync all checkpoints from RTX 4090 to Windows"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context.")
        
        async with self.session.post(f"{self.base_url}/checkpoints/sync-to-windows") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "error", "code": resp.status}


class RTX4090Pool:
    """Pool of RTX 4090 clients for load balancing and failover"""
    
    def __init__(self, hosts: Optional[List[str]] = None):
        """
        Initialize pool of RTX 4090 servers
        
        Args:
            hosts: List of host IPs. If None, uses RTX4090_HOSTS env var or defaults
        """
        if hosts is None:
            hosts_env = os.getenv("RTX4090_HOSTS", "192.168.68.125")
            hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
        
        self.hosts = hosts
        self.clients = [RTX4090Client(host=h) for h in hosts]
        self.current = 0
        self.health_status: Dict[str, bool] = {h: True for h in hosts}
    
    def get_client(self) -> RTX4090Client:
        """Get next client using round-robin with health check"""
        attempts = 0
        while attempts < len(self.clients):
            client = self.clients[self.current]
            host = self.hosts[self.current]
            self.current = (self.current + 1) % len(self.clients)
            
            if self.health_status.get(host, True):
                return client
            
            attempts += 1
        
        # If all unhealthy, return first anyway (failover)
        return self.clients[0]
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """Check health of all servers in pool"""
        results = {}
        for host, client in zip(self.hosts, self.clients):
            try:
                async with client:
                    health = await client.health_check()
                    results[host] = {"healthy": True, "info": health}
                    self.health_status[host] = True
            except Exception as e:
                results[host] = {"healthy": False, "error": str(e)}
                self.health_status[host] = False
        
        return results
    
    def get_healthy_count(self) -> int:
        """Get number of healthy servers"""
        return sum(1 for h in self.hosts if self.health_status.get(h, True))


class RTX4090Inference:
    """High-level inference interface for RTX 4090"""
    
    def __init__(self, client: Optional[RTX4090Client] = None):
        self.client = client or RTX4090Client()
    
    async def generate_with_fallback(
        self,
        prompt: str,
        fallback_fn=None,
        **kwargs
    ) -> str:
        """Generate with fallback to local function if RTX 4090 fails"""
        try:
            async with self.client:
                return await self.client.generate_text(prompt, **kwargs)
        except Exception as e:
            if fallback_fn:
                return await fallback_fn(prompt, **kwargs)
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        concurrency: int = 3
    ) -> List[str]:
        """Generate multiple prompts with concurrency control"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def generate_one(prompt: str) -> str:
            async with semaphore:
                async with self.client:
                    return await self.client.generate_text(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
        
        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)


# Global client instance for easy access
_default_client: Optional[RTX4090Client] = None
_default_pool: Optional[RTX4090Pool] = None


def get_rtx4090_client() -> RTX4090Client:
    """Get or create default RTX 4090 client"""
    global _default_client
    if _default_client is None:
        _default_client = RTX4090Client()
    return _default_client


def get_rtx4090_pool() -> RTX4090Pool:
    """Get or create default RTX 4090 pool"""
    global _default_pool
    if _default_pool is None:
        _default_pool = RTX4090Pool()
    return _default_pool


# Test function
async def test_rtx4090_connection():
    """Test connection to RTX 4090 server"""
    print("🧪 Testing RTX 4090 connection...")
    
    async with RTX4090Client() as client:
        # Health check
        health = await client.health_check()
        print(f"✅ Health Check: {health.get('name', 'Unknown')}")
        print(f"   Version: {health.get('version', 'N/A')}")
        print(f"   Device: {health.get('gpu', 'N/A')}")
        
        # Status
        status = await client.get_status()
        print(f"   Status: {status.get('status', 'unknown')}")
        
        # List checkpoints
        checkpoints = await client.list_checkpoints()
        print(f"   Checkpoints: {len(checkpoints)}")
        
        # Test generation
        print("\n📝 Testing text generation...")
        response = await client.generate_text(
            prompt="ما هي أهمية الذكاء الاصطناعي في المستقبل؟",
            max_tokens=256,
            temperature=0.7
        )
        print(f"Response: {response[:200]}...")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_rtx4090_connection())
