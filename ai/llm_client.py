"""
LLM Client - Unified interface for OpenAI, Anthropic, and local models
"""
import os
from typing import Optional, Dict, List, AsyncGenerator
from dataclasses import dataclass
import aiohttp

@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Dict
    source: str  # openai, anthropic, local


class OpenAIClient:
    """OpenAI GPT-4/GPT-3.5 client"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
    
    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=model,
                    usage=data.get("usage", {}),
                    source="openai"
                )


class AnthropicClient:
    """Anthropic Claude client"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1"
    
    async def generate(
        self,
        prompt: str,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                return LLMResponse(
                    text=data["content"][0]["text"],
                    model=model,
                    usage=data.get("usage", {}),
                    source="anthropic"
                )


class LLMManager:
    """Manager for multiple LLM providers with fallback"""
    
    def __init__(self):
        self.openai = OpenAIClient() if os.getenv("OPENAI_API_KEY") else None
        self.anthropic = AnthropicClient() if os.getenv("ANTHROPIC_API_KEY") else None
        self.primary = os.getenv("LLM_PRIMARY", "openai")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate with automatic fallback"""
        
        # Try primary
        if self.primary == "openai" and self.openai:
            try:
                return await self.openai.generate(prompt, system_prompt=system_prompt, **kwargs)
            except Exception as e:
                print(f"OpenAI failed: {e}, trying fallback...")
        
        # Try Anthropic
        if self.anthropic:
            try:
                return await self.anthropic.generate(prompt, system_prompt=system_prompt, **kwargs)
            except Exception as e:
                print(f"Anthropic failed: {e}")
        
        # Try OpenAI as fallback
        if self.openai and self.primary != "openai":
            return await self.openai.generate(prompt, system_prompt=system_prompt, **kwargs)
        
        raise Exception("All LLM providers failed")
    
    async def generate_for_wise_man(
        self,
        wise_man_name: str,
        message: str,
        context: List[Dict] = None
    ) -> str:
        """Generate response for specific wise man"""
        
        # Build system prompt based on wise man
        system_prompts = {
            "حكيم القرار": "You are a decisive strategic advisor. Give clear, actionable advice.",
            "حكيم البصيرة": "You are an analytical advisor who sees patterns in data.",
            "حكيم المستقبل": "You are a visionary who thinks long-term.",
        }
        
        system_prompt = system_prompts.get(wise_man_name, "You are a wise advisor.")
        
        # Build context
        context_str = ""
        if context:
            context_str = "Previous conversation:\n" + "\n".join([
                f"{msg['role']}: {msg['message']}" for msg in context[-5:]
            ]) + "\n\n"
        
        prompt = f"{context_str}User: {message}\n\nRespond as {wise_man_name} would:"
        
        response = await self.generate(prompt, system_prompt)
        return response.text


# Global instance
llm_manager = LLMManager()
