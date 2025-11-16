"""Unified LLM connector for TransformerLab, external APIs, and local models."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMResponse:
    """Response from an LLM inference call.
    
    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        source: The source of the response (api, transformerlab, ollama).
        metadata: Additional metadata (tokens, latency, etc.).
    """
    content: str
    model: str
    source: str
    metadata: Dict[str, Any]


class LLMConnector:
    """Unified connector for various LLM inference endpoints.
    
    Supports:
    - TransformerLab local models
    - External APIs (OpenAI, Anthropic, Together, xAI)
    - Ollama local endpoints
    """
    
    def __init__(
        self,
        connection_type: str = "transformerlab_local",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        ollama_endpoint: str = "http://localhost:11434",
        transformerlab_endpoint: str = "http://localhost:8000",
        timeout: int = 60,
    ):
        """Initialize the LLM connector.
        
        Args:
            connection_type: Type of connection (api, transformerlab_local, ollama).
            model: Model identifier.
            api_key: API key for external services.
            api_endpoint: Custom API endpoint URL.
            ollama_endpoint: Ollama server endpoint.
            transformerlab_endpoint: TransformerLab server endpoint.
            timeout: Request timeout in seconds.
        """
        self.connection_type = connection_type.lower()
        self.model = model or "default"
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.ollama_endpoint = ollama_endpoint
        self.transformerlab_endpoint = transformerlab_endpoint
        self.timeout = timeout
        
        # Auto-detect API endpoints based on model name
        if not self.api_endpoint and self.connection_type == "api":
            self.api_endpoint = self._detect_api_endpoint()
    
    def _detect_api_endpoint(self) -> str:
        """Auto-detect API endpoint based on model name."""
        model_lower = self.model.lower()
        
        if "gpt" in model_lower or "openai" in model_lower:
            return "https://api.openai.com/v1/chat/completions"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "https://api.anthropic.com/v1/messages"
        elif "grok" in model_lower or "xai" in model_lower:
            return "https://api.x.ai/v1/chat/completions"
        elif "together" in model_lower:
            return "https://api.together.xyz/v1/chat/completions"
        else:
            # Default to OpenAI-compatible endpoint
            return "https://api.openai.com/v1/chat/completions"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the configured LLM.
        
        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            LLMResponse containing the generated content.
        """
        if self.connection_type == "api":
            return self._generate_api(prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif self.connection_type == "ollama":
            return self._generate_ollama(prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif self.connection_type == "transformerlab_local":
            return self._generate_transformerlab(prompt, system_prompt, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")
    
    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using external API (OpenAI-compatible format)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Handle Anthropic API format (different from OpenAI)
        if "anthropic" in self.api_endpoint:
            return self._generate_anthropic(messages, temperature, max_tokens, **kwargs)
        
        # OpenAI-compatible format
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            metadata = {
                "usage": data.get("usage", {}),
                "model": data.get("model", self.model)
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                source="api",
                metadata=metadata
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                source="api",
                metadata={"error": str(e)}
            )
    
    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic API (different format)."""
        # Extract system prompt if present
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("content", [{}])[0].get("text", "")
            metadata = {
                "usage": data.get("usage", {}),
                "model": data.get("model", self.model)
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                source="api",
                metadata=metadata
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                source="api",
                metadata={"error": str(e)}
            )
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama local endpoint."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("response", "")
            metadata = {
                "model": data.get("model", self.model),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0)
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                source="ollama",
                metadata=metadata
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                source="ollama",
                metadata={"error": str(e)}
            )
    
    def _generate_transformerlab(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using TransformerLab local inference."""
        # TransformerLab API format (adjust based on actual API)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.transformerlab_endpoint}/api/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            metadata = {
                "model": data.get("model", self.model),
                "usage": data.get("usage", {})
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                source="transformerlab",
                metadata=metadata
            )
        except Exception as e:
            # Fallback: try to use transformerlab SDK if available
            try:
                from transformerlab.sdk.v1.inference import generate
                result = generate(
                    model=self.model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return LLMResponse(
                    content=result,
                    model=self.model,
                    source="transformerlab",
                    metadata={}
                )
            except Exception as sdk_error:
                return LLMResponse(
                    content=f"Error: {str(e)} | SDK Error: {str(sdk_error)}",
                    model=self.model,
                    source="transformerlab",
                    metadata={"error": str(e), "sdk_error": str(sdk_error)}
                )


def create_connector_from_config(config: Dict[str, Any]) -> LLMConnector:
    """Create an LLM connector from configuration dictionary.
    
    Args:
        config: Configuration dictionary with connection details.
        
    Returns:
        Configured LLMConnector instance.
    """
    connection_type = config.get("connection_type", "transformerlab_local")
    model = config.get("model", config.get("model_hint", "default"))
    
    # Get API key from environment variable if specified
    api_key_env = config.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else config.get("api_key")
    
    return LLMConnector(
        connection_type=connection_type,
        model=model,
        api_key=api_key,
        api_endpoint=config.get("api_endpoint"),
        ollama_endpoint=config.get("ollama_endpoint", "http://localhost:11434"),
        transformerlab_endpoint=config.get("transformerlab_endpoint", "http://localhost:8000"),
        timeout=config.get("timeout", 60)
    )
