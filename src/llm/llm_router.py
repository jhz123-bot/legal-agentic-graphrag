from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.cache.cache_manager import get_cache_manager
from src.config.settings import settings
from src.llm.bailian_provider import BailianProvider
from src.llm.base_provider import BaseLLMProvider
from src.llm.ollama_provider import OllamaProvider


class CachedLLMProvider(BaseLLMProvider):
    def __init__(self, backend: BaseLLMProvider) -> None:
        self.backend = backend
        self.cache_manager = get_cache_manager()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        cache_enabled = settings.enable_cache and settings.enable_llm_cache
        cache_key = json.dumps(
            {
                "mode": "chat",
                "messages": messages,
                "model": model,
                "temperature": temperature,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_enabled:
            cached = self.cache_manager.llm_cache.get(cache_key)
            if cached is not None:
                return cached
        out = self.backend.chat(messages=messages, model=model, temperature=temperature, **kwargs)
        if cache_enabled:
            self.cache_manager.llm_cache.set(cache_key, out)
        return out

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        cache_enabled = settings.enable_cache and settings.enable_llm_cache
        cache_key = json.dumps(
            {
                "mode": "generate",
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_enabled:
            cached = self.cache_manager.llm_cache.get(cache_key)
            if cached is not None:
                return cached
        out = self.backend.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        if cache_enabled:
            self.cache_manager.llm_cache.set(cache_key, out)
        return out


def get_llm_provider(provider: Optional[str] = None, timeout: int = 60) -> BaseLLMProvider:
    selected = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    backend: BaseLLMProvider
    if selected in {"bailian", "dashscope"}:
        backend = BailianProvider(timeout=timeout)
    else:
        backend = OllamaProvider(timeout=timeout)
    if settings.enable_cache and settings.enable_llm_cache:
        return CachedLLMProvider(backend)
    return backend
