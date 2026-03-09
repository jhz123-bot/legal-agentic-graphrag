from __future__ import annotations

from typing import Optional

from src.cache.common import CacheStats, LRUCacheStore, stable_hash


class LLMCache:
    def __init__(self, max_size: int = 1024) -> None:
        self.store = LRUCacheStore(max_size=max_size)
        self.stats = CacheStats()

    def _key(self, prompt: str) -> str:
        return stable_hash({"prompt": prompt or ""})

    def get(self, prompt: str) -> Optional[str]:
        key = self._key(prompt)
        value = self.store.get(key)
        if value is None:
            self.stats.miss += 1
            return None
        self.stats.hits += 1
        return str(value)

    def set(self, prompt: str, response: str) -> None:
        key = self._key(prompt)
        self.store.set(key, response)
