from __future__ import annotations

from typing import Optional

import numpy as np

from src.cache.common import CacheStats, LRUCacheStore, stable_hash


class EmbeddingCache:
    def __init__(self, max_size: int = 4096) -> None:
        self.store = LRUCacheStore(max_size=max_size)
        self.stats = CacheStats()

    def _key(self, text: str) -> str:
        return stable_hash({"text": text or ""})

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        value = self.store.get(key)
        if value is None:
            self.stats.miss += 1
            return None
        self.stats.hits += 1
        return np.asarray(value, dtype=np.float32)

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self.store.set(key, np.asarray(embedding, dtype=np.float32))
