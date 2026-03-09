from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.cache.common import CacheStats, LRUCacheStore, stable_hash


class RetrievalCache:
    def __init__(self, max_size: int = 512) -> None:
        self.store = LRUCacheStore(max_size=max_size)
        self.stats = CacheStats()

    def _key(self, query: str, strategy: str, subqueries: Optional[List[str]] = None) -> str:
        return stable_hash(
            {
                "query": query or "",
                "strategy": strategy or "graph",
                "subqueries": subqueries or [],
            }
        )

    def get(self, query: str, strategy: str, subqueries: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        key = self._key(query, strategy, subqueries=subqueries)
        value = self.store.get(key)
        if value is None:
            self.stats.miss += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, query: str, strategy: str, result: Dict[str, Any], subqueries: Optional[List[str]] = None) -> None:
        key = self._key(query, strategy, subqueries=subqueries)
        self.store.set(key, result)
