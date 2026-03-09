from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.cache.common import CacheStats, LRUCacheStore, stable_hash


class RerankCache:
    def __init__(self, max_size: int = 1024) -> None:
        self.store = LRUCacheStore(max_size=max_size)
        self.stats = CacheStats()

    def _evidence_ids(self, evidence_paths: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for p in evidence_paths:
            ids.append(
                stable_hash(
                    {
                        "source": p.get("source", ""),
                        "target": p.get("target", ""),
                        "relation": p.get("relation", ""),
                        "evidence": str(p.get("evidence", ""))[:200],
                    }
                )
            )
        return ids

    def _key(self, query: str, evidence_paths: List[Dict[str, Any]]) -> str:
        return stable_hash({"query": query or "", "evidence_ids": self._evidence_ids(evidence_paths)})

    def get(self, query: str, evidence_paths: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        key = self._key(query, evidence_paths)
        value = self.store.get(key)
        if value is None:
            self.stats.miss += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, query: str, evidence_paths: List[Dict[str, Any]], ranked_results: List[Dict[str, Any]]) -> None:
        key = self._key(query, evidence_paths)
        self.store.set(key, ranked_results)
