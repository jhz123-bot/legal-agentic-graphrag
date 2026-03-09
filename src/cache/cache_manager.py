from __future__ import annotations

from typing import Dict

from src.cache.embedding_cache import EmbeddingCache
from src.cache.llm_cache import LLMCache
from src.cache.rerank_cache import RerankCache
from src.cache.retrieval_cache import RetrievalCache
from src.config.settings import settings


class CacheManager:
    def __init__(self) -> None:
        self.embedding_cache = EmbeddingCache(max_size=settings.embedding_cache_size)
        self.retrieval_cache = RetrievalCache(max_size=settings.retrieval_cache_size)
        self.rerank_cache = RerankCache(max_size=settings.rerank_cache_size)
        self.llm_cache = LLMCache(max_size=settings.llm_cache_size)

    def get_stats(self) -> Dict[str, int]:
        return {
            "embedding_hits": self.embedding_cache.stats.hits,
            "embedding_miss": self.embedding_cache.stats.miss,
            "retrieval_hits": self.retrieval_cache.stats.hits,
            "retrieval_miss": self.retrieval_cache.stats.miss,
            "rerank_hits": self.rerank_cache.stats.hits,
            "rerank_miss": self.rerank_cache.stats.miss,
            "llm_hits": self.llm_cache.stats.hits,
            "llm_miss": self.llm_cache.stats.miss,
        }


_GLOBAL_CACHE_MANAGER: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _GLOBAL_CACHE_MANAGER
    if _GLOBAL_CACHE_MANAGER is None:
        _GLOBAL_CACHE_MANAGER = CacheManager()
    return _GLOBAL_CACHE_MANAGER
