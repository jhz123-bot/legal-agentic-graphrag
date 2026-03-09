from typing import List

import numpy as np

from src.cache.cache_manager import get_cache_manager
from src.config.settings import settings
from src.embedding.embedding_router import get_embedding_provider
from src.ingestion.text_chunker import TextChunk


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", provider: str | None = None) -> None:
        self.model_name = model_name
        self.provider_name = provider
        self.provider = get_embedding_provider(provider=provider, model_name=model_name, timeout=60)
        self.cache_manager = get_cache_manager()

    def embed_text(self, text: str) -> np.ndarray:
        if settings.enable_cache and settings.enable_embedding_cache:
            cached = self.cache_manager.embedding_cache.get(text)
            if cached is not None:
                return np.asarray(cached, dtype=np.float32)
        vector = np.asarray(self.provider.embed_text(text), dtype=np.float32)
        if settings.enable_cache and settings.enable_embedding_cache:
            self.cache_manager.embedding_cache.set(text, vector)
        return vector

    def embed_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        texts = [c.text for c in chunks]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if not (settings.enable_cache and settings.enable_embedding_cache):
            return np.asarray(self.provider.embed_texts(texts), dtype=np.float32)

        cache = self.cache_manager.embedding_cache
        vectors: List[np.ndarray | None] = [None] * len(texts)
        missing_idx: List[int] = []
        missing_texts: List[str] = []
        for i, t in enumerate(texts):
            cached = cache.get(t)
            if cached is None:
                missing_idx.append(i)
                missing_texts.append(t)
            else:
                vectors[i] = np.asarray(cached, dtype=np.float32)

        if missing_texts:
            fresh = np.asarray(self.provider.embed_texts(missing_texts), dtype=np.float32)
            for idx, vec in zip(missing_idx, fresh):
                vectors[idx] = vec
                cache.set(texts[idx], vec)

        if any(v is None for v in vectors):
            raise RuntimeError("embedding cache fill failed: some vectors are missing")
        return np.asarray(vectors, dtype=np.float32)
