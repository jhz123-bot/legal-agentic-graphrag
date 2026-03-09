from __future__ import annotations

import os
from typing import Optional

from src.embedding.bailian_provider import BailianEmbeddingProvider
from src.embedding.base_provider import BaseEmbeddingProvider
from src.embedding.local_provider import LocalEmbeddingProvider


def get_embedding_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: int = 60,
) -> BaseEmbeddingProvider:
    selected = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()
    if selected in {"bailian", "dashscope"}:
        return BailianEmbeddingProvider(model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"), timeout=timeout)
    return LocalEmbeddingProvider(model_name=model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
