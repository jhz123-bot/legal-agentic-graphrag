import hashlib
import os
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.text_chunker import TextChunk


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.fallback_dim = 384
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            if os.getenv("ALLOW_REMOTE_MODELS", "0") == "1":
                try:
                    self.model = SentenceTransformer(model_name)
                except Exception:
                    self.model = None
            else:
                self.model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self.model is not None:
            emb = self.model.encode([text], normalize_embeddings=True)
            return np.asarray(emb[0], dtype=np.float32)
        return self._hash_embed(text)

    def embed_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        texts = [c.text for c in chunks]
        if self.model is not None:
            emb = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        return np.vstack([self._hash_embed(t) for t in texts]).astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.fallback_dim, dtype=np.float32)
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.fallback_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
