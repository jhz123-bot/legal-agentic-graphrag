from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import request
from urllib.error import HTTPError, URLError

import numpy as np

from src.embedding.base_provider import BaseEmbeddingProvider
from src.embedding.local_provider import LocalEmbeddingProvider


class BailianEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
        self.timeout = timeout
        self.local_fallback = LocalEmbeddingProvider(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Bailian embedding HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Bailian embedding connection error: {exc.reason}") from exc

    def embed_text(self, text: str) -> np.ndarray:
        arr = self.embed_texts([text])
        return arr[0] if arr.size else self.local_fallback.embed_text(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        payload = {"model": self.model, "input": texts}
        try:
            data = self._post_json("/embeddings", payload)
            vectors: List[np.ndarray] = []
            for item in data.get("data", []):
                embedding = item.get("embedding", [])
                vec = np.asarray(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
            if len(vectors) == len(texts) and vectors:
                return np.vstack(vectors)
        except Exception:
            pass
        return self.local_fallback.embed_texts(texts)

