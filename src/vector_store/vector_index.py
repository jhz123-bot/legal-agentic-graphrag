from typing import Dict, List

import faiss
import numpy as np

from src.ingestion.text_chunker import TextChunk


class FaissVectorIndex:
    def __init__(self) -> None:
        self.index = None
        self.chunks: List[TextChunk] = []
        self.dimension = 0

    def build_index(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        self.dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)
        hits: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            hits.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_type": getattr(c, "doc_type", ""),
                    "source_type": getattr(c, "source_type", ""),
                    "law_name": getattr(c, "law_name", ""),
                    "article_no": getattr(c, "article_no", ""),
                    "case_id": getattr(c, "case_id", ""),
                    "section": getattr(c, "section", ""),
                    "title": c.title,
                    "text": c.text,
                    "score": float(score),
                }
            )
        return hits
