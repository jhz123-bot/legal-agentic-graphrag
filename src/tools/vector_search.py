from typing import Any, Dict

from src.embedding.embedding_model import EmbeddingModel
from src.vector_store.vector_index import FaissVectorIndex


class VectorSearchTool:
    name = "vector_search"

    def __init__(self, embedding_model: EmbeddingModel, vector_index: FaissVectorIndex) -> None:
        self.embedding_model = embedding_model
        self.vector_index = vector_index

    def run(self, query: str) -> Dict[str, Any]:
        query_vec = self.embedding_model.embed_text(query)
        hits = self.vector_index.search(query_vec, top_k=5)
        return {
            "tool": self.name,
            "hits": hits,
            "summary": f"vector hits={len(hits)}",
        }
