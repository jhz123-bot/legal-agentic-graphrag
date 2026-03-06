from typing import Dict, List

from src.embedding.embedding_model import EmbeddingModel
from src.retrieval.evidence_ranker import rank_evidence
from src.retrieval.graph_retriever import GraphRetriever
from src.vector_store.vector_index import FaissVectorIndex


class HybridRetriever:
    def __init__(self, vector_index: FaissVectorIndex, embedding_model: EmbeddingModel, graph_retriever: GraphRetriever) -> None:
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str, top_k_vector: int = 5, top_k_nodes: int = 6, top_k_edges: int = 10, top_k_ranked: int = 8) -> Dict:
        query_vec = self.embedding_model.embed_text(query)
        vector_hits = self.vector_index.search(query_vec, top_k=top_k_vector)
        graph_result = self.graph_retriever.retrieve(query, top_k_nodes=top_k_nodes, top_k_edges=top_k_edges)

        candidate_paths: List[Dict] = []
        for edge in graph_result.get("edges", []):
            candidate_paths.append({**edge, "distance": 1})
        for hit in vector_hits:
            candidate_paths.append(
                {
                    "source": f"chunk::{hit['chunk_id']}",
                    "target": f"doc::{hit['doc_id']}",
                    "relation": "VECTOR_MATCH",
                    "evidence": hit["text"],
                    "distance": 2,
                    "vector_score": hit["score"],
                }
            )

        ranked = rank_evidence(query=query, candidate_paths=candidate_paths, top_k=top_k_ranked)
        return {
            "query_mentions": graph_result.get("query_mentions", []),
            "nodes": graph_result.get("nodes", []),
            "edges": graph_result.get("edges", []),
            "vector_hits": vector_hits,
            "ranked_paths": ranked,
        }
