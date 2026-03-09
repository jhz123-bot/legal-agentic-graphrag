import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

import numpy as np

from src.config.settings import settings
from src.embedding.embedding_model import EmbeddingModel


RELATION_PRIOR = {
    "APPLIES_TO": 1.0,
    "REFERENCES_STATUTE": 0.9,
    "CITES": 0.9,
    "INVOLVES_PARTY": 0.75,
    "MENTIONED_WITH": 0.6,
    "VECTOR_MATCH": 0.45,
}


@dataclass
class RankWeights:
    w_semantic: float = settings.w_semantic
    w_vector: float = settings.w_vector
    w_bm25: float = settings.w_bm25
    w_graph: float = settings.w_graph


class UnifiedEvidenceRanker:
    """Coarse ranker for unified graph/vector evidence objects."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, weights: Optional[RankWeights] = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModel(model_name="all-MiniLM-L6-v2")
        self.weights = weights or RankWeights()

    def _tokenize(self, text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", (text or "").lower())
        return {t for t in cleaned.split() if t}

    def _lexical_similarity(self, query: str, evidence_text: str) -> float:
        q = self._tokenize(query)
        e = self._tokenize(evidence_text)
        if not q or not e:
            return 0.0
        overlap = len(q.intersection(e))
        return overlap / max(1, len(q))

    def _embedding_similarity(self, query: str, evidence_text: str) -> float:
        if not query or not evidence_text:
            return 0.0
        # Main path: embedding cosine similarity.
        try:
            q_vec = self.embedding_model.embed_text(query)
            e_vec = self.embedding_model.embed_text(evidence_text)
            q_norm = float(np.linalg.norm(q_vec))
            e_norm = float(np.linalg.norm(e_vec))
            if q_norm <= 0.0 or e_norm <= 0.0:
                return self._lexical_similarity(query, evidence_text)
            cos = float(np.dot(q_vec, e_vec) / (q_norm * e_norm))
            return max(0.0, min(1.0, (cos + 1.0) / 2.0))
        except Exception:
            return self._lexical_similarity(query, evidence_text)

    def _graph_score(self, evidence: Dict[str, Any]) -> float:
        if evidence.get("evidence_type") != "graph":
            return 0.0
        distance = float(evidence.get("distance", 1.0))
        return 1.0 / (1.0 + max(distance - 1.0, 0.0))

    def _relation_weight(self, evidence: Dict[str, Any]) -> float:
        relation = str(evidence.get("relation", "MENTIONED_WITH"))
        return RELATION_PRIOR.get(relation, 0.5)

    def _vector_score(self, evidence: Dict[str, Any]) -> float:
        raw = float(evidence.get("vector_score", 0.0) or 0.0)
        if evidence.get("evidence_type") not in {"vector", "hybrid"}:
            return 0.0
        # Normalize inner-product-like score into [0, 1].
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def _bm25_score(self, evidence: Dict[str, Any]) -> float:
        raw = float(evidence.get("bm25_score", 0.0) or 0.0)
        if raw <= 0:
            return 0.0
        # Smooth BM25 raw score to [0, 1).
        return max(0.0, min(1.0, raw / (raw + 8.0)))

    def _score_one(self, query: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        evidence_text = str(evidence.get("text") or evidence.get("evidence") or "")
        semantic_score = self._embedding_similarity(query, evidence_text)
        graph_score = self._graph_score(evidence)
        vector_score = self._vector_score(evidence)
        bm25_score = self._bm25_score(evidence)

        score = (
            self.weights.w_semantic * semantic_score
            + self.weights.w_vector * vector_score
            + self.weights.w_bm25 * bm25_score
            + self.weights.w_graph * graph_score
        )

        return {
            **evidence,
            "semantic_score": round(semantic_score, 4),
            "graph_score": round(graph_score, 4),
            "vector_score": round(vector_score, 4),
            "bm25_score": round(bm25_score, 4),
            "score": round(score, 4),
            "score_factors": {
                "semantic_score": round(semantic_score, 4),
                "graph_score": round(graph_score, 4),
                "vector_score": round(vector_score, 4),
                "bm25_score": round(bm25_score, 4),
                "relation_weight": round(self._relation_weight(evidence), 4),
                "weights": {
                    "w_semantic": self.weights.w_semantic,
                    "w_vector": self.weights.w_vector,
                    "w_bm25": self.weights.w_bm25,
                    "w_graph": self.weights.w_graph,
                },
            },
        }

    def rank(self, query: str, candidate_evidence: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not candidate_evidence:
            return []
        ranked = [self._score_one(query, ev) for ev in candidate_evidence]
        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[: max(1, top_k)]


def rank_evidence(query: str, candidate_paths: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    # Backward-compatible function wrapper.
    ranker = UnifiedEvidenceRanker()
    return ranker.rank(query=query, candidate_evidence=candidate_paths, top_k=top_k)


def make_evidence_ranking_node(ranker: UnifiedEvidenceRanker):
    def evidence_ranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
        candidate = state.get("candidate_evidence", [])
        query_used = state.get("rewritten_query") or state["user_query"]
        coarse_top_k = settings.coarse_rank_top_k
        ranked = ranker.rank(query_used, candidate, top_k=max(1, coarse_top_k))
        evidence_pack = dict(state.get("evidence_pack", {}))
        evidence_pack["ranked_paths"] = ranked

        logs = list(state.get("logs", []))
        top_score = ranked[0].get("score", 0.0) if ranked else 0.0
        logs.append(
            "evidence_ranking: "
            f"ranked={len(ranked)}, top_score={top_score}, coarse_top_k={coarse_top_k}, query_used={query_used}"
        )
        return {
            "ranked_evidence": ranked,
            "evidence_pack": evidence_pack,
            "logs": logs,
        }

    return evidence_ranking_node
