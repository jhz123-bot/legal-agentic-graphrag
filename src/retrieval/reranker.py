import os
import re
from typing import Dict, List

from src.cache.cache_manager import get_cache_manager
from src.config.settings import settings
from sentence_transformers import CrossEncoder


def _lexical_score(query: str, evidence: str) -> float:
    q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    e_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", evidence.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens.intersection(e_tokens)) / len(q_tokens)


class EvidenceReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.cache_manager = get_cache_manager()
        self.last_cache_hit = False
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            if os.getenv("ALLOW_REMOTE_MODELS", "0") == "1":
                try:
                    self.model = CrossEncoder(model_name)
                except Exception:
                    self.model = None

    def rerank(self, query: str, evidence_paths: List[Dict], top_k: int | None = None) -> List[Dict]:
        if not evidence_paths:
            return []
        cache_enabled = settings.enable_cache and settings.enable_rerank_cache
        self.last_cache_hit = False
        if cache_enabled:
            cached = self.cache_manager.rerank_cache.get(query=query, evidence_paths=evidence_paths)
            if cached is not None:
                self.last_cache_hit = True
                if top_k is not None:
                    return cached[:top_k]
                return cached

        reranked = []
        if self.model is not None:
            pairs = [(query, p.get("evidence", "")) for p in evidence_paths]
            scores = self.model.predict(pairs).tolist()
            for path, score in zip(evidence_paths, scores):
                reranked.append({**path, "rerank_score": float(score)})
        else:
            for path in evidence_paths:
                score = _lexical_score(query, path.get("evidence", ""))
                reranked.append({**path, "rerank_score": float(score)})

        reranked.sort(key=lambda x: (x.get("rerank_score", 0.0), x.get("score", 0.0)), reverse=True)
        if cache_enabled:
            self.cache_manager.rerank_cache.set(query=query, evidence_paths=evidence_paths, ranked_results=reranked)
        if top_k is not None:
            return reranked[:top_k]
        return reranked


def make_reranker_node(reranker: EvidenceReranker):
    def reranker_node(state: Dict) -> Dict:
        ranked = state.get("ranked_evidence", [])
        query_used = state.get("rewritten_query") or state["user_query"]
        # Deep rerank only the top-N coarse-ranked evidence for efficiency.
        rerank_input_top_k = settings.rerank_input_top_k
        rerank_top_k = settings.rerank_top_k
        rerank_pool = ranked[: max(1, rerank_input_top_k)]
        reranked = reranker.rerank(query_used, rerank_pool, top_k=max(1, rerank_top_k))
        evidence_pack = dict(state.get("evidence_pack", {}))
        evidence_pack["reranked_paths"] = reranked
        evidence_pack["ranked_paths"] = ranked
        logs = list(state.get("logs", []))
        top_score = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
        if settings.enable_cache and settings.enable_rerank_cache:
            if reranker.last_cache_hit:
                logs.append("rerank_cache_hit")
            else:
                logs.append("rerank_cache_miss")
        logs.append(
            f"reranker: reranked={len(reranked)}, top_rerank_score={round(float(top_score), 4)}, "
            f"rerank_input_top_k={rerank_input_top_k}, rerank_top_k={rerank_top_k}, query_used={query_used}"
        )
        return {"ranked_evidence": reranked, "evidence_pack": evidence_pack, "logs": logs}

    return reranker_node
