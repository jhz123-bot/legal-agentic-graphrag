import math
import re
from typing import Dict, List


RELATION_PRIOR = {
    "APPLIES_TO": 1.0,
    "REFERENCES_STATUTE": 0.9,
    "INVOLVES_PARTY": 0.7,
    "MENTIONED_WITH": 0.6,
}


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return {t for t in cleaned.split() if t}


def _semantic_similarity(query: str, evidence_text: str) -> float:
    q = _tokenize(query)
    e = _tokenize(evidence_text)
    if not q or not e:
        return 0.0
    overlap = len(q.intersection(e))
    return overlap / max(1, len(q))


def _graph_distance_score(path: Dict) -> float:
    # Default distance 1 for direct edge evidence.
    distance = float(path.get("distance", 1))
    return 1.0 / (1.0 + max(distance - 1.0, 0.0))


def _relation_weight(path: Dict) -> float:
    relation = path.get("relation", "MENTIONED_WITH")
    return RELATION_PRIOR.get(relation, 0.5)


def rank_evidence(
    query: str,
    candidate_paths: List[Dict],
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.25,
) -> List[Dict]:
    ranked: List[Dict] = []
    for path in candidate_paths:
        evidence_text = path.get("evidence", "")
        semantic = _semantic_similarity(query, evidence_text)
        distance = _graph_distance_score(path)
        rel_w = _relation_weight(path)
        score = alpha * semantic + beta * distance + gamma * rel_w
        ranked.append(
            {
                **path,
                "score": round(score, 4),
                "score_factors": {
                    "semantic_similarity": round(semantic, 4),
                    "graph_distance_score": round(distance, 4),
                    "relation_weight": round(rel_w, 4),
                },
            }
        )

    ranked.sort(key=lambda p: p["score"], reverse=True)
    return ranked[: max(1, top_k)]
