from __future__ import annotations

import re
from typing import Any, Dict, List

from src.embedding.embedding_model import EmbeddingModel

_EMBED_MODEL: EmbeddingModel | None = None


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()) if len(t) >= 1}


def _overlap(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _embedding_similarity(a: str, b: str, model: EmbeddingModel | None) -> float:
    if model is None or not a or not b:
        return 0.0
    try:
        import numpy as np

        va = model.embed_text(a)
        vb = model.embed_text(b)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na <= 0 or nb <= 0:
            return 0.0
        cos = float(np.dot(va, vb) / (na * nb))
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))
    except Exception:
        return 0.0


def check_grounding(answer: Dict[str, Any], evidence_list: List[Dict[str, Any]], reasoning_trace: Dict[str, Any]) -> Dict[str, Any]:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            _EMBED_MODEL = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        except Exception:
            _EMBED_MODEL = None
    claims = reasoning_trace.get("claims") or []
    if claims and isinstance(claims[0], dict):
        claim_texts = [str(c.get("claim", "")) for c in claims]
        claim_support = {str(c.get("claim", "")): c.get("supporting_evidence_ids", []) for c in claims}
    else:
        claim_texts = [str(c) for c in claims]
        claim_support = {str(c): [] for c in claim_texts}

    if not claim_texts:
        short_answer = str(answer.get("short_answer", ""))
        claim_texts = [short_answer] if short_answer else []

    ev_by_id = {str(ev.get("evidence_id", "")): ev for ev in evidence_list if ev.get("evidence_id")}

    unsupported_claims: List[str] = []
    unsupported_spans: List[Dict[str, Any]] = []
    grounded_scores: List[float] = []

    for claim in claim_texts:
        ids = claim_support.get(claim, [])
        if ids:
            snippets = [str(ev_by_id.get(i, {}).get("text") or ev_by_id.get(i, {}).get("evidence") or "") for i in ids]
        else:
            snippets = [str(ev.get("text") or ev.get("evidence") or "") for ev in evidence_list[:3]]
        score = 0.0
        for s in snippets:
            lexical = _overlap(claim, s)
            semantic = _embedding_similarity(claim, s, _EMBED_MODEL)
            merged = 0.5 * lexical + 0.5 * semantic
            if merged > score:
                score = merged
        grounded_scores.append(score)
        if score < 0.08:
            unsupported_claims.append(claim)
            claim_tokens = _tokens(claim)
            evidence_blob = " ".join(snippets)
            missing_tokens = [t for t in claim_tokens if t and t not in _tokens(evidence_blob)]
            unsupported_spans.append(
                {
                    "claim": claim,
                    "missing_tokens": missing_tokens[:8],
                }
            )

    grounding_score = sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0
    grounded = grounding_score >= 0.12 and len(unsupported_claims) == 0

    return {
        "grounded": grounded,
        "grounding_score": round(grounding_score, 4),
        "unsupported_claims": unsupported_claims,
        "unsupported_spans": unsupported_spans,
        "unsupported_claim_count": len(unsupported_claims),
        "missing_evidence": len(evidence_list) == 0,
    }
