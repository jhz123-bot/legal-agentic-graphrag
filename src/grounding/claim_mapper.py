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


def _split_claims(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[。！？!?；;]|以及|并且|且|同时", text)
    claims = [p.strip() for p in parts if p and p.strip()]
    return claims[:4]


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


def map_claims_to_evidence(
    query: str,
    intermediate_conclusion: str,
    reasoning_steps: List[str],
    evidence_list: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            _EMBED_MODEL = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        except Exception:
            _EMBED_MODEL = None

    claims: List[str] = []
    if intermediate_conclusion:
        claims.extend(_split_claims(intermediate_conclusion.strip()))
    for step in reasoning_steps[:3]:
        s = (step or "").strip()
        if s:
            for part in _split_claims(s):
                if part and part not in claims:
                    claims.append(part)
    if not claims and query:
        claims.append(query)

    mapped_claims: List[Dict[str, Any]] = []
    unsupported: List[str] = []
    evidence_ids_used: List[str] = []

    for claim in claims:
        scored: List[tuple[float, Dict[str, Any]]] = []
        for ev in evidence_list:
            text = str(ev.get("text") or ev.get("evidence") or "")
            lexical = _overlap(claim, text)
            semantic = _embedding_similarity(claim, text, _EMBED_MODEL)
            score = 0.55 * lexical + 0.45 * semantic
            scored.append((score, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        support = [ev for s, ev in scored[:top_k] if s > 0.03]
        support_ids = [str(ev.get("evidence_id") or "") for ev in support if ev.get("evidence_id")]
        support_level = "supported" if support_ids else "unsupported"
        if support_ids and len(support_ids) < 2:
            support_level = "weakly_supported"
        if support_level == "unsupported":
            unsupported.append(claim)

        evidence_ids_used.extend(support_ids)
        mapped_claims.append(
            {
                "claim": claim,
                "supporting_evidence_ids": support_ids,
                "support_level": support_level,
            }
        )

    # Deduplicate while preserving order.
    seen = set()
    dedup_ids: List[str] = []
    for eid in evidence_ids_used:
        if eid and eid not in seen:
            seen.add(eid)
            dedup_ids.append(eid)

    return {
        "claims": mapped_claims,
        "supporting_evidence_ids": dedup_ids,
        "unsupported_claims": unsupported,
    }
