from __future__ import annotations

from typing import Dict, List


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    top_k: int = 50,
) -> List[Dict]:
    """Fuse vector + BM25 ranked lists with RRF.

    score = sum(1 / (k + rank_i))
    """
    score_map: Dict[str, Dict] = {}

    for rank, item in enumerate(vector_results, start=1):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        row = score_map.setdefault(
            cid,
            {
                "chunk_id": cid,
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "vector_score": float(item.get("score", 0.0)),
                "bm25_score": 0.0,
                "rrf_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": 0,
            },
        )
        row["vector_score"] = max(float(item.get("score", 0.0)), float(row.get("vector_score", 0.0)))
        row["vector_rank"] = min(row.get("vector_rank", rank), rank)
        row["rrf_score"] += 1.0 / (k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        cid = str(item.get("chunk_id", ""))
        if not cid:
            continue
        row = score_map.setdefault(
            cid,
            {
                "chunk_id": cid,
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "vector_score": 0.0,
                "bm25_score": float(item.get("score", 0.0)),
                "rrf_score": 0.0,
                "vector_rank": 0,
                "bm25_rank": rank,
            },
        )
        row["bm25_score"] = max(float(item.get("score", 0.0)), float(row.get("bm25_score", 0.0)))
        if row.get("bm25_rank", 0) <= 0:
            row["bm25_rank"] = rank
        else:
            row["bm25_rank"] = min(row.get("bm25_rank", rank), rank)
        row["rrf_score"] += 1.0 / (k + rank)

    fused = list(score_map.values())
    fused.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return fused[: max(1, top_k)]
