from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agents.query_decomposer import decompose_query
from src.cache.cache_manager import get_cache_manager
from src.citation.citation_utils import attach_citation_metadata
from src.config.settings import settings
from src.graph.entity_linker import EntityLinker
from src.query.query_rewriter import rewrite_query as rewrite_retrieval_query
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.graph_retriever import GraphRetriever


def _to_graph_evidence(edge: Dict[str, Any]) -> Dict[str, Any]:
    text = edge.get("evidence", "")
    source = edge.get("source", "")
    target = edge.get("target", "")
    source_name = edge.get("source_name", source)
    target_name = edge.get("target_name", target)
    title = f"{source_name}->{target_name}"
    return {
        "evidence_type": "graph",
        "source": source,
        "target": target,
        "source_name": source_name,
        "target_name": target_name,
        "relation": edge.get("relation", "MENTIONED_WITH"),
        "source_type": "graph",
        "doc_id": str(edge.get("source_doc_id", "") or edge.get("target_doc_id", "")),
        "chunk_id": "",
        "title": title,
        "law_name": "",
        "article_no": "",
        "case_id": "",
        "court": "",
        "section": "",
        "text": text,
        "evidence": text,
        "distance": float(edge.get("distance", 1.0) or 1.0),
        "retrieval_score": 0.0,
        "vector_score": 0.0,
        "bm25_score": 0.0,
        "semantic_score": 0.0,
        "graph_score": 1.0,
        "final_score": 0.0,
        "score": 0.0,
        "score_factors": {},
    }


def _to_fused_evidence(hit: Dict[str, Any]) -> Dict[str, Any]:
    text = hit.get("text", "")
    score = float(hit.get("rrf_score", 0.0) or 0.0)
    return {
        "evidence_type": "vector",
        "source": f"chunk::{hit.get('chunk_id', '')}",
        "target": f"doc::{hit.get('doc_id', '')}",
        "relation": "HYBRID_MATCH",
        "source_type": str(hit.get("source_type") or hit.get("doc_type") or "vector"),
        "doc_id": str(hit.get("doc_id", "")),
        "chunk_id": str(hit.get("chunk_id", "")),
        "title": str(hit.get("title", "")),
        "law_name": str(hit.get("law_name", "")),
        "article_no": str(hit.get("article_no", "")),
        "case_id": str(hit.get("case_id", "")),
        "court": str(hit.get("court", "")),
        "section": str(hit.get("section", "")),
        "retrieval_score": score,
        "text": text,
        "evidence": text,
        "distance": 2.0,
        "vector_score": float(hit.get("vector_score", 0.0) or 0.0),
        "bm25_score": float(hit.get("bm25_score", 0.0) or 0.0),
        "semantic_score": 0.0,
        "graph_score": 0.0,
        "final_score": score,
        "score": score,
        "score_factors": {
            "rrf_score": score,
            "vector_rank": int(hit.get("vector_rank", 0) or 0),
            "bm25_rank": int(hit.get("bm25_rank", 0) or 0),
        },
    }


def _dedupe_candidate_evidence(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for c in candidates:
        key = (
            c.get("evidence_type"),
            c.get("source"),
            c.get("target"),
            c.get("relation"),
            (c.get("evidence") or "")[:120],
        )
        if key not in deduped:
            deduped[key] = c
            continue
        # keep higher fused score when duplicate appears
        if c.get("final_score", c.get("score", 0.0)) > deduped[key].get("final_score", deduped[key].get("score", 0.0)):
            deduped[key] = c
    return list(deduped.values())


def _ensure_subqueries(query: str, plan_subqueries: List[str] | None) -> List[str]:
    if plan_subqueries:
        out = [q for q in plan_subqueries if q]
        if out:
            return list(dict.fromkeys(out))

    decomposition = decompose_query(query)
    subs = [query]
    for sq in decomposition.get("subqueries", []):
        if sq and sq not in subs:
            subs.append(sq)
    return subs


def _extract_entities_from_evidence(candidates: List[Dict[str, Any]], max_entities: int = 24) -> List[str]:
    out: List[str] = []
    statute_pat = re.compile(r"(刑法第[一二三四五六七八九十百千万零两〇0-9]+条|民法典第[一二三四五六七八九十百千万零两〇0-9]+条)")
    concept_hints = [
        "盗窃罪",
        "抢劫罪",
        "诈骗罪",
        "职务侵占",
        "违约责任",
        "侵权责任",
        "租赁合同",
        "买卖合同",
        "可得利益",
        "补足出资",
        "勤勉义务",
        "举证责任",
        "不动产登记",
        "物权变动",
        "预期违约",
    ]

    for ev in candidates[:40]:
        if not isinstance(ev, dict):
            continue
        for key in ("law_name", "article_no", "case_id", "title", "source_name", "target_name", "section"):
            v = str(ev.get(key, "")).strip()
            if v and v not in out:
                out.append(v)
            if len(out) >= max_entities:
                return out

        text = str(ev.get("text") or ev.get("evidence") or "")
        for m in statute_pat.findall(text):
            if m not in out:
                out.append(m)
            if len(out) >= max_entities:
                return out
        for hint in concept_hints:
            if hint in text and hint not in out:
                out.append(hint)
            if len(out) >= max_entities:
                return out
    return out


def make_retrieval_node(graph_store, hybrid_retriever=None):
    retriever = GraphRetriever(graph_store)
    linker = EntityLinker()
    cache_manager = get_cache_manager()

    def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
        linker.link(graph_store)
        effective_query = state.get("rewritten_query") or state["user_query"]
        strategy = (
            state.get("retrieval_strategy")
            or state.get("plan", {}).get("retrieval_strategy")
            or state.get("router_decision", {}).get("retrieval_strategy")
            or "graph"
        )
        strategy_fallback = None
        if strategy in {"vector", "hybrid"} and hybrid_retriever is None:
            strategy_fallback = f"{strategy}->graph(no_hybrid_retriever)"
            strategy = "graph"
        subqueries = _ensure_subqueries(effective_query, state.get("plan", {}).get("subqueries"))

        retrieval_query = rewrite_retrieval_query(effective_query)
        retrieval_subqueries = [rewrite_retrieval_query(sq) for sq in subqueries]

        cache_enabled = settings.enable_cache and settings.enable_retrieval_cache

        if strategy == "direct_answer":
            logs = list(state.get("logs", []))
            logs.append("retrieval: strategy=direct_answer, skip retrieval")
            return {
                "linked_entities": [],
                "evidence_pack": {"query_mentions": [], "nodes": [], "edges": [], "vector_hits": [], "bm25_hits": [], "fused_hits": [], "ranked_paths": []},
                "candidate_evidence": [],
                "ranked_evidence": [],
                "retrieval_strategy": strategy,
                "logs": logs,
            }

        if cache_enabled:
            cached = cache_manager.retrieval_cache.get(
                query=retrieval_query,
                strategy=strategy,
                subqueries=retrieval_subqueries,
            )
            if cached is not None:
                logs = list(state.get("logs", []))
                logs.append(
                    "retrieval_cache_hit: "
                    f"strategy={strategy}, query={effective_query}, retrieval_query={retrieval_query}, subqueries={len(retrieval_subqueries)}"
                )
                return {
                    "linked_entities": cached.get("linked_entities", []),
                    "evidence_pack": cached.get("evidence_pack", {}),
                    "candidate_evidence": cached.get("candidate_evidence", []),
                    "ranked_evidence": [],
                    "retrieval_strategy": strategy,
                    "logs": logs,
                }

        merged_nodes: Dict[str, Dict[str, Any]] = {}
        merged_edges: List[Dict[str, Any]] = []
        merged_mentions: List[Dict[str, Any]] = []
        merged_vector_hits: List[Dict[str, Any]] = []
        merged_bm25_hits: List[Dict[str, Any]] = []

        bm25_retriever = getattr(hybrid_retriever, "bm25_retriever", None) if hybrid_retriever is not None else None

        for subq in retrieval_subqueries:
            partial_graph = {"query_mentions": [], "nodes": [], "edges": []}
            if strategy in {"graph", "hybrid"}:
                partial_graph = retriever.retrieve(subq, top_k_nodes=10, top_k_edges=20)

            partial_vector_hits: List[Dict[str, Any]] = []
            if hybrid_retriever is not None and strategy in {"vector", "hybrid"}:
                query_vec = hybrid_retriever.embedding_model.embed_text(subq)
                partial_vector_hits = hybrid_retriever.vector_index.search(query_vec, top_k=settings.retrieval_top_k_vector)

            partial_bm25_hits: List[Dict[str, Any]] = []
            if bm25_retriever is not None and strategy in {"vector", "hybrid"}:
                partial_bm25_hits = bm25_retriever.search(subq, top_k=settings.retrieval_top_k_bm25)

            merged_mentions.extend(partial_graph.get("query_mentions", []))
            for node in partial_graph.get("nodes", []):
                merged_nodes[node["node_id"]] = node
            merged_edges.extend(partial_graph.get("edges", []))
            merged_vector_hits.extend(partial_vector_hits)
            merged_bm25_hits.extend(partial_bm25_hits)

        fused_hits = reciprocal_rank_fusion(
            vector_results=merged_vector_hits,
            bm25_results=merged_bm25_hits,
            k=settings.fusion_rrf_k,
            top_k=settings.retrieval_top_k_fusion,
        )

        candidate_evidence: List[Dict[str, Any]] = []
        candidate_evidence.extend(_to_graph_evidence(edge) for edge in merged_edges)
        candidate_evidence.extend(_to_fused_evidence(hit) for hit in fused_hits)
        candidate_evidence = _dedupe_candidate_evidence(candidate_evidence)
        candidate_evidence = attach_citation_metadata(candidate_evidence)

        retrieval_result = {
            "query_mentions": merged_mentions,
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
            "vector_hits": merged_vector_hits,
            "bm25_hits": merged_bm25_hits,
            "fused_hits": fused_hits,
            "candidate_evidence": candidate_evidence,
            "ranked_paths": [],
        }
        linked_entities = [node["name"] for node in retrieval_result.get("nodes", [])]
        ev_entities = _extract_entities_from_evidence(candidate_evidence, max_entities=24)
        for ent in ev_entities:
            if ent not in linked_entities:
                linked_entities.append(ent)

        if cache_enabled:
            cache_manager.retrieval_cache.set(
                query=retrieval_query,
                strategy=strategy,
                subqueries=retrieval_subqueries,
                result={
                    "linked_entities": linked_entities,
                    "evidence_pack": retrieval_result,
                    "candidate_evidence": candidate_evidence,
                },
            )

        logs = list(state.get("logs", []))
        if strategy_fallback:
            logs.append(f"retrieval_strategy_fallback: {strategy_fallback}")
        if cache_enabled:
            logs.append(
                "retrieval_cache_miss: "
                f"strategy={strategy}, query={effective_query}, retrieval_query={retrieval_query}, subqueries={len(retrieval_subqueries)}"
            )
        logs.append(
            "retrieval: "
            f"strategy={strategy}, "
            f"query={effective_query}, "
            f"retrieval_query={retrieval_query}, "
            f"subqueries={retrieval_subqueries}, "
            f"graph_hits={len(merged_edges)}, vector_hits={len(merged_vector_hits)}, bm25_hits={len(merged_bm25_hits)}, fused_hits={len(fused_hits)}, "
            f"candidate_evidence={len(candidate_evidence)}"
        )
        if fused_hits:
            logs.append(
                "retrieval_fusion: top_docs="
                + str([{"chunk_id": h.get("chunk_id"), "rrf": round(float(h.get("rrf_score", 0.0)), 4)} for h in fused_hits[:5]])
            )

        return {
            "linked_entities": linked_entities,
            "evidence_pack": retrieval_result,
            "candidate_evidence": candidate_evidence,
            "ranked_evidence": [],
            "retrieval_strategy": strategy,
            "logs": logs,
        }

    return retrieval_node
