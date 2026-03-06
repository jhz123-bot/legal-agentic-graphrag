from typing import Any, Dict

from src.graph.entity_linker import EntityLinker
from src.retrieval.evidence_ranker import rank_evidence
from src.retrieval.graph_retriever import GraphRetriever


def make_retrieval_node(graph_store, hybrid_retriever=None):
    retriever = GraphRetriever(graph_store)
    linker = EntityLinker()

    def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Reuse Round 1 linker and retriever directly.
        linker.link(graph_store)
        queries = state.get("plan", {}).get("subqueries") or [state["user_query"]]
        route = state.get("router_decision", {}).get("route", "graph_reasoning")
        merged_nodes = {}
        merged_edges = []
        merged_mentions = []
        merged_vector_hits = []

        for subq in queries:
            if hybrid_retriever is not None and route in {"hybrid_retrieval", "vector_retrieval"}:
                top_k_nodes = 2 if route == "vector_retrieval" else 6
                top_k_edges = 3 if route == "vector_retrieval" else 10
                partial = hybrid_retriever.retrieve(
                    subq,
                    top_k_vector=6,
                    top_k_nodes=top_k_nodes,
                    top_k_edges=top_k_edges,
                    top_k_ranked=6,
                )
                merged_vector_hits.extend(partial.get("vector_hits", []))
            else:
                partial = retriever.retrieve(subq, top_k_nodes=6, top_k_edges=10)
            merged_mentions.extend(partial.get("query_mentions", []))
            for node in partial.get("nodes", []):
                merged_nodes[node["node_id"]] = node
            merged_edges.extend(partial.get("edges", []))

        retrieval_result = {
            "query_mentions": merged_mentions,
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
            "vector_hits": merged_vector_hits,
        }

        if hybrid_retriever is not None:
            candidate_paths = list(retrieval_result.get("edges", []))
            for hit in retrieval_result.get("vector_hits", []):
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
            ranked_paths = rank_evidence(query=state["user_query"], candidate_paths=candidate_paths, top_k=6)
        else:
            ranked_paths = rank_evidence(
                query=state["user_query"],
                candidate_paths=retrieval_result.get("edges", []),
                top_k=6,
            )
        retrieval_result["ranked_paths"] = ranked_paths
        linked_entities = [node["name"] for node in retrieval_result.get("nodes", [])]

        logs = list(state.get("logs", []))
        logs.append(
            "retrieval: "
            f"nodes={len(retrieval_result.get('nodes', []))}, "
            f"edges={len(retrieval_result.get('edges', []))}, "
            f"vector_hits={len(retrieval_result.get('vector_hits', []))}, "
            f"ranked={len(ranked_paths)}"
        )
        return {
            "linked_entities": linked_entities,
            "evidence_pack": retrieval_result,
            "ranked_evidence": ranked_paths,
            "logs": logs,
        }

    return retrieval_node


def evidence_ranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Ranking is computed in retrieval for efficiency; this node makes the step explicit in workflow.
    logs = list(state.get("logs", []))
    logs.append("evidence_ranking: using precomputed ranked evidence paths")
    return {"logs": logs}
