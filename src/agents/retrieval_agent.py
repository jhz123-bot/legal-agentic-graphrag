from typing import Any, Dict

from src.graph.entity_linker import EntityLinker
from src.retrieval.evidence_ranker import rank_evidence
from src.retrieval.graph_retriever import GraphRetriever


def make_retrieval_node(graph_store):
    retriever = GraphRetriever(graph_store)
    linker = EntityLinker()

    def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Reuse Round 1 linker and retriever directly.
        linker.link(graph_store)
        queries = state.get("plan", {}).get("subqueries") or [state["user_query"]]
        merged_nodes = {}
        merged_edges = []
        merged_mentions = []

        for subq in queries:
            partial = retriever.retrieve(subq, top_k_nodes=6, top_k_edges=10)
            merged_mentions.extend(partial.get("query_mentions", []))
            for node in partial.get("nodes", []):
                merged_nodes[node["node_id"]] = node
            merged_edges.extend(partial.get("edges", []))

        retrieval_result = {
            "query_mentions": merged_mentions,
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
        }

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
