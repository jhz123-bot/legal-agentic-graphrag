from typing import Any, Dict

from src.retrieval.strategy_selector import select_retrieval_strategy


def route_query(query: str) -> Dict[str, Any]:
    strategy = select_retrieval_strategy(query)
    retrieval_strategy = strategy["retrieval_strategy"]

    if retrieval_strategy == "vector":
        return {
            "route": "vector_retrieval",
            "retrieval_strategy": "vector",
            "reason": strategy["reason"],
            "preferred_tools": ["vector_search", "legal_article_lookup"],
        }
    if retrieval_strategy == "hybrid":
        return {
            "route": "hybrid_retrieval",
            "retrieval_strategy": "hybrid",
            "reason": strategy["reason"],
            "preferred_tools": ["case_lookup", "graph_search", "vector_search"],
        }
    if retrieval_strategy == "direct_answer":
        return {
            "route": "direct_answer",
            "retrieval_strategy": "direct_answer",
            "reason": strategy["reason"],
            "preferred_tools": [],
        }

    return {
        "route": "graph_reasoning",
        "retrieval_strategy": "graph",
        "reason": strategy["reason"],
        "preferred_tools": ["graph_search", "legal_article_lookup"],
    }


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    effective_query = state.get("rewritten_query") or state["user_query"]
    decision = route_query(effective_query)
    logs = list(state.get("logs", []))
    logs.append(
        "router: "
        f"route={decision['route']}, "
        f"retrieval_strategy={decision.get('retrieval_strategy', 'graph')}, "
        f"reason={decision['reason']}, "
        f"query_used={effective_query}"
    )
    return {
        "router_decision": decision,
        "retrieval_strategy": decision.get("retrieval_strategy", "graph"),
        "logs": logs,
    }
