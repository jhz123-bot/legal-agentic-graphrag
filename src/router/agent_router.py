from typing import Any, Dict


def route_query(query: str) -> Dict[str, Any]:
    q = query.lower()
    if any(k in query for k in ["定义", "是什么", "含义", "概念"]):
        return {
            "route": "vector_retrieval",
            "reason": "definition-like query",
            "preferred_tools": ["vector_search", "legal_article_lookup"],
        }
    if any(k in query for k in ["案例", "判决", "分析", "为何", "为什么", "如何认定"]):
        return {
            "route": "hybrid_retrieval",
            "reason": "case-analysis query",
            "preferred_tools": ["case_lookup", "graph_search", "vector_search"],
        }
    return {
        "route": "graph_reasoning",
        "reason": "fact/legal reasoning query",
        "preferred_tools": ["graph_search", "legal_article_lookup"],
    }


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = route_query(state["user_query"])
    logs = list(state.get("logs", []))
    logs.append(f"router: route={decision['route']}, reason={decision['reason']}")
    return {"router_decision": decision, "logs": logs}
