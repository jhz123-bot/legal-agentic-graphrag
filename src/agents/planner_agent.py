from typing import Any, Dict, List

from src.agents.query_decomposer import decompose_query
from src.graph.entity_extraction import extract_entities


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = state["user_query"]
    decomposition = decompose_query(user_query)
    subqueries = [user_query]
    for subq in decomposition.get("subqueries", []):
        if subq not in subqueries:
            subqueries.append(subq)

    mentions = extract_entities(user_query, doc_id="query")
    target_entities: List[str] = []
    for mention in mentions:
        if mention.name not in target_entities:
            target_entities.append(mention.name)

    query_lower = user_query.lower()
    needs_reasoning = any(token in query_lower for token in ["why", "how", "because", "explain", "decide", "holding"])
    intent = "fact_lookup"
    if needs_reasoning:
        intent = "legal_reasoning"

    plan = {
        "intent": intent,
        "target_entities": target_entities,
        "subqueries": subqueries,
        "needs_retrieval": True,
        "needs_reasoning": True,
    }

    logs = list(state.get("logs", []))
    logs.append(
        f"planner: intent={intent}, targets={target_entities}, "
        f"decompose_method={decomposition.get('method', 'unknown')}"
    )
    return {"plan": plan, "decomposition": decomposition, "logs": logs}
