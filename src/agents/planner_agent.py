from typing import Any, Dict, List

from src.agents.query_decomposer import decompose_query
from src.graph.entity_extraction import extract_entities
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt
from src.retrieval.strategy_selector import select_retrieval_strategy


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def query_decomposition_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = state.get("rewritten_query") or state["user_query"]
    plan = dict(state.get("plan", {}))
    need_decomposition = bool(plan.get("need_decomposition", False))
    if need_decomposition:
        decomposition = decompose_query(user_query)
        subqueries = [user_query]
        for subq in decomposition.get("subqueries", []):
            if subq not in subqueries:
                subqueries.append(subq)
    else:
        decomposition = {"original_query": user_query, "subqueries": [user_query], "method": "skip"}
        subqueries = [user_query]

    logs = list(state.get("logs", []))
    logs.append(
        "query_decomposition: "
        f"method={decomposition.get('method', 'unknown')}, "
        f"subqueries={len(subqueries)}, "
        f"need_decomposition={need_decomposition}"
    )
    if plan:
        plan["subqueries"] = subqueries
    return {"decomposition": decomposition, "subqueries": subqueries, "plan": plan, "logs": logs}


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = state.get("rewritten_query") or state["user_query"]
    conversation_context = state.get("conversation_context", [])
    conversation_summary = state.get("conversation_summary", "")
    fact_memory = state.get("fact_memory", {})
    subqueries = state.get("subqueries", [user_query])

    mentions = extract_entities(user_query, doc_id="query")
    target_entities: List[str] = []
    for mention in mentions:
        if mention.name not in target_entities:
            target_entities.append(mention.name)

    strategy_from_router = state.get("router_decision", {}).get("retrieval_strategy")
    rule_strategy = strategy_from_router or select_retrieval_strategy(user_query).get("retrieval_strategy", "graph")

    llm_used = False
    plan: Dict[str, Any] = {}
    try:
        provider = get_llm_provider(timeout=20)
        context_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in conversation_context[-8:]
        ) or "无"
        prompt = render_prompt("planner", query=user_query, rule_strategy=rule_strategy)
        prompt += f"\n\n最近对话上下文：\n{context_text}\n"
        if conversation_summary:
            prompt += f"\n对话摘要：{conversation_summary}\n"
        if fact_memory:
            prompt += f"\n关键事实：{fact_memory}\n"
        parsed = provider.generate_json(prompt=prompt, temperature=0.0)
        if parsed:
            llm_used = True
            strategy = str(parsed.get("retrieval_strategy", rule_strategy))
            if strategy not in {"graph", "vector", "hybrid", "direct_answer"}:
                strategy = rule_strategy
            plan = {
                "intent": str(parsed.get("intent", "fact_lookup")),
                "target_entities": target_entities,
                "subqueries": subqueries,
                "retrieval_strategy": strategy,
                "need_retrieval": _to_bool(parsed.get("need_retrieval", True), True),
                "need_reasoning": _to_bool(parsed.get("need_reasoning", strategy != "direct_answer"), strategy != "direct_answer"),
                "need_decomposition": _to_bool(parsed.get("need_decomposition", False), False),
            }
    except Exception:
        llm_used = False

    if not plan:
        query_lower = user_query.lower()
        needs_reasoning = any(token in query_lower for token in ["why", "how", "because", "explain", "decide", "holding", "如何", "为什么"])
        intent = "legal_reasoning" if needs_reasoning else "fact_lookup"
        plan = {
            "intent": intent,
            "target_entities": target_entities,
            "subqueries": subqueries,
            "retrieval_strategy": rule_strategy,
            "need_retrieval": True,
            "need_reasoning": rule_strategy != "direct_answer",
            "need_decomposition": False,
        }

    # Keep router-first strategy to avoid planner LLM overriding deterministic routing.
    if strategy_from_router in {"graph", "vector", "hybrid", "direct_answer"}:
        plan["retrieval_strategy"] = strategy_from_router

    plan["needs_retrieval"] = plan.get("need_retrieval", True)
    plan["needs_reasoning"] = plan.get("need_reasoning", True)
    strategy = plan.get("retrieval_strategy", rule_strategy)

    logs = list(state.get("logs", []))
    logs.append(
        "planner: "
        f"intent={plan.get('intent')}, "
        f"targets={target_entities}, "
        f"subqueries={len(subqueries)}, "
        f"retrieval_strategy={strategy}, "
        f"llm_used={llm_used}, "
        f"context_turns={len(conversation_context)}, "
        f"has_summary={bool(conversation_summary)}, "
        f"has_facts={bool(fact_memory)}"
    )
    return {"plan": plan, "retrieval_strategy": strategy, "logs": logs}
