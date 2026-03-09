import re
from typing import Dict, List, Optional

from src.llm.base_provider import BaseLLMProvider
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt


def _rule_based_decompose(query: str) -> List[str]:
    parts = re.split(r"[，。；;]\s*|以及|并且|同时|和", query)
    subqueries = [p.strip() for p in parts if p.strip()]
    if not subqueries:
        return [query.strip()]
    if len(subqueries) == 1 and "什么" not in subqueries[0] and "如何" not in subqueries[0]:
        return [query.strip()]
    return subqueries[:3]


def _llm_decompose(query: str, provider: BaseLLMProvider) -> Optional[List[str]]:
    prompt = render_prompt("query_decomposition", query=query)
    try:
        text = provider.generate(prompt=prompt, temperature=0.0)
    except Exception:
        return None
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if len(line) > 2]
    if not lines:
        return None
    return lines[:3]


def decompose_query(query: str) -> Dict[str, List[str] | str]:
    provider = get_llm_provider(timeout=20)
    llm_subqueries = _llm_decompose(query, provider)
    if llm_subqueries:
        return {"original_query": query, "subqueries": llm_subqueries, "method": "llm"}
    return {"original_query": query, "subqueries": _rule_based_decompose(query), "method": "rule"}
