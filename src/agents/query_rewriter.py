from __future__ import annotations

from typing import List

from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt


def _rule_rewrite(current_query: str, entities: List[str]) -> str:
    if not current_query.strip():
        return ""
    if any(k in current_query for k in ["那", "这个", "上述", "该情形", "这种情况"]) and entities:
        return f"在涉及{entities[0]}的情形下，{current_query}"
    return current_query.strip()


def rewrite_followup_query(
    current_query: str,
    history: str,
    summary: str,
    entities: List[str],
) -> dict:
    try:
        provider = get_llm_provider(timeout=20)
        prompt = render_prompt(
            "query_rewrite",
            current_query=current_query,
            history=history or "",
            summary=summary or "",
            entities=entities or [],
        )
        rewritten = provider.generate(prompt=prompt, temperature=0.0).strip()
        if rewritten:
            return {"rewritten_query": rewritten, "method": "llm"}
    except Exception:
        pass
    return {"rewritten_query": _rule_rewrite(current_query, entities), "method": "rule"}

