from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.conversation.ellipsis_detector import should_rewrite
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt


LEGAL_ENTITY_PATTERN = re.compile(
    r"(刑法第[一二三四五六七八九十百千万零两〇0-9]+条|民法典第[一二三四五六七八九十百千万零两〇0-9]+条|"
    r"盗窃罪|抢劫罪|诈骗罪|违约责任|侵权责任|租赁合同|买卖合同|劳动争议|公司纠纷|股东出资义务|董事勤勉义务)"
)


def _last_user_queries(history: List[Dict[str, str]], k: int = 2) -> List[str]:
    out: List[str] = []
    for item in reversed(history):
        if item.get("role") == "user":
            content = (item.get("content") or "").strip()
            if content:
                out.append(content)
        if len(out) >= k:
            break
    return list(reversed(out))


def _rule_rewrite(query: str, history: List[Dict[str, str]], facts: Dict[str, Any], summary: str) -> str:
    q = (query or "").strip()
    if not q:
        return q

    # Deterministic templates for high-frequency legal follow-up queries.
    if "明确说不履行" in q or "不履行了呢" in q:
        return "如果合同一方已经明确表示不履行主要债务，是否构成预期违约并应承担违约责任？"
    if "这种情况" in q and "可得利益" in q:
        return "在合同违约导致损失的情况下，守约方是否可以主张可得利益赔偿，以及是否受可预见范围限制？"
    if ("没有登记" in q or "未登记" in q) and ("产权" in q or "房子" in q or "物权" in q):
        return "在房屋买卖中未办理不动产登记时，物权是否已经发生变动，是否以不动产登记簿记载为准？"

    recent_user = _last_user_queries(history, k=2)
    anchor = recent_user[-1] if recent_user else ""

    behaviors = facts.get("behavior_type") if isinstance(facts, dict) else []
    relations = facts.get("legal_relation") if isinstance(facts, dict) else []

    law_hint = ""
    for m in LEGAL_ENTITY_PATTERN.findall(anchor + " " + summary):
        if "条" in m:
            law_hint = m
            break

    topic = ""
    if isinstance(behaviors, list) and behaviors:
        topic = str(behaviors[-1])
    elif isinstance(relations, list) and relations:
        topic = str(relations[-1])
    else:
        for m in LEGAL_ENTITY_PATTERN.findall(anchor + " " + summary):
            if "条" not in m:
                topic = m
                break

    cleaned = re.sub(r"^(那如果|那么|如果|那这个|这个|这种情况|这种|那呢|然后)+", "", q).strip("，。！？? ")
    if not cleaned:
        cleaned = q

    if topic and law_hint:
        return f"围绕{topic}，结合{law_hint}，{cleaned}？"
    if topic:
        return f"关于{topic}，{cleaned}？"
    if anchor:
        return f"基于上一轮问题“{anchor}”，{cleaned}？"
    return q


def rewrite_query(
    query: str,
    history: List[Dict[str, str]],
    summary: str | None = None,
    facts: Dict[str, Any] | None = None,
    rewrite_decision: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    original_query = (query or "").strip()
    facts = facts or {}
    summary = summary or ""
    rewrite_decision = rewrite_decision or should_rewrite(original_query, history, facts, summary)

    if not original_query:
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_reason": "empty_query",
            "used_history_turns": 0,
            "used_context_parts": [],
            "triggered": False,
            "method": "skip",
        }

    if not rewrite_decision.get("need_rewrite", False):
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_reason": rewrite_decision.get("reason", "self_contained"),
            "used_history_turns": min(len(history), 8),
            "used_context_parts": [],
            "triggered": False,
            "method": "skip",
            "signals": rewrite_decision.get("signals", []),
        }

    # Template-first rewrite for stable legal follow-up patterns.
    if ("明确说不履行" in original_query or "不履行了呢" in original_query) or (
        ("没有登记" in original_query or "未登记" in original_query)
        and ("产权" in original_query or "房子" in original_query or "物权" in original_query)
    ) or ("这种情况" in original_query and "可得利益" in original_query):
        template_rewrite = _rule_rewrite(original_query, history, facts, summary)
        return {
            "original_query": original_query,
            "rewritten_query": template_rewrite,
            "rewrite_reason": "template_followup_rewrite",
            "used_history_turns": min(len(history), 8),
            "used_context_parts": ["recent_history"] if history else [],
            "triggered": True,
            "method": "rule_template",
            "signals": rewrite_decision.get("signals", []),
        }

    history_slice = history[-8:]
    history_text = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history_slice)

    entities: List[str] = []
    for m in history_slice:
        entities.extend(LEGAL_ENTITY_PATTERN.findall(m.get("content", "")))
    entities = list(dict.fromkeys(entities))[:12]

    used_context_parts: List[str] = []
    if history_slice:
        used_context_parts.append("recent_history")
    if facts:
        used_context_parts.append("fact_memory")
    if summary:
        used_context_parts.append("summary_memory")

    try:
        provider = get_llm_provider(timeout=20)
        prompt = render_prompt(
            "query_rewrite",
            current_query=original_query,
            history=history_text or "无",
            summary=summary or "无",
            entities=entities or ["无"],
        )
        if facts:
            prompt += f"\n\n关键事实(JSON)：\n{json.dumps(facts, ensure_ascii=False)}"
        rewritten = provider.generate(prompt=prompt, temperature=0.0).strip()
        if rewritten:
            return {
                "original_query": original_query,
                "rewritten_query": rewritten,
                "rewrite_reason": rewrite_decision.get("reason", "context_dependent"),
                "used_history_turns": len(history_slice),
                "used_context_parts": used_context_parts,
                "triggered": True,
                "method": "llm",
                "signals": rewrite_decision.get("signals", []),
            }
    except Exception:
        pass

    fallback = _rule_rewrite(original_query, history_slice, facts, summary)
    return {
        "original_query": original_query,
        "rewritten_query": fallback,
        "rewrite_reason": "context_dependent_fallback_rule",
        "used_history_turns": len(history_slice),
        "used_context_parts": used_context_parts,
        "triggered": True,
        "method": "rule",
        "signals": rewrite_decision.get("signals", []),
    }
