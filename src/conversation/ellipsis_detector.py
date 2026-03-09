from __future__ import annotations

from typing import Any, Dict, List

PRONOUN_PATTERNS = [
    "这个",
    "那个",
    "这种情况",
    "那如果",
    "如果这样",
    "那呢",
    "他",
    "她",
    "它",
]
FOLLOWUP_PATTERNS = ["那", "那么", "然后", "如果", "这种", "这样", "那如果"]
LEGAL_ENTITY_HINTS = [
    "刑法",
    "民法典",
    "公司法",
    "劳动合同法",
    "盗窃",
    "诈骗",
    "违约",
    "侵权",
    "租赁",
    "买卖",
    "借贷",
    "法条",
    "条文",
]


def should_rewrite(query: str, history: List[Dict[str, str]], facts: Dict[str, Any], summary: str) -> Dict[str, Any]:
    q = (query or "").strip()
    signals: List[str] = []
    user_turn_count = len([h for h in history if h.get("role") == "user" and (h.get("content") or "").strip()])

    if len(q) <= 10:
        signals.append("short_query")

    if any(p in q for p in PRONOUN_PATTERNS):
        signals.append("pronoun_or_reference")

    if any(q.startswith(p) for p in FOLLOWUP_PATTERNS):
        signals.append("followup_pattern")

    has_legal_entity = any(k in q for k in LEGAL_ENTITY_HINTS)
    if not has_legal_entity:
        signals.append("missing_legal_entity")

    if user_turn_count > 0 and any(tok in q for tok in ["那", "这个", "这种", "如果", "上述", "该情形"]):
        signals.append("depends_on_previous_turn")

    fact_dense = False
    if isinstance(facts, dict):
        bt = facts.get("behavior_type") or []
        lr = facts.get("legal_relation") or []
        if bt or lr or facts.get("amount") or facts.get("time"):
            fact_dense = True

    # Strong standalone follow-up patterns should still trigger rewrite even if
    # benchmark/demo context is not explicitly passed in this turn.
    strong_followup = any(tok in q for tok in ["如果对方", "那没有登记", "那没有", "这种情况下", "明确说不履行"])
    need_rewrite = (bool(signals) and user_turn_count > 0) or strong_followup
    if fact_dense and ("short_query" in signals or "pronoun_or_reference" in signals):
        need_rewrite = True

    reason = "context_dependent" if need_rewrite else "self_contained"
    return {
        "need_rewrite": need_rewrite,
        "reason": reason,
        "signals": signals,
    }
