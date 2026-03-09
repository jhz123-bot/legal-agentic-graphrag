from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


_SUGGESTION_LAYER_MAP = {
    "data_expansion_suggestion": "data",
    "cleaning_suggestion": "chunking",
    "rewrite_rule_suggestion": "rewrite",
    "retrieval_strategy_suggestion": "retrieval",
    "ranking_suggestion": "ranking",
}

_FAILURE_LAYER_WEIGHT = {
    "no_data": 1.0,
    "bad_chunking": 0.9,
    "rewrite_error": 0.9,
    "retrieval_strategy_error": 0.8,
    "retrieval_miss": 0.8,
    "ranking_error": 0.7,
    "reasoning_error": 0.6,
    "answer_hallucination": 0.6,
}

_PRIORITY_WEIGHT = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}

_STATUS_WEIGHT = {
    "open": 1.0,
    "in_progress": 0.6,
    "closed": 0.0,
}


def summarize_stage_bottlenecks(analysis_summary: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    dist = analysis_summary.get("failure_type_distribution", {}) or {}
    total = sum(int(v) for v in dist.values()) or 1
    ranked = sorted(
        (
            {
                "failure_type": str(k),
                "count": int(v),
                "ratio": float(v) / float(total),
                "severity_weight": _FAILURE_LAYER_WEIGHT.get(str(k), 0.5),
            }
            for k, v in dist.items()
        ),
        key=lambda x: (x["ratio"] * x["severity_weight"], x["count"]),
        reverse=True,
    )
    return ranked[: max(1, top_n)]


def build_action_queue(
    suggestions: List[Dict[str, Any]],
    analysis_summary: Dict[str, Any],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    suggestions = suggestions or []
    bottlenecks = summarize_stage_bottlenecks(analysis_summary, top_n=5)
    layer_pressure = Counter()
    for b in bottlenecks:
        layer_key = b["failure_type"]
        layer_pressure[layer_key] += float(b["ratio"]) * float(b["severity_weight"])

    rows: List[Dict[str, Any]] = []
    for s in suggestions:
        suggestion_type = str(s.get("suggestion_type", ""))
        status = str(s.get("status", "open"))
        if status == "closed":
            continue

        priority = str(s.get("suggested_priority", "medium"))
        layer = _SUGGESTION_LAYER_MAP.get(suggestion_type, "other")

        # Map suggestion type to failure-type pressure.
        if layer == "data":
            pressure = layer_pressure.get("no_data", 0.0)
        elif layer == "chunking":
            pressure = layer_pressure.get("bad_chunking", 0.0)
        elif layer == "rewrite":
            pressure = layer_pressure.get("rewrite_error", 0.0)
        elif layer == "retrieval":
            pressure = layer_pressure.get("retrieval_strategy_error", 0.0) + layer_pressure.get("retrieval_miss", 0.0)
        elif layer == "ranking":
            pressure = layer_pressure.get("ranking_error", 0.0)
        else:
            pressure = 0.1

        score = (
            _PRIORITY_WEIGHT.get(priority, 0.7) * 0.5
            + _STATUS_WEIGHT.get(status, 1.0) * 0.2
            + min(1.0, pressure) * 0.3
        )

        row = dict(s)
        row["blocking_layer"] = layer
        row["action_score"] = round(float(score), 4)
        rows.append(row)

    rows.sort(key=lambda x: (x.get("action_score", 0.0), x.get("suggested_priority", "medium")), reverse=True)
    return rows[: max(1, top_n)]

