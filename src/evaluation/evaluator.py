from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def evaluate_overall(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "retrieval_hit_rate": 0.0,
            "answer_relevance": 0.0,
            "citation_correctness": 0.0,
            "citation_coverage": 0.0,
            "grounding_score": 0.0,
            "context_resolution_accuracy": 0.0,
            "rewrite_trigger_accuracy": 0.0,
            "rewritten_query_quality": 0.0,
            "average_latency": 0.0,
        }
    return {
        "sample_count": len(results),
        "retrieval_hit_rate": _avg([float(r.get("evidence_path_hit_rate", 0.0)) for r in results]),
        "answer_relevance": _avg([float(r.get("answer_keyword_match_rate", 0.0)) for r in results]),
        "citation_correctness": _avg([float(r.get("citation_correctness", 0.0)) for r in results]),
        "citation_coverage": _avg([float(r.get("citation_coverage", 0.0)) for r in results]),
        "grounding_score": _avg([float(r.get("grounding_score", 0.0)) for r in results]),
        "context_resolution_accuracy": _avg([float(r.get("context_resolution_accuracy", 0.0)) for r in results]),
        "rewrite_trigger_accuracy": _avg([float(r.get("rewrite_trigger_accuracy", 0.0)) for r in results]),
        "rewritten_query_quality": _avg([float(r.get("rewritten_query_quality", 0.0)) for r in results]),
        "average_latency": _avg([float(r.get("latency", 0.0)) for r in results]),
    }


def _bucket(results: List[Dict[str, Any]], key: str, values: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for v in values:
        sub = [r for r in results if str(r.get(key, "")) == v]
        out[v] = evaluate_overall(sub)
    return out


def evaluate_by_question_type(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _bucket(
        results,
        key="question_type",
        values=["statute_lookup", "concept_definition", "case_reasoning", "multiturn_followup"],
    )


def evaluate_by_source_type(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _bucket(
        results,
        key="source_type",
        values=["statute_only", "case_only", "hybrid"],
    )


def evaluate_multiturn(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    subset = [r for r in results if bool(r.get("requires_multiturn", False))]
    return evaluate_overall(subset)
