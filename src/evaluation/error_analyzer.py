from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def _contains_any(hay: str, needles: List[str]) -> bool:
    h = (hay or "").lower()
    return any((n or "").lower() in h for n in needles if n)


def analyze_single_error(sample: Dict[str, Any], prediction: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    # sample: normalized benchmark sample
    # prediction: per-example metric record
    # trace: raw state from workflow
    primary = "none"
    secondary: List[str] = []
    details: List[str] = []

    rewrite_info = trace.get("rewrite_info", {})
    rewrite_decision = trace.get("rewrite_decision", {})
    rewritten_query = trace.get("rewritten_query", sample.get("query", ""))
    requires_multiturn = bool(sample.get("requires_multiturn", False))

    expected_entities = sample.get("expected_entities", [])
    expected_evidence_ids = set(str(x) for x in (sample.get("expected_evidence_ids", []) or []) if x)
    linked_entities = trace.get("linked_entities", [])
    evidence_pack = trace.get("evidence_pack", {})
    candidate = evidence_pack.get("candidate_evidence", [])
    ranked = trace.get("ranked_evidence", evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", [])))
    reasoning_trace = trace.get("reasoning_trace", {})
    final_answer = trace.get("final_answer", {})
    candidate_ids = {
        str(e.get("evidence_id", ""))
        for e in candidate
        if isinstance(e, dict) and str(e.get("evidence_id", "")).strip()
    }
    ranked_ids = {
        str(e.get("evidence_id", ""))
        for e in ranked
        if isinstance(e, dict) and str(e.get("evidence_id", "")).strip()
    }

    # 1) rewrite error
    if requires_multiturn:
        triggered = bool(rewrite_info.get("triggered", False))
        if not triggered and rewrite_decision.get("need_rewrite", False):
            primary = "rewrite_error"
            details.append("multi-turn问题应触发rewrite但未触发")
        elif expected_entities and not _contains_any(rewritten_query, expected_entities):
            primary = "rewrite_error"
            details.append("rewritten_query未包含关键上下文实体")

    # 2) retrieval error
    if primary == "none":
        if expected_evidence_ids and candidate_ids and not (expected_evidence_ids & candidate_ids):
            primary = "retrieval_error"
            details.append("目标evidence_id未被召回")
        elif not candidate and prediction.get("evidence_path_hit_rate", 0.0) <= 0.0:
            primary = "retrieval_error"
            details.append("候选证据为空或证据命中率为0")
        elif expected_entities and not linked_entities:
            primary = "retrieval_error"
            details.append("未检索到实体链接")

    # 3) ranking error
    if primary == "none":
        if expected_evidence_ids and ranked_ids and not (expected_evidence_ids & ranked_ids) and (expected_evidence_ids & candidate_ids):
            primary = "ranking_error"
            details.append("目标evidence已召回但未进入排序结果")
        elif candidate and not ranked:
            primary = "ranking_error"
            details.append("有召回无排序结果")
        elif candidate and ranked and prediction.get("citation_correctness", 0.0) <= 0.0:
            primary = "ranking_error"
            details.append("目标证据可能未进入top-k")

    # 4) reasoning error
    if primary == "none":
        unsupported = reasoning_trace.get("unsupported_claims", [])
        if unsupported and prediction.get("grounding_score", 0.0) < 0.1:
            primary = "reasoning_error"
            details.append("claim-evidence映射存在未支撑结论")

    # 5) answer hallucination
    if primary == "none":
        grounding = final_answer.get("grounding", {})
        if not grounding.get("grounded", True):
            primary = "answer_hallucination"
            details.append("最终回答grounding不足")
        elif prediction.get("citation_correctness", 0.0) < 0.2 and prediction.get("answer_keyword_match_rate", 0.0) < 0.3:
            primary = "answer_hallucination"
            details.append("低citation正确率且答案相关性低")

    if primary == "none" and prediction.get("answer_keyword_match_rate", 0.0) < 0.2:
        primary = "reasoning_error"
        details.append("答案关键词覆盖偏低")

    return {
        "id": sample.get("id"),
        "query": sample.get("query"),
        "question_type": sample.get("question_type"),
        "source_type": sample.get("source_type"),
        "primary_error_type": primary,
        "secondary_error_types": secondary,
        "details": details,
    }


def aggregate_errors(error_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [e for e in error_records if e.get("primary_error_type") and e.get("primary_error_type") != "none"]
    counter = Counter(e.get("primary_error_type") for e in valid)
    total = sum(counter.values()) or 1
    ratio = {k: v / total for k, v in counter.items()}

    bottleneck = "none"
    if counter:
        bottleneck = counter.most_common(1)[0][0]

    return {
        "error_type_distribution": dict(counter),
        "error_type_ratio": ratio,
        "stage_bottleneck": bottleneck,
    }
