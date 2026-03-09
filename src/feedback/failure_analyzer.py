from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


TOPIC_HINTS = {
    "租赁": "租赁纠纷",
    "承租": "租赁纠纷",
    "出租": "租赁纠纷",
    "盗窃": "盗窃罪",
    "诈骗": "诈骗罪",
    "侵权": "侵权责任",
    "违约": "违约责任",
    "买卖": "买卖合同",
    "公司": "公司纠纷",
    "劳动": "劳动争议",
}


def _infer_topic(query: str) -> str:
    q = query or ""
    for k, v in TOPIC_HINTS.items():
        if k in q:
            return v
    return "其他"


def analyze_failure(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = record.get("evidence_pack_summary", {}) or {}
    reason = str(record.get("failure_reason", "") or "")
    query = str(record.get("original_query", "") or "")
    ranked_ids = record.get("ranked_evidence_ids", []) or []
    retrieval_strategy = str(record.get("retrieval_strategy", "") or "")

    candidate_count = int(summary.get("candidate_evidence_count", 0) or 0)
    ranked_count = int(summary.get("ranked_evidence_count", 0) or 0)
    vector_hits = int(summary.get("vector_hits", 0) or 0)
    graph_nodes = int(summary.get("graph_nodes", 0) or 0)

    primary = "retrieval_miss"
    secondary: List[str] = []
    analysis_reason = "召回命中不足。"

    if candidate_count == 0 and vector_hits == 0 and graph_nodes == 0:
        primary = "no_data"
        analysis_reason = "图谱和向量均无可用证据，疑似知识库覆盖不足。"
    elif candidate_count == 0:
        primary = "retrieval_miss"
        analysis_reason = "候选证据为空，疑似召回失败。"
    elif candidate_count > 0 and ranked_count == 0:
        primary = "ranking_error"
        analysis_reason = "候选证据存在但排序结果为空。"
    elif candidate_count > 0 and ranked_count > 0 and len(ranked_ids) == 0:
        primary = "ranking_error"
        analysis_reason = "存在排序结果但无有效证据ID，疑似排序链路映射问题。"
    elif "rewrite" in reason:
        primary = "rewrite_error"
        analysis_reason = "query 改写偏离上下文或未正确补全。"
    elif "reasoning" in reason:
        primary = "reasoning_error"
        analysis_reason = "证据存在但推理链条不稳定。"
    elif "hallucination" in reason:
        primary = "answer_hallucination"
        analysis_reason = "答案出现证据未支持内容。"
    elif retrieval_strategy == "graph" and "案例" in query:
        primary = "retrieval_strategy_error"
        analysis_reason = "案例问题使用 graph 策略可能不合适。"
    elif retrieval_strategy == "vector" and ("法条" in query or "哪一条" in query):
        primary = "retrieval_strategy_error"
        analysis_reason = "法条定位问题使用 vector 策略可能不合适。"

    if candidate_count > 0 and ranked_count > 0 and len(query) > 40 and graph_nodes == 0:
        secondary.append("bad_chunking")

    return {
        "failure_id": record.get("failure_id"),
        "original_query": query,
        "question_type": record.get("question_type"),
        "source_type": record.get("source_type"),
        "topic": _infer_topic(query),
        "primary_failure_type": primary,
        "secondary_failure_types": secondary,
        "analysis_reason": analysis_reason,
    }


def batch_analyze_failures(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    analyses = [analyze_failure(r) for r in records]
    counter = Counter(a.get("primary_failure_type", "unknown") for a in analyses)
    topic_counter = Counter(a.get("topic", "其他") for a in analyses)
    total = len(analyses) or 1

    return {
        "count": len(analyses),
        "analysis_records": analyses,
        "failure_type_distribution": dict(counter),
        "failure_type_ratio": {k: v / total for k, v in counter.items()},
        "topic_distribution": dict(topic_counter),
    }
