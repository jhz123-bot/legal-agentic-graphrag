from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from src.feedback.status_tracker import build_suggestion_id


def suggest_improvements(failure_analysis_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    if not failure_analysis_records:
        return {"suggestions": suggestions, "by_type": {}}

    by_type = Counter(r.get("primary_failure_type", "unknown") for r in failure_analysis_records)
    by_topic = Counter(r.get("topic", "其他") for r in failure_analysis_records)
    by_question_type = Counter(r.get("question_type", "unknown") for r in failure_analysis_records)
    by_source_type = Counter(r.get("source_type", "unknown") for r in failure_analysis_records)

    topic_failures: Dict[str, List[str]] = defaultdict(list)
    for r in failure_analysis_records:
        topic_failures[r.get("topic", "其他")].append(r.get("original_query", ""))

    for topic, count in by_topic.most_common(5):
        if count >= 2:
            suggestions.append(
                {
                    "suggestion_type": "data_expansion_suggestion",
                    "target": topic,
                    "reason": f"失败样本中 {topic} 出现 {count} 次，知识覆盖可能不足。",
                    "supporting_failed_queries": topic_failures.get(topic, [])[:5],
                    "suggested_priority": "high" if count >= 4 else "medium",
                    "status": "open",
                    "suggested_owner": "data_team",
                    "execution_hint": "补充该主题法条/案例/FAQ，优先覆盖高频问法。",
                }
            )

    if by_type.get("bad_chunking", 0) > 0:
        suggestions.append(
            {
                "suggestion_type": "cleaning_suggestion",
                "target": "chunking_rules",
                "reason": "检测到 bad_chunking 失败，建议优化法条/案例结构化切分。",
                "suggested_priority": "high",
                "status": "open",
                "suggested_owner": "ingestion_team",
                "execution_hint": "调整按条/款/项与案情/法院认为切分策略。",
            }
        )

    if by_type.get("rewrite_error", 0) > 0:
        suggestions.append(
            {
                "suggestion_type": "rewrite_rule_suggestion",
                "target": "ellipsis_and_followup_patterns",
                "reason": f"rewrite_error 出现 {by_type.get('rewrite_error', 0)} 次，建议扩展追问改写规则。",
                "suggested_priority": "high",
                "status": "open",
                "suggested_owner": "conversation_team",
                "execution_hint": "新增省略问法模式，并增强事实补全。",
            }
        )

    if by_type.get("retrieval_strategy_error", 0) > 0:
        suggestions.append(
            {
                "suggestion_type": "retrieval_strategy_suggestion",
                "target": "router_strategy_rules",
                "reason": f"retrieval_strategy_error 出现 {by_type.get('retrieval_strategy_error', 0)} 次，建议调整 graph/vector/hybrid 路由规则。",
                "suggested_priority": "medium",
                "status": "open",
                "suggested_owner": "retrieval_team",
                "execution_hint": "按问题类型重设策略优先级并回归验证。",
            }
        )

    if by_type.get("ranking_error", 0) > 0 or by_type.get("retrieval_miss", 0) > 0:
        suggestions.append(
            {
                "suggestion_type": "ranking_suggestion",
                "target": "ranking_weights_and_topk",
                "reason": "检索/排序失败占比较高，建议调参 evidence ranking 与 rerank top-k。",
                "suggested_priority": "medium",
                "status": "open",
                "suggested_owner": "ranking_team",
                "execution_hint": "按 source/question 分桶调权重与 top-k。",
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in suggestions:
        s["suggestion_id"] = build_suggestion_id(s)
        grouped[s["suggestion_type"]].append(s)

    return {
        "suggestions": suggestions,
        "by_type": dict(grouped),
        "failure_type_distribution": dict(by_type),
        "topic_distribution": dict(by_topic),
        "question_type_distribution": dict(by_question_type),
        "source_type_distribution": dict(by_source_type),
    }
