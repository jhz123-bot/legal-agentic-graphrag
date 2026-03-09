from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.feedback.action_planner import build_action_queue, summarize_stage_bottlenecks


def _make_candidate(candidate_type: str, topic: str, queries: List[str], priority: str) -> Dict[str, Any]:
    return {
        "candidate_type": candidate_type,
        "topic": topic,
        "supporting_failed_queries": queries[:5],
        "suggested_priority": priority,
        "status": "open",
    }


def generate_closed_loop_candidates(analysis_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    topic_queries: Dict[str, List[str]] = defaultdict(list)
    for r in analysis_records:
        topic_queries[r.get("topic", "其他")].append(r.get("original_query", ""))

    faq_candidates: List[Dict[str, Any]] = []
    statute_candidates: List[Dict[str, Any]] = []
    case_candidates: List[Dict[str, Any]] = []

    for topic, queries in topic_queries.items():
        q_blob = " ".join(queries)
        count = len(queries)
        priority = "high" if count >= 3 else "medium"

        if any(x in q_blob for x in ["是什么", "怎么办", "流程", "可以吗", "通常由谁"]):
            faq_candidates.append(_make_candidate("faq", topic, queries, priority))
        if any(x in q_blob for x in ["法条", "哪一条", "依据", "条文"]):
            statute_candidates.append(_make_candidate("statute", topic, queries, priority))
        if any(x in q_blob for x in ["案件", "法院", "责任认定", "案例"]):
            case_candidates.append(_make_candidate("case", topic, queries, priority))

        # fallback when no explicit pattern but high frequency
        if count >= 3 and not any([faq_candidates, statute_candidates, case_candidates]):
            faq_candidates.append(_make_candidate("faq", topic, queries, "medium"))

    return {
        "faq_expansion_candidates": faq_candidates,
        "statute_gap_candidates": statute_candidates,
        "case_gap_candidates": case_candidates,
    }


def generate_data_feedback_loop_report(
    failure_records: List[Dict[str, Any]],
    analysis_summary: Dict[str, Any],
    suggestion_summary: Dict[str, Any],
    eval_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    eval_summary = eval_summary or {}
    total_eval = int(eval_summary.get("num_examples", len(failure_records)) or len(failure_records))
    total_fail = len(failure_records)
    if total_eval < total_fail:
        total_eval = total_fail

    q_counter = Counter(r.get("question_type", "unknown") for r in failure_records)
    s_counter = Counter(r.get("source_type", "unknown") for r in failure_records)
    top_causes = analysis_summary.get("failure_type_distribution", {})
    top_topics = analysis_summary.get("topic_distribution", {})

    analysis_records = analysis_summary.get("analysis_records", [])
    candidates = generate_closed_loop_candidates(analysis_records)
    status_counter = Counter(
        str((s or {}).get("status", "open"))
        for s in (suggestion_summary.get("suggestions", []) or [])
    )
    stage_bottlenecks = summarize_stage_bottlenecks(analysis_summary, top_n=3)
    action_queue = build_action_queue(
        suggestion_summary.get("suggestions", []) or [],
        analysis_summary=analysis_summary,
        top_n=10,
    )

    return {
        "failure_summary": {
            "total_failures": total_fail,
            "failure_rate": (total_fail / total_eval) if total_eval else 0.0,
            "by_question_type": dict(q_counter),
            "by_source_type": dict(s_counter),
        },
        "top_failure_causes": top_causes,
        "stage_bottleneck_summary": stage_bottlenecks,
        "top_weak_topics": top_topics,
        "suggested_actions": suggestion_summary.get("suggestions", []),
        "action_status_summary": dict(status_counter),
        "action_queue": action_queue,
        "closed_loop_candidates": candidates,
    }


def save_feedback_loop_outputs(
    report: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    report_path = root / "data_feedback_loop_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates = report.get("closed_loop_candidates", {})
    faq_path = root / "faq_expansion_candidates.json"
    statute_path = root / "statute_gap_candidates.json"
    case_path = root / "case_gap_candidates.json"
    faq_path.write_text(json.dumps(candidates.get("faq_expansion_candidates", []), ensure_ascii=False, indent=2), encoding="utf-8")
    statute_path.write_text(json.dumps(candidates.get("statute_gap_candidates", []), ensure_ascii=False, indent=2), encoding="utf-8")
    case_path.write_text(json.dumps(candidates.get("case_gap_candidates", []), ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Data Feedback Loop 摘要",
        f"- 失败总数: {report.get('failure_summary', {}).get('total_failures', 0)}",
        f"- 失败率: {report.get('failure_summary', {}).get('failure_rate', 0.0):.4f}",
        f"- 高频失败原因: {report.get('top_failure_causes', {})}",
        f"- 瓶颈层摘要: {report.get('stage_bottleneck_summary', [])}",
        f"- 弱主题: {report.get('top_weak_topics', {})}",
        f"- 待执行动作TOP3: {report.get('action_queue', [])[:3]}",
    ]
    summary_path = root / "data_feedback_loop_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "report_path": str(report_path),
        "faq_candidates_path": str(faq_path),
        "statute_candidates_path": str(statute_path),
        "case_candidates_path": str(case_path),
        "summary_path": str(summary_path),
    }
