from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List


def _top_failing_samples(results: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    scored = []
    for r in results:
        score = (
            0.35 * float(r.get("answer_keyword_match_rate", 0.0))
            + 0.25 * float(r.get("evidence_path_hit_rate", 0.0))
            + 0.2 * float(r.get("citation_correctness", 0.0))
            + 0.2 * float(r.get("grounding_score", 0.0))
        )
        scored.append((score, r))
    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored[:top_n]]


def _top_failing_by_bucket(results: List[Dict[str, Any]], key: str, top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        bucket = str(row.get(key, "unknown") or "unknown")
        grouped.setdefault(bucket, []).append(row)
    return {bucket: _top_failing_samples(rows, top_n=top_n) for bucket, rows in grouped.items()}


def _error_matrix_by_question_type(error_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Counter] = {}
    for row in error_records:
        qtype = str(row.get("question_type", "unknown") or "unknown")
        etype = str(row.get("primary_error_type", "none") or "none")
        if etype == "none":
            continue
        matrix.setdefault(qtype, Counter())
        matrix[qtype][etype] += 1
    return {qtype: dict(counter) for qtype, counter in matrix.items()}


def _error_matrix_by_source_type(error_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Counter] = {}
    for row in error_records:
        stype = str(row.get("source_type", "unknown") or "unknown")
        etype = str(row.get("primary_error_type", "none") or "none")
        if etype == "none":
            continue
        matrix.setdefault(stype, Counter())
        matrix[stype][etype] += 1
    return {stype: dict(counter) for stype, counter in matrix.items()}


def _bucket_bottleneck_summary(bucket_metrics: Dict[str, Any], prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bucket, metrics in bucket_metrics.items():
        count = int(metrics.get("sample_count", 0) or 0)
        if count <= 0:
            out[bucket] = f"{prefix}{bucket}暂无样本。"
            continue
        retrieval = float(metrics.get("retrieval_hit_rate", 0.0))
        relevance = float(metrics.get("answer_relevance", 0.0))
        citation = float(metrics.get("citation_correctness", 0.0))
        grounding = float(metrics.get("grounding_score", 0.0))
        scores = {
            "retrieval": retrieval,
            "answer_relevance": relevance,
            "citation": citation,
            "grounding": grounding,
        }
        weakest = min(scores, key=scores.get)
        out[bucket] = f"{prefix}{bucket}主要短板在 {weakest}（{round(scores[weakest], 4)}）。"
    return out


def _strategy_drift_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    mismatch = 0
    counter: Counter = Counter()
    for r in results:
        expected = str(r.get("expected_retrieval_strategy", "") or "")
        actual = str(r.get("actual_retrieval_strategy", "") or "")
        if not expected:
            continue
        hit = expected == actual
        if not hit:
            mismatch += 1
            counter[f"{expected}->{actual or 'unknown'}"] += 1
        rows.append({"id": r.get("id"), "expected": expected, "actual": actual, "matched": hit})
    total = len(rows)
    return {
        "sample_count": total,
        "match_count": total - mismatch,
        "mismatch_count": mismatch,
        "match_rate": (total - mismatch) / total if total else 0.0,
        "mismatch_patterns": dict(counter),
        "details": rows,
    }


def generate_error_analysis_report(
    overall_metrics: Dict[str, Any],
    question_type_summary: Dict[str, Any],
    source_type_summary: Dict[str, Any],
    multiturn_summary: Dict[str, Any],
    errors_summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    error_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_fail = _top_failing_samples(results, top_n=10)
    bottleneck = errors_summary.get("stage_bottleneck", "none")
    ratio = errors_summary.get("error_type_ratio", {})
    stage_line = "当前未识别明显瓶颈。"
    if bottleneck != "none":
        stage_line = (
            f"当前主要瓶颈集中在 {bottleneck}，"
            f"占比约 {round(float(ratio.get(bottleneck, 0.0)) * 100, 2)}%。"
        )

    return {
        "overall_metrics": overall_metrics,
        "metrics_by_question_type": question_type_summary,
        "metrics_by_source_type": source_type_summary,
        "multiturn_summary": multiturn_summary,
        "error_type_distribution": errors_summary.get("error_type_distribution", {}),
        "error_type_ratio": ratio,
        "stage_bottleneck_summary": {
            "stage_bottleneck": bottleneck,
            "summary": stage_line,
        },
        "top_failing_samples": top_fail,
        "top_failing_by_question_type": _top_failing_by_bucket(results, key="question_type", top_n=3),
        "top_failing_by_source_type": _top_failing_by_bucket(results, key="source_type", top_n=3),
        "error_matrix_by_question_type": _error_matrix_by_question_type(error_records),
        "error_matrix_by_source_type": _error_matrix_by_source_type(error_records),
        "question_type_bottleneck_summary": _bucket_bottleneck_summary(question_type_summary, prefix="题型 "),
        "source_type_bottleneck_summary": _bucket_bottleneck_summary(source_type_summary, prefix="来源 "),
        "strategy_drift_summary": _strategy_drift_summary(results),
        "error_records": error_records,
    }


def save_error_analysis_report(report: Dict[str, Any], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_evaluation_markdown(summary: Dict[str, Any], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report = summary.get("error_analysis_report", {})
    overall = summary.get("overall_metrics", {})
    stage = summary.get("stage_bottleneck_summary", {})
    drift = report.get("strategy_drift_summary", {})
    lines = [
        "# 评测摘要",
        "",
        "## Overall",
        f"- 样本数: {overall.get('sample_count', 0)}",
        f"- 检索命中率: {overall.get('retrieval_hit_rate', 0.0):.4f}",
        f"- 答案相关性: {overall.get('answer_relevance', 0.0):.4f}",
        f"- 引用正确率: {overall.get('citation_correctness', 0.0):.4f}",
        f"- Grounding分数: {overall.get('grounding_score', 0.0):.4f}",
        "",
        "## 主要瓶颈",
        f"- 阶段: {stage.get('stage_bottleneck', 'none')}",
        f"- 说明: {stage.get('summary', '')}",
        "",
        "## 检索策略偏差",
        f"- 策略匹配率: {drift.get('match_rate', 0.0):.4f}",
        f"- 不匹配数: {drift.get('mismatch_count', 0)} / {drift.get('sample_count', 0)}",
        "",
        "## 题型短板",
    ]
    for k, v in (report.get("question_type_bottleneck_summary", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## 来源短板"])
    for k, v in (report.get("source_type_bottleneck_summary", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
