import json
import argparse
from pathlib import Path

from src.benchmark.benchmark_runner import run_agent_benchmark, save_benchmark_summary
from src.evaluation.report_generator import save_error_analysis_report, save_evaluation_markdown
from src.feedback import (
    FailureCollector,
    apply_action_status,
    archive_failure_records,
    batch_analyze_failures,
    generate_data_feedback_loop_report,
    load_action_status,
    save_action_status,
    update_action_status,
    save_feedback_loop_outputs,
    suggest_improvements,
)
from src.visualization.benchmark_plot import plot_benchmark_results


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Legal Agentic GraphRAG benchmark")
    parser.add_argument("--small", action="store_true", help="Run a small benchmark (at most 3 queries)")
    parser.add_argument("--small-size", type=int, default=3, help="Sample size used with --small")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --small sampling")
    parser.add_argument("--set-status", type=str, default="", help="Set status for one suggestion_id")
    parser.add_argument("--status", type=str, default="", choices=["open", "in_progress", "closed"], help="Target status for --set-status")
    parser.add_argument("--status-only", action="store_true", help="Only update action status and exit")
    args = parser.parse_args()

    root = Path(__file__).parent
    feedback_dir = root / "outputs" / "feedback"
    status_path = feedback_dir / "action_status.json"

    if args.set_status:
        if not args.status:
            raise SystemExit("--set-status 需要同时提供 --status")
        update_action_status(status_path, args.set_status, args.status)
        print(json.dumps({"updated": True, "suggestion_id": args.set_status, "status": args.status, "action_status_path": str(status_path)}, ensure_ascii=False, indent=2))
        if args.status_only:
            return

    dataset_path = root / "data" / "benchmark" / "legal_benchmark.json"
    summary = run_agent_benchmark(
        dataset_path,
        max_examples=args.small_size if args.small else None,
        seed=args.seed,
    )
    plot_files = plot_benchmark_results(summary, output_dir=root / "outputs" / "plots")
    summary["plot_files"] = plot_files
    output_dir = root / "outputs" / "evaluation"
    save_benchmark_summary(summary, output_dir / "citation_eval_summary.json")
    save_error_analysis_report(summary.get("error_analysis_report", {}), output_dir / "error_analysis_report.json")
    save_evaluation_markdown(summary, output_dir / "summary.md")

    error_records = (summary.get("error_analysis_report", {}) or {}).get("error_records", [])
    error_map = {int(r.get("id")): r for r in error_records if str(r.get("id", "")).isdigit()}

    collector = FailureCollector()
    for row in summary.get("results", []):
        is_fail, reasons = collector.is_failure_result(row)
        if not is_fail:
            continue
        row_id = row.get("id")
        err = error_map.get(int(row_id)) if str(row_id).isdigit() else {}
        failure_reason = str((err or {}).get("primary_error_type") or ",".join(reasons) or "unknown")
        trace = {
            "rewritten_query": row.get("rewritten_query", row.get("query", "")),
            "actual_retrieval_strategy": row.get("actual_retrieval_strategy", ""),
            "evidence_pack_summary": row.get("evidence_pack_summary", {}),
            "ranked_evidence_ids": row.get("ranked_evidence_ids", []),
            "final_answer": row.get("final_answer", {}),
            "predicted_error_type": (err or {}).get("primary_error_type", ""),
        }
        collector.collect_failure(sample=row, trace=trace, failure_reason=failure_reason, metadata={"reasons": reasons})

    raw_failures_path = feedback_dir / "failure_records.json"
    collector.save_failure_records(raw_failures_path)
    benchmark_failure_records = collector.records()
    runtime_failure_records = _load_jsonl(feedback_dir / "runtime_failures.jsonl")
    failure_records = benchmark_failure_records + runtime_failure_records

    archive_meta = archive_failure_records(benchmark_failure_records, feedback_dir / "failures")
    analyzed = batch_analyze_failures(failure_records)
    suggestions = suggest_improvements(analyzed.get("analysis_records", []))
    existing_status = load_action_status(status_path)
    suggestions["suggestions"] = apply_action_status(suggestions.get("suggestions", []), existing_status)
    save_action_status(status_path, suggestions.get("suggestions", []))
    loop_report = generate_data_feedback_loop_report(
        failure_records=failure_records,
        analysis_summary=analyzed,
        suggestion_summary=suggestions,
        eval_summary=summary,
    )
    loop_paths = save_feedback_loop_outputs(loop_report, feedback_dir)
    summary["feedback_loop"] = {
        "raw_failure_records_path": str(raw_failures_path),
        "runtime_failure_records_path": str(feedback_dir / "runtime_failures.jsonl"),
        "benchmark_failure_count": len(benchmark_failure_records),
        "runtime_failure_count": len(runtime_failure_records),
        "combined_failure_count": len(failure_records),
        "archive": archive_meta,
        "analysis_count": analyzed.get("count", 0),
        "suggestion_count": len(suggestions.get("suggestions", [])),
        "action_status_path": str(status_path),
        "stage_bottleneck_summary": loop_report.get("stage_bottleneck_summary", []),
        "top_action_queue": loop_report.get("action_queue", [])[:3],
        "paths": loop_paths,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
