from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.feedback.failure_archive import archive_failure_records
from src.feedback.failure_collector import FailureCollector


def detect_runtime_failure(trace: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    evidence_pack = trace.get("evidence_pack", {}) or {}
    candidate = trace.get("candidate_evidence", evidence_pack.get("candidate_evidence", [])) or []
    ranked = trace.get("ranked_evidence", evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", []))) or []
    grounding = (trace.get("final_answer", {}) or {}).get("grounding", {}) or {}
    short_answer = str((trace.get("final_answer", {}) or {}).get("short_answer", ""))
    confidence = float((trace.get("final_answer", {}) or {}).get("confidence", 0.0) or 0.0)

    if not candidate and not ranked:
        reasons.append("retrieval_zero_hit")
    if not bool(grounding.get("grounded", False)) and float(grounding.get("grounding_score", 0.0) or 0.0) < 0.1:
        reasons.append("low_grounding")
    if any(token in short_answer for token in ["证据不足", "无法判断", "无法回答", "建议补充"]):
        reasons.append("insufficient_evidence")
    if confidence < 0.35:
        reasons.append("low_confidence")

    return (len(reasons) > 0, reasons)


def record_runtime_failure(
    query: str,
    trace: Dict[str, Any],
    output_root: str | Path,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    should_record, reasons = detect_runtime_failure(trace)
    if not should_record:
        return None

    collector = FailureCollector()
    sample = {
        "query": query,
        "question_type": str(trace.get("question_type", "runtime")),
        "source_type": str(trace.get("source_type", "runtime")),
    }
    reason = reasons[0] if reasons else "runtime_failure"
    rec = collector.collect_failure(sample=sample, trace=trace, failure_reason=reason, metadata=metadata or {"reasons": reasons})

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    dedup_index_path = root / "runtime_failures_dedup_index.json"
    day_key = datetime.utcnow().strftime("%Y-%m-%d")
    dedup_key = hashlib.md5(f"{day_key}|{query}|{reason}".encode("utf-8")).hexdigest()
    dedup_index: Dict[str, Any] = {}
    if dedup_index_path.exists():
        try:
            dedup_index = json.loads(dedup_index_path.read_text(encoding="utf-8"))
        except Exception:
            dedup_index = {}
    if dedup_key in dedup_index:
        return {
            "record": rec,
            "jsonl_path": str(root / "runtime_failures.jsonl"),
            "archive": None,
            "deduplicated": True,
        }

    jsonl_path = root / "runtime_failures.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    archive_meta = archive_failure_records([rec], root / "runtime_failures_archive")
    dedup_index[dedup_key] = {
        "timestamp": rec.get("timestamp"),
        "query": query,
        "failure_reason": reason,
        "failure_id": rec.get("failure_id"),
    }
    dedup_index_path.write_text(json.dumps(dedup_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "record": rec,
        "jsonl_path": str(jsonl_path),
        "archive": archive_meta,
        "deduplicated": False,
    }
