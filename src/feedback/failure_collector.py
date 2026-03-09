from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class FailureRecord:
    failure_id: str
    timestamp: str
    original_query: str
    rewritten_query: str
    conversation_context: List[Dict[str, Any]]
    retrieval_strategy: str
    evidence_pack_summary: Dict[str, Any]
    ranked_evidence_ids: List[str]
    final_answer: Dict[str, Any]
    failure_reason: str
    predicted_error_type: str
    question_type: str
    source_type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureCollector:
    def __init__(self) -> None:
        self._records: List[FailureRecord] = []

    @staticmethod
    def is_failure_result(result: Dict[str, Any]) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        if float(result.get("evidence_path_hit_rate", 0.0)) <= 0.0:
            reasons.append("retrieval_zero_hit")
        if float(result.get("answer_keyword_match_rate", 0.0)) < 0.2:
            reasons.append("low_answer_relevance")
        if float(result.get("citation_correctness", 0.0)) < 0.2:
            reasons.append("citation_incorrect")
        if float(result.get("grounding_score", 0.0)) < 0.1:
            reasons.append("low_grounding")
        return (len(reasons) > 0, reasons)

    def collect_failure(
        self,
        sample: Dict[str, Any],
        trace: Dict[str, Any],
        failure_reason: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        evidence_pack_summary = trace.get("evidence_pack_summary", {})
        if not evidence_pack_summary:
            evidence_pack_summary = {
                "candidate_evidence_count": len(trace.get("candidate_evidence", [])),
                "ranked_evidence_count": len(trace.get("ranked_evidence", [])),
                "graph_nodes": len((trace.get("evidence_pack") or {}).get("nodes", [])),
                "graph_edges": len((trace.get("evidence_pack") or {}).get("edges", [])),
                "vector_hits": len((trace.get("evidence_pack") or {}).get("vector_hits", [])),
            }

        final_answer_obj = trace.get("final_answer", {})
        if not isinstance(final_answer_obj, dict):
            final_answer_obj = {"short_answer": str(final_answer_obj)}

        record = FailureRecord(
            failure_id=str(uuid.uuid4()),
            timestamp=now,
            original_query=str(sample.get("query", "") or trace.get("query", "")),
            rewritten_query=str(trace.get("rewritten_query", sample.get("query", ""))),
            conversation_context=list(trace.get("conversation_context", [])),
            retrieval_strategy=str(trace.get("actual_retrieval_strategy", trace.get("retrieval_strategy", ""))),
            evidence_pack_summary=evidence_pack_summary,
            ranked_evidence_ids=list(trace.get("ranked_evidence_ids", [])),
            final_answer=final_answer_obj,
            failure_reason=failure_reason,
            predicted_error_type=str(trace.get("predicted_error_type", "")),
            question_type=str(sample.get("question_type", "")),
            source_type=str(sample.get("source_type", "")),
            metadata=metadata,
        )
        self._records.append(record)
        return record.to_dict()

    def records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def save_failure_records(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in self._records]
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
