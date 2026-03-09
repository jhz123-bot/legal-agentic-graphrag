from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


QUESTION_TYPES = {
    "statute_lookup",
    "concept_definition",
    "case_reasoning",
    "multiturn_followup",
}

SOURCE_TYPES = {
    "statute_only",
    "case_only",
    "hybrid",
}

_CATEGORY_QTYPE_HINTS = {
    "法条定位": "statute_lookup",
    "娉曟潯瀹氫綅": "statute_lookup",  # mojibake fallback
    "案例支撑": "case_reasoning",
    "妗堜緥鏀拺": "case_reasoning",  # mojibake fallback
    "多步推理": "case_reasoning",
    "澶氭鎺ㄧ悊": "case_reasoning",  # mojibake fallback
    "multiturn": "multiturn_followup",
    "FAQ": "concept_definition",
    "faq": "concept_definition",
}

_CATEGORY_STYPE_HINTS = {
    "法条定位": "statute_only",
    "娉曟潯瀹氫綅": "statute_only",  # mojibake fallback
    "案例支撑": "case_only",
    "妗堜緥鏀拺": "case_only",  # mojibake fallback
    "多步推理": "hybrid",
    "澶氭鎺ㄧ悊": "hybrid",  # mojibake fallback
    "FAQ": "statute_only",
    "faq": "statute_only",
}


@dataclass
class BenchmarkSample:
    id: int
    query: str
    expected_answer_keywords: List[str]
    expected_entities: List[str]
    expected_evidence_ids: List[str]
    expected_evidence_paths: List[str]
    question_type: str
    source_type: str
    requires_multiturn: bool
    expected_retrieval_strategy: str
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_question_type(raw: Dict[str, Any]) -> str:
    q = str(raw.get("query", ""))
    c_raw = str(raw.get("category", ""))
    c = c_raw.lower()
    if raw.get("requires_multiturn"):
        return "multiturn_followup"
    for hint, qtype in _CATEGORY_QTYPE_HINTS.items():
        if hint.lower() in c:
            return qtype
    if "faq" in c or "一定要" in q or "是不是" in q:
        return "concept_definition"
    if "案例" in c or "法院" in q or "案件" in q:
        return "case_reasoning"
    if "法条" in c or "哪一条" in q or "依据哪条" in q:
        return "statute_lookup"
    return "concept_definition"


def _infer_source_type(raw: Dict[str, Any]) -> str:
    q = str(raw.get("query", ""))
    c_raw = str(raw.get("category", ""))
    c = c_raw.lower()
    for hint, stype in _CATEGORY_STYPE_HINTS.items():
        if hint.lower() in c:
            return stype

    strategy = str(raw.get("expected_retrieval_strategy", "")).lower()
    if strategy in {"graph", "vector", "hybrid"}:
        if strategy == "graph":
            return "statute_only"
        if strategy == "vector":
            return "case_only"
        return "hybrid"

    if "案例" in c or "法院" in q or "案件" in q:
        if "法条" in c or "依据" in q:
            return "hybrid"
        return "case_only"
    if "法条" in c or "哪一条" in q or "依据哪条" in q:
        return "statute_only"
    return "hybrid"


def _infer_strategy(question_type: str, source_type: str) -> str:
    if question_type == "multiturn_followup":
        return "hybrid"
    if source_type == "statute_only":
        return "graph"
    if source_type == "case_only":
        return "vector"
    return "hybrid"


def normalize_benchmark_sample(raw: Dict[str, Any], index: int) -> BenchmarkSample:
    inferred_qtype = _infer_question_type(raw)
    raw_qtype = str(raw.get("question_type", "") or "")
    qtype = str(raw_qtype or inferred_qtype)
    if qtype not in QUESTION_TYPES:
        qtype = inferred_qtype
    # Correct known legacy over-default labels when category gives stronger signal.
    if (not raw_qtype or raw_qtype == "concept_definition") and qtype == "concept_definition" and inferred_qtype in {
        "statute_lookup",
        "case_reasoning",
        "multiturn_followup",
    }:
        qtype = inferred_qtype

    inferred_stype = _infer_source_type(raw)
    raw_stype = str(raw.get("source_type", "") or "")
    stype = str(raw_stype or inferred_stype)
    if stype not in SOURCE_TYPES:
        stype = inferred_stype
    # Correct known legacy over-default labels when stronger source signal exists.
    strategy_hint = str(raw.get("expected_retrieval_strategy", "")).lower()
    if (
        (not raw_stype or raw_stype == "hybrid")
        and stype == "hybrid"
        and inferred_stype in {"statute_only", "case_only"}
        and strategy_hint in {"", "graph", "vector"}
    ):
        stype = inferred_stype

    requires_multiturn = bool(raw.get("requires_multiturn", qtype == "multiturn_followup"))

    sample = BenchmarkSample(
        id=int(raw.get("id", index + 1)),
        query=str(raw.get("query", "")).strip(),
        expected_answer_keywords=list(raw.get("expected_answer_keywords", []) or []),
        expected_entities=list(raw.get("expected_entities", []) or []),
        expected_evidence_ids=list(raw.get("expected_evidence_ids", []) or []),
        expected_evidence_paths=list(raw.get("expected_evidence_paths", raw.get("expected_paths", [])) or []),
        question_type=qtype,
        source_type=stype,
        requires_multiturn=requires_multiturn,
        expected_retrieval_strategy=str(raw.get("expected_retrieval_strategy") or _infer_strategy(qtype, stype)),
        category=str(raw.get("category", "")),
    )
    return sample


def normalize_benchmark_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample = normalize_benchmark_sample(row, i)
        out.append(sample.to_dict())
    return out
