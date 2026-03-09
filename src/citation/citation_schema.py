from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CitationSchema:
    evidence_id: str
    source_type: str
    doc_id: str
    chunk_id: str
    title: str
    law_name: str
    article_no: str
    case_id: str
    court: str
    section: str
    text: str
    retrieval_score: float
    rerank_score: float


def empty_citation() -> Dict[str, Any]:
    return {
        "evidence_id": "",
        "source_type": "unknown",
        "doc_id": "",
        "chunk_id": "",
        "title": "",
        "law_name": "",
        "article_no": "",
        "case_id": "",
        "court": "",
        "section": "",
        "text": "",
        "retrieval_score": 0.0,
        "rerank_score": 0.0,
    }
