from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from src.citation.citation_schema import empty_citation
from src.config.settings import settings
from src.llm.llm_router import get_llm_provider


LAW_ARTICLE_PATTERN = re.compile(r"(刑法|民法典|公司法|劳动合同法|劳动争议调解仲裁法)?第([一二三四五六七八九十百千万零两〇0-9]+)条")
COURT_PATTERN = re.compile(r"([\u4e00-\u9fff]{2,}(?:人民法院|法院))")
CASE_ID_PATTERN = re.compile(r"((?:\(|（)?[0-9]{4}(?:\)|）)?.{0,8}号)")

_DIGITS_ZH = "零一二三四五六七八九"


def _int_to_zh(num: int) -> str:
    if num <= 0:
        return "零"
    units = ["", "十", "百", "千"]
    parts = []
    s = str(num)
    n = len(s)
    for i, ch in enumerate(s):
        d = int(ch)
        u = units[n - i - 1]
        if d == 0:
            if parts and parts[-1] != "零":
                parts.append("零")
            continue
        parts.append(_DIGITS_ZH[d] + u)
    out = "".join(parts).rstrip("零")
    out = out.replace("一十", "十")
    return out or "零"


def _infer_from_doc_id(doc_id: str) -> tuple[str, str]:
    d = (doc_id or "").lower()
    m = re.search(r"(law|statute)_(criminal|civil|company|labor|labour)_([0-9]{1,4})", d)
    if m:
        branch = m.group(2)
        no = int(m.group(3))
        zh_no = _int_to_zh(no)
        if branch == "criminal":
            return "中华人民共和国刑法", f"刑法第{zh_no}条"
        if branch == "civil":
            return "中华人民共和国民法典", f"民法典第{zh_no}条"
        if branch == "company":
            return "中华人民共和国公司法", f"公司法第{zh_no}条"
        return "中华人民共和国劳动合同法", f"劳动合同法第{zh_no}条"
    return "", ""


def _clean_graph_label(label: str) -> str:
    text = (label or "").strip()
    if "::" in text:
        text = text.split("::", 1)[-1]
    return text


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    m = pattern.search(text or "")
    if not m:
        return ""
    if m.lastindex and m.lastindex > 1:
        return "".join([g for g in m.groups() if g])
    return m.group(1) if m.lastindex else m.group(0)


def _normalize_source_type(evidence: Dict[str, Any]) -> str:
    st = str(evidence.get("source_type") or evidence.get("doc_type") or "").lower()
    if st in {"law", "statute", "laws"}:
        return "statute"
    if st in {"case", "judgment", "cases"}:
        return "case"
    if st in {"faq", "qa"}:
        return "faq"
    if evidence.get("evidence_type") == "graph":
        return "graph"
    if evidence.get("evidence_type") == "vector":
        return "vector"
    return st or "unknown"


def _infer_law_and_article(evidence: Dict[str, Any]) -> tuple[str, str]:
    law_name = str(evidence.get("law_name") or "")
    article_no = str(evidence.get("article_no") or evidence.get("article") or "")
    doc_law, doc_article = _infer_from_doc_id(str(evidence.get("doc_id", "")))
    if not law_name and doc_law:
        law_name = doc_law
    if not article_no and doc_article:
        article_no = doc_article
    text = " ".join([
        str(evidence.get("title") or ""),
        str(_clean_graph_label(str(evidence.get("source_name", "")))),
        str(_clean_graph_label(str(evidence.get("target_name", "")))),
        str(evidence.get("text") or evidence.get("evidence") or ""),
        str(evidence.get("source") or ""),
        str(evidence.get("target") or ""),
    ])
    if not article_no:
        m = LAW_ARTICLE_PATTERN.search(text)
        if m:
            prefix = m.group(1) or ""
            article_no = f"第{m.group(2)}条"
            if prefix:
                article_no = f"{prefix}{article_no}"
            if not law_name and prefix:
                law_name = f"中华人民共和国{prefix}" if not prefix.startswith("中华人民共和国") else prefix
    if not law_name and "刑法" in article_no:
        law_name = "中华人民共和国刑法"
    elif not law_name and "民法典" in article_no:
        law_name = "中华人民共和国民法典"
    elif not law_name and "公司法" in article_no:
        law_name = "中华人民共和国公司法"
    return law_name, article_no


def _build_evidence_id(evidence: Dict[str, Any]) -> str:
    key = "|".join(
        [
            str(evidence.get("source_type") or ""),
            str(evidence.get("doc_id") or ""),
            str(evidence.get("chunk_id") or ""),
            str(evidence.get("source") or ""),
            str(evidence.get("target") or ""),
            str(evidence.get("relation") or ""),
            str(evidence.get("text") or evidence.get("evidence") or "")[:160],
        ]
    )
    return hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_citation_metadata(evidence: Dict[str, Any]) -> Dict[str, Any]:
    base = empty_citation()
    text = str(evidence.get("text") or evidence.get("evidence") or "")
    law_name, article_no = _infer_law_and_article(evidence)

    source_type = _normalize_source_type(evidence)
    doc_id = str(evidence.get("doc_id") or "")
    chunk_id = str(evidence.get("chunk_id") or "")
    source_name = _clean_graph_label(str(evidence.get("source_name") or evidence.get("source") or ""))
    target_name = _clean_graph_label(str(evidence.get("target_name") or evidence.get("target") or ""))
    title = str(evidence.get("title") or doc_id or evidence.get("source") or "")
    if source_type == "graph" and not title:
        title = f"{source_name}->{target_name}".strip("->")
    if source_type == "graph" and title and "::" in title:
        title = f"{source_name}->{target_name}".strip("->")
    case_id = str(evidence.get("case_id") or _first_match(CASE_ID_PATTERN, text))
    court = str(evidence.get("court") or _first_match(COURT_PATTERN, text))
    section = str(evidence.get("section") or "")
    retrieval_score = float(evidence.get("score", evidence.get("retrieval_score", 0.0)) or 0.0)
    rerank_score = float(evidence.get("rerank_score", 0.0) or 0.0)

    base.update(
        {
            "evidence_id": str(evidence.get("evidence_id") or _build_evidence_id(evidence)),
            "source_type": source_type,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "title": title,
            "law_name": law_name,
            "article_no": article_no,
            "case_id": case_id,
            "court": court,
            "section": section,
            "text": text,
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score,
        }
    )

    if source_type == "faq":
        base["faq_id"] = str(evidence.get("faq_id") or doc_id)
        base["topic"] = str(evidence.get("topic") or evidence.get("section") or "")

    if settings.enable_llm_citation_validation:
        base = _llm_validate_and_enrich(base)

    return base


def _llm_validate_and_enrich(citation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        provider = get_llm_provider(timeout=12)
        prompt = (
            "你是法律引用元数据校验器。请基于给定JSON补全并纠正字段，严格返回JSON，不要解释。\n"
            "只允许字段: law_name, article_no, case_id, court, source_type, title。\n"
            "若无法判断请返回空字符串，不要编造。\n"
            f"输入JSON:\n{citation}"
        )
        parsed = provider.generate_json(prompt=prompt, temperature=0.0)
        if not parsed:
            return citation
        updated = dict(citation)
        for k in ["law_name", "article_no", "case_id", "court", "source_type", "title"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                updated[k] = v.strip()
        return updated
    except Exception:
        return citation


def attach_citation_metadata(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in evidence_list:
        citation = build_citation_metadata(ev)
        merged = dict(ev)
        merged.update(citation)
        merged["citation"] = citation
        out.append(merged)
    return out


def evidence_to_citation_item(evidence: Dict[str, Any], snippet_len: int = 120) -> Dict[str, Any]:
    citation = build_citation_metadata(evidence)
    return {
        "evidence_id": citation.get("evidence_id", ""),
        "source_type": citation.get("source_type", "unknown"),
        "title": citation.get("title", ""),
        "source": _clean_graph_label(str(evidence.get("source_name") or evidence.get("source", ""))),
        "target": _clean_graph_label(str(evidence.get("target_name") or evidence.get("target", ""))),
        "relation": str(evidence.get("relation", "")),
        "law_name": citation.get("law_name", ""),
        "article_no": citation.get("article_no", ""),
        "case_id": citation.get("case_id", ""),
        "court": citation.get("court", ""),
        "doc_id": citation.get("doc_id", ""),
        "chunk_id": citation.get("chunk_id", ""),
        "section": citation.get("section", ""),
        "snippet": str(citation.get("text", ""))[:snippet_len],
    }


def summarize_citations(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_count: Dict[str, int] = {}
    laws: List[str] = []
    cases: List[str] = []
    for c in citations:
        st = str(c.get("source_type", "unknown"))
        source_count[st] = source_count.get(st, 0) + 1
        law = f"{c.get('law_name', '')}{c.get('article_no', '')}".strip()
        if law and law not in laws:
            laws.append(law)
        case_key = c.get("case_id") or c.get("title")
        if case_key and case_key not in cases:
            cases.append(str(case_key))
    return {
        "citations_count": len(citations),
        "top_citation_sources": source_count,
        "major_laws": laws[:5],
        "major_cases": cases[:5],
    }
