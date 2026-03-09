import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.ingestion.legal_text_cleaner import clean_legal_text


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    source_type: str
    law_name: str
    article_no: str
    case_id: str
    section: str
    article: str  # backward-compatible alias of article_no
    title: str
    text: str
    start_char: int
    end_char: int


def _detect_doc_type(doc: dict) -> str:
    doc_id = (doc.get("doc_id") or "").lower()
    text = doc.get("text", "")
    title = (doc.get("title") or "").lower()
    source_type = (doc.get("source_type") or "").lower()

    if source_type in {"law", "laws", "statute"}:
        return "law"
    if source_type in {"case", "cases", "judgment"}:
        return "case"
    if source_type in {"faq", "qa"}:
        return "faq"

    if "law" in doc_id or "law" in title or ("第" in text and "条" in text):
        return "law"
    if "case" in doc_id or "case" in title or any(k in text for k in ["案情", "法院认为", "本院认为", "裁判结果", "争议焦点"]):
        return "case"
    return "generic"


def _infer_law_name(doc: dict, text: str) -> str:
    if doc.get("law_name"):
        return str(doc["law_name"])
    if "中华人民共和国刑法" in text or "刑法" in text:
        return "中华人民共和国刑法"
    if "中华人民共和国民法典" in text or "民法典" in text:
        return "中华人民共和国民法典"
    if "中华人民共和国公司法" in text or "公司法" in text:
        return "中华人民共和国公司法"
    if "劳动合同法" in text:
        return "中华人民共和国劳动合同法"
    if "劳动争议调解仲裁法" in text:
        return "中华人民共和国劳动争议调解仲裁法"
    return ""


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    spans: List[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    step = chunk_size - chunk_overlap
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            spans.append((i, j, chunk))
        i += step
    return spans


def _structured_split_by_headers(text: str, headers: List[str]) -> List[Tuple[str, int, int, str]]:
    pattern = re.compile(r"(" + "|".join(re.escape(h) for h in headers) + r")(?:[:：])?")
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    sections: List[Tuple[str, int, int, str]] = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append(("案情", 0, matches[0].start(), prefix))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[start:end].strip()
        if sec_text:
            sections.append((m.group(1), start, end, sec_text))
    return sections


def _law_structured_chunks(text: str, law_name: str) -> List[Tuple[str, str, int, int, str]]:
    article_pattern = re.compile(r"(第[一二三四五六七八九十百千万零两〇0-9]+(?:条|款|项))")
    matches = list(article_pattern.finditer(text))
    if not matches:
        return []

    chunks: List[Tuple[str, str, int, int, str]] = []
    current_article = ""
    for i, m in enumerate(matches):
        head = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        if head.endswith("条"):
            current_article = head
            article = head
        elif current_article:
            article = current_article
        else:
            article = head

        if law_name:
            if "刑法" in law_name:
                article_no = f"刑法{article}"
            elif "民法典" in law_name:
                article_no = f"民法典{article}"
            elif "公司法" in law_name:
                article_no = f"公司法{article}"
            elif "劳动合同法" in law_name:
                article_no = f"劳动合同法{article}"
            elif "劳动争议调解仲裁法" in law_name:
                article_no = f"劳动争议调解仲裁法{article}"
            else:
                article_no = article
        else:
            article_no = article

        chunks.append((head, article_no, start, end, body))
    return chunks


def _case_structured_chunks(text: str) -> List[Tuple[str, str, int, int, str]]:
    headers = ["案情", "争议焦点", "法院认为", "本院认为", "裁判理由", "裁判结果", "裁判要旨", "判决结果", "审理查明"]
    sections = _structured_split_by_headers(text, headers)
    return [(sec, "", s, e, t) for sec, s, e, t in sections]


def _build_chunk(
    doc: dict,
    idx: int,
    doc_type: str,
    section: str,
    article_no: str,
    start: int,
    end: int,
    text: str,
    law_name: str,
) -> TextChunk:
    source_type = str(doc.get("source_type") or doc_type)
    case_id = str(doc.get("case_id") or (doc.get("doc_id") if doc_type == "case" else ""))
    return TextChunk(
        chunk_id=f"{doc['doc_id']}_{idx}",
        doc_id=doc["doc_id"],
        doc_type=doc_type,
        source_type=source_type,
        law_name=law_name,
        article_no=article_no,
        case_id=case_id,
        section=section,
        article=article_no,
        title=doc["title"],
        text=text,
        start_char=start,
        end_char=end,
    )


def chunk_documents(docs: List[dict], chunk_size: int = 400, chunk_overlap: int = 80) -> List[TextChunk]:
    chunks: List[TextChunk] = []

    for doc in docs:
        cleaned_text = clean_legal_text(doc.get("text", ""))
        doc_type = _detect_doc_type(doc)
        law_name = _infer_law_name(doc, cleaned_text)

        structured: List[Tuple[str, str, int, int, str]] = []
        if doc_type == "law":
            structured = _law_structured_chunks(cleaned_text, law_name=law_name)
        elif doc_type == "case":
            structured = _case_structured_chunks(cleaned_text)

        if structured:
            for idx, (section, article_no, start, end, sec_text) in enumerate(structured):
                chunks.append(
                    _build_chunk(
                        doc=doc,
                        idx=idx,
                        doc_type=doc_type,
                        section=section,
                        article_no=article_no,
                        start=start,
                        end=end,
                        text=sec_text,
                        law_name=law_name,
                    )
                )
            continue

        spans = chunk_text(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, (start, end, seg_text) in enumerate(spans):
            chunks.append(
                _build_chunk(
                    doc=doc,
                    idx=idx,
                    doc_type=doc_type,
                    section="sliding_window",
                    article_no="",
                    start=start,
                    end=end,
                    text=seg_text,
                    law_name=law_name,
                )
            )

    return chunks
