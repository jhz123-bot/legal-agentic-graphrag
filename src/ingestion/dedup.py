from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[，。；：！？,.!?、（）()\-]", "", t)
    return t


def _char_jaccard(a: str, b: str) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class DedupResult:
    kept: List[Dict]
    removed: int
    reasons: Dict[str, int]


def dedup_statutes(rows: List[Dict]) -> DedupResult:
    kept: List[Dict] = []
    reasons = {"duplicate_article_no": 0, "duplicate_title": 0, "duplicate_content": 0}

    seen_article = set()
    seen_title = set()
    seen_content = set()

    for row in rows:
        article_no = _normalize_text(str(row.get("article_no", "")))
        title = _normalize_text(str(row.get("title", "")))
        content = _normalize_text(str(row.get("content", "")))

        if article_no and article_no in seen_article:
            reasons["duplicate_article_no"] += 1
            continue
        if title and title in seen_title:
            reasons["duplicate_title"] += 1
            continue
        if content and content in seen_content:
            reasons["duplicate_content"] += 1
            continue

        if article_no:
            seen_article.add(article_no)
        if title:
            seen_title.add(title)
        if content:
            seen_content.add(content)
        kept.append(row)

    return DedupResult(kept=kept, removed=len(rows) - len(kept), reasons=reasons)


def dedup_cases(rows: List[Dict]) -> DedupResult:
    kept: List[Dict] = []
    reasons = {"duplicate_case_id": 0, "duplicate_title": 0, "duplicate_content": 0}

    seen_case_id = set()
    seen_title = set()
    seen_content = set()

    for row in rows:
        case_id = _normalize_text(str(row.get("case_id", "")))
        title = _normalize_text(str(row.get("title", "")))
        content = _normalize_text(str(row.get("content", "")))

        if case_id and case_id in seen_case_id:
            reasons["duplicate_case_id"] += 1
            continue
        if title and title in seen_title:
            reasons["duplicate_title"] += 1
            continue
        if content and content in seen_content:
            reasons["duplicate_content"] += 1
            continue

        if case_id:
            seen_case_id.add(case_id)
        if title:
            seen_title.add(title)
        if content:
            seen_content.add(content)
        kept.append(row)

    return DedupResult(kept=kept, removed=len(rows) - len(kept), reasons=reasons)


def dedup_faqs(rows: List[Dict], near_dup_threshold: float = 0.985) -> DedupResult:
    kept: List[Dict] = []
    reasons = {"duplicate_question_exact": 0, "duplicate_question_near": 0}

    normalized_questions: List[Tuple[str, Dict]] = []

    for row in rows:
        q = _normalize_text(str(row.get("question", "")))
        if not q:
            kept.append(row)
            continue

        exact_dup = any(q == old_q for old_q, _ in normalized_questions)
        if exact_dup:
            reasons["duplicate_question_exact"] += 1
            continue

        near_dup = False
        topic = _normalize_text(str(row.get("topic", "")))
        for old_q, old_row in normalized_questions:
            old_topic = _normalize_text(str(old_row.get("topic", "")))
            if q in old_q or old_q in q:
                near_dup = True
                break
            # Strict near-duplicate policy: same topic + very high sequence similarity.
            if topic and old_topic and topic == old_topic:
                if SequenceMatcher(None, q, old_q).ratio() >= near_dup_threshold:
                    near_dup = True
                    break
            elif _char_jaccard(q, old_q) >= 0.995:
                near_dup = True
                break
        if near_dup:
            reasons["duplicate_question_near"] += 1
            continue

        normalized_questions.append((q, row))
        kept.append(row)

    return DedupResult(kept=kept, removed=len(rows) - len(kept), reasons=reasons)
