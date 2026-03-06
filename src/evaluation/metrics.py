from typing import Iterable, List


def _norm(text: str) -> str:
    return text.lower().strip()


def compute_entity_hit_rate(expected_entities: List[str], linked_entities: List[str]) -> float:
    if not expected_entities:
        return 1.0
    linked_norm = [_norm(ent) for ent in linked_entities]
    hits = 0
    for expected in expected_entities:
        e = _norm(expected)
        if any(e in candidate for candidate in linked_norm):
            hits += 1
    return hits / len(expected_entities)


def compute_evidence_path_hit_rate(expected_paths: List[str], evidence_pack: dict) -> float:
    if not expected_paths:
        return 1.0
    haystack: List[str] = []
    for node in evidence_pack.get("nodes", []):
        haystack.append(_norm(node.get("name", "")))
        haystack.extend(_norm(m) for m in node.get("mentions", []))
    for edge in evidence_pack.get("ranked_paths", evidence_pack.get("edges", [])):
        haystack.append(_norm(edge.get("source", "")))
        haystack.append(_norm(edge.get("target", "")))
        haystack.append(_norm(edge.get("relation", "")))
        haystack.append(_norm(edge.get("evidence", "")))

    hits = 0
    for expected in expected_paths:
        e = _norm(expected)
        if any(e in item for item in haystack):
            hits += 1
    return hits / len(expected_paths)


def compute_answer_keyword_match_rate(expected_keywords: Iterable[str], answer_text: str) -> float:
    expected_keywords = list(expected_keywords)
    if not expected_keywords:
        return 1.0
    answer_norm = _norm(answer_text)
    hits = sum(1 for keyword in expected_keywords if _norm(keyword) in answer_norm)
    return hits / len(expected_keywords)


def compute_reflection_trigger_rate(triggered_flags: List[bool]) -> float:
    if not triggered_flags:
        return 0.0
    return sum(1 for flag in triggered_flags if flag) / len(triggered_flags)
