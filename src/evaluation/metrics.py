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


def compute_citation_correctness(citations: List[dict], evidence_pack: dict) -> float:
    if not citations:
        return 0.0
    valid_ids = set()
    for ev in evidence_pack.get("candidate_evidence", []):
        eid = ev.get("evidence_id")
        if eid:
            valid_ids.add(str(eid))
    for ev in evidence_pack.get("ranked_paths", []):
        eid = ev.get("evidence_id")
        if eid:
            valid_ids.add(str(eid))
    for ev in evidence_pack.get("reranked_paths", []):
        eid = ev.get("evidence_id")
        if eid:
            valid_ids.add(str(eid))

    if not valid_ids:
        return 0.0
    hits = 0
    for c in citations:
        eid = str(c.get("evidence_id", ""))
        if eid and eid in valid_ids:
            hits += 1
    return hits / len(citations)


def compute_citation_coverage(claims: List[dict], citations: List[dict], grounded_evidence: List[str] | None = None) -> float:
    if not claims:
        if grounded_evidence:
            citation_ids = {str(c.get("evidence_id", "")) for c in citations if c.get("evidence_id")}
            grounded_ids = {str(i) for i in grounded_evidence if i}
            if not grounded_ids:
                return 0.0
            return len(citation_ids.intersection(grounded_ids)) / len(grounded_ids)
        return 1.0
    citation_ids = {str(c.get("evidence_id", "")) for c in citations if c.get("evidence_id")}
    if not citation_ids:
        return 0.0
    covered = 0
    total = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        total += 1
        support_ids = {str(i) for i in claim.get("supporting_evidence_ids", []) if i}
        if support_ids and support_ids.intersection(citation_ids):
            covered += 1
            continue
        # fallback: lexical coverage between claim text and citation snippet/title
        claim_text = _norm(str(claim.get("claim", "")))
        if claim_text:
            for c in citations:
                blob = _norm(" ".join([str(c.get("snippet", "")), str(c.get("title", "")), str(c.get("law_name", "")), str(c.get("article_no", ""))]))
                if claim_text and any(tok and tok in blob for tok in claim_text.split()):
                    covered += 1
                    break
    if total == 0:
        if grounded_evidence:
            grounded_ids = {str(i) for i in grounded_evidence if i}
            if not grounded_ids:
                return 0.0
            return len(citation_ids.intersection(grounded_ids)) / len(grounded_ids)
        return 1.0
    return covered / total
