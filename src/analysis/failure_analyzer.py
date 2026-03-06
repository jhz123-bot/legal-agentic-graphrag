from typing import Any, Dict, List


def analyze_failure(example: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    query = example.get("query", "")
    expected_entities = example.get("expected_entities", [])
    expected_keywords = example.get("expected_answer_keywords", [])

    linked_entities: List[str] = state.get("linked_entities", [])
    evidence_pack = state.get("evidence_pack", {})
    ranked = evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", []))
    reasoning_trace = state.get("reasoning_trace", {})
    verification = state.get("verification_result", {})
    final_answer = state.get("final_answer", {})

    def _contains_expected_entities() -> bool:
        if not expected_entities:
            return True
        lowers = [e.lower() for e in linked_entities]
        hits = 0
        for ent in expected_entities:
            if any(ent.lower() in got for got in lowers):
                hits += 1
        return hits >= max(1, len(expected_entities) // 2)

    def _contains_keywords() -> bool:
        if not expected_keywords:
            return True
        answer_blob = " ".join(
            [
                final_answer.get("short_answer", ""),
                final_answer.get("reasoning_summary", ""),
            ]
        ).lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer_blob)
        return hits >= max(1, len(expected_keywords) // 2)

    failure_type = "none"
    details = "no obvious failure"

    if not linked_entities:
        failure_type = "entity_linking_error"
        details = "no linked entities produced"
    elif not _contains_expected_entities():
        failure_type = "entity_linking_error"
        details = "expected entities not matched in linked entities"
    elif not ranked:
        failure_type = "retrieval_failure"
        details = "no relevant graph/vector evidence path found"
    elif ranked and all(float(x.get("score", 0.0)) < 0.2 for x in ranked[:3]):
        failure_type = "ranking_error"
        details = "top ranked evidence confidence is too low"
    elif not reasoning_trace.get("structured_steps"):
        failure_type = "reasoning_failure"
        details = "missing structured reasoning steps"
    elif verification.get("decision") in {"re-retrieve", "re-reason"} and int(verification.get("reflection_round", 0)) >= 1:
        failure_type = "reflection_failure"
        details = "reflection requested correction but outcome remains unstable"
    elif not _contains_keywords():
        failure_type = "reasoning_failure"
        details = "final answer misses expected legal keywords"

    return {
        "query": query,
        "failure_type": failure_type,
        "details": details,
    }
