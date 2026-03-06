from typing import Dict, List


def evaluate_reflection_policy(
    target_entities: List[str],
    linked_entities: List[str],
    evidence_pack: Dict,
    reasoning_trace: Dict,
    threshold: float = 0.6,
    min_steps: int = 2,
) -> Dict:
    if target_entities:
        hits = 0
        for t in target_entities:
            if any(t.lower() in e.lower() for e in linked_entities):
                hits += 1
        entity_coverage = hits / len(target_entities)
    else:
        entity_coverage = 1.0

    evidence_count = len(evidence_pack.get("ranked_paths", evidence_pack.get("edges", [])))
    evidence_consistency = min(1.0, evidence_count / 5.0)

    structured_steps = reasoning_trace.get("structured_steps", [])
    reasoning_depth = min(1.0, len(structured_steps) / 3.0)

    confidence_score = 0.4 * entity_coverage + 0.3 * evidence_consistency + 0.3 * reasoning_depth

    decision = "pass"
    if confidence_score < threshold:
        decision = "re-retrieve"
    elif len(structured_steps) < min_steps:
        decision = "re-reason"

    return {
        "confidence_score": round(confidence_score, 4),
        "entity_coverage": round(entity_coverage, 4),
        "evidence_consistency": round(evidence_consistency, 4),
        "reasoning_depth": round(reasoning_depth, 4),
        "decision": decision,
    }
