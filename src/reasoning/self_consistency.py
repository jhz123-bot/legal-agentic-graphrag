from collections import Counter
from typing import Dict, List


def generate_reasoning_paths(reasoning_trace: Dict, evidence_paths: List[Dict], num_paths: int = 3) -> List[str]:
    base = reasoning_trace.get("intermediate_conclusion", "")
    if not base:
        base = "当前证据不足，无法稳定回答。"
    paths: List[str] = []
    for i in range(max(1, num_paths)):
        if i < len(evidence_paths):
            ev = evidence_paths[i].get("relation", "UNKNOWN")
            paths.append(f"{base} [path{i+1}:{ev}]")
        else:
            paths.append(base)
    return paths


def select_consensus_answer(candidates: List[str]) -> Dict:
    counter = Counter(candidates)
    answer, count = counter.most_common(1)[0]
    confidence = count / len(candidates)
    return {
        "candidates": candidates,
        "consensus_answer": answer,
        "vote_count": count,
        "confidence": round(confidence, 4),
    }


def self_consistency_node(state: Dict) -> Dict:
    reasoning_trace = state.get("reasoning_trace", {})
    evidence_paths = state.get("ranked_evidence", [])
    candidates = generate_reasoning_paths(reasoning_trace, evidence_paths, num_paths=3)
    result = select_consensus_answer(candidates)

    updated_trace = dict(reasoning_trace)
    updated_trace["self_consistency"] = result
    updated_trace["intermediate_conclusion"] = result["consensus_answer"]
    updated_trace["confidence"] = result["confidence"]

    logs = list(state.get("logs", []))
    logs.append(f"self_consistency: confidence={result['confidence']}")
    return {"reasoning_trace": updated_trace, "self_consistency_result": result, "logs": logs}
