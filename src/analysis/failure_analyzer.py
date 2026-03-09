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
    details = "未发现明显失败"

    if not linked_entities:
        failure_type = "entity_linking_error"
        details = "未产生有效实体链接结果"
    elif not _contains_expected_entities():
        failure_type = "entity_linking_error"
        details = "期望实体未在已链接实体中命中"
    elif not ranked:
        failure_type = "retrieval_failure"
        details = "未检索到相关图谱或向量证据路径"
    elif ranked and all(float(x.get("score", 0.0)) < 0.2 for x in ranked[:3]):
        failure_type = "ranking_error"
        details = "排序后高位证据分数偏低，可信度不足"
    elif not reasoning_trace.get("structured_steps"):
        failure_type = "reasoning_failure"
        details = "缺少结构化推理步骤"
    elif verification.get("decision") in {"re-retrieve", "re-reason"} and int(verification.get("reflection_round", 0)) >= 1:
        failure_type = "reflection_failure"
        details = "反思已触发纠偏但结果仍不稳定"
    elif not _contains_keywords():
        failure_type = "reasoning_failure"
        details = "最终回答未覆盖关键法律术语"

    return {
        "query": query,
        "failure_type": failure_type,
        "details": details,
    }
