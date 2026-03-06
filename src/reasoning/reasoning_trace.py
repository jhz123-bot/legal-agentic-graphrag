from typing import Dict, List


def build_reasoning_trace(plan: Dict, ranked_paths: List[Dict]) -> List[Dict]:
    trace: List[Dict] = []
    intent = plan.get("intent", "unknown")

    for idx, path in enumerate(ranked_paths[:3], start=1):
        evidence = path.get("evidence", "")
        relation = path.get("relation", "MENTIONED_WITH")
        target = path.get("target", "")
        conclusion = f"基于关系 {relation}，可推断 {target} 与问题意图 {intent} 相关。"
        trace.append(
            {
                "step": idx,
                "evidence": evidence,
                "relation": relation,
                "conclusion": conclusion,
            }
        )

    if not trace:
        trace.append(
            {
                "step": 1,
                "evidence": "未检索到足够证据",
                "relation": "NONE",
                "conclusion": "当前证据不足，需补充检索或重新推理。",
            }
        )
    return trace
