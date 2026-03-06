from typing import Any, Dict, List

from src.reasoning.reasoning_trace import build_reasoning_trace


def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence_pack = state.get("evidence_pack", {})
    nodes = evidence_pack.get("nodes", [])
    ranked_paths = state.get("ranked_evidence", evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", [])))
    plan = state.get("plan", {})

    text_steps: List[str] = []
    text_steps.append(f"目标：解决问题意图 '{plan.get('intent', 'unknown')}'。")
    text_steps.append(f"观察到 {len(nodes)} 个相关实体与 {len(ranked_paths)} 条证据路径。")

    if nodes:
        top_names = ", ".join(node["name"] for node in nodes[:3])
        text_steps.append(f"核心实体：{top_names}。")

    if ranked_paths:
        rel = ", ".join(path.get("relation", "MENTIONED_WITH") for path in ranked_paths[:3])
        text_steps.append(f"高频关系：{rel}。")

    structured_steps = build_reasoning_trace(plan=plan, ranked_paths=ranked_paths)

    if nodes or ranked_paths:
        intermediate = "证据显示该问题可基于当前检索结果进行回答。"
    else:
        intermediate = "证据不足，建议补充检索或改写问题。"
        text_steps.append("检测到证据稀疏。")

    entities = [node.get("name", "") for node in nodes]
    evidence_paths = [
        {
            "source": p.get("source"),
            "target": p.get("target"),
            "relation": p.get("relation"),
            "score": p.get("score", p.get("rerank_score", 0.0)),
        }
        for p in ranked_paths
    ]

    confidence = 0.3
    if entities:
        confidence += 0.2
    if len(evidence_paths) >= 3:
        confidence += 0.3
    elif evidence_paths:
        confidence += 0.15
    confidence = min(0.95, round(confidence, 4))

    structured_output = {
        "entities": entities,
        "evidence": evidence_paths,
        "reasoning_steps": structured_steps,
        "confidence": confidence,
    }

    logs = list(state.get("logs", []))
    logs.append(f"reasoning: structured_output_confidence={confidence}")
    return {
        "reasoning_trace": {
            "steps": text_steps,
            "structured_steps": structured_steps,
            "intermediate_conclusion": intermediate,
            "structured_output": structured_output,
            "confidence": confidence,
        },
        "logs": logs,
    }
