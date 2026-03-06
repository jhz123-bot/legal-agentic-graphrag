from typing import Any, Dict, List

from src.reasoning.reasoning_trace import build_reasoning_trace


def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence_pack = state.get("evidence_pack", {})
    nodes = evidence_pack.get("nodes", [])
    ranked_paths = state.get("ranked_evidence", evidence_pack.get("ranked_paths", []))
    plan = state.get("plan", {})

    steps: List[str] = []
    steps.append(f"目标：解决问题意图 '{plan.get('intent', 'unknown')}'。")
    steps.append(f"观察到 {len(nodes)} 个相关实体与 {len(ranked_paths)} 条排序后证据路径。")

    if nodes:
        top_names = ", ".join(node["name"] for node in nodes[:3])
        steps.append(f"核心实体：{top_names}。")
    if ranked_paths:
        relation_names = ", ".join(path.get("relation", "MENTIONED_WITH") for path in ranked_paths[:3])
        steps.append(f"主要关系模式：{relation_names}。")

    structured_steps = build_reasoning_trace(plan=plan, ranked_paths=ranked_paths)

    if nodes or ranked_paths:
        intermediate = "证据显示当前问题可基于已检索法律图谱进行回答。"
    else:
        intermediate = "证据不足，当前答案置信度较低，建议补充检索。"
        steps.append("检测到证据稀疏。")

    logs = list(state.get("logs", []))
    logs.append("reasoning: generated structured reasoning trace")
    return {
        "reasoning_trace": {
            "steps": steps,
            "structured_steps": structured_steps,
            "intermediate_conclusion": intermediate,
        },
        "logs": logs,
    }
