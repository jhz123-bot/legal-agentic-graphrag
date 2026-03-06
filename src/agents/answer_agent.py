from typing import Any, Dict

from src.retrieval.evidence_formatter import format_evidence


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence_pack = state.get("evidence_pack", {})
    reasoning_trace = state.get("reasoning_trace", {})
    verification = state.get("verification_result", {})

    short_answer = reasoning_trace.get(
        "intermediate_conclusion",
        "证据不足，暂时无法给出高置信度法律答案。",
    )
    evidence_summary = format_evidence(evidence_pack) if evidence_pack else "无可用证据。"
    reasoning_summary = " | ".join(reasoning_trace.get("steps", [])[:3]) or "无显式推理步骤。"
    structured_trace = reasoning_trace.get("structured_steps", [])
    structured_output = reasoning_trace.get("structured_output", {})

    uncertainty_note = "不确定性较低。"
    if not verification.get("evidence_sufficient", False):
        uncertainty_note = "不确定性较高：检索证据可能不完整。"
    elif verification.get("missing_targets"):
        uncertainty_note = "中等不确定性：部分目标实体未被链接。"

    final_answer = {
        "short_answer": short_answer,
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "reasoning_trace": structured_trace,
        "structured_output": structured_output,
        "confidence": reasoning_trace.get("confidence", structured_output.get("confidence", 0.0)),
        "reflection_decision": verification.get("decision", "pass"),
        "uncertainty_note": uncertainty_note,
    }

    logs = list(state.get("logs", []))
    logs.append("answer: compiled final answer payload")
    return {"final_answer": final_answer, "logs": logs}
