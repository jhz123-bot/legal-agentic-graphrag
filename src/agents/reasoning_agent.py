from typing import Any, Dict, List

from src.grounding.claim_mapper import map_claims_to_evidence
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt
from src.reasoning.reasoning_trace import build_reasoning_trace


def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    effective_query = state.get("rewritten_query") or state.get("user_query", "")
    evidence_pack = state.get("evidence_pack", {})
    nodes = evidence_pack.get("nodes", [])
    ranked_paths = state.get("ranked_evidence", evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", [])))
    plan = state.get("plan", {})

    structured_steps = build_reasoning_trace(plan=plan, ranked_paths=ranked_paths)
    intermediate = "证据显示该问题可基于当前检索结果进行回答。" if (nodes or ranked_paths) else "证据不足，建议补充检索或改写问题。"
    text_steps: List[str] = []
    llm_used = False
    conversation_context = state.get("conversation_context", [])
    conversation_summary = state.get("conversation_summary", "")
    fact_memory = state.get("fact_memory", {})

    evidence_snippets = []
    for i, p in enumerate(ranked_paths[:6], start=1):
        evidence_snippets.append(
            {
                "id": i,
                "relation": p.get("relation", ""),
                "source": p.get("source", ""),
                "target": p.get("target", ""),
                "evidence": str(p.get("evidence", ""))[:180],
            }
        )

    try:
        provider = get_llm_provider(timeout=20)
        prompt = render_prompt(
            "reasoning",
            query=effective_query,
            intent=plan.get("intent", "unknown"),
            evidence=evidence_snippets,
        )
        if conversation_summary:
            prompt += f"\n\n对话摘要：\n{conversation_summary}\n"
        if fact_memory:
            prompt += f"\n关键事实：\n{fact_memory}\n"
        if conversation_context:
            ctx = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in conversation_context[-8:]
            )
            prompt += f"\n最近对话上下文：\n{ctx}\n"
        parsed = provider.generate_json(prompt=prompt, temperature=0.0)
        if parsed:
            llm_steps = parsed.get("steps", [])
            if isinstance(llm_steps, list):
                text_steps = [str(s) for s in llm_steps if str(s).strip()][:5]
            if parsed.get("intermediate_conclusion"):
                intermediate = str(parsed.get("intermediate_conclusion"))
            llm_used = True
    except Exception:
        llm_used = False

    if not text_steps:
        text_steps.append(f"目标：解决问题意图 '{plan.get('intent', 'unknown')}'。")
        text_steps.append(f"观察到 {len(nodes)} 个相关实体与 {len(ranked_paths)} 条证据路径。")
        if nodes:
            top_names = ", ".join(node["name"] for node in nodes[:3])
            text_steps.append(f"核心实体：{top_names}。")
        if ranked_paths:
            rel = ", ".join(path.get("relation", "MENTIONED_WITH") for path in ranked_paths[:3])
            text_steps.append(f"高频关系：{rel}。")
        if not (nodes or ranked_paths):
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

    claim_mapping = map_claims_to_evidence(
        query=effective_query,
        intermediate_conclusion=intermediate,
        reasoning_steps=text_steps,
        evidence_list=ranked_paths,
        top_k=3,
    )
    structured_output["claims"] = claim_mapping.get("claims", [])
    structured_output["supporting_evidence_ids"] = claim_mapping.get("supporting_evidence_ids", [])
    structured_output["unsupported_claims"] = claim_mapping.get("unsupported_claims", [])

    logs = list(state.get("logs", []))
    logs.append(
        f"reasoning: structured_output_confidence={confidence}, llm_used={llm_used}, "
        f"context_turns={len(conversation_context)}, has_summary={bool(conversation_summary)}, "
        f"has_facts={bool(fact_memory)}, query_used={effective_query}"
    )
    logs.append(
        "reasoning_grounding: "
        f"claims={len(claim_mapping.get('claims', []))}, "
        f"evidence_ids_used={claim_mapping.get('supporting_evidence_ids', [])}, "
        f"unsupported_claim_count={len(claim_mapping.get('unsupported_claims', []))}"
    )
    return {
        "reasoning_trace": {
            "steps": text_steps,
            "structured_steps": structured_steps,
            "intermediate_conclusion": intermediate,
            "claims": claim_mapping.get("claims", []),
            "supporting_evidence_ids": claim_mapping.get("supporting_evidence_ids", []),
            "unsupported_claims": claim_mapping.get("unsupported_claims", []),
            "structured_output": structured_output,
            "confidence": confidence,
        },
        "logs": logs,
    }
