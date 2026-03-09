from typing import Any, Dict, List

from src.agents.reflection_policy import evaluate_reflection_policy
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt


def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("plan", {})
    evidence_pack = state.get("evidence_pack", {})
    reasoning_trace = state.get("reasoning_trace", {})
    linked_entities = state.get("linked_entities", [])
    reflection_round = int(state.get("reflection_round", 0))

    missing_targets: List[str] = []
    for target in plan.get("target_entities", []):
        if not any(target.lower() in ent.lower() for ent in linked_entities):
            missing_targets.append(target)

    evidence_sufficient = len(evidence_pack.get("nodes", [])) >= 1 and len(
        evidence_pack.get("ranked_paths", evidence_pack.get("edges", []))
    ) >= 1
    reasoning_coherent = bool(reasoning_trace.get("steps")) and bool(reasoning_trace.get("intermediate_conclusion"))

    policy = evaluate_reflection_policy(
        target_entities=plan.get("target_entities", []),
        linked_entities=linked_entities,
        evidence_pack=evidence_pack,
        reasoning_trace=reasoning_trace,
        threshold=0.6,
        min_steps=2,
    )
    rule_decision = policy.get("decision", "pass")
    unsupported_claims = state.get("reasoning_trace", {}).get("unsupported_claims", [])
    if unsupported_claims and rule_decision == "pass":
        # If reasoning has explicit unsupported claims, prefer another retrieval pass.
        rule_decision = "re-retrieve"

    llm_decision = ""
    llm_used = False
    try:
        provider = get_llm_provider(timeout=20)
        prompt = render_prompt(
            "reflection",
            query=state.get("user_query", ""),
            target_entities=plan.get("target_entities", []),
            linked_entities=linked_entities,
            evidence_stats={
                "nodes": len(evidence_pack.get("nodes", [])),
                "ranked_paths": len(state.get("ranked_evidence", [])),
            },
            reasoning_conclusion=reasoning_trace.get("intermediate_conclusion", ""),
        )
        parsed = provider.generate_json(prompt=prompt, temperature=0.0)
        cand = str(parsed.get("decision", "")).strip()
        if cand in {"pass", "re-retrieve", "re-reason"}:
            llm_decision = cand
            llm_used = True
    except Exception:
        llm_used = False

    decision = llm_decision or rule_decision
    if reflection_round >= 1 and decision in {"re-retrieve", "re-reason"}:
        decision = "pass"

    verification_result = {
        "decision": decision,
        "evidence_sufficient": evidence_sufficient,
        "reasoning_coherent": reasoning_coherent,
        "missing_targets": missing_targets,
        "policy": policy,
        "reflection_round": reflection_round,
        "llm_used": llm_used,
        "rule_decision": rule_decision,
    }

    logs = list(state.get("logs", []))
    logs.append(
        f"reflection: decision={decision}, "
        f"rule_decision={rule_decision}, "
        f"confidence={policy.get('confidence_score')}, "
        f"missing_targets={missing_targets}, unsupported_claim_count={len(unsupported_claims)}, "
        f"llm_used={llm_used}"
    )
    return {
        "verification_result": verification_result,
        "reflection_round": reflection_round + 1,
        "logs": logs,
    }


def reflection_router(state: Dict[str, Any]) -> str:
    decision = state.get("verification_result", {}).get("decision", "pass")
    if decision == "re-retrieve":
        return "retrieval"
    if decision == "re-reason":
        return "reasoning"
    return "answer"
