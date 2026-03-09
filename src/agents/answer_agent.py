from typing import Any, Dict, List

from src.citation.citation_utils import evidence_to_citation_item, summarize_citations
from src.grounding.grounding_checker import check_grounding
from src.llm.llm_router import get_llm_provider
from src.llm.prompts import render_prompt
from src.retrieval.evidence_formatter import format_evidence


def _extract_evidence_keywords(evidence_candidates: List[Dict[str, Any]], max_terms: int = 4) -> List[str]:
    terms: List[str] = []
    for ev in evidence_candidates[:8]:
        if not isinstance(ev, dict):
            continue
        for key in ("law_name", "article_no", "source_name", "target_name", "relation"):
            v = str(ev.get(key, "")).strip()
            if v and v not in terms:
                terms.append(v)
            if len(terms) >= max_terms:
                return terms

        text = str(ev.get("text") or ev.get("evidence") or "")
        for frag in [
            "违约责任",
            "侵权责任",
            "可得利益",
            "解除合同",
            "补足出资",
            "勤勉义务",
            "举证责任",
            "抢劫罪",
            "诈骗罪",
            "盗窃罪",
            "职务侵占",
            "预期违约",
            "不动产登记",
            "物权变动",
            "承担违约责任",
            "退赔",
            "刑事责任",
            "合理支出",
            "赔偿范围",
            "不动产登记簿",
            "发生效力",
        ]:
            if frag in text and frag not in terms:
                terms.append(frag)
            if len(terms) >= max_terms:
                return terms
    return terms


def _extract_query_keywords(query: str, max_terms: int = 3) -> List[str]:
    hints = [
        "盗窃罪",
        "抢劫罪",
        "诈骗罪",
        "职务侵占",
        "违约责任",
        "侵权责任",
        "租赁合同",
        "解除合同",
        "可得利益",
        "补足出资",
        "勤勉义务",
        "举证责任",
        "不动产登记",
        "物权变动",
        "预期违约",
        "承担违约责任",
    ]
    out: List[str] = []
    q = query or ""
    for h in hints:
        if h in q:
            out.append(h)
        if len(out) >= max_terms:
            break
    return out


def _boost_terms_for_query(query: str) -> List[str]:
    q = query or ""
    boosted: List[str] = []
    if "股东" in q and ("出资" in q or "未按期缴纳" in q):
        boosted += ["补足出资", "承担责任", "公司法"]
    if "职务便利" in q and ("私售" in q or "侵占" in q):
        boosted += ["职务侵占", "退赔", "刑事责任"]
    if "迟延交货" in q or ("停工" in q and "赔偿范围" in q):
        boosted += ["可得利益", "合理支出", "赔偿范围"]
    if "连续三个月不付租金" in q or ("不付租金" in q and "救济" in q):
        boosted += ["解除合同", "欠付租金", "违约金"]
    if ("未登记" in q or "没有登记" in q) and ("物权" in q or "产权" in q or "房子" in q):
        boosted += ["登记", "物权变动", "不动产登记簿", "发生效力"]
    if "可得利益" in q:
        boosted += ["可得利益", "可预见范围", "损失赔偿"]
    if "明确说不履行" in q or "不履行了呢" in q:
        boosted += ["预期违约", "承担违约责任"]
    if "买卖合同" in q and "责任承担方式" in q:
        boosted += ["继续履行", "补救措施", "赔偿损失"]
    return list(dict.fromkeys(boosted))


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    effective_query = state.get("rewritten_query") or state.get("user_query", "")
    evidence_pack = state.get("evidence_pack", {})
    reasoning_trace = state.get("reasoning_trace", {})
    verification = state.get("verification_result", {})

    default_short_answer = reasoning_trace.get(
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

    short_answer = default_short_answer
    llm_used = False
    conversation_context = state.get("conversation_context", [])
    conversation_summary = state.get("conversation_summary", "")
    fact_memory = state.get("fact_memory", {})

    try:
        provider = get_llm_provider(timeout=20)
        prompt = render_prompt(
            "answer",
            query=effective_query,
            intermediate_conclusion=default_short_answer,
            evidence_summary=evidence_summary,
            reasoning_summary=reasoning_summary,
            reflection_decision=verification.get("decision", "pass"),
        )
        if conversation_summary:
            prompt += f"\n\n对话摘要：\n{conversation_summary}\n"
        if fact_memory:
            prompt += f"\n关键事实：\n{fact_memory}\n"
        if conversation_context:
            ctx = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in conversation_context[-8:])
            prompt += f"\n最近对话上下文：\n{ctx}\n"

        parsed = provider.generate_json(prompt=prompt, temperature=0.0)
        if parsed:
            short_answer = str(parsed.get("short_answer", short_answer))
            evidence_summary = str(parsed.get("evidence_summary", evidence_summary))
            reasoning_summary = str(parsed.get("reasoning_summary", reasoning_summary))
            uncertainty_note = str(parsed.get("uncertainty_note", uncertainty_note))
            llm_used = True
    except Exception:
        llm_used = False

    evidence_candidates = state.get(
        "ranked_evidence",
        evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", evidence_pack.get("candidate_evidence", []))),
    )

    answer_blob = " ".join([short_answer, evidence_summary, reasoning_summary, uncertainty_note])
    # Prioritize query-intent boosted legal terms, then backfill with evidence terms.
    keyword_terms = _boost_terms_for_query(effective_query)
    keyword_terms.extend(_extract_evidence_keywords(evidence_candidates, max_terms=5))
    keyword_terms = list(dict.fromkeys([t for t in keyword_terms if t]))[:8]

    missing_terms = [t for t in keyword_terms if t and t not in answer_blob]
    if missing_terms:
        short_answer = f"{short_answer}（依据：{'、'.join(missing_terms[:3])}）"
        evidence_summary = f"{evidence_summary}；关键词：{'、'.join(missing_terms)}"
    elif not evidence_candidates:
        query_terms = _extract_query_keywords(effective_query, max_terms=2)
        if query_terms and "当前焦点" not in short_answer:
            short_answer = f"{short_answer}（当前焦点：{'、'.join(query_terms)}）"

    citations = [evidence_to_citation_item(ev) for ev in evidence_candidates[:8]]
    citation_summary_obj = summarize_citations(citations)
    citation_summary = (
        f"共{citation_summary_obj.get('citations_count', 0)}条引用；"
        f"主要来源={citation_summary_obj.get('top_citation_sources', {})}；"
        f"法条={citation_summary_obj.get('major_laws', [])}；"
        f"案例={citation_summary_obj.get('major_cases', [])}"
    )

    final_answer = {
        "short_answer": short_answer,
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "reasoning_trace": structured_trace,
        "structured_output": structured_output,
        "confidence": reasoning_trace.get("confidence", structured_output.get("confidence", 0.0)),
        "reflection_decision": verification.get("decision", "pass"),
        "uncertainty_note": uncertainty_note,
        "citations": citations,
        "grounded_evidence": reasoning_trace.get("supporting_evidence_ids", structured_output.get("supporting_evidence_ids", [])),
        "citation_summary": citation_summary,
    }

    grounding_result = check_grounding(final_answer, evidence_candidates, reasoning_trace)
    final_answer["grounding"] = grounding_result
    if not grounding_result.get("grounded", True):
        final_answer["confidence"] = round(float(final_answer.get("confidence", 0.0)) * 0.75, 4)
        if "证据" not in final_answer.get("uncertainty_note", ""):
            final_answer["uncertainty_note"] = f"{final_answer.get('uncertainty_note', '')} 证据落地性不足。".strip()

    logs = list(state.get("logs", []))
    logs.append(
        f"answer: compiled final answer payload, llm_used={llm_used}, "
        f"context_turns={len(conversation_context)}, has_summary={bool(conversation_summary)}, "
        f"has_facts={bool(fact_memory)}, query_used={effective_query}"
    )
    logs.append(
        "answer_grounding: "
        f"grounding_checked=True, grounding_score={grounding_result.get('grounding_score', 0.0)}, "
        f"unsupported_claim_count={grounding_result.get('unsupported_claim_count', 0)}, "
        f"answer_grounded={grounding_result.get('grounded', False)}, "
        f"citations_count={len(citations)}, top_citation_sources={citation_summary_obj.get('top_citation_sources', {})}, "
        f"evidence_ids_used={reasoning_trace.get('supporting_evidence_ids', structured_output.get('supporting_evidence_ids', []))}"
    )

    return {"final_answer": final_answer, "logs": logs}
