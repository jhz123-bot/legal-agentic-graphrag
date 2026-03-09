from __future__ import annotations

import argparse
from pathlib import Path

from run_demo import build_hybrid_components, load_env_file
from src.agents.workflow import run_agentic_graphrag
from src.config.settings import settings
from src.conversation.context_builder import build_context_for_reasoning, build_context_for_rewrite
from src.feedback.runtime_feedback import record_runtime_failure
from src.memory.memory_manager import MemoryManager
from src.utils.encoding_utils import configure_stdio_encoding, safe_print


def _print_round_output(user_query: str, state: dict, memory_state: dict, append_result: dict) -> None:
    final_answer = state.get("final_answer", {}) or {}
    rewrite_info = state.get("rewrite_info", {}) or {}
    rewrite_decision = state.get("rewrite_decision", {}) or {}
    rewritten_query = state.get("rewritten_query", user_query)

    answer_text = final_answer.get("short_answer", "未生成回答。")
    confidence = final_answer.get("confidence", 0.0)
    evidence_summary = final_answer.get("evidence_summary", "")
    citations = final_answer.get("citations", [])
    grounding = final_answer.get("grounding", {})

    safe_print(f"\n用户问题：{user_query}")
    safe_print(
        "改写决策："
        f"need_rewrite={rewrite_decision.get('need_rewrite', False)}, "
        f"reason={rewrite_decision.get('reason', '')}, "
        f"signals={rewrite_decision.get('signals', [])}"
    )
    if rewrite_info.get("triggered") and rewritten_query != user_query:
        safe_print(f"改写后问题：{rewritten_query}")
        safe_print(f"改写方式：{rewrite_info.get('method', 'unknown')}")

    safe_print(f"最终回答：{answer_text}")
    safe_print(f"置信度：{confidence}")
    if evidence_summary:
        safe_print(f"证据摘要：{evidence_summary}")
    if citations:
        safe_print(f"引用数量：{len(citations)}")
        for c in citations[:3]:
            display_ref = c.get("case_id") or c.get("title") or f"{c.get('source', '')}->{c.get('target', '')}"
            safe_print(
                f"- [{c.get('source_type')}] {c.get('law_name', '')}{c.get('article_no', '')} "
                f"{display_ref} ({c.get('chunk_id', '')})"
            )
        citation_summary = final_answer.get("citation_summary", "")
        if citation_summary:
            safe_print(f"引用摘要：{citation_summary}")
    safe_print(
        f"Grounding：grounded={grounding.get('grounded', False)}, "
        f"score={grounding.get('grounding_score', 0.0)}, "
        f"unsupported_claim_count={grounding.get('unsupported_claim_count', 0)}"
    )

    safe_print(
        "记忆状态："
        f"short_term_turn_count={memory_state.get('short_term_turn_count', 0)}, "
        f"short_term_tokens={memory_state.get('short_term_tokens', 0)}, "
        f"has_conflict={memory_state.get('has_conflict', False)}, "
        f"conflict_fields={memory_state.get('conflict_fields', [])}, "
        f"fact_extraction_mode={memory_state.get('fact_extraction_mode', 'rule')}, "
        f"summary_build_mode={memory_state.get('summary_build_mode', 'rule')}, "
        f"extraction_confidence={memory_state.get('extraction_confidence', 0.0)}"
    )

    safe_print(
        "记忆日志："
        f"summary_updated={append_result.get('summary_updated', False)}, "
        f"summary_trigger_reason={append_result.get('summary_trigger_reason', 'none')}, "
        f"summary_build_mode={append_result.get('summary_build_mode', 'rule')}, "
        f"fact_extraction_mode={append_result.get('fact_extraction_mode', 'rule')}, "
        f"fact_extraction_confidence={append_result.get('fact_extraction_confidence', 0.0)}, "
        f"summary_extraction_confidence={append_result.get('summary_extraction_confidence', 0.0)}, "
        f"fact_updates={append_result.get('fact_updates', {})}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Agentic GraphRAG Multi-turn Chat")
    parser.add_argument("--docs", type=str, default="data/legal_docs", help="文档目录，支持 TXT/PDF/DOCX")
    args = parser.parse_args()

    if settings.enable_encoding_fix:
        configured, current_encoding = configure_stdio_encoding()
        safe_print(
            f"[encoding] stdio_encoding_configured={configured}, current_stdout_encoding={current_encoding}"
        )

    root = Path(__file__).parent
    load_env_file(root / ".env")
    docs_dir = (root / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)

    safe_print("初始化知识库中...")
    graph_store, hybrid_retriever, doc_count, chunk_count, *_ = build_hybrid_components(docs_dir)
    safe_print(f"已加载文档 {doc_count} 份，分块 {chunk_count} 个。")
    safe_print("进入多轮问答。输入 exit 或 quit 退出。")

    memory_manager = MemoryManager()

    while True:
        try:
            user_query = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("\n会话结束。")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            safe_print("会话结束。")
            break

        rewrite_ctx = build_context_for_rewrite(memory_manager)
        reasoning_ctx = build_context_for_reasoning(memory_manager)

        state = run_agentic_graphrag(
            query=user_query,
            graph_store=graph_store,
            hybrid_retriever=hybrid_retriever,
            conversation_history=rewrite_ctx.get("recent_history", []),
            conversation_context=rewrite_ctx.get("context_window", []),
            conversation_summary=reasoning_ctx.get("summary", ""),
            fact_memory=reasoning_ctx.get("facts", {}),
            short_term_memory=rewrite_ctx.get("recent_history", []),
            summary_memory=reasoning_ctx.get("summary", ""),
            memory_state=memory_manager.get_memory_state(),
        )

        if settings.enable_runtime_feedback:
            record_runtime_failure(
                query=user_query,
                trace=state,
                output_root=root / "outputs" / "feedback",
                metadata={"entry": "run_chat"},
            )

        final_answer = state.get("final_answer", {}) or {}
        assistant_text = final_answer.get("short_answer", "未生成回答。")

        append_result = memory_manager.append_turn(user_query, assistant_text)
        memory_state = memory_manager.get_memory_state()

        _print_round_output(user_query, state, memory_state, append_result)


if __name__ == "__main__":
    main()
