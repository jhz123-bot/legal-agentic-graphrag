from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.answer_agent import answer_node
from src.agents.planner_agent import planner_node, query_decomposition_node
from src.agents.reasoning_agent import reasoning_node
from src.agents.reflection_agent import reflection_node, reflection_router
from src.agents.retrieval_agent import make_retrieval_node
from src.common.models import Document
from src.graph.entity_linker import EntityLinker
from src.graph.graph_builder import GraphBuilder
from src.graph.store import InMemoryGraphStore
from src.reasoning.react_loop import make_tool_calling_node, react_loop_node
from src.reasoning.self_consistency import self_consistency_node
from src.retrieval.evidence_ranker import UnifiedEvidenceRanker, make_evidence_ranking_node
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import EvidenceReranker, make_reranker_node
from src.router.agent_router import router_node
from src.tools.case_lookup import CaseLookupTool
from src.tools.graph_search import GraphSearchTool
from src.tools.legal_article_lookup import LegalArticleLookupTool
from src.tools.vector_search import VectorSearchTool
from src.conversation.ellipsis_detector import should_rewrite
from src.conversation.query_rewriter import rewrite_query


class AgenticGraphRAGState(TypedDict, total=False):
    user_query: str
    conversation_history: List[Dict[str, str]]
    conversation_context: List[Dict[str, str]]
    conversation_summary: str
    fact_memory: Dict[str, Any]
    short_term_memory: List[Dict[str, str]]
    summary_memory: str
    memory_state: Dict[str, Any]
    rewrite_decision: Dict[str, Any]
    rewritten_query: str
    rewrite_info: Dict[str, Any]
    router_decision: Dict[str, Any]
    retrieval_strategy: str
    workflow_branch: str
    decomposition: Dict[str, Any]
    subqueries: List[str]
    react_plan: List[Dict[str, Any]]
    react_trace: List[Dict[str, str]]
    tool_results: List[Dict[str, Any]]
    linked_entities: List[str]
    plan: Dict[str, Any]
    evidence_pack: Dict[str, Any]
    candidate_evidence: List[Dict[str, Any]]
    ranked_evidence: List[Dict[str, Any]]
    reasoning_trace: Dict[str, Any]
    self_consistency_result: Dict[str, Any]
    verification_result: Dict[str, Any]
    final_answer: Dict[str, Any]
    logs: List[str]
    reflection_round: int


def _strategy_branch_router(state: AgenticGraphRAGState) -> str:
    strategy = (
        state.get("retrieval_strategy")
        or state.get("plan", {}).get("retrieval_strategy")
        or state.get("router_decision", {}).get("retrieval_strategy")
        or "graph"
    )
    if strategy == "direct_answer":
        return "direct"
    if strategy == "vector":
        return "vector"
    if strategy == "hybrid":
        return "hybrid"
    return "graph"


def _query_rewrite_node(state: AgenticGraphRAGState) -> Dict[str, Any]:
    user_query = state.get("user_query", "")
    context = state.get("conversation_history", []) or state.get("conversation_context", [])
    summary = state.get("conversation_summary", "")
    facts = state.get("fact_memory", {})
    rewrite_decision = should_rewrite(query=user_query, history=context, facts=facts, summary=summary)
    rewrite = rewrite_query(
        query=user_query,
        history=context,
        summary=summary,
        facts=facts,
        rewrite_decision=rewrite_decision,
    )
    rewritten_query = str(rewrite.get("rewritten_query", user_query)).strip() or user_query
    logs = list(state.get("logs", []))
    logs.append(
        "rewrite_gate: "
        f"need_rewrite={rewrite_decision.get('need_rewrite', False)}, "
        f"reason={rewrite_decision.get('reason', 'unknown')}, "
        f"signals={rewrite_decision.get('signals', [])}"
    )
    logs.append(
        "query_rewrite: "
        f"triggered={rewrite.get('triggered', False)}, "
        f"method={rewrite.get('method', 'none')}, "
        f"context_sources={rewrite.get('used_context_parts', [])}, "
        f"original={user_query}, "
        f"rewritten={rewritten_query}"
    )
    return {
        "rewrite_decision": rewrite_decision,
        "rewritten_query": rewritten_query,
        "rewrite_info": rewrite,
        "logs": logs,
    }


def _set_graph_branch(_: AgenticGraphRAGState) -> Dict[str, Any]:
    return {"retrieval_strategy": "graph", "workflow_branch": "graph"}


def _set_vector_branch(_: AgenticGraphRAGState) -> Dict[str, Any]:
    return {"retrieval_strategy": "vector", "workflow_branch": "vector"}


def _set_hybrid_branch(_: AgenticGraphRAGState) -> Dict[str, Any]:
    return {"retrieval_strategy": "hybrid", "workflow_branch": "hybrid"}


def _direct_answer_prep_node(state: AgenticGraphRAGState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append("branch: direct_answer, skip retrieval/reasoning/reflection")
    return {
        "retrieval_strategy": "direct_answer",
        "workflow_branch": "direct",
        "linked_entities": [],
        "evidence_pack": {"query_mentions": [], "nodes": [], "edges": [], "vector_hits": [], "ranked_paths": []},
        "ranked_evidence": [],
        "reasoning_trace": {
            "steps": ["该问题命中 direct_answer 分支，未触发检索。"],
            "structured_steps": [],
            "intermediate_conclusion": "该问题为通用咨询，已走直接回答路径。",
            "structured_output": {"entities": [], "evidence": [], "reasoning_steps": [], "confidence": 0.35},
            "confidence": 0.35,
        },
        "verification_result": {
            "decision": "pass",
            "evidence_sufficient": True,
            "reasoning_coherent": True,
            "missing_targets": [],
            "policy": {"confidence_score": 0.35},
            "reflection_round": state.get("reflection_round", 0),
        },
        "logs": logs,
    }


def _load_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        docs.append(Document(doc_id=path.stem, title=path.stem, text=path.read_text(encoding="utf-8")))
    return docs


def build_graph_store(data_dir: Path) -> InMemoryGraphStore:
    docs = _load_documents(data_dir)
    return build_graph_store_from_documents(docs)


def build_graph_store_from_documents(docs: List[Document]) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    graph_store = builder.build(docs)
    setattr(graph_store, "build_debug_stats", builder.debug_stats)
    linker = EntityLinker()
    linker.link(graph_store)
    return graph_store


def _build_tools(graph_store: InMemoryGraphStore, hybrid_retriever=None) -> Dict[str, Any]:
    graph_retriever = GraphRetriever(graph_store)
    tools: Dict[str, Any] = {
        "graph_search": GraphSearchTool(graph_retriever),
        "case_lookup": CaseLookupTool(graph_retriever),
        "legal_article_lookup": LegalArticleLookupTool(graph_retriever),
    }
    if hybrid_retriever is not None:
        tools["vector_search"] = VectorSearchTool(
            embedding_model=hybrid_retriever.embedding_model,
            vector_index=hybrid_retriever.vector_index,
        )
    return tools


def create_agentic_app(
    data_dir: Path | None = None,
    graph_store: InMemoryGraphStore | None = None,
    hybrid_retriever=None,
):
    effective_graph_store = graph_store
    if effective_graph_store is None:
        if data_dir is None:
            raise ValueError("data_dir or graph_store must be provided")
        effective_graph_store = build_graph_store(data_dir)

    tools = _build_tools(effective_graph_store, hybrid_retriever=hybrid_retriever)
    ranker_embedding = hybrid_retriever.embedding_model if hybrid_retriever is not None else None
    evidence_ranker = UnifiedEvidenceRanker(embedding_model=ranker_embedding)
    reranker = EvidenceReranker()

    graph = StateGraph(AgenticGraphRAGState)
    graph.add_node("query_rewrite", _query_rewrite_node)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("query_decomposition", query_decomposition_node)
    graph.add_node("branch_graph", _set_graph_branch)
    graph.add_node("branch_vector", _set_vector_branch)
    graph.add_node("branch_hybrid", _set_hybrid_branch)
    graph.add_node("branch_direct", _direct_answer_prep_node)
    graph.add_node("react_loop", react_loop_node)
    graph.add_node("tool_calling", make_tool_calling_node(tools))
    graph.add_node("retrieval", make_retrieval_node(effective_graph_store, hybrid_retriever=hybrid_retriever))
    graph.add_node("evidence_ranking", make_evidence_ranking_node(evidence_ranker))
    graph.add_node("reranker", make_reranker_node(reranker))
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("self_consistency", self_consistency_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "query_rewrite")
    graph.add_edge("query_rewrite", "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "query_decomposition")
    graph.add_conditional_edges(
        "query_decomposition",
        _strategy_branch_router,
        {
            "direct": "branch_direct",
            "graph": "branch_graph",
            "vector": "branch_vector",
            "hybrid": "branch_hybrid",
        },
    )
    graph.add_edge("branch_graph", "react_loop")
    graph.add_edge("branch_vector", "react_loop")
    graph.add_edge("branch_hybrid", "react_loop")
    graph.add_edge("branch_direct", "answer")
    graph.add_edge("react_loop", "tool_calling")
    graph.add_edge("tool_calling", "retrieval")
    graph.add_edge("retrieval", "evidence_ranking")
    graph.add_edge("evidence_ranking", "reranker")
    graph.add_edge("reranker", "reasoning")
    graph.add_edge("reasoning", "self_consistency")
    graph.add_edge("self_consistency", "reflection")
    graph.add_conditional_edges(
        "reflection",
        reflection_router,
        {
            "retrieval": "retrieval",
            "reasoning": "reasoning",
            "answer": "answer",
        },
    )
    graph.add_edge("answer", END)
    return graph.compile()


def run_agentic_graphrag(
    query: str,
    data_dir: Path | None = None,
    graph_store: InMemoryGraphStore | None = None,
    hybrid_retriever=None,
    conversation_history: List[Dict[str, str]] | None = None,
    conversation_context: List[Dict[str, str]] | None = None,
    conversation_summary: str | None = None,
    fact_memory: Dict[str, Any] | None = None,
    short_term_memory: List[Dict[str, str]] | None = None,
    summary_memory: str | None = None,
    memory_state: Dict[str, Any] | None = None,
    history_window_turns: int = 6,
) -> AgenticGraphRAGState:
    root = Path(__file__).resolve().parents[2]
    effective_data_dir = data_dir or (root / "data" / "sample_legal_docs")
    app = create_agentic_app(data_dir=effective_data_dir, graph_store=graph_store, hybrid_retriever=hybrid_retriever)

    history_in = conversation_history or []
    history_msg_limit = max(2, history_window_turns * 2)
    history_window = history_in[-history_msg_limit:]

    initial_state: AgenticGraphRAGState = {
        "user_query": query,
        "conversation_history": history_window,
        "conversation_context": conversation_context or [],
        "conversation_summary": conversation_summary or "",
        "fact_memory": fact_memory or {},
        "short_term_memory": short_term_memory or history_window,
        "summary_memory": summary_memory or (conversation_summary or ""),
        "memory_state": memory_state or {},
        "rewrite_decision": {},
        "rewritten_query": query,
        "rewrite_info": {},
        "router_decision": {},
        "retrieval_strategy": "graph",
        "workflow_branch": "",
        "decomposition": {},
        "subqueries": [],
        "react_plan": [],
        "react_trace": [],
        "tool_results": [],
        "linked_entities": [],
        "plan": {},
        "evidence_pack": {},
        "candidate_evidence": [],
        "ranked_evidence": [],
        "reasoning_trace": {},
        "self_consistency_result": {},
        "verification_result": {},
        "final_answer": {},
        "logs": [],
        "reflection_round": 0,
    }
    return app.invoke(initial_state)
