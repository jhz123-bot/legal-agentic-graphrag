from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.answer_agent import answer_node
from src.agents.planner_agent import planner_node, query_decomposition_node
from src.agents.reasoning_agent import reasoning_node
from src.agents.reflection_agent import reflection_node, reflection_router
from src.agents.retrieval_agent import evidence_ranking_node, make_retrieval_node
from src.common.models import Document
from src.graph.entity_linker import EntityLinker
from src.graph.graph_builder import GraphBuilder
from src.graph.store import InMemoryGraphStore
from src.reasoning.react_loop import make_tool_calling_node, react_loop_node
from src.reasoning.self_consistency import self_consistency_node
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import EvidenceReranker, make_reranker_node
from src.router.agent_router import router_node
from src.tools.case_lookup import CaseLookupTool
from src.tools.graph_search import GraphSearchTool
from src.tools.legal_article_lookup import LegalArticleLookupTool
from src.tools.vector_search import VectorSearchTool


class AgenticGraphRAGState(TypedDict, total=False):
    user_query: str
    router_decision: Dict[str, Any]
    decomposition: Dict[str, Any]
    subqueries: List[str]
    react_plan: List[Dict[str, Any]]
    react_trace: List[Dict[str, str]]
    tool_results: List[Dict[str, Any]]
    linked_entities: List[str]
    plan: Dict[str, Any]
    evidence_pack: Dict[str, Any]
    ranked_evidence: List[Dict[str, Any]]
    reasoning_trace: Dict[str, Any]
    self_consistency_result: Dict[str, Any]
    verification_result: Dict[str, Any]
    final_answer: Dict[str, Any]
    logs: List[str]
    reflection_round: int


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
    reranker = EvidenceReranker()

    graph = StateGraph(AgenticGraphRAGState)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("query_decomposition", query_decomposition_node)
    graph.add_node("react_loop", react_loop_node)
    graph.add_node("tool_calling", make_tool_calling_node(tools))
    graph.add_node("retrieval", make_retrieval_node(effective_graph_store, hybrid_retriever=hybrid_retriever))
    graph.add_node("evidence_ranking", evidence_ranking_node)
    graph.add_node("reranker", make_reranker_node(reranker))
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("self_consistency", self_consistency_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "query_decomposition")
    graph.add_edge("query_decomposition", "react_loop")
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
) -> AgenticGraphRAGState:
    root = Path(__file__).resolve().parents[2]
    effective_data_dir = data_dir or (root / "data" / "sample_legal_docs")
    app = create_agentic_app(data_dir=effective_data_dir, graph_store=graph_store, hybrid_retriever=hybrid_retriever)

    initial_state: AgenticGraphRAGState = {
        "user_query": query,
        "router_decision": {},
        "decomposition": {},
        "subqueries": [],
        "react_plan": [],
        "react_trace": [],
        "tool_results": [],
        "linked_entities": [],
        "plan": {},
        "evidence_pack": {},
        "ranked_evidence": [],
        "reasoning_trace": {},
        "self_consistency_result": {},
        "verification_result": {},
        "final_answer": {},
        "logs": [],
        "reflection_round": 0,
    }
    return app.invoke(initial_state)
