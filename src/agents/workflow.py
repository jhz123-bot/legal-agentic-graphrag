from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.answer_agent import answer_node
from src.agents.planner_agent import planner_node
from src.agents.reasoning_agent import reasoning_node
from src.agents.reflection_agent import reflection_node, reflection_router
from src.agents.retrieval_agent import evidence_ranking_node, make_retrieval_node
from src.common.models import Document
from src.graph.entity_linker import EntityLinker
from src.graph.graph_builder import GraphBuilder
from src.graph.store import InMemoryGraphStore


class AgenticGraphRAGState(TypedDict, total=False):
    user_query: str
    decomposition: Dict[str, Any]
    linked_entities: List[str]
    plan: Dict[str, Any]
    evidence_pack: Dict[str, Any]
    ranked_evidence: List[Dict[str, Any]]
    reasoning_trace: Dict[str, Any]
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
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    graph_store = builder.build(docs)
    linker = EntityLinker()
    linker.link(graph_store)
    return graph_store


def create_agentic_app(data_dir: Path):
    graph_store = build_graph_store(data_dir)

    graph = StateGraph(AgenticGraphRAGState)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", make_retrieval_node(graph_store))
    graph.add_node("evidence_ranking", evidence_ranking_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "evidence_ranking")
    graph.add_edge("evidence_ranking", "reasoning")
    graph.add_edge("reasoning", "reflection")
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


def run_agentic_graphrag(query: str, data_dir: Path | None = None) -> AgenticGraphRAGState:
    root = Path(__file__).resolve().parents[2]
    effective_data_dir = data_dir or (root / "data" / "sample_legal_docs")
    app = create_agentic_app(effective_data_dir)

    initial_state: AgenticGraphRAGState = {
        "user_query": query,
        "decomposition": {},
        "linked_entities": [],
        "plan": {},
        "evidence_pack": {},
        "ranked_evidence": [],
        "reasoning_trace": {},
        "verification_result": {},
        "final_answer": {},
        "logs": [],
        "reflection_round": 0,
    }
    return app.invoke(initial_state)
