from typing import Any, Dict

from src.retrieval.graph_retriever import GraphRetriever


class CaseLookupTool:
    name = "case_lookup"

    def __init__(self, graph_retriever: GraphRetriever) -> None:
        self.graph_retriever = graph_retriever

    def run(self, query: str) -> Dict[str, Any]:
        result = self.graph_retriever.retrieve(query, top_k_nodes=8, top_k_edges=6)
        case_nodes = [n for n in result.get("nodes", []) if n.get("entity_type") == "CASE"]
        return {
            "tool": self.name,
            "cases": case_nodes,
            "summary": f"case hits={len(case_nodes)}",
        }
