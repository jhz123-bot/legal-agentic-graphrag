from typing import Any, Dict

from src.retrieval.graph_retriever import GraphRetriever


class GraphSearchTool:
    name = "graph_search"

    def __init__(self, graph_retriever: GraphRetriever) -> None:
        self.graph_retriever = graph_retriever

    def run(self, query: str) -> Dict[str, Any]:
        result = self.graph_retriever.retrieve(query, top_k_nodes=5, top_k_edges=8)
        return {
            "tool": self.name,
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
            "summary": f"graph nodes={len(result.get('nodes', []))}, edges={len(result.get('edges', []))}",
        }
