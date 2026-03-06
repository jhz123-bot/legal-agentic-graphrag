from typing import Any, Dict

from src.retrieval.graph_retriever import GraphRetriever


class LegalArticleLookupTool:
    name = "legal_article_lookup"

    def __init__(self, graph_retriever: GraphRetriever) -> None:
        self.graph_retriever = graph_retriever

    def run(self, query: str) -> Dict[str, Any]:
        result = self.graph_retriever.retrieve(query, top_k_nodes=8, top_k_edges=10)
        statute_nodes = [n for n in result.get("nodes", []) if n.get("entity_type") == "STATUTE"]
        return {
            "tool": self.name,
            "articles": statute_nodes,
            "summary": f"article hits={len(statute_nodes)}",
        }
