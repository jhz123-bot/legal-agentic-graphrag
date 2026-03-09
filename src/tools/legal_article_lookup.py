import re
from typing import Any, Dict, List

from src.retrieval.graph_retriever import GraphRetriever

ARTICLE_PATTERN = re.compile(r"((?:《?中华人民共和国(?:刑法|民法典)》?|刑法|民法典)?第[一二三四五六七八九十百千万零两〇0-9]+条)")


class LegalArticleLookupTool:
    name = "legal_article_lookup"

    def __init__(self, graph_retriever: GraphRetriever) -> None:
        self.graph_retriever = graph_retriever

    def _normalize(self, text: str) -> str:
        s = text.replace("《", "").replace("》", "")
        s = s.replace("中华人民共和国", "")
        if s.startswith("第"):
            if "二百六十四" in s:
                return f"刑法{s}"
            if "五百七十七" in s:
                return f"民法典{s}"
        return s

    def run(self, query: str) -> Dict[str, Any]:
        result = self.graph_retriever.retrieve(query, top_k_nodes=12, top_k_edges=12)
        statute_nodes = [n for n in result.get("nodes", []) if n.get("entity_type") == "STATUTE"]

        refs = [self._normalize(m) for m in ARTICLE_PATTERN.findall(query)]
        if refs:
            matched: List[Dict[str, Any]] = []
            for node in self.graph_retriever.graph_store.nodes.values():
                if node.entity_type != "STATUTE":
                    continue
                node_texts = [node.name] + list(node.aliases)
                if any(ref in txt for ref in refs for txt in node_texts):
                    matched.append(
                        {
                            "node_id": node.node_id,
                            "name": node.name,
                            "entity_type": node.entity_type,
                            "aliases": node.aliases,
                            "mentions": [m.sentence for m in node.mentions[:3]],
                        }
                    )
            statute_nodes = matched if matched else statute_nodes
        elif not statute_nodes and ("法条" in query or "条文" in query):
            # fallback: return top statute nodes from whole graph
            all_statutes = [n for n in self.graph_retriever.graph_store.nodes.values() if n.entity_type == "STATUTE"]
            statute_nodes = [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "entity_type": n.entity_type,
                    "aliases": n.aliases,
                    "mentions": [m.sentence for m in n.mentions[:3]],
                }
                for n in all_statutes[:8]
            ]

        return {
            "tool": self.name,
            "articles": statute_nodes,
            "summary": f"article hits={len(statute_nodes)}",
        }
