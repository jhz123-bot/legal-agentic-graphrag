import re
from typing import Dict, List

from src.graph.entity_extraction import extract_entities
from src.graph.store import InMemoryGraphStore


class GraphRetriever:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self.graph_store = graph_store

    def _fallback_tokens(self, question: str) -> List[str]:
        zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", question)
        en_terms = [t for t in re.split(r"\W+", question.lower()) if len(t) > 1]
        return zh_terms + en_terms

    def retrieve(self, question: str, top_k_nodes: int = 5, top_k_edges: int = 10) -> Dict[str, List[dict]]:
        query_mentions = extract_entities(question, doc_id="query")
        candidate_nodes = {}

        for mention in query_mentions:
            matches = self.graph_store.find_nodes_by_name(mention.name)
            for node in matches:
                score = 2
                if node.entity_type == mention.entity_type:
                    score = 3
                candidate_nodes[node.node_id] = max(candidate_nodes.get(node.node_id, 0), score)

        if not candidate_nodes:
            tokens = self._fallback_tokens(question)
            for node in self.graph_store.nodes.values():
                node_text = f"{node.name} {' '.join(node.aliases)}".lower()
                if any(token.lower() in node_text for token in tokens):
                    candidate_nodes[node.node_id] = 1

        ranked_nodes = sorted(candidate_nodes.items(), key=lambda x: x[1], reverse=True)[:top_k_nodes]
        node_ids = [node_id for node_id, _ in ranked_nodes]

        nodes_payload = []
        for node_id in node_ids:
            node = self.graph_store.nodes[node_id]
            nodes_payload.append(
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "entity_type": node.entity_type,
                    "aliases": node.aliases,
                    "mentions": [m.sentence for m in node.mentions[:3]],
                }
            )

        edges_payload = []
        for node_id in node_ids:
            for edge in self.graph_store.neighbors(node_id):
                if len(edges_payload) >= top_k_edges:
                    break
                source_node = self.graph_store.nodes.get(edge.source)
                target_node = self.graph_store.nodes.get(edge.target)
                source_doc_id = ""
                target_doc_id = ""
                if source_node:
                    source_doc_id = source_node.metadata.get("first_doc_id", "")
                    if not source_doc_id and source_node.mentions:
                        source_doc_id = source_node.mentions[0].doc_id
                if target_node:
                    target_doc_id = target_node.metadata.get("first_doc_id", "")
                    if not target_doc_id and target_node.mentions:
                        target_doc_id = target_node.mentions[0].doc_id
                edges_payload.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "source_name": source_node.name if source_node else edge.source,
                        "target_name": target_node.name if target_node else edge.target,
                        "source_doc_id": source_doc_id,
                        "target_doc_id": target_doc_id,
                        "relation": edge.relation,
                        "evidence": edge.evidence or "",
                    }
                )
            if len(edges_payload) >= top_k_edges:
                break

        return {
            "query_mentions": [{"name": m.name, "entity_type": m.entity_type} for m in query_mentions],
            "nodes": nodes_payload,
            "edges": edges_payload,
        }
