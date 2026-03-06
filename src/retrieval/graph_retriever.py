from typing import Dict, List

from src.graph.entity_extraction import extract_entities
from src.graph.store import InMemoryGraphStore


class GraphRetriever:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self.graph_store = graph_store

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
            for node in self.graph_store.nodes.values():
                if any(token in node.name.lower() for token in question.lower().split()):
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
                edges_payload.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
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
