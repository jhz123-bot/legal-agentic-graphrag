from collections import defaultdict
from typing import Dict, List

from src.common.models import GraphEdge, GraphNode


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)

    def upsert_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if not existing:
            self.nodes[node.node_id] = node
            return
        for alias in node.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        existing.mentions.extend(node.mentions)
        existing.metadata.update(node.metadata)

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge)

    def neighbors(self, node_id: str) -> List[GraphEdge]:
        return self.adjacency.get(node_id, [])

    def find_nodes_by_name(self, query: str) -> List[GraphNode]:
        q = query.lower().strip()
        results = []
        for node in self.nodes.values():
            if q in node.name.lower():
                results.append(node)
                continue
            if any(q in alias.lower() for alias in node.aliases):
                results.append(node)
        return results
