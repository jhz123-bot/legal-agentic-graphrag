import re
from collections import defaultdict
from typing import Dict, List

from src.common.models import GraphEdge, GraphNode
from src.graph.store import InMemoryGraphStore


def _canonicalize(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()
    tokens = [t for t in cleaned.split() if t not in {"the", "inc", "corp", "corporation", "llc"}]
    return " ".join(tokens)


class EntityLinker:
    def link(self, graph_store: InMemoryGraphStore) -> InMemoryGraphStore:
        buckets: Dict[str, List[GraphNode]] = defaultdict(list)
        for node in graph_store.nodes.values():
            key = f"{node.entity_type}:{_canonicalize(node.name)}"
            buckets[key].append(node)

        new_nodes: Dict[str, GraphNode] = {}
        id_map: Dict[str, str] = {}

        for group in buckets.values():
            canonical = group[0]
            merged = GraphNode(
                node_id=canonical.node_id,
                name=canonical.name,
                entity_type=canonical.entity_type,
                aliases=[],
                mentions=[],
                metadata=dict(canonical.metadata),
            )
            for node in group:
                id_map[node.node_id] = canonical.node_id
                if node.name != canonical.name and node.name not in merged.aliases:
                    merged.aliases.append(node.name)
                for alias in node.aliases:
                    if alias != canonical.name and alias not in merged.aliases:
                        merged.aliases.append(alias)
                merged.mentions.extend(node.mentions)
                merged.metadata.update(node.metadata)
            new_nodes[canonical.node_id] = merged

        new_edges: List[GraphEdge] = []
        seen = set()
        for edge in graph_store.edges:
            src = id_map.get(edge.source, edge.source)
            tgt = id_map.get(edge.target, edge.target)
            key = (src, tgt, edge.relation, edge.evidence)
            if key in seen:
                continue
            seen.add(key)
            new_edges.append(
                GraphEdge(
                    source=src,
                    target=tgt,
                    relation=edge.relation,
                    weight=edge.weight,
                    evidence=edge.evidence,
                )
            )

        graph_store.nodes = new_nodes
        graph_store.edges = []
        graph_store.adjacency.clear()
        for edge in new_edges:
            graph_store.add_edge(edge)
        return graph_store
