from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.models import EntityMention, GraphEdge, GraphNode
from src.graph.store import InMemoryGraphStore


def _dict_to_node(row: Dict[str, Any]) -> GraphNode:
    mentions = [
        EntityMention(
            name=str(m.get("name", "")),
            entity_type=str(m.get("entity_type", "")),
            doc_id=str(m.get("doc_id", "")),
            sentence=str(m.get("sentence", "")),
        )
        for m in row.get("mentions", [])
    ]
    return GraphNode(
        node_id=str(row.get("node_id", "")),
        name=str(row.get("name", "")),
        entity_type=str(row.get("entity_type", "")),
        aliases=[str(x) for x in row.get("aliases", [])],
        mentions=mentions,
        metadata={str(k): str(v) for k, v in (row.get("metadata", {}) or {}).items()},
    )


def _dict_to_edge(row: Dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        source=str(row.get("source", "")),
        target=str(row.get("target", "")),
        relation=str(row.get("relation", "MENTIONED_WITH")),
        weight=float(row.get("weight", 1.0) or 1.0),
        evidence=row.get("evidence"),
    )


def load_graph(graph_dir: str | Path = "data/graph") -> InMemoryGraphStore:
    """Load persisted graph artifacts from disk."""
    base = Path(graph_dir)
    nodes_path = base / "graph_nodes.json"
    edges_path = base / "graph_edges.json"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing graph nodes file: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing graph edges file: {edges_path}")

    nodes_payload = json.loads(nodes_path.read_text(encoding="utf-8"))
    edges_payload = json.loads(edges_path.read_text(encoding="utf-8"))

    store = InMemoryGraphStore()
    for row in nodes_payload:
        store.upsert_node(_dict_to_node(row))
    for row in edges_payload:
        store.add_edge(_dict_to_edge(row))
    return store

