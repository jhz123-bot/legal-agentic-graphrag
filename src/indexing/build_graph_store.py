from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.models import Document, EntityMention, GraphEdge, GraphNode
from src.graph.entity_linker import EntityLinker
from src.graph.graph_builder import GraphBuilder
from src.graph.store import InMemoryGraphStore
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_chunker import chunk_documents


def _node_to_dict(node: GraphNode) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "name": node.name,
        "entity_type": node.entity_type,
        "aliases": node.aliases,
        "mentions": [
            {
                "name": m.name,
                "entity_type": m.entity_type,
                "doc_id": m.doc_id,
                "sentence": m.sentence,
            }
            for m in node.mentions
        ],
        "metadata": node.metadata,
    }


def _edge_to_dict(edge: GraphEdge) -> Dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation,
        "weight": edge.weight,
        "evidence": edge.evidence,
    }


def build_graph_store(
    docs_dir: str | Path,
    graph_dir: str | Path = "data/graph",
) -> Dict[str, Any]:
    """Offline graph build + persistence.

    Steps:
    1) load docs
    2) chunk docs
    3) build graph
    4) entity linking
    5) save nodes/edges
    """
    docs_path = Path(docs_dir)
    out_dir = Path(graph_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_docs = load_documents_from_dir(docs_path)
    if not loaded_docs:
        raise ValueError(f"No TXT/PDF/DOCX documents found in {docs_path}")

    chunks = chunk_documents(loaded_docs)
    graph_documents = [Document(doc_id=c.chunk_id, title=c.title, text=c.text) for c in chunks]

    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    graph_store = builder.build(graph_documents)
    linker = EntityLinker()
    linker.link(graph_store)

    nodes_path = out_dir / "graph_nodes.json"
    edges_path = out_dir / "graph_edges.json"
    debug_path = out_dir / "graph_build_debug.json"

    nodes_payload = [_node_to_dict(n) for n in graph_store.nodes.values()]
    edges_payload = [_edge_to_dict(e) for e in graph_store.edges]
    nodes_path.write_text(json.dumps(nodes_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    edges_path.write_text(json.dumps(edges_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_path.write_text(json.dumps(builder.debug_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "docs_dir": str(docs_path),
        "graph_dir": str(out_dir),
        "doc_count": len(loaded_docs),
        "chunk_count": len(chunks),
        "node_count": len(graph_store.nodes),
        "edge_count": len(graph_store.edges),
        "artifacts": {
            "graph_nodes": str(nodes_path),
            "graph_edges": str(edges_path),
            "graph_build_debug": str(debug_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist graph store artifacts")
    parser.add_argument("--docs", default="data/legal_docs", help="document directory (TXT/PDF/DOCX)")
    parser.add_argument("--graph-dir", default="data/graph", help="graph artifact output directory")
    args = parser.parse_args()

    summary = build_graph_store(docs_dir=args.docs, graph_dir=args.graph_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

