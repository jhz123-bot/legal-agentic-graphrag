from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx


def plot_reasoning_tree(reasoning_trace: Dict[str, Any], output_path: str | Path) -> None:
    steps = reasoning_trace.get("structured_steps", [])
    g = nx.DiGraph()
    g.add_node("Q", label="Query")
    prev = "Q"
    for step in steps:
        node_id = f"S{step.get('step')}"
        label = f"{step.get('relation', 'REL')}\\n{step.get('conclusion', '')[:28]}"
        g.add_node(node_id, label=label)
        g.add_edge(prev, node_id)
        prev = node_id

    pos = nx.spring_layout(g, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(g, pos, node_color="#dbeafe", node_size=2200)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", arrowsize=16)
    nx.draw_networkx_labels(g, pos, labels=nx.get_node_attributes(g, "label"), font_size=9)
    plt.title("Reasoning Tree")
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_evidence_graph(evidence_pack: Dict[str, Any], output_path: str | Path) -> None:
    paths = evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", evidence_pack.get("edges", [])))
    g = nx.DiGraph()
    for i, p in enumerate(paths[:20]):
        src = p.get("source", f"src_{i}")
        tgt = p.get("target", f"tgt_{i}")
        rel = p.get("relation", "REL")
        g.add_node(src)
        g.add_node(tgt)
        g.add_edge(src, tgt, label=rel)

    pos = nx.spring_layout(g, seed=7)
    plt.figure(figsize=(11, 7))
    nx.draw_networkx_nodes(g, pos, node_color="#fde68a", node_size=1600)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=14, alpha=0.8)
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "label"), font_size=7)
    plt.title("Evidence Path Graph")
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
