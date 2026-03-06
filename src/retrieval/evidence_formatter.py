from typing import Dict, List


def format_evidence(retrieval_result: Dict[str, List[dict]]) -> str:
    lines = []
    lines.append("Relevant entities:")
    for node in retrieval_result.get("nodes", []):
        lines.append(f"- {node['name']} ({node['entity_type']})")
        for mention in node.get("mentions", [])[:2]:
            lines.append(f"  Evidence: {mention}")

    lines.append("")
    lines.append("Relevant graph relations:")
    for edge in retrieval_result.get("edges", []):
        lines.append(f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}")
        if edge.get("evidence"):
            lines.append(f"  Evidence: {edge['evidence']}")

    return "\n".join(lines).strip()
