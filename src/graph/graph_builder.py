import re
from collections import defaultdict
from typing import Dict, List

from src.common.models import Document, GraphEdge, GraphNode
from src.graph.entity_extraction import extract_entities
from src.graph.store import InMemoryGraphStore


def _normalize_node_id(name: str, entity_type: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{entity_type.lower()}::{base}"


class GraphBuilder:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self.graph_store = graph_store

    def build(self, documents: List[Document]) -> InMemoryGraphStore:
        for doc in documents:
            mentions = extract_entities(doc.text, doc.doc_id)
            sentence_to_node_ids: Dict[str, List[str]] = defaultdict(list)

            for mention in mentions:
                node_id = _normalize_node_id(mention.name, mention.entity_type)
                node = GraphNode(
                    node_id=node_id,
                    name=mention.name,
                    entity_type=mention.entity_type,
                    mentions=[mention],
                    metadata={"first_doc_id": doc.doc_id},
                )
                self.graph_store.upsert_node(node)
                sentence_to_node_ids[mention.sentence].append(node_id)

            for sentence, node_ids in sentence_to_node_ids.items():
                unique_ids = list(dict.fromkeys(node_ids))
                for i in range(len(unique_ids)):
                    for j in range(i + 1, len(unique_ids)):
                        left = self.graph_store.nodes[unique_ids[i]]
                        right = self.graph_store.nodes[unique_ids[j]]
                        relation = "MENTIONED_WITH"
                        if "STATUTE" in {left.entity_type, right.entity_type}:
                            relation = "REFERENCES_STATUTE"
                        elif "PARTY" in {left.entity_type, right.entity_type}:
                            relation = "INVOLVES_PARTY"
                        self.graph_store.add_edge(
                            GraphEdge(
                                source=left.node_id,
                                target=right.node_id,
                                relation=relation,
                                evidence=sentence,
                            )
                        )
                        self.graph_store.add_edge(
                            GraphEdge(
                                source=right.node_id,
                                target=left.node_id,
                                relation=relation,
                                evidence=sentence,
                            )
                        )

        return self.graph_store
