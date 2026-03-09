import os
import re
from collections import defaultdict
from hashlib import md5
from typing import Dict, List

from src.common.models import Document, EntityMention, GraphEdge, GraphNode
from src.graph.entity_extraction import extract_entities
from src.graph.llm_triple_extractor import LLMTripleExtractor
from src.graph.store import InMemoryGraphStore


def _debug(msg: str) -> None:
    if os.getenv("DEBUG_CHINESE_PIPELINE", "0") == "1":
        print(f"[graph_builder] {msg}")


def _normalize_node_id(name: str, entity_type: str) -> str:
    base = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", name.lower()).strip("_")
    if not base:
        base = md5(name.encode("utf-8")).hexdigest()[:12]
    return f"{entity_type.lower()}::{base}"


class GraphBuilder:
    def __init__(self, graph_store: InMemoryGraphStore) -> None:
        self.graph_store = graph_store
        self.llm_extractor = LLMTripleExtractor()
        self.debug_stats: List[Dict[str, object]] = []

    def _map_llm_entity_type(self, typ: str) -> str:
        return {
            "Crime": "LEGAL_CONCEPT",
            "LawArticle": "STATUTE",
            "Case": "CASE",
            "Court": "PARTY",
            "Circumstance": "LEGAL_CONCEPT",
            "Liability": "LEGAL_CONCEPT",
            "Party": "PARTY",
        }.get(typ, "LEGAL_CONCEPT")

    def _is_rule_extraction_sufficient(self, rule_mentions: List[EntityMention]) -> bool:
        if not rule_mentions:
            return False
        types = {m.entity_type for m in rule_mentions}
        has_statute = "STATUTE" in types
        has_legal_concept = "LEGAL_CONCEPT" in types
        # Rule extraction is considered enough when it can already connect
        # legal concept + statute, or it captures rich mentions in one chunk.
        return (has_statute and has_legal_concept) or len(rule_mentions) >= 4

    def build(self, documents: List[Document]) -> InMemoryGraphStore:
        for doc in documents:
            rule_mentions = extract_entities(doc.text, doc.doc_id)
            use_llm_fallback = not self._is_rule_extraction_sufficient(rule_mentions)
            llm_payload = self.llm_extractor.extract_triples(doc.text) if use_llm_fallback else {"entities": [], "triples": []}
            llm_mentions: List[EntityMention] = []
            entity_type_map: Dict[str, str] = {}
            first_sentence = doc.text.strip().splitlines()[0] if doc.text.strip() else doc.text

            for ent in llm_payload.get("entities", []):
                name = str(ent.get("name", "")).strip()
                if not name:
                    continue
                mapped_type = self._map_llm_entity_type(str(ent.get("type", "")))
                entity_type_map[name] = mapped_type
                llm_mentions.append(
                    EntityMention(
                        name=name,
                        entity_type=mapped_type,
                        doc_id=doc.doc_id,
                        sentence=first_sentence,
                    )
                )

            mentions = rule_mentions + llm_mentions
            deduped_mentions = {}
            for m in mentions:
                deduped_mentions[(m.name, m.entity_type, m.doc_id, m.sentence)] = m
            mentions = list(deduped_mentions.values())

            sentence_to_node_ids: Dict[str, List[str]] = defaultdict(list)
            _debug(
                f"doc={doc.doc_id} rule={len(rule_mentions)} use_llm_fallback={use_llm_fallback} "
                f"llm_entities={len(llm_payload.get('entities', []))} "
                f"llm_triples={len(llm_payload.get('triples', []))} merged={len(mentions)}"
            )

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

            # LLM triples enrich relation extraction and are optional.
            for triple in llm_payload.get("triples", []):
                subj = str(triple.get("subject", "")).strip()
                rel = str(triple.get("relation", "")).strip()
                obj = str(triple.get("object", "")).strip()
                if not subj or not obj or not rel:
                    continue

                subj_type = entity_type_map.get(subj, "LEGAL_CONCEPT")
                obj_type = entity_type_map.get(obj, "LEGAL_CONCEPT")
                subj_id = _normalize_node_id(subj, subj_type)
                obj_id = _normalize_node_id(obj, obj_type)

                if subj_id not in self.graph_store.nodes:
                    self.graph_store.upsert_node(
                        GraphNode(
                            node_id=subj_id,
                            name=subj,
                            entity_type=subj_type,
                            mentions=[],
                            metadata={"first_doc_id": doc.doc_id},
                        )
                    )
                if obj_id not in self.graph_store.nodes:
                    self.graph_store.upsert_node(
                        GraphNode(
                            node_id=obj_id,
                            name=obj,
                            entity_type=obj_type,
                            mentions=[],
                            metadata={"first_doc_id": doc.doc_id},
                        )
                    )

                self.graph_store.add_edge(
                    GraphEdge(
                        source=subj_id,
                        target=obj_id,
                        relation=rel,
                        evidence=doc.text[:200],
                    )
                )

            for sentence, node_ids in sentence_to_node_ids.items():
                unique_ids = list(dict.fromkeys(node_ids))
                for i in range(len(unique_ids)):
                    for j in range(i + 1, len(unique_ids)):
                        left = self.graph_store.nodes[unique_ids[i]]
                        right = self.graph_store.nodes[unique_ids[j]]
                        relation = "MENTIONED_WITH"

                        names = {left.name, right.name}
                        if (
                            any(x in names for x in {"盗窃罪", "违约责任", "诈骗罪", "侵权责任", "抢劫罪", "职务侵占", "股东出资义务", "董事勤勉义务", "租赁合同", "不动产登记", "预期违约", "损失赔偿"})
                            and (left.entity_type == "STATUTE" or right.entity_type == "STATUTE")
                        ):
                            relation = "APPLIES_TO"
                        elif {"LEGAL_CONCEPT", "STATUTE"} == {left.entity_type, right.entity_type}:
                            # Generic legal-concept <-> statute anchoring for Chinese legal QA.
                            relation = "APPLIES_TO"
                        elif "构成" in sentence and (left.entity_type == "STATUTE" or right.entity_type == "STATUTE"):
                            relation = "APPLIES_TO"
                        elif (left.entity_type == "CASE" and right.entity_type == "STATUTE") or (
                            right.entity_type == "CASE" and left.entity_type == "STATUTE"
                        ):
                            relation = "CITES"
                        elif "STATUTE" in {left.entity_type, right.entity_type}:
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

            self.debug_stats.append(
                {
                    "doc_id": doc.doc_id,
                    "rule_entities": len(rule_mentions),
                    "use_llm_fallback": use_llm_fallback,
                    "llm_entities": len(llm_payload.get("entities", [])),
                    "merged_entities": len(mentions),
                    "llm_triples": len(llm_payload.get("triples", [])),
                    "sample_llm_triples": llm_payload.get("triples", [])[:3],
                }
            )

        _debug(f"graph built nodes={len(self.graph_store.nodes)} edges={len(self.graph_store.edges)}")
        return self.graph_store
