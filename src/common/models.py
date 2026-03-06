from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class EntityMention:
    name: str
    entity_type: str
    doc_id: str
    sentence: str


@dataclass
class GraphNode:
    node_id: str
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    mentions: List[EntityMention] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence: Optional[str] = None
