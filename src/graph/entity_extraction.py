import re
from typing import List

from src.common.models import EntityMention


CASE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+ v\. [A-Z][a-zA-Z ]+)\b")
SECTION_PATTERN = re.compile(r"\b(Section\s+\d+)\b", flags=re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b")
CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

LEGAL_CONCEPTS = {
    "breach of contract",
    "negligence",
    "duty of care",
    "causation",
    "wrongful eviction",
    "injunctive relief",
    "damages",
    "foreseeability",
}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_entities(text: str, doc_id: str) -> List[EntityMention]:
    mentions: List[EntityMention] = []
    sentences = _split_sentences(text)

    for sentence in sentences:
        for match in CASE_PATTERN.findall(sentence):
            mentions.append(EntityMention(name=match.strip(), entity_type="CASE", doc_id=doc_id, sentence=sentence))

        for match in SECTION_PATTERN.findall(sentence):
            mentions.append(EntityMention(name=match.strip().title(), entity_type="STATUTE", doc_id=doc_id, sentence=sentence))

        for match in DATE_PATTERN.findall(sentence):
            mentions.append(EntityMention(name=match.strip(), entity_type="DATE", doc_id=doc_id, sentence=sentence))

        lowered = sentence.lower()
        for concept in LEGAL_CONCEPTS:
            if concept in lowered:
                mentions.append(
                    EntityMention(
                        name=concept.title(),
                        entity_type="LEGAL_CONCEPT",
                        doc_id=doc_id,
                        sentence=sentence,
                    )
                )

        for match in CAPITALIZED_PHRASE_PATTERN.findall(sentence):
            clean = match.strip()
            if " v. " in clean:
                continue
            if clean.lower().startswith("section "):
                continue
            mentions.append(EntityMention(name=clean, entity_type="PARTY", doc_id=doc_id, sentence=sentence))

    deduped = {}
    for m in mentions:
        key = (m.name.lower(), m.entity_type, m.doc_id, m.sentence)
        deduped[key] = m
    return list(deduped.values())
