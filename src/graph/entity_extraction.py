import os
import re
from typing import List

from src.common.models import EntityMention


CASE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+ v\. [A-Z][a-zA-Z ]+)\b")
SECTION_PATTERN = re.compile(r"\b(Section\s+\d+)\b", flags=re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b")
CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

ZH_LEGAL_CONCEPTS = {
    "盗窃罪": "盗窃罪",
    "盗窃行为": "盗窃罪",
    "抢劫罪": "抢劫罪",
    "抢劫行为": "抢劫罪",
    "抢劫": "抢劫罪",
    "诈骗罪": "诈骗罪",
    "诈骗行为": "诈骗罪",
    "诈骗": "诈骗罪",
    "侵权责任": "侵权责任",
    "共同侵权": "共同侵权",
    "违约责任": "违约责任",
    "预期违约": "预期违约",
    "损失赔偿": "损失赔偿",
    "违约金": "违约金",
    "租赁合同": "租赁合同",
    "买卖合同": "买卖合同",
    "职务侵占": "职务侵占",
    "职务侵占罪": "职务侵占",
    "不动产登记": "不动产登记",
    "物权变动": "物权变动",
    "举证责任": "举证责任",
    "出资义务": "股东出资义务",
    "勤勉义务": "董事勤勉义务",
    "暴力方式夺取财物": "抢劫罪",
    "持械抢夺财物": "抢劫罪",
    "虚构投资项目骗取钱款": "诈骗罪",
    "骗取钱款": "诈骗罪",
    "利用职务便利侵占单位财物": "职务侵占",
    "未按期缴纳出资": "股东出资义务",
    "迟延履行": "违约责任",
    "明确表示不履行主要债务": "预期违约",
    "违约损失赔偿范围": "损失赔偿",
    "违约金明显过高": "违约金",
    "催告后承租人仍不付租金": "租赁合同",
    "连续三个月不付租金": "租赁合同",
    "董事违反勤勉义务": "董事勤勉义务",
    "股东未按期缴纳出资": "股东出资义务",
}

ZH_PARTY_PATTERN = re.compile(r"(张某|李某|王某|赵某|钱某|孙某|周某|吴某|郑某|刘某|陈某|甲公司|乙公司|丙公司|某公司|法院|原告|被告)")
ZH_ARTICLE_PATTERN = re.compile(
    r"((?:《?中华人民共和国(?:刑法|民法典)》?|刑法|民法典)?第[一二三四五六七八九十百千万零两〇0-9]+条)"
)
ZH_STATUTE_NAME_PATTERN = re.compile(
    r"(中华人民共和国刑法|中华人民共和国民法典|中华人民共和国公司法|中华人民共和国劳动合同法|中华人民共和国劳动法|中华人民共和国民事诉讼法|刑法|民法典|公司法|劳动合同法|劳动法|民事诉讼法|消费者权益保护法|电子商务法|保险法|个人信息保护法)"
)

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


def _debug(msg: str) -> None:
    if os.getenv("DEBUG_CHINESE_PIPELINE", "0") == "1":
        print(f"[entity_extraction] {msg}")


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？；;])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalize_article(article: str, sentence: str) -> str:
    raw = article.replace("《", "").replace("》", "").strip()
    if raw.startswith("中华人民共和国刑法"):
        return raw.replace("中华人民共和国", "")
    if raw.startswith("中华人民共和国民法典"):
        return raw.replace("中华人民共和国", "")

    if raw.startswith("第"):
        if "二百六十四" in raw:
            return f"刑法{raw}"
        if "五百七十七" in raw:
            return f"民法典{raw}"
        if "刑法" in sentence:
            return f"刑法{raw}"
        if "民法典" in sentence:
            return f"民法典{raw}"
    return raw


def _normalize_statute_name(name: str) -> str:
    s = name.replace("《", "").replace("》", "").strip()
    if s.startswith("中华人民共和国"):
        s = s.replace("中华人民共和国", "", 1)
    return s


def extract_entities(text: str, doc_id: str) -> List[EntityMention]:
    mentions: List[EntityMention] = []
    sentences = _split_sentences(text)

    if doc_id.startswith("case"):
        case_name = f"案例:{doc_id}"
        seed_sentence = sentences[0] if sentences else text[:50]
        mentions.append(EntityMention(name=case_name, entity_type="CASE", doc_id=doc_id, sentence=seed_sentence))

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

        for match in ZH_ARTICLE_PATTERN.findall(sentence):
            normalized = _normalize_article(match, sentence)
            mentions.append(EntityMention(name=normalized, entity_type="STATUTE", doc_id=doc_id, sentence=sentence))

        for match in ZH_STATUTE_NAME_PATTERN.findall(sentence):
            mentions.append(
                EntityMention(
                    name=_normalize_statute_name(match),
                    entity_type="STATUTE",
                    doc_id=doc_id,
                    sentence=sentence,
                )
            )

        for phrase, canonical in ZH_LEGAL_CONCEPTS.items():
            if phrase in sentence:
                mentions.append(EntityMention(name=canonical, entity_type="LEGAL_CONCEPT", doc_id=doc_id, sentence=sentence))

        for match in ZH_PARTY_PATTERN.findall(sentence):
            mentions.append(EntityMention(name=match, entity_type="PARTY", doc_id=doc_id, sentence=sentence))

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

    out = list(deduped.values())
    _debug(f"doc_id={doc_id}, sentences={len(sentences)}, entities={len(out)}")
    return out
