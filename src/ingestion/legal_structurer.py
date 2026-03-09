from __future__ import annotations

import re
from typing import Dict


LAW_NAMES = [
    "中华人民共和国刑法",
    "中华人民共和国民法典",
    "中华人民共和国公司法",
    "中华人民共和国劳动合同法",
    "最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释",
    "最高人民法院关于审理民间借贷案件适用法律若干问题的规定",
]


def structure_statute(item: Dict, content: str) -> Dict:
    title = str(item.get("title", "")).strip()
    law_name = str(item.get("law_name", "")).strip()
    if not law_name:
        for candidate in LAW_NAMES:
            if candidate in title or candidate in content:
                law_name = candidate
                break

    article_no = str(item.get("article_no", "")).strip()
    if not article_no:
        m = re.search(r"第[一二三四五六七八九十百千万零两〇0-9]+条", title) or re.search(
            r"第[一二三四五六七八九十百千万零两〇0-9]+条", content
        )
        article_no = m.group(0) if m else ""

    chapter = str(item.get("chapter", "")).strip()
    if not chapter:
        m = re.search(r"第[一二三四五六七八九十百千万零两〇0-9]+[章节编]", content)
        chapter = m.group(0) if m else ""

    return {
        "law_name": law_name,
        "chapter": chapter,
        "article_no": article_no,
    }


def structure_case(item: Dict, content: str) -> Dict:
    case_id = str(item.get("case_id", "")).strip() or str(item.get("doc_id", "")).strip()
    court = str(item.get("court", "")).strip()
    if not court:
        m = re.search(r"([\u4e00-\u9fff]{2,}(?:人民法院|法院))", content)
        court = m.group(1) if m else ""

    case_type = str(item.get("case_type", "")).strip()
    if not case_type:
        for k in ["盗窃", "诈骗", "违约", "租赁", "侵权", "公司", "劳动"]:
            if k in content:
                case_type = f"{k}纠纷"
                break

    dispute_focus = str(item.get("dispute_focus", "")).strip()
    if not dispute_focus:
        m = re.search(r"争议焦点[：:](.+)", content)
        dispute_focus = m.group(1).strip() if m else ""

    judgment = str(item.get("judgment", "")).strip()
    if not judgment:
        m = re.search(r"(?:裁判结果|判决如下)[：:](.+)", content)
        judgment = m.group(1).strip() if m else ""

    return {
        "case_id": case_id,
        "court": court,
        "case_type": case_type,
        "dispute_focus": dispute_focus,
        "judgment": judgment,
    }


def structure_faq(item: Dict) -> Dict:
    topic = str(item.get("topic", "")).strip()
    if topic:
        return {"topic": topic}

    question = str(item.get("question", ""))
    for t in ["盗窃罪", "诈骗罪", "合同违约", "租赁纠纷", "侵权责任", "公司纠纷", "劳动争议"]:
        if t in question:
            return {"topic": t}
    return {"topic": "一般法律咨询"}
