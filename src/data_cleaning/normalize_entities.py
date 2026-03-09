import re
from typing import List, Optional


_ALIAS_MAP = {
    "房东": "出租人",
    "房主": "出租人",
    "租客": "承租人",
    "租户": "承租人",
    "盗窃行为": "盗窃罪",
    "诈骗行为": "诈骗罪",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove common OCR junk chars.
    t = t.replace("�", "").replace("\u3000", " ")

    # Remove empty lines and trim spaces per line.
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    t = "\n".join(lines)

    # Collapse duplicated punctuation.
    t = re.sub(r"([，。；：！？,.!?;:])\1+", r"\1", t)
    # Collapse repeated spaces.
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def normalize_entity_alias(text: str) -> str:
    out = text
    for raw, canonical in _ALIAS_MAP.items():
        out = out.replace(raw, canonical)
    return out


def _law_short_name(law_name: str) -> str:
    if "刑法" in law_name:
        return "刑法"
    if "民法典" in law_name:
        return "民法典"
    if "公司法" in law_name:
        return "公司法"
    if "劳动合同法" in law_name:
        return "劳动合同法"
    if "劳动争议调解仲裁法" in law_name:
        return "劳动争议调解仲裁法"
    return law_name.strip()


def normalize_article_name(article: str, default_law_name: str = "") -> str:
    if not article:
        return ""
    a = article.strip().replace("《", "").replace("》", "")
    a = a.replace("中华人民共和国", "")

    m = re.search(r"第[一二三四五六七八九十百千万零两〇0-9]+条", a)
    if not m:
        return a
    art = m.group(0)

    if "刑法" in a:
        return f"刑法{art}"
    if "民法典" in a:
        return f"民法典{art}"
    if "公司法" in a:
        return f"公司法{art}"
    if "劳动合同法" in a:
        return f"劳动合同法{art}"
    if "劳动争议调解仲裁法" in a:
        return f"劳动争议调解仲裁法{art}"

    if default_law_name:
        short = _law_short_name(default_law_name)
        if short:
            return f"{short}{art}"
    return art


def normalize_articles_in_text(text: str, default_law_name: str = "") -> str:
    if not text:
        return ""

    def repl(match: re.Match[str]) -> str:
        return normalize_article_name(match.group(0), default_law_name=default_law_name)

    pattern = re.compile(
        r"(?:《?中华人民共和国(?:刑法|民法典|公司法|劳动合同法|劳动争议调解仲裁法)》?)?"
        r"(?:刑法|民法典|公司法|劳动合同法|劳动争议调解仲裁法)?"
        r"第[一二三四五六七八九十百千万零两〇0-9]+条"
    )
    return pattern.sub(repl, text)


def normalize_keywords(keywords: List[str], default_law_name: str = "") -> List[str]:
    out: List[str] = []
    seen = set()
    for kw in keywords or []:
        k = clean_text(str(kw))
        k = normalize_entity_alias(k)
        if re.search(r"第[一二三四五六七八九十百千万零两〇0-9]+条", k):
            k = normalize_article_name(k, default_law_name=default_law_name)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out
