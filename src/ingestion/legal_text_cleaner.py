from __future__ import annotations

import re
from typing import Iterable, Optional

# Phase-1 baseline patterns for legal text cleaning.
_PAGE_PATTERNS = [
    r"^第\s*\d+\s*页\s*(共\s*\d+\s*页)?$",
    r"^页码[:：]?\s*\d+$",
    r"^Page\s*\d+(\s*/\s*\d+)?$",
]

_HEADER_FOOTER_PATTERNS = [
    r"^中华人民共和国.+$",
    r"^最高人民法院.+$",
    r"^中国裁判文书网.+$",
    r"^（?以下称.+）?$",
]

_PUNCT_MAP = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
        "(": "（",
        ")": "）",
    }
)


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    for p in _PAGE_PATTERNS:
        if re.match(p, stripped, flags=re.IGNORECASE):
            return True
    for p in _HEADER_FOOTER_PATTERNS:
        if re.match(p, stripped):
            return True
    return False


def _dedup_adjacent_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    prev = ""
    for line in lines:
        if line == prev:
            continue
        out.append(line)
        prev = line
    return out


def clean_legal_text(text: str, title: Optional[str] = None) -> str:
    """Basic phase-1 cleaning for statutes/cases/faqs."""
    if not isinstance(text, str):
        return ""

    out = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    out = out.replace("\u3000", " ").replace("\t", " ").replace("�", "")
    out = out.translate(_PUNCT_MAP)

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in out.split("\n")]
    lines = [ln for ln in lines if not _is_noise_line(ln)]

    # Remove repeated title lines.
    if title:
        normalized_title = re.sub(r"\s+", " ", title).strip()
        lines = [ln for ln in lines if ln != normalized_title]

    # Remove repeated template lines used by notices/import artifacts.
    template_noise = {
        "本页无正文",
        "以下无正文",
        "特此公告",
        "以上事实，有证据在卷佐证。",
    }
    lines = [ln for ln in lines if ln not in template_noise]
    lines = _dedup_adjacent_lines(lines)

    # Phase-2 enhancement: deduplicate repeated paragraphs globally.
    uniq: list[str] = []
    seen = set()
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        uniq.append(ln)
    lines = uniq

    out = "\n".join(lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"([，。；：！？])\1+", r"\1", out)
    out = re.sub(r"[ ]{2,}", " ", out)
    return out.strip()
