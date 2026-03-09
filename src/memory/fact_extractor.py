from __future__ import annotations

import re
from typing import Any, Dict, List

from src.llm.llm_router import get_llm_provider


_AMOUNT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?\s*(?:\u5143|\u4e07\u5143|\u4e07|\u5343\u5143))")
_TIME_PATTERN = re.compile(r"([0-9]{4}\u5e74(?:[0-9]{1,2}\u6708)?(?:[0-9]{1,2}\u65e5)?)")

EVENT_WORDS = [
    "\u76d7\u7a83\u7f6a",
    "\u8bc8\u9a97\u7f6a",
    "\u8fdd\u7ea6\u8d23\u4efb",
    "\u4fb5\u6743\u8d23\u4efb",
    "\u79df\u8d41\u7ea0\u7eb7",
    "\u52b3\u52a8\u4e89\u8bae",
    "\u516c\u53f8\u7ea0\u7eb7",
]
EVENT_ALIASES = {
    "\u76d7\u7a83": "\u76d7\u7a83\u7f6a",
    "\u8bc8\u9a97": "\u8bc8\u9a97\u7f6a",
    "\u8fdd\u7ea6": "\u8fdd\u7ea6\u8d23\u4efb",
    "\u4fb5\u6743": "\u4fb5\u6743\u8d23\u4efb",
}

RELATION_WORDS = [
    "\u79df\u8d41",
    "\u4e70\u5356",
    "\u501f\u8d37",
    "\u52b3\u52a8",
    "\u4fb5\u6743",
    "\u8fdd\u7ea6",
]

ACTOR_WORDS = [
    "\u5f20\u67d0",
    "\u674e\u67d0",
    "\u738b\u67d0",
    "\u7532\u516c\u53f8",
    "\u4e59\u516c\u53f8",
    "\u67d0\u516c\u53f8",
    "\u6cd5\u9662",
]

ROLE_WORDS = [
    "\u51fa\u79df\u4eba",
    "\u627f\u79df\u4eba",
    "\u4e70\u65b9",
    "\u5356\u65b9",
    "\u52b3\u52a8\u8005",
    "\u7528\u4eba\u5355\u4f4d",
]


def _dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def extract_facts_rule(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {
            "fact_type": "unknown",
            "amount": None,
            "time": None,
            "relationship": [],
            "actor": [],
            "role": [],
            "event": [],
            "confidence": 0.35,
            "extractor_type": "rule",
        }

    normalized = content.replace("\u623f\u4e1c", "\u51fa\u79df\u4eba").replace("\u79df\u5ba2", "\u627f\u79df\u4eba")
    amounts = _AMOUNT_PATTERN.findall(normalized)
    times = _TIME_PATTERN.findall(normalized)

    events = [w for w in EVENT_WORDS if (w in normalized or w.replace("\u8d23\u4efb", "") in normalized)]
    for alias, canonical in EVENT_ALIASES.items():
        if alias in normalized and canonical not in events:
            events.append(canonical)
    relationships = [w for w in RELATION_WORDS if w in normalized]
    actors = [w for w in ACTOR_WORDS if w in normalized]
    roles = [w for w in ROLE_WORDS if w in normalized]

    fact_type = "legal_fact"
    if events:
        fact_type = "event_fact"
    elif relationships:
        fact_type = "relationship_fact"

    confidence = 0.45
    if events:
        confidence += 0.2
    if amounts:
        confidence += 0.1
    if times:
        confidence += 0.1
    if actors or roles:
        confidence += 0.1

    return {
        "fact_type": fact_type,
        "amount": amounts[-1] if amounts else None,
        "time": times[-1] if times else None,
        "relationship": _dedupe(relationships),
        "actor": _dedupe(actors),
        "role": _dedupe(roles),
        "event": _dedupe(events),
        "confidence": round(min(0.95, confidence), 4),
        "extractor_type": "rule",
    }


def _extract_facts_llm(text: str) -> Dict[str, Any]:
    provider = get_llm_provider(timeout=20)
    prompt = (
        "\u4f60\u662f\u6cd5\u5f8b\u4e8b\u5b9e\u62bd\u53d6\u5668\u3002\u8bf7\u4ece\u8f93\u5165\u6587\u672c\u62bd\u53d6\u7ed3\u6784\u5316\u4e8b\u5b9e\uff0c\u8f93\u51fa\u4e25\u683cJSON\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\\n"
        "\u5b57\u6bb5: fact_type, amount, time, relationship, actor, role, event, confidence\u3002\\n"
        "\u8981\u6c42:\\n"
        "1. \u4ec5\u6839\u636e\u6587\u672c\uff0c\u4e0d\u8981\u7f16\u9020\u3002\\n"
        "2. relationship/actor/role/event \u5fc5\u987b\u662f\u6570\u7ec4\u3002\\n"
        "3. confidence \u53d6 0~1 \u5c0f\u6570\u3002\\n"
        f"\u8f93\u5165\u6587\u672c:\\n{text}"
    )
    parsed = provider.generate_json(prompt=prompt, temperature=0.0)
    if not parsed:
        return {}
    return {
        "fact_type": str(parsed.get("fact_type", "unknown")),
        "amount": parsed.get("amount"),
        "time": parsed.get("time"),
        "relationship": parsed.get("relationship") if isinstance(parsed.get("relationship"), list) else [],
        "actor": parsed.get("actor") if isinstance(parsed.get("actor"), list) else [],
        "role": parsed.get("role") if isinstance(parsed.get("role"), list) else [],
        "event": parsed.get("event") if isinstance(parsed.get("event"), list) else [],
        "confidence": float(parsed.get("confidence", 0.6) or 0.6),
        "extractor_type": "llm",
    }


def extract_facts(text: str, use_llm: bool = False) -> Dict[str, Any]:
    rule = extract_facts_rule(text)
    if not use_llm:
        rule["fact_extraction_mode"] = "rule"
        rule["extraction_confidence"] = float(rule.get("confidence", 0.0) or 0.0)
        return rule

    try:
        llm = _extract_facts_llm(text)
        if llm:
            merged = dict(rule)
            for key in ["fact_type", "amount", "time"]:
                if llm.get(key):
                    merged[key] = llm[key]
            for key in ["relationship", "actor", "role", "event"]:
                merged[key] = _dedupe((rule.get(key) or []) + (llm.get(key) or []))
            merged["confidence"] = round(
                max(float(rule.get("confidence", 0.0) or 0.0), float(llm.get("confidence", 0.0) or 0.0)),
                4,
            )
            merged["extractor_type"] = "hybrid"
            merged["fact_extraction_mode"] = "hybrid"
            merged["extraction_confidence"] = merged["confidence"]
            return merged
    except Exception:
        pass

    rule["fact_extraction_mode"] = "rule"
    rule["extraction_confidence"] = float(rule.get("confidence", 0.0) or 0.0)
    return rule
