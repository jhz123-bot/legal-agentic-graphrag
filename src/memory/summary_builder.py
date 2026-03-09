from __future__ import annotations

from typing import Any, Dict, List

from src.llm.llm_router import get_llm_provider


def build_summary_rule(history: List[Dict[str, str]]) -> Dict[str, Any]:
    if not history:
        return {
            "topic": "\u672a\u660e\u786e",
            "discussed_issues": [],
            "cited_laws": [],
            "unresolved_questions": [],
            "summary_text": "\u6682\u65e0\u5bf9\u8bdd\u6458\u8981\u3002",
            "confidence": 0.35,
            "summary_build_mode": "rule",
        }

    merged = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history)
    topic = "\u672a\u660e\u786e"
    for t in ["\u76d7\u7a83", "\u8bc8\u9a97", "\u8fdd\u7ea6", "\u79df\u8d41", "\u4fb5\u6743", "\u52b3\u52a8", "\u516c\u53f8", "\u4e70\u5356", "\u501f\u8d37"]:
        if t in merged:
            topic = t
            break

    discussed_issues: List[str] = []
    for kw in ["\u8d23\u4efb", "\u8d54\u507f", "\u8bc1\u636e", "\u6784\u6210\u8981\u4ef6", "\u8fdd\u7ea6", "\u8fc7\u9519", "\u635f\u5bb3", "\u89e3\u9664"]:
        if kw in merged:
            discussed_issues.append(kw)

    cited_laws: List[str] = []
    for law in ["\u5211\u6cd5", "\u6c11\u6cd5\u5178", "\u516c\u53f8\u6cd5", "\u52b3\u52a8\u5408\u540c\u6cd5"]:
        if law in merged:
            cited_laws.append(law)

    unresolved_questions: List[str] = []
    for m in reversed(history):
        c = (m.get("content") or "").strip()
        if m.get("role") == "user" and (c.endswith("\uff1f") or c.endswith("?")):
            unresolved_questions.append(c)
            if len(unresolved_questions) >= 2:
                break

    summary_text = (
        f"\u54a8\u8be2\u4e3b\u9898\uff1a{topic}\uff1b"
        f"\u5df2\u8ba8\u8bba\u95ee\u9898\uff1a{('\u3001'.join(discussed_issues) if discussed_issues else '\u5f85\u8865\u5145')}\uff1b"
        f"\u5df2\u5f15\u7528\u6cd5\u6761\uff1a{('\u3001'.join(cited_laws) if cited_laws else '\u5f85\u8865\u5145')}\uff1b"
        f"\u5f85\u89e3\u51b3\uff1a{('\uff1b'.join(unresolved_questions) if unresolved_questions else '\u6682\u65e0')}"
    )

    return {
        "topic": topic,
        "discussed_issues": discussed_issues,
        "cited_laws": cited_laws,
        "unresolved_questions": unresolved_questions,
        "summary_text": summary_text,
        "confidence": 0.55,
        "summary_build_mode": "rule",
    }


def _build_summary_llm(history: List[Dict[str, str]]) -> Dict[str, Any]:
    provider = get_llm_provider(timeout=20)
    history_text = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history[-12:])
    prompt = (
        "\u4f60\u662f\u6cd5\u5f8b\u5bf9\u8bdd\u6458\u8981\u5668\u3002\u6839\u636e\u591a\u8f6e\u5386\u53f2\uff0c\u8f93\u51fa\u4e25\u683cJSON\uff0c\u4e0d\u8981\u89e3\u91ca\u3002\\n"
        "\u5b57\u6bb5: topic, discussed_issues, cited_laws, unresolved_questions, summary_text, confidence\u3002\\n"
        "\u8981\u6c42:\\n"
        "1. \u4fdd\u7559\u6cd5\u5f8b\u672f\u8bed\u3002\\n"
        "2. \u4e0d\u8981\u7f16\u9020\u5386\u53f2\u4e2d\u4e0d\u5b58\u5728\u7684\u6cd5\u6761\u6216\u4e8b\u5b9e\u3002\\n"
        "3. discussed_issues/cited_laws/unresolved_questions \u5fc5\u987b\u662f\u6570\u7ec4\u3002\\n"
        f"\u5386\u53f2:\\n{history_text}"
    )
    parsed = provider.generate_json(prompt=prompt, temperature=0.0)
    if not parsed:
        return {}
    return {
        "topic": str(parsed.get("topic", "\u672a\u660e\u786e")),
        "discussed_issues": parsed.get("discussed_issues") if isinstance(parsed.get("discussed_issues"), list) else [],
        "cited_laws": parsed.get("cited_laws") if isinstance(parsed.get("cited_laws"), list) else [],
        "unresolved_questions": parsed.get("unresolved_questions") if isinstance(parsed.get("unresolved_questions"), list) else [],
        "summary_text": str(parsed.get("summary_text", "")),
        "confidence": float(parsed.get("confidence", 0.65) or 0.65),
        "summary_build_mode": "llm",
    }


def build_summary(history: List[Dict[str, str]], use_llm: bool = False) -> Dict[str, Any]:
    rule = build_summary_rule(history)
    if not use_llm:
        return rule

    try:
        llm = _build_summary_llm(history)
        if llm:
            merged = dict(rule)
            for key in ["topic", "summary_text"]:
                if llm.get(key):
                    merged[key] = llm[key]
            for key in ["discussed_issues", "cited_laws", "unresolved_questions"]:
                merged[key] = list(dict.fromkeys((rule.get(key) or []) + (llm.get(key) or [])))
            merged["confidence"] = round(max(float(rule.get("confidence", 0.0)), float(llm.get("confidence", 0.0))), 4)
            merged["summary_build_mode"] = "hybrid"
            return merged
    except Exception:
        pass

    return rule
