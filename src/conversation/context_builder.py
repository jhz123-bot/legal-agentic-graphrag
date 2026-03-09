from __future__ import annotations

import json
from typing import Any, Dict, List

from src.config.settings import settings
from src.memory.memory_manager import MemoryManager
from src.memory.short_term_memory import estimate_tokens


def build_context_for_rewrite(memory_manager: MemoryManager) -> Dict[str, Any]:
    payload = memory_manager.build_context_payload()
    recent_history = payload.get("recent_history", [])
    facts = payload.get("facts", {})

    context_window: List[Dict[str, str]] = []
    if facts:
        context_window.append(
            {
                "role": "system",
                "content": f"关键事实：{json.dumps(facts, ensure_ascii=False)}",
            }
        )
    context_window.extend(recent_history)

    return {
        "recent_history": recent_history,
        "summary": "",
        "facts": facts,
        "context_window": context_window,
        "memory_context_size": sum(estimate_tokens(m.get("content", "")) for m in context_window),
    }


def build_context_for_reasoning(memory_manager: MemoryManager) -> Dict[str, Any]:
    payload = memory_manager.build_context_payload()
    recent_history = payload.get("recent_history", [])
    summary = payload.get("summary", "")
    facts = payload.get("facts", {})

    context_window: List[Dict[str, str]] = []
    if summary:
        context_window.append({"role": "system", "content": f"对话摘要：{summary}"})
    if facts:
        context_window.append(
            {
                "role": "system",
                "content": f"关键事实：{json.dumps(facts, ensure_ascii=False)}",
            }
        )
    context_window.extend(recent_history[-max(2, settings.short_term_window_size * 2) :])

    return {
        "recent_history": recent_history,
        "summary": summary,
        "facts": facts,
        "context_window": context_window,
        "memory_context_size": sum(estimate_tokens(m.get("content", "")) for m in context_window),
    }


def build_conversation_context(memory_manager: MemoryManager, for_stage: str = "reasoning") -> Dict[str, Any]:
    if for_stage == "rewrite":
        return build_context_for_rewrite(memory_manager)
    return build_context_for_reasoning(memory_manager)


def render_context_text(context_window: List[Dict[str, str]]) -> str:
    if not context_window:
        return "无"
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in context_window)
