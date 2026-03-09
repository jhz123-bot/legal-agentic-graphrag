from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.settings import settings
from src.memory.fact_memory import FactMemory
from src.memory.short_term_memory import ShortTermMemory, estimate_tokens
from src.memory.summary_memory import SummaryMemory


@dataclass
class MemoryManager:
    short_term: ShortTermMemory = field(
        default_factory=lambda: ShortTermMemory(
            max_turns=settings.memory_max_turns,
            max_tokens=settings.memory_max_tokens,
        )
    )
    summary_memory: SummaryMemory = field(default_factory=SummaryMemory)
    fact_memory: FactMemory = field(default_factory=FactMemory)
    turn_counter: int = 0

    def append_turn(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        self.turn_counter += 1
        self.short_term.append_turn(user_msg, assistant_msg)

        fact_user = self.fact_memory.update_from_turn(user_msg, source_turn=self.turn_counter, role="user")
        fact_assistant = self.fact_memory.update_from_turn(assistant_msg, source_turn=self.turn_counter, role="assistant")

        summary_updated = False
        trigger_reason = "none"
        history = self.short_term.history
        total_tokens = self.short_term.token_size()
        user_turns = len([m for m in history if m.get("role") == "user"])

        if user_turns >= settings.summary_trigger_turns:
            result = self.summary_memory.update_summary(history)
            summary_updated = bool(result.get("updated"))
            trigger_reason = "turn_threshold"
        elif total_tokens >= settings.summary_trigger_tokens:
            result = self.summary_memory.update_summary(history)
            summary_updated = bool(result.get("updated"))
            trigger_reason = "token_threshold"

        # If summary updated, keep only recent window in short-term.
        if summary_updated:
            self.short_term.history = self.short_term.get_recent_turns(settings.short_term_window_size)

        return {
            "summary_updated": summary_updated,
            "summary_trigger_reason": trigger_reason,
            "summary_build_mode": result.get("summary_build_mode", "rule") if 'result' in locals() else "rule",
            "summary_extraction_confidence": result.get("extraction_confidence", 0.0) if 'result' in locals() else 0.0,
            "fact_updates": {
                "user": fact_user,
                "assistant": fact_assistant,
            },
            "fact_extraction_mode": fact_assistant.get("fact_extraction_mode")
            or fact_user.get("fact_extraction_mode")
            or "rule",
            "fact_extraction_confidence": max(
                float(fact_user.get("extraction_confidence", 0.0) or 0.0),
                float(fact_assistant.get("extraction_confidence", 0.0) or 0.0),
            ),
        }

    def build_context_payload(self) -> Dict[str, Any]:
        recent_history = self.short_term.get_recent_turns(settings.short_term_window_size)
        summary = self.summary_memory.get_summary()
        facts = self.fact_memory.get_facts()
        summary_payload = self.summary_memory.get_summary_payload()

        context_window: List[Dict[str, str]] = []
        if summary:
            context_window.append({"role": "system", "content": f"对话摘要：{summary}"})
        context_window.extend(recent_history)

        return {
            "recent_history": recent_history,
            "summary": summary,
            "facts": facts,
            "summary_payload": summary_payload,
            "context_window": context_window,
            "memory_context_size": estimate_tokens(summary) + sum(estimate_tokens(m.get("content", "")) for m in recent_history),
        }

    def get_memory_state(self) -> Dict[str, Any]:
        facts = self.fact_memory.get_facts()
        conflict = facts.get("conflict", {})
        return {
            "short_term_turn_count": len([m for m in self.short_term.history if m.get("role") == "user"]),
            "short_term_message_count": len(self.short_term.history),
            "short_term_tokens": self.short_term.token_size(),
            "summary": self.summary_memory.get_summary(),
            "summary_payload": self.summary_memory.get_summary_payload(),
            "fact_snapshot": facts,
            "has_conflict": bool(conflict.get("has_conflict", False)),
            "conflict_fields": conflict.get("conflict_fields", []),
            "fact_extraction_mode": facts.get("fact_extraction_mode", "rule"),
            "summary_build_mode": self.summary_memory.get_summary_payload().get("summary_build_mode", "rule"),
            "extraction_confidence": max(
                float(facts.get("extraction_confidence", 0.0) or 0.0),
                float(self.summary_memory.get_summary_payload().get("confidence", 0.0) or 0.0),
            ),
        }
