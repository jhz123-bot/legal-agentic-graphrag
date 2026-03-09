from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.memory.short_term_memory import ShortTermMemory


@dataclass
class ConversationMemory:
    """Backward-compatible alias around ShortTermMemory."""

    max_messages: int = 20
    _stm: ShortTermMemory = field(init=False)

    def __post_init__(self) -> None:
        self._stm = ShortTermMemory(max_turns=max(1, self.max_messages // 2), max_tokens=10_000)

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._stm.history

    def append_message(self, role: str, content: str) -> None:
        self._stm.append_message(role, content)

    def append_user(self, content: str) -> None:
        self._stm.append_message("user", content)

    def append_assistant(self, content: str) -> None:
        self._stm.append_message("assistant", content)

    def get_recent_turns(self, k: int = 6) -> List[Dict[str, str]]:
        return self._stm.get_recent_turns(k)

    def build_context_window(self, k: int = 6) -> List[Dict[str, str]]:
        return self._stm.get_recent_turns(k)

    def turn_count(self) -> int:
        return len([m for m in self._stm.history if m.get("role") == "user"])
