from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.config.settings import settings


def estimate_tokens(text: str) -> int:
    # Lightweight approximation for Chinese/English mixed text.
    if not text:
        return 0
    return max(1, int(len(text) * 0.6))


@dataclass
class ShortTermMemory:
    max_turns: int = settings.memory_max_turns
    max_tokens: int = settings.memory_max_tokens
    history: List[Dict[str, str]] = field(default_factory=list)

    def append_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.trim_by_turn_limit()
        self.trim_by_token_limit()

    def append_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.append_message("user", user_msg)
        self.append_message("assistant", assistant_msg)

    def get_recent_turns(self, k: int = settings.short_term_window_size) -> List[Dict[str, str]]:
        msg_count = max(2, int(k) * 2)
        return self.history[-msg_count:]

    def trim_by_turn_limit(self) -> None:
        max_msgs = max(2, self.max_turns * 2)
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def trim_by_token_limit(self) -> None:
        while self.history and self.token_size() > self.max_tokens:
            # Drop oldest message first.
            self.history.pop(0)

    def token_size(self) -> int:
        return sum(estimate_tokens(m.get("content", "")) for m in self.history)
