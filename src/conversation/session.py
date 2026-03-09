from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConversationSession:
    history: List[Dict[str, str]] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

