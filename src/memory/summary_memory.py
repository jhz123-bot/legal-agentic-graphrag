from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config.settings import settings
from src.memory.short_term_memory import estimate_tokens
from src.memory.summary_builder import build_summary


@dataclass
class SummaryMemory:
    summary: str = ""
    summary_payload: Dict[str, Any] = field(default_factory=dict)

    def update_summary(self, history: List[Dict[str, str]]) -> Dict[str, object]:
        if not history:
            return {
                "updated": False,
                "summary": self.summary,
                "reason": "empty_history",
                "summary_build_mode": "rule",
                "extraction_confidence": 0.0,
            }

        payload = build_summary(history, use_llm=settings.enable_llm_memory_extraction)
        summary_text = str(payload.get("summary_text", "")).strip() or self.summary
        changed = summary_text != self.summary or payload != self.summary_payload
        if changed:
            self.summary = summary_text
            self.summary_payload = payload

        return {
            "updated": changed,
            "summary": self.summary,
            "tokens": estimate_tokens(self.summary),
            "summary_build_mode": payload.get("summary_build_mode", "rule"),
            "extraction_confidence": float(payload.get("confidence", 0.0) or 0.0),
            "summary_payload": payload,
        }

    def get_summary(self) -> str:
        return self.summary

    def get_summary_payload(self) -> Dict[str, Any]:
        return dict(self.summary_payload)
