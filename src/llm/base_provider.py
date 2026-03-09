from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages=messages, model=model, temperature=temperature, **kwargs)

    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raw = self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = text.removesuffix("```").strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

