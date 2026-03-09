from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from src.llm.base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 60) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.timeout = timeout

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama connection error: {exc.reason}") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        payload.update(kwargs)
        data = self._post_json("/api/chat", payload)
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        return str(msg.get("content", "")).strip()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)
        data = self._post_json("/api/generate", payload)
        return str(data.get("response", "")).strip() if isinstance(data, dict) else ""

