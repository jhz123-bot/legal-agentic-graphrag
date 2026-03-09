from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from src.llm.base_provider import BaseLLMProvider


class BailianProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.timeout = timeout

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Bailian HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Bailian connection error: {exc.reason}") from exc

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
            "temperature": temperature,
        }
        payload.update(kwargs)
        data = self._post_json("/chat/completions", payload)
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()

