from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheStats:
    hits: int = 0
    miss: int = 0


class LRUCacheStore:
    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max(1, int(max_size))
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
