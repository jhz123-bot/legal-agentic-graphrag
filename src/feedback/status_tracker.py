from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


VALID_STATUS = {"open", "in_progress", "closed"}


def _normalize_status(value: str) -> str:
    v = (value or "open").strip().lower()
    return v if v in VALID_STATUS else "open"


def build_suggestion_id(suggestion: Dict[str, Any]) -> str:
    key = "|".join(
        [
            str(suggestion.get("suggestion_type", "")),
            str(suggestion.get("target", "")),
            str(suggestion.get("reason", "")),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def load_action_status(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): _normalize_status(str(v)) for k, v in data.items()}
    except Exception:
        return {}


def apply_action_status(suggestions: List[Dict[str, Any]], status_map: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in suggestions:
        row = dict(s)
        sid = str(row.get("suggestion_id") or build_suggestion_id(row))
        row["suggestion_id"] = sid
        row["status"] = _normalize_status(status_map.get(sid, row.get("status", "open")))
        out.append(row)
    return out


def save_action_status(path: str | Path, suggestions: List[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, str] = {}
    for s in suggestions:
        sid = str(s.get("suggestion_id") or build_suggestion_id(s))
        payload[sid] = _normalize_status(str(s.get("status", "open")))
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def update_action_status(path: str | Path, suggestion_id: str, status: str) -> Path:
    p = Path(path)
    mapping = load_action_status(p)
    mapping[str(suggestion_id)] = _normalize_status(status)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
