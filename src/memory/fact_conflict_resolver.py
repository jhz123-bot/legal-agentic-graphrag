from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from src.config.settings import settings


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def detect_conflict(old_fact: Dict[str, Any], new_fact: Dict[str, Any]) -> bool:
    if not old_fact:
        return False
    return old_fact.get("value") != new_fact.get("value") and new_fact.get("value") is not None


def _score(fact: Dict[str, Any]) -> float:
    confidence = _to_float(fact.get("confidence"), 0.5)
    source_type = str(fact.get("source_type", "unknown"))
    rel = settings.source_reliability_weights.get(source_type, settings.source_reliability_weights.get("unknown", 0.5))

    turn = _to_float(fact.get("source_turn"), 0.0)
    recency = min(1.0, 0.5 + turn / 20.0)
    return round(0.55 * confidence + 0.3 * rel + 0.15 * recency, 6)


def resolve_conflict(
    old_fact: Dict[str, Any],
    new_fact: Dict[str, Any],
    strategy: str = "confidence_weighted",
) -> Dict[str, Any]:
    if not old_fact:
        return {"chosen": new_fact, "resolution_strategy": strategy, "uncertain_fact": False}

    if strategy == "latest_wins":
        chosen = new_fact
        return {"chosen": chosen, "resolution_strategy": "latest_wins", "uncertain_fact": False}

    old_score = _score(old_fact)
    new_score = _score(new_fact)
    uncertain = abs(new_score - old_score) < 0.05

    if new_score >= old_score:
        chosen = dict(new_fact)
        chosen["confidence"] = max(_to_float(new_fact.get("confidence"), 0.0), _to_float(old_fact.get("confidence"), 0.0))
    else:
        chosen = dict(old_fact)

    chosen["resolved_at"] = datetime.utcnow().isoformat()

    return {
        "chosen": chosen,
        "resolution_strategy": "confidence_weighted",
        "scores": {"old": old_score, "new": new_score},
        "uncertain_fact": uncertain,
    }
