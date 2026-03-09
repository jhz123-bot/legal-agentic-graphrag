from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from src.config.settings import settings
from src.memory.fact_conflict_resolver import detect_conflict, resolve_conflict
from src.memory.fact_extractor import extract_facts


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _mk_fact_record(
    key: str,
    value: Any,
    confidence: float,
    source_type: str,
    source_turn: int,
    extractor_type: str,
) -> Dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "confidence": float(confidence),
        "source_type": source_type,
        "source_turn": int(source_turn),
        "timestamp": _now_iso(),
        "extractor_type": extractor_type,
    }


@dataclass
class FactMemory:
    facts: Dict[str, Any] = field(
        default_factory=lambda: {
            "fact_records": {},
            "normalized": {
                "amount": None,
                "time": None,
                "relationship": [],
                "actor": [],
                "role": [],
                "event": [],
            },
            "conflict": {
                "has_conflict": False,
                "conflict_fields": [],
                "events": [],
            },
            "fact_extraction_mode": "rule",
            "extraction_confidence": 0.0,
        }
    )

    def _source_type_from_role(self, role: str) -> str:
        role = (role or "unknown").lower()
        if role == "assistant":
            return "assistant"
        if role == "user":
            return "user"
        return "unknown"

    def _merge_list_fact(self, key: str, values: List[Any], confidence: float, source_type: str, source_turn: int, extractor_type: str) -> Dict[str, Any]:
        changed = False
        existing = self.facts["normalized"].get(key) or []
        merged = list(existing)
        for v in values:
            if v and v not in merged:
                merged.append(v)
        if merged != existing:
            changed = True
            self.facts["normalized"][key] = merged
        if changed:
            self.facts["fact_records"][key] = _mk_fact_record(
                key=key,
                value=merged,
                confidence=confidence,
                source_type=source_type,
                source_turn=source_turn,
                extractor_type=extractor_type,
            )
        return {"updated": changed, "conflict": False, "resolved_value": merged}

    def _update_scalar_fact(self, key: str, value: Any, confidence: float, source_type: str, source_turn: int, extractor_type: str) -> Dict[str, Any]:
        if value in (None, ""):
            return {"updated": False, "conflict": False}

        current_record = self.facts["fact_records"].get(key)
        new_record = _mk_fact_record(
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_turn=source_turn,
            extractor_type=extractor_type,
        )

        if not current_record:
            self.facts["fact_records"][key] = new_record
            self.facts["normalized"][key] = value
            return {"updated": True, "conflict": False, "resolved_value": value}

        has_conflict = detect_conflict(current_record, new_record)
        if not has_conflict:
            # Same value but possibly higher confidence.
            if float(new_record.get("confidence", 0.0)) > float(current_record.get("confidence", 0.0)):
                self.facts["fact_records"][key] = new_record
            return {"updated": False, "conflict": False, "resolved_value": current_record.get("value")}

        result = resolve_conflict(
            old_fact=current_record,
            new_fact=new_record,
            strategy=settings.fact_conflict_strategy,
        )
        chosen = result.get("chosen", new_record)
        self.facts["fact_records"][key] = chosen
        self.facts["normalized"][key] = chosen.get("value")

        conflict_state = self.facts["conflict"]
        conflict_state["has_conflict"] = True
        if key not in conflict_state["conflict_fields"]:
            conflict_state["conflict_fields"].append(key)
        if settings.enable_conflict_logging:
            conflict_state["events"].append(
                {
                    "fact_conflict_detected": True,
                    "conflict_key": key,
                    "old_value": current_record.get("value"),
                    "new_value": new_record.get("value"),
                    "chosen_value": chosen.get("value"),
                    "resolution_strategy": result.get("resolution_strategy", settings.fact_conflict_strategy),
                    "scores": result.get("scores", {}),
                    "uncertain_fact": bool(result.get("uncertain_fact", False)),
                }
            )

        return {
            "updated": True,
            "conflict": True,
            "resolved_value": chosen.get("value"),
            "resolution_strategy": result.get("resolution_strategy", settings.fact_conflict_strategy),
        }

    def update_from_turn(self, text: str, source_turn: int, role: str = "user") -> Dict[str, Any]:
        extracted = extract_facts(text, use_llm=settings.enable_llm_memory_extraction)
        mode = extracted.get("fact_extraction_mode", "rule")
        confidence = float(extracted.get("extraction_confidence", extracted.get("confidence", 0.0)) or 0.0)
        extractor_type = str(extracted.get("extractor_type", mode))
        source_type = self._source_type_from_role(role)

        self.facts["fact_extraction_mode"] = mode
        self.facts["extraction_confidence"] = confidence

        changes: Dict[str, Any] = {}
        conflicts: List[Dict[str, Any]] = []

        scalar_keys = ["amount", "time"]
        list_keys = ["relationship", "actor", "role", "event"]

        for k in scalar_keys:
            result = self._update_scalar_fact(
                key=k,
                value=extracted.get(k),
                confidence=confidence,
                source_type=source_type,
                source_turn=source_turn,
                extractor_type=extractor_type,
            )
            if result.get("updated"):
                changes[k] = self.facts["normalized"].get(k)
            if result.get("conflict"):
                conflicts.append(
                    {
                        "fact_conflict_detected": True,
                        "conflict_key": k,
                        "chosen_value": result.get("resolved_value"),
                        "resolution_strategy": result.get("resolution_strategy", settings.fact_conflict_strategy),
                    }
                )

        for k in list_keys:
            vals = extracted.get(k) if isinstance(extracted.get(k), list) else []
            result = self._merge_list_fact(
                key=k,
                values=vals,
                confidence=confidence,
                source_type=source_type,
                source_turn=source_turn,
                extractor_type=extractor_type,
            )
            if result.get("updated"):
                changes[k] = self.facts["normalized"].get(k)

        return {
            "updated": bool(changes),
            "changes": changes,
            "conflicts": conflicts,
            "fact_extraction_mode": mode,
            "extraction_confidence": confidence,
        }

    def get_facts(self) -> Dict[str, Any]:
        out = dict(self.facts["normalized"])
        out["conflict"] = self.facts["conflict"]
        out["fact_extraction_mode"] = self.facts.get("fact_extraction_mode", "rule")
        out["extraction_confidence"] = self.facts.get("extraction_confidence", 0.0)
        out["fact_records"] = self.facts.get("fact_records", {})
        return out

    # Backward-compatible helpers.
    def to_dict(self) -> Dict[str, Any]:
        return self.get_facts()

    def update_from_message(self, role: str, text: str, source_turn: int = 0) -> Dict[str, Any]:
        return self.update_from_turn(text=text, source_turn=source_turn or 0, role=role)
