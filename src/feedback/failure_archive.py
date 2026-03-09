from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _norm_reason(reason: str) -> str:
    value = (reason or "unknown").strip().lower().replace(" ", "_")
    return value or "unknown"


def archive_failure_records(records: List[Dict[str, Any]], output_root: str | Path) -> Dict[str, Any]:
    root = Path(output_root)
    date_key = datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = root / date_key
    day_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        reason = _norm_reason(str(r.get("failure_reason", "unknown")))
        grouped.setdefault(reason, []).append(r)

    archived_files: Dict[str, str] = {}
    for reason, rows in grouped.items():
        path = day_dir / f"{reason}_failures.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        archived_files[reason] = str(path)

    index = {
        "date": date_key,
        "total_records": len(records),
        "files": archived_files,
        "by_question_type": {},
        "by_source_type": {},
    }

    for r in records:
        qtype = str(r.get("question_type", "unknown") or "unknown")
        stype = str(r.get("source_type", "unknown") or "unknown")
        index["by_question_type"][qtype] = index["by_question_type"].get(qtype, 0) + 1
        index["by_source_type"][stype] = index["by_source_type"].get(stype, 0) + 1

    index_path = day_dir / "failure_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = root / "latest_failure_index.json"
    latest_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "archive_root": str(root),
        "day_dir": str(day_dir),
        "index_path": str(index_path),
        "latest_index_path": str(latest_path),
        "archived_files": archived_files,
        "count": len(records),
    }
