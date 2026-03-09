import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.benchmark_schema import normalize_benchmark_dataset


def load_benchmark_dataset(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("examples", [])
    return normalize_benchmark_dataset(rows)
