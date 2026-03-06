import json
from pathlib import Path
from typing import Any, Dict, List


def load_benchmark_dataset(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    return payload.get("examples", [])
