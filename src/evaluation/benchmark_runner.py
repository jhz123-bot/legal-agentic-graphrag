import json
from pathlib import Path
from typing import Any, Dict, List

from src.benchmark.benchmark_runner import run_agent_benchmark


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    return payload.get("examples", [])


def run_benchmark(dataset_path: Path) -> Dict[str, Any]:
    # Keep compatibility with old API but return the upgraded evaluation summary.
    return run_agent_benchmark(dataset_path)
