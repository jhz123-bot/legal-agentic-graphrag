import json
from pathlib import Path

from src.evaluation.benchmark_runner import run_benchmark


def main() -> None:
    root = Path(__file__).parent
    dataset_path = root / "data" / "sample_benchmark" / "legal_benchmark.json"
    summary = run_benchmark(dataset_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
