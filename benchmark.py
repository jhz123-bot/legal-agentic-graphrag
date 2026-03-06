import json
from pathlib import Path

from src.benchmark.benchmark_runner import run_agent_benchmark
from src.visualization.benchmark_plot import plot_benchmark_results


def main() -> None:
    root = Path(__file__).parent
    dataset_path = root / "data" / "benchmark" / "legal_benchmark.json"
    summary = run_agent_benchmark(dataset_path)
    plot_files = plot_benchmark_results(summary, output_dir=root / "outputs" / "plots")
    summary["plot_files"] = plot_files
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
