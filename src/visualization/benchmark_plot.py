from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_benchmark_results(summary: Dict[str, Any], output_dir: str | Path = "outputs/plots") -> List[str]:
    out = _ensure_dir(output_dir)
    results = summary.get("results", [])
    if not results:
        return []

    xs = list(range(1, len(results) + 1))
    acc = [r.get("answer_keyword_match_rate", 0.0) for r in results]
    lat = [r.get("latency", 0.0) for r in results]
    refl = [1 if r.get("reflection_triggered", False) else 0 for r in results]

    files: List[str] = []

    plt.figure(figsize=(8, 4))
    plt.plot(xs, acc, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Query Index")
    plt.ylabel("Answer Keyword Match")
    plt.title("Accuracy Over Queries")
    p1 = out / "accuracy_over_queries.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    files.append(str(p1))

    plt.figure(figsize=(8, 4))
    plt.hist(lat, bins=min(10, len(lat)))
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title("Latency Distribution")
    p2 = out / "latency_distribution.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    files.append(str(p2))

    plt.figure(figsize=(8, 4))
    plt.bar(xs, refl)
    plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Query Index")
    plt.ylabel("Reflection Triggered")
    plt.title("Reflection Trigger Frequency")
    p3 = out / "reflection_trigger_frequency.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()
    files.append(str(p3))

    return files
