import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.agents.workflow import run_agentic_graphrag
from src.evaluation.metrics import (
    compute_answer_keyword_match_rate,
    compute_entity_hit_rate,
    compute_evidence_path_hit_rate,
    compute_reflection_trigger_rate,
)


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    return payload.get("examples", [])


def run_benchmark(dataset_path: Path) -> Dict[str, Any]:
    examples = load_benchmark(dataset_path)
    results = []
    latencies = []
    reflection_trigger_flags = []

    for index, example in enumerate(examples):
        query = example["query"]
        expected_entities = example.get("expected_entities", [])
        expected_paths = example.get("expected_evidence_paths", expected_entities)
        expected_keywords = example.get("expected_answer_keywords", [])

        start = time.perf_counter()
        state = run_agentic_graphrag(query=query)
        latency = time.perf_counter() - start
        latencies.append(latency)

        final_answer = state.get("final_answer", {})
        answer_blob = " ".join(
            [
                final_answer.get("short_answer", ""),
                final_answer.get("reasoning_summary", ""),
                final_answer.get("uncertainty_note", ""),
            ]
        )

        entity_hit = compute_entity_hit_rate(expected_entities, state.get("linked_entities", []))
        evidence_hit = compute_evidence_path_hit_rate(expected_paths, state.get("evidence_pack", {}))
        keyword_match = compute_answer_keyword_match_rate(expected_keywords, answer_blob)
        logs = state.get("logs", [])
        reflection_triggered = any("decision=re-retrieve" in log or "decision=re-reason" in log for log in logs)
        reflection_trigger_flags.append(reflection_triggered)

        results.append(
            {
                "index": index,
                "query": query,
                "entity_hit_rate": entity_hit,
                "evidence_path_hit_rate": evidence_hit,
                "answer_keyword_match_rate": keyword_match,
                "reflection_triggered": reflection_triggered,
                "latency_sec": latency,
            }
        )

    if not results:
        return {
            "num_examples": 0,
            "entity_hit_rate": 0.0,
            "evidence_path_hit_rate": 0.0,
            "answer_keyword_match_rate": 0.0,
            "reflection_trigger_rate": 0.0,
            "average_latency": 0.0,
            "results": [],
        }

    return {
        "num_examples": len(results),
        "entity_hit_rate": sum(r["entity_hit_rate"] for r in results) / len(results),
        "evidence_path_hit_rate": sum(r["evidence_path_hit_rate"] for r in results) / len(results),
        "answer_keyword_match_rate": sum(r["answer_keyword_match_rate"] for r in results) / len(results),
        "reflection_trigger_rate": compute_reflection_trigger_rate(reflection_trigger_flags),
        "average_latency": sum(latencies) / len(latencies),
        "results": results,
    }
