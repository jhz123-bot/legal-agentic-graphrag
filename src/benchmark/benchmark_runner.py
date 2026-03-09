import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np

from src.agents.workflow import run_agentic_graphrag
from src.benchmark.dataset_loader import load_benchmark_dataset
from src.common.models import Document
from src.embedding.embedding_model import EmbeddingModel
from src.evaluation.error_analyzer import analyze_single_error, aggregate_errors
from src.evaluation.evaluator import (
    evaluate_by_question_type,
    evaluate_by_source_type,
    evaluate_multiturn,
    evaluate_overall,
)
from src.evaluation.metrics import (
    compute_answer_keyword_match_rate,
    compute_citation_correctness,
    compute_citation_coverage,
    compute_entity_hit_rate,
    compute_evidence_path_hit_rate,
    compute_reflection_trigger_rate,
)
from src.evaluation.report_generator import generate_error_analysis_report
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_chunker import chunk_documents
from src.indexing.graph_loader import load_graph
from src.indexing.vector_index_loader import load_vector_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector_store.vector_index import FaissVectorIndex
from src.agents.workflow import build_graph_store_from_documents

_EVAL_EMBEDDER: EmbeddingModel | None = None


def _select_examples_for_small(examples: List[Dict[str, Any]], max_examples: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """Prefer bucket coverage for --small mode instead of taking the first N only."""
    if max_examples <= 0 or not examples:
        return []

    selected: List[Dict[str, Any]] = []
    used_ids = set()
    priority_qtypes = ["statute_lookup", "case_reasoning", "multiturn_followup", "concept_definition"]
    rng = random.Random(seed if seed is not None else 0)

    # First pass: one sample per question type (if available).
    for qtype in priority_qtypes:
        if len(selected) >= max_examples:
            break
        candidates = [row for row in examples if row.get("id") not in used_ids and str(row.get("question_type", "")) == qtype]
        if candidates:
            pick = rng.choice(candidates)
            selected.append(pick)
            used_ids.add(pick.get("id"))

    # Second pass: fill remaining with original order.
    if len(selected) < max_examples:
        for row in examples:
            rid = row.get("id")
            if rid in used_ids:
                continue
            selected.append(row)
            used_ids.add(rid)
            if len(selected) >= max_examples:
                break
    return selected


def _lexical_overlap_score(text_a: str, text_b: str) -> float:
    a = set((text_a or "").strip())
    b = set((text_b or "").strip())
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _semantic_similarity(text_a: str, text_b: str) -> float:
    global _EVAL_EMBEDDER
    try:
        if _EVAL_EMBEDDER is None:
            _EVAL_EMBEDDER = EmbeddingModel()
        va = _EVAL_EMBEDDER.embed_text(text_a)
        vb = _EVAL_EMBEDDER.embed_text(text_b)
        return _cosine_similarity(va, vb)
    except Exception:
        # Fallback: keep benchmark running even if embedding provider is unavailable.
        return _lexical_overlap_score(text_a, text_b)


def _multiturn_metrics(sample: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, float]:
    if not bool(sample.get("requires_multiturn", False)):
        return {
            "context_resolution_accuracy": 0.0,
            "rewrite_trigger_accuracy": 0.0,
            "rewritten_query_quality": 0.0,
        }

    expected_entities = sample.get("expected_entities", [])
    rewrite_info = state.get("rewrite_info", {})
    original_query = str(sample.get("query", ""))
    rewritten_query = str(state.get("rewritten_query", sample.get("query", "")))
    final_answer = state.get("final_answer", {})
    answer_blob = " ".join([
        str(final_answer.get("short_answer", "")),
        str(final_answer.get("reasoning_summary", "")),
    ])

    triggered = bool(rewrite_info.get("triggered", False))
    rewrite_trigger_accuracy = 1.0 if triggered else 0.0

    rq_hits = 0
    ans_hits = 0
    for e in expected_entities:
        if e and e in rewritten_query:
            rq_hits += 1
        if e and e in answer_blob:
            ans_hits += 1
    denom = max(1, len(expected_entities))
    entity_coverage = rq_hits / denom
    # Baseline quality: entity coverage + semantic similarity with original query to avoid rewrite drift.
    semantic_score = _semantic_similarity(original_query, rewritten_query)
    rewritten_query_quality = 0.7 * entity_coverage + 0.3 * semantic_score
    context_resolution_accuracy = max(rq_hits, ans_hits) / denom

    return {
        "context_resolution_accuracy": context_resolution_accuracy,
        "rewrite_trigger_accuracy": rewrite_trigger_accuracy,
        "rewritten_query_quality": rewritten_query_quality,
    }


def _build_hybrid_components_for_benchmark() -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[2]
    docs_dir = root / "data" / "sample_legal_docs"
    index_dir = root / "data" / "index"
    graph_dir = root / "data" / "graph"

    loaded_docs = load_documents_from_dir(docs_dir)
    if not loaded_docs:
        return None, None

    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    try:
        vector_index, _, chunks = load_vector_index(index_dir)
    except Exception:
        chunks = chunk_documents(loaded_docs)
        embeddings = embedding_model.embed_chunks(chunks)
        vector_index = FaissVectorIndex()
        vector_index.build_index(chunks, embeddings)

    try:
        graph_store = load_graph(graph_dir)
    except Exception:
        graph_documents = [Document(doc_id=c.chunk_id, title=c.title, text=c.text) for c in chunks]
        graph_store = build_graph_store_from_documents(graph_documents)

    graph_retriever = GraphRetriever(graph_store)
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)
    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        embedding_model=embedding_model,
        graph_retriever=graph_retriever,
        bm25_retriever=bm25_retriever,
    )
    return graph_store, hybrid_retriever


def run_agent_benchmark(dataset_path: str | Path, max_examples: int | None = None, seed: int | None = None) -> Dict[str, Any]:
    examples = load_benchmark_dataset(dataset_path)
    if max_examples is not None:
        examples = _select_examples_for_small(examples, max(0, max_examples), seed=seed)

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    reflection_flags: List[bool] = []
    error_records: List[Dict[str, Any]] = []
    graph_store, hybrid_retriever = _build_hybrid_components_for_benchmark()

    for idx, sample in enumerate(examples):
        query = sample.get("query", "")
        expected_entities = sample.get("expected_entities", [])
        expected_paths = sample.get("expected_evidence_paths", sample.get("expected_entities", []))
        expected_keywords = sample.get("expected_answer_keywords", [])

        start = time.perf_counter()
        state = run_agentic_graphrag(query=query, graph_store=graph_store, hybrid_retriever=hybrid_retriever)
        latency = time.perf_counter() - start
        latencies.append(latency)

        final_answer = state.get("final_answer", {})
        answer_blob = " ".join(
            [
                str(final_answer.get("short_answer", "")),
                str(final_answer.get("reasoning_summary", "")),
                str(final_answer.get("uncertainty_note", "")),
            ]
        )

        entity_hit = compute_entity_hit_rate(expected_entities, state.get("linked_entities", []))
        evidence_hit = compute_evidence_path_hit_rate(expected_paths, state.get("evidence_pack", {}))
        keyword_hit = compute_answer_keyword_match_rate(expected_keywords, answer_blob)
        reflection_triggered = state.get("verification_result", {}).get("decision") in {"re-retrieve", "re-reason"}
        reflection_flags.append(bool(reflection_triggered))

        citations = final_answer.get("citations", [])
        claims = state.get("reasoning_trace", {}).get("claims", [])
        citation_correctness = compute_citation_correctness(citations, state.get("evidence_pack", {}))
        grounded_evidence = final_answer.get("grounded_evidence", [])
        citation_coverage = compute_citation_coverage(claims, citations, grounded_evidence=grounded_evidence)
        grounding_score = float(final_answer.get("grounding", {}).get("grounding_score", 0.0) or 0.0)
        evidence_pack = state.get("evidence_pack", {}) or {}
        ranked_evidence = state.get("ranked_evidence", evidence_pack.get("reranked_paths", evidence_pack.get("ranked_paths", [])))
        ranked_evidence_ids = [
            str(x.get("evidence_id", ""))
            for x in ranked_evidence
            if isinstance(x, dict) and str(x.get("evidence_id", "")).strip()
        ]

        mt = _multiturn_metrics(sample, state)

        row = {
            "index": idx,
            "id": sample.get("id", idx + 1),
            "query": query,
            "question_type": sample.get("question_type", "concept_definition"),
            "source_type": sample.get("source_type", "hybrid"),
            "requires_multiturn": bool(sample.get("requires_multiturn", False)),
            "expected_retrieval_strategy": sample.get("expected_retrieval_strategy", ""),
            "actual_retrieval_strategy": state.get("retrieval_strategy", ""),
            "entity_hit_rate": entity_hit,
            "evidence_path_hit_rate": evidence_hit,
            "answer_keyword_match_rate": keyword_hit,
            "reflection_triggered": bool(reflection_triggered),
            "latency": latency,
            "confidence": final_answer.get("confidence", 0.0),
            "citation_correctness": citation_correctness,
            "citation_coverage": citation_coverage,
            "grounding_score": grounding_score,
            **mt,
            "rewritten_query": state.get("rewritten_query", query),
            "evidence_pack_summary": {
                "graph_nodes": len(evidence_pack.get("nodes", [])),
                "graph_edges": len(evidence_pack.get("edges", [])),
                "vector_hits": len(evidence_pack.get("vector_hits", [])),
                "candidate_evidence_count": len(evidence_pack.get("candidate_evidence", [])),
                "ranked_evidence_count": len(ranked_evidence),
            },
            "ranked_evidence_ids": ranked_evidence_ids,
            "final_answer": {
                "short_answer": final_answer.get("short_answer", ""),
                "confidence": final_answer.get("confidence", 0.0),
                "reflection_decision": final_answer.get("reflection_decision", ""),
                "grounding": final_answer.get("grounding", {}),
            },
            "predicted_error_type": "",  # backfilled later by error analyzer output if needed
        }
        results.append(row)

        err = analyze_single_error(sample, row, state)
        error_records.append(err)

    overall = evaluate_overall(results)
    overall["reflection_trigger_rate"] = compute_reflection_trigger_rate(reflection_flags)

    by_qtype = evaluate_by_question_type(results)
    by_source = evaluate_by_source_type(results)
    multiturn_summary = evaluate_multiturn(results)

    errors_summary = aggregate_errors(error_records)
    report = generate_error_analysis_report(
        overall_metrics=overall,
        question_type_summary=by_qtype,
        source_type_summary=by_source,
        multiturn_summary=multiturn_summary,
        errors_summary=errors_summary,
        results=results,
        error_records=error_records,
    )

    # Keep backward-compatible top-level keys.
    return {
        "num_examples": len(results),
        "entity_hit_rate": float(mean([r.get("entity_hit_rate", 0.0) for r in results])) if results else 0.0,
        "evidence_path_hit_rate": float(mean([r.get("evidence_path_hit_rate", 0.0) for r in results])) if results else 0.0,
        "answer_keyword_match_rate": float(mean([r.get("answer_keyword_match_rate", 0.0) for r in results])) if results else 0.0,
        "latency": overall.get("average_latency", 0.0),
        "reflection_trigger_rate": overall.get("reflection_trigger_rate", 0.0),
        "citation_correctness": overall.get("citation_correctness", 0.0),
        "citation_coverage": overall.get("citation_coverage", 0.0),
        "grounding_score": overall.get("grounding_score", 0.0),
        "results": results,
        "failures": [e for e in error_records if e.get("primary_error_type") != "none"],
        "overall_metrics": overall,
        "metrics_by_question_type": by_qtype,
        "metrics_by_source_type": by_source,
        "multiturn_summary": multiturn_summary,
        "error_type_distribution": errors_summary.get("error_type_distribution", {}),
        "error_type_ratio": errors_summary.get("error_type_ratio", {}),
        "stage_bottleneck_summary": report.get("stage_bottleneck_summary", {}),
        "error_analysis_report": report,
    }


def save_benchmark_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
