from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default


def _env_json_dict(name: str, default: Dict[str, float]) -> Dict[str, float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            out: Dict[str, float] = {}
            for k, v in parsed.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out or default
    except Exception:
        return default
    return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass
class Settings:
    # Cache controls.
    enable_cache: bool = _env_bool("ENABLE_CACHE", True)
    enable_embedding_cache: bool = _env_bool("ENABLE_EMBEDDING_CACHE", True)
    enable_retrieval_cache: bool = _env_bool("ENABLE_RETRIEVAL_CACHE", True)
    enable_rerank_cache: bool = _env_bool("ENABLE_RERANK_CACHE", True)
    enable_llm_cache: bool = _env_bool("ENABLE_LLM_CACHE", True)

    embedding_cache_size: int = _env_int("EMBEDDING_CACHE_SIZE", 4096)
    retrieval_cache_size: int = _env_int("RETRIEVAL_CACHE_SIZE", 512)
    rerank_cache_size: int = _env_int("RERANK_CACHE_SIZE", 1024)
    llm_cache_size: int = _env_int("LLM_CACHE_SIZE", 1024)

    # Retrieval pipeline tuning.
    retrieval_top_k_vector: int = _env_int("RETRIEVAL_TOP_K_VECTOR", 50)
    retrieval_top_k_bm25: int = _env_int("RETRIEVAL_TOP_K_BM25", 50)
    retrieval_top_k_fusion: int = _env_int("RETRIEVAL_TOP_K_FUSION", 50)
    fusion_rrf_k: int = _env_int("FUSION_RRF_K", 60)
    coarse_rank_top_k: int = _env_int("COARSE_RANK_TOP_K", 20)
    rerank_top_k: int = _env_int("RERANK_TOP_K", 10)
    rerank_input_top_k: int = _env_int("RERANK_INPUT_TOP_K", 20)

    # Unified evidence ranking weights.
    w_semantic: float = _env_float("W_SEMANTIC", 0.45)
    w_vector: float = _env_float("W_VECTOR", 0.2)
    w_bm25: float = _env_float("W_BM25", 0.15)
    w_graph: float = _env_float("W_GRAPH", 0.2)

    # Multi-turn memory controls.
    memory_max_turns: int = _env_int("MEMORY_MAX_TURNS", 12)
    memory_max_tokens: int = _env_int("MEMORY_MAX_TOKENS", 1800)
    short_term_window_size: int = _env_int("SHORT_TERM_WINDOW_SIZE", 6)
    summary_trigger_turns: int = _env_int("SUMMARY_TRIGGER_TURNS", 6)
    summary_trigger_tokens: int = _env_int("SUMMARY_TRIGGER_TOKENS", 1000)
    fact_expire_policy: str = os.getenv("FACT_EXPIRE_POLICY", "none")
    enable_llm_memory_extraction: bool = _env_bool("ENABLE_LLM_MEMORY_EXTRACTION", False)
    enable_llm_citation_validation: bool = _env_bool("ENABLE_LLM_CITATION_VALIDATION", False)
    enable_encoding_fix: bool = _env_bool("ENABLE_ENCODING_FIX", True)
    enable_runtime_feedback: bool = _env_bool("ENABLE_RUNTIME_FEEDBACK", True)
    fact_conflict_strategy: str = os.getenv("FACT_CONFLICT_STRATEGY", "confidence_weighted")
    enable_conflict_logging: bool = _env_bool("ENABLE_CONFLICT_LOGGING", True)
    source_reliability_weights: Dict[str, float] = field(
        default_factory=lambda: _env_json_dict(
            "SOURCE_RELIABILITY_WEIGHTS",
            {
                "statute": 1.0,
                "case": 0.9,
                "assistant_summary": 0.75,
                "assistant": 0.7,
                "user": 0.6,
                "unknown": 0.5,
            },
        )
    )


settings = Settings()
