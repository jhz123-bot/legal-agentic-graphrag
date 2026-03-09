from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from src.ingestion.text_chunker import TextChunk
from src.vector_store.vector_index import FaissVectorIndex


def _dict_to_chunk(row: Dict[str, Any]) -> TextChunk:
    return TextChunk(
        chunk_id=str(row.get("chunk_id", "")),
        doc_id=str(row.get("doc_id", "")),
        doc_type=str(row.get("doc_type", "generic")),
        source_type=str(row.get("source_type", "generic")),
        law_name=str(row.get("law_name", "")),
        article_no=str(row.get("article_no", "")),
        case_id=str(row.get("case_id", "")),
        section=str(row.get("section", "")),
        article=str(row.get("article", row.get("article_no", ""))),
        title=str(row.get("title", "")),
        text=str(row.get("text", "")),
        start_char=int(row.get("start_char", 0) or 0),
        end_char=int(row.get("end_char", 0) or 0),
    )


def load_vector_index(index_dir: str | Path = "data/index") -> Tuple[FaissVectorIndex, np.ndarray, List[TextChunk]]:
    """Load persisted vector artifacts.

    Expected files in index_dir:
    - faiss.index
    - chunk_embeddings.npy
    - chunk_metadata.json
    """
    base = Path(index_dir)
    faiss_path = base / "faiss.index"
    emb_path = base / "chunk_embeddings.npy"
    meta_path = base / "chunk_metadata.json"

    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing index file: {faiss_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    embeddings = np.load(str(emb_path))
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [_dict_to_chunk(x) for x in raw_meta]

    if embeddings.ndim != 2:
        raise ValueError(f"chunk_embeddings.npy must be 2D, got shape={embeddings.shape}")
    if len(chunks) != int(embeddings.shape[0]):
        raise ValueError(
            f"metadata length ({len(chunks)}) != embedding rows ({embeddings.shape[0]})"
        )

    index = faiss.read_index(str(faiss_path))
    vector_index = FaissVectorIndex()
    vector_index.index = index
    vector_index.chunks = chunks
    vector_index.dimension = int(embeddings.shape[1])
    return vector_index, embeddings.astype(np.float32), chunks

