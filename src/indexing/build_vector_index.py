from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from src.embedding.embedding_model import EmbeddingModel
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_chunker import TextChunk, chunk_documents
from src.vector_store.vector_index import FaissVectorIndex


def _chunk_to_dict(chunk: TextChunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "source_type": chunk.source_type,
        "law_name": chunk.law_name,
        "article_no": chunk.article_no,
        "case_id": chunk.case_id,
        "section": chunk.section,
        "article": chunk.article,
        "title": chunk.title,
        "text": chunk.text,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def build_vector_index(
    docs_dir: str | Path,
    index_dir: str | Path = "data/index",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_provider: str | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> Dict[str, Any]:
    """Offline vector index build pipeline.

    Steps:
    1) load documents
    2) chunk documents
    3) compute embeddings
    4) build FAISS index
    5) save artifacts:
       - faiss.index
       - chunk_embeddings.npy
       - chunk_metadata.json
    """
    docs_path = Path(docs_dir)
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_docs = load_documents_from_dir(docs_path)
    if not loaded_docs:
        raise ValueError(f"No TXT/PDF/DOCX documents found in {docs_path}")

    chunks = chunk_documents(loaded_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks generated from input documents")

    embedding_model = EmbeddingModel(model_name=embedding_model_name, provider=embedding_provider)
    embeddings = embedding_model.embed_chunks(chunks)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Invalid embeddings shape={embeddings.shape}, expected ({len(chunks)}, dim)"
        )

    vector_index = FaissVectorIndex()
    vector_index.build_index(chunks, embeddings)
    if vector_index.index is None:
        raise RuntimeError("FAISS index build failed")

    faiss_path = out_dir / "faiss.index"
    embedding_path = out_dir / "chunk_embeddings.npy"
    metadata_path = out_dir / "chunk_metadata.json"

    faiss.write_index(vector_index.index, str(faiss_path))
    np.save(str(embedding_path), embeddings.astype(np.float32))

    metadata: List[Dict[str, Any]] = [_chunk_to_dict(c) for c in chunks]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "docs_dir": str(docs_path),
        "index_dir": str(out_dir),
        "doc_count": len(loaded_docs),
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": embedding_model_name,
        "embedding_provider": embedding_provider or "default",
        "artifacts": {
            "faiss_index": str(faiss_path),
            "chunk_embeddings": str(embedding_path),
            "chunk_metadata": str(metadata_path),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist vector index artifacts")
    parser.add_argument("--docs", default="data/legal_docs", help="document directory (TXT/PDF/DOCX)")
    parser.add_argument("--index-dir", default="data/index", help="artifact output directory")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="embedding model name")
    parser.add_argument("--embedding-provider", default=None, help="embedding provider override")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    args = parser.parse_args()

    summary = build_vector_index(
        docs_dir=args.docs,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        embedding_provider=args.embedding_provider,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
