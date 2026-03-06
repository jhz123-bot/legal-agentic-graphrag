from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    spans: List[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    step = chunk_size - chunk_overlap
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            spans.append((i, j, chunk))
        i += step
    return spans


def chunk_documents(docs: List[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for doc in docs:
        spans = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, (start, end, text) in enumerate(spans):
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc['doc_id']}_{idx}",
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )
    return chunks
