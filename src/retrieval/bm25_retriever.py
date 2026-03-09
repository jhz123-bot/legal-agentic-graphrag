from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List

from src.ingestion.text_chunker import TextChunk


def _tokenize(text: str) -> List[str]:
    # Mixed tokenizer for Chinese legal text: words + character bigrams.
    words = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
    tokens: List[str] = []
    for w in words:
        if len(w) <= 2:
            tokens.append(w)
            continue
        tokens.append(w)
        if re.search(r"[\u4e00-\u9fff]", w):
            for i in range(len(w) - 1):
                tokens.append(w[i : i + 2])
    return tokens


@dataclass
class _DocStat:
    tf: Dict[str, int]
    length: int


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.doc_stats: List[_DocStat] = []
        self.df: Dict[str, int] = {}
        self.avgdl: float = 0.0

    def build_index(self, docs: List[TextChunk]) -> None:
        self.chunks = docs
        self.doc_stats = []
        self.df = {}
        total_len = 0

        for chunk in docs:
            toks = _tokenize(chunk.text)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_stats.append(_DocStat(tf=tf, length=len(toks)))
            total_len += len(toks)
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = (total_len / len(docs)) if docs else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.df.get(term, 0)
        if n == 0:
            return 0.0
        # BM25 idf variant with +1 for stability.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.chunks:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        hits: List[Dict] = []
        for idx, stat in enumerate(self.doc_stats):
            score = 0.0
            dl = max(1, stat.length)
            for term in q_terms:
                tf = stat.tf.get(term, 0)
                if tf <= 0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score += idf * (tf * (self.k1 + 1)) / max(1e-9, denom)

            if score <= 0:
                continue
            c = self.chunks[idx]
            hits.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text,
                    "score": float(score),
                }
            )

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[: max(1, top_k)]
