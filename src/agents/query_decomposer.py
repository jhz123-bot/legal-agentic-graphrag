import os
import re
from typing import Dict, List, Optional

from src.llm.ollama_client import OllamaClient


def _rule_based_decompose(query: str) -> List[str]:
    parts = re.split(r"[？?；;，,]\s*|以及|并且|同时|并", query)
    subqueries = [p.strip() for p in parts if p.strip()]
    if not subqueries:
        return [query.strip()]
    if len(subqueries) == 1 and "什么" not in subqueries[0] and "如何" not in subqueries[0]:
        return [query.strip()]
    return subqueries[:3]


def _llm_decompose(query: str, client: OllamaClient) -> Optional[List[str]]:
    prompt = (
        "请将这个法律问答问题拆解为1到3个更小的检索子问题，"
        "每行一个，不要编号，不要解释：\n"
        f"{query}"
    )
    try:
        text = client.generate(prompt=prompt)
    except Exception:
        return None
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if len(line) > 2]
    if not lines:
        return None
    return lines[:3]


def decompose_query(query: str) -> Dict[str, List[str] | str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    client = OllamaClient(base_url=base_url, model=model, timeout=20)

    llm_subqueries = _llm_decompose(query, client)
    if llm_subqueries:
        return {"original_query": query, "subqueries": llm_subqueries, "method": "llm"}
    return {"original_query": query, "subqueries": _rule_based_decompose(query), "method": "rule"}
