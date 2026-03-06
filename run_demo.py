import os
from pathlib import Path
from typing import List

from src.common.models import Document
from src.graph.entity_linker import EntityLinker
from src.graph.graph_builder import GraphBuilder
from src.graph.store import InMemoryGraphStore
from src.llm.ollama_client import OllamaClient
from src.retrieval.evidence_formatter import format_evidence
from src.retrieval.graph_retriever import GraphRetriever


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        docs.append(Document(doc_id=path.stem, title=path.stem, text=path.read_text(encoding="utf-8")))
    return docs


def answer_with_fallback(client: OllamaClient, question: str, evidence_text: str) -> str:
    system_prompt = (
        "You are a legal QA assistant. Answer only from supplied evidence. "
        "If evidence is incomplete, state uncertainty briefly."
    )
    user_prompt = f"Question:\n{question}\n\nEvidence:\n{evidence_text}\n\nAnswer:"
    try:
        return client.generate(prompt=user_prompt, system_prompt=system_prompt)
    except Exception as exc:
        return (
            "Fallback answer (Ollama unavailable): Based on retrieved graph evidence, "
            f"the likely answer is that the dispute centers on entities and relations shown above. Error: {exc}"
        )


def main() -> None:
    root = Path(__file__).parent
    load_env_file(root / ".env")
    data_dir = root / "data" / "sample_legal_docs"

    question = "What did the court decide in Smith v. Acme Corp about breach of contract?"

    print("Step 1/5: Loading sample legal documents...")
    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} documents.")

    print("Step 2/5: Building knowledge graph...")
    store = InMemoryGraphStore()
    builder = GraphBuilder(store)
    graph_store = builder.build(documents)
    print(f"Graph has {len(graph_store.nodes)} nodes and {len(graph_store.edges)} edges before linking.")

    print("Step 3/5: Running entity linking...")
    linker = EntityLinker()
    graph_store = linker.link(graph_store)
    print(f"Graph has {len(graph_store.nodes)} nodes and {len(graph_store.edges)} edges after linking.")

    print("Step 4/5: Retrieving graph evidence...")
    retriever = GraphRetriever(graph_store)
    retrieved = retriever.retrieve(question=question, top_k_nodes=5, top_k_edges=8)
    evidence_text = format_evidence(retrieved)
    print(evidence_text)

    print("Step 5/5: Generating answer...")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    client = OllamaClient(base_url=ollama_base_url, model=ollama_model)
    answer = answer_with_fallback(client, question, evidence_text)

    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
