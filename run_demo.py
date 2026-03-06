import argparse
import os
from pathlib import Path

from src.agents.workflow import build_graph_store_from_documents, run_agentic_graphrag
from src.common.models import Document
from src.embedding.embedding_model import EmbeddingModel
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector_store.vector_index import FaissVectorIndex


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


def build_hybrid_components(docs_dir: Path):
    loaded_docs = load_documents_from_dir(docs_dir)
    if not loaded_docs:
        raise ValueError(f"在目录 {docs_dir} 下未找到 TXT/PDF/DOCX 文档。")

    documents = [Document(doc_id=d["doc_id"], title=d["title"], text=d["text"]) for d in loaded_docs]
    graph_store = build_graph_store_from_documents(documents)

    chunks = chunk_documents(loaded_docs, chunk_size=500, chunk_overlap=100)
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_model.embed_chunks(chunks)

    vector_index = FaissVectorIndex()
    vector_index.build_index(chunks, embeddings)

    graph_retriever = GraphRetriever(graph_store)
    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        embedding_model=embedding_model,
        graph_retriever=graph_retriever,
    )
    return graph_store, hybrid_retriever, len(loaded_docs), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Agentic GraphRAG Demo")
    parser.add_argument("--docs", type=str, default="data/legal_docs", help="文档目录，支持 TXT/PDF/DOCX")
    parser.add_argument(
        "--query",
        type=str,
        default="在 Zhangsan v. Jia Corp 买卖合同纠纷中，法院如何认定违约责任，并依据哪些证据？",
        help="演示问题",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    load_env_file(root / ".env")

    docs_dir = (root / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)

    print("步骤 1/6：加载并解析文档（TXT/PDF/DOCX）...")
    graph_store, hybrid_retriever, doc_count, chunk_count = build_hybrid_components(docs_dir)
    print(f"- 已加载文档数：{doc_count}")
    print(f"- 已生成文本分块数：{chunk_count}")

    print("步骤 2/6：运行 Agent Framework 工作流...")
    state = run_agentic_graphrag(args.query, graph_store=graph_store, hybrid_retriever=hybrid_retriever)

    print("\n步骤 3/6：Router 决策")
    router_decision = state.get("router_decision", {})
    print(f"- route: {router_decision.get('route')}")
    print(f"- reason: {router_decision.get('reason')}")

    print("\n步骤 4/6：ReAct 过程（Thought / Action / Observation）")
    for i, step in enumerate(state.get("react_trace", []), start=1):
        print(f"{i}. Thought: {step.get('thought')}")
        print(f"   Action: {step.get('action')}")
        print(f"   Observation: {step.get('observation')}")

    print("\n步骤 5/6：检索与推理摘要")
    evidence_pack = state.get("evidence_pack", {})
    print(f"- 图谱节点数：{len(evidence_pack.get('nodes', []))}")
    print(f"- 图谱边数：{len(evidence_pack.get('edges', []))}")
    print(f"- 向量命中数：{len(evidence_pack.get('vector_hits', []))}")
    print(f"- 排序证据数：{len(state.get('ranked_evidence', []))}")

    final_answer = state.get("final_answer", {})
    print("\n步骤 6/6：Final Answer")
    print(f"问题：{args.query}")
    print(f"答案：{final_answer.get('short_answer')}")
    print(f"置信度：{final_answer.get('confidence')}")
    print(f"反思决策：{final_answer.get('reflection_decision')}")

    structured = final_answer.get("structured_output", {})
    if structured:
        print("\n结构化输出：")
        print(f"- entities: {structured.get('entities', [])}")
        print(f"- evidence paths: {len(structured.get('evidence', []))}")
        print(f"- reasoning steps: {len(structured.get('reasoning_steps', []))}")
        for step in structured.get("reasoning_steps", []):
            print(
                f"  step={step.get('step')} relation={step.get('relation')} "
                f"conclusion={step.get('conclusion')}"
            )
        print(f"- confidence: {structured.get('confidence')}")

    if os.getenv("SHOW_DEMO_LOGS", "0") == "1":
        print("\n调试日志：")
        for line in state.get("logs", []):
            print(f"- {line}")


if __name__ == "__main__":
    main()
