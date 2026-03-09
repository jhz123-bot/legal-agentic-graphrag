import argparse
import os
from pathlib import Path

from src.agents.workflow import build_graph_store_from_documents, run_agentic_graphrag
from src.cache.cache_manager import get_cache_manager
from src.common.models import Document
from src.config.settings import settings
from src.embedding.embedding_model import EmbeddingModel
from src.feedback.runtime_feedback import record_runtime_failure
from src.ingestion.document_loader import load_documents_from_dir
from src.ingestion.text_chunker import chunk_documents
from src.indexing.graph_loader import load_graph
from src.indexing.vector_index_loader import load_vector_index
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.encoding_utils import configure_stdio_encoding
from src.vector_store.vector_index import FaissVectorIndex

DEMO_QUERIES = [
    "盗窃行为通常适用哪一条法律条文？",
    "买卖合同违约时，法院通常如何认定违约责任？",
    "房屋租赁纠纷中，承租人逾期支付租金应承担什么责任？",
]


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


def build_hybrid_components(docs_dir: Path, index_dir: Path | None = None, graph_dir: Path | None = None):
    loaded_docs = load_documents_from_dir(docs_dir)
    if not loaded_docs:
        raise ValueError(f"在目录 {docs_dir} 下未找到 TXT/PDF/DOCX 文档。")

    effective_index_dir = index_dir or (Path(__file__).parent / "data" / "index")
    embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    index_load_mode = "prebuilt"
    try:
        vector_index, _, chunks = load_vector_index(effective_index_dir)
    except Exception:
        index_load_mode = "rebuild"
        chunks = chunk_documents(loaded_docs)
        embeddings = embedding_model.embed_chunks(chunks)
        vector_index = FaissVectorIndex()
        vector_index.build_index(chunks, embeddings)

    effective_graph_dir = graph_dir or (Path(__file__).parent / "data" / "graph")
    graph_load_mode = "prebuilt"
    try:
        graph_store = load_graph(effective_graph_dir)
    except Exception:
        graph_load_mode = "rebuild"
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
    chunk_type_stats = {}
    for c in chunks:
        chunk_type_stats[c.doc_type] = chunk_type_stats.get(c.doc_type, 0) + 1
    sample_chunk = None
    if chunks:
        c = chunks[0]
        sample_chunk = {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "doc_type": c.doc_type,
            "source_type": c.source_type,
            "law_name": c.law_name,
            "article_no": c.article_no,
            "case_id": c.case_id,
            "section": c.section,
            "article": c.article,
            "text_preview": c.text[:80],
        }
    build_debug = getattr(graph_store, "build_debug_stats", [])
    build_debug = [{"index_load_mode": index_load_mode, "graph_load_mode": graph_load_mode}] + build_debug
    return graph_store, hybrid_retriever, len(loaded_docs), len(chunks), chunk_type_stats, sample_chunk, build_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Agentic GraphRAG Demo")
    parser.add_argument("--docs", type=str, default="data/legal_docs", help="文档目录，支持 TXT/PDF/DOCX")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="演示问题",
    )
    args = parser.parse_args()

    configured, current_encoding = configure_stdio_encoding()
    print(f"[encoding] stdio_encoding_configured={configured}, current_stdout_encoding={current_encoding}")

    root = Path(__file__).parent
    load_env_file(root / ".env")

    docs_dir = (root / args.docs).resolve() if not Path(args.docs).is_absolute() else Path(args.docs)

    selected_query = args.query.strip() if args.query.strip() else DEMO_QUERIES[0]
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")

    print("步骤 1/6：加载并解析文档（TXT/PDF/DOCX）...")
    print(f"- LLM 提供方：{llm_provider}")
    print(f"- Embedding 提供方：{embedding_provider}")
    graph_store, hybrid_retriever, doc_count, chunk_count, chunk_type_stats, sample_chunk, build_debug = build_hybrid_components(docs_dir)
    print(f"- 已加载文档数：{doc_count}")
    print(f"- 已生成文本分块数：{chunk_count}")
    print(f"- 分块类型统计：{chunk_type_stats}")
    if sample_chunk:
        print(f"- 示例分块元数据：{sample_chunk}")
    if build_debug:
        if build_debug[0].get("index_load_mode"):
            print(f"- 向量索引加载模式：{build_debug[0].get('index_load_mode')}")
        if build_debug[0].get("graph_load_mode"):
            print(f"- 图谱加载模式：{build_debug[0].get('graph_load_mode')}")
        for d in build_debug:
            if "rule_entities" not in d:
                continue
            print(
                "- 抽取调试（首个文档）："
                f"rule_entities={d.get('rule_entities')}, "
                f"llm_entities={d.get('llm_entities')}, "
                f"merged_entities={d.get('merged_entities')}, "
                f"llm_triples={d.get('llm_triples')}"
            )
            if d.get("sample_llm_triples"):
                print(f"- 示例LLM三元组：{d.get('sample_llm_triples')[0]}")
            break
    print(f"- 全图规模：nodes={len(graph_store.nodes)}, edges={len(graph_store.edges)}")
    print("内置演示问题：")
    for idx, q in enumerate(DEMO_QUERIES, start=1):
        print(f"  {idx}. {q}")

    print("步骤 2/6：运行 Agent Framework 工作流...")
    state = run_agentic_graphrag(selected_query, graph_store=graph_store, hybrid_retriever=hybrid_retriever)

    print("\n步骤 3/6：路由决策")
    router_decision = state.get("router_decision", {})
    retrieval_strategy = state.get("retrieval_strategy") or router_decision.get("retrieval_strategy")
    workflow_branch = state.get("workflow_branch", "")
    print(f"用户问题：{selected_query}")
    print(f"- 路由类型：{router_decision.get('route')}")
    print(f"- 检索策略：{retrieval_strategy}")
    print(f"- 工作流分支：{workflow_branch}")
    print(f"- 路由依据：{router_decision.get('reason')}")

    print("\n步骤 4/6：推理过程（Thought / Action / Observation）")
    for i, step in enumerate(state.get("react_trace", []), start=1):
        print(f"{i}. Thought: {step.get('thought')}")
        print(f"   Action: {step.get('action')}")
        print(f"   Observation: {step.get('observation')}")

    print("\n步骤 5/6：检索与推理摘要")
    evidence_pack = state.get("evidence_pack", {})
    print(f"- 图谱节点数：{len(evidence_pack.get('nodes', []))}")
    print(f"- 图谱边数：{len(evidence_pack.get('edges', []))}")
    print(f"- 向量命中数：{len(evidence_pack.get('vector_hits', []))}")
    print(f"- 候选证据数：{len(state.get('candidate_evidence', evidence_pack.get('candidate_evidence', [])))}")
    print(f"- 排序证据数：{len(state.get('ranked_evidence', []))}")
    if state.get("ranked_evidence"):
        top_ev = state["ranked_evidence"][0]
        print(f"- Top证据类型：{top_ev.get('evidence_type')}")
        print(f"- Top证据评分因子：{top_ev.get('score_factors', {})}")

    final_answer = state.get("final_answer", {})
    grounding = final_answer.get("grounding", {})
    citations = final_answer.get("citations", [])
    print("\n步骤 6/6：最终回答")
    print(f"用户问题：{selected_query}")
    print(f"最终回答：{final_answer.get('short_answer')}")
    print(f"置信度：{final_answer.get('confidence')}")
    print(f"反思决策：{final_answer.get('reflection_decision')}")
    print(f"Grounding score: {grounding.get('grounding_score', 0.0)}")
    print(f"Answer grounded: {grounding.get('grounded', False)}")
    print(f"Citations count: {len(citations)}")
    for c in citations[:3]:
        display_ref = c.get("case_id") or c.get("title") or f"{c.get('source', '')}->{c.get('target', '')}"
        print(f"- [{c.get('source_type')}] {c.get('law_name', '')}{c.get('article_no', '')} {display_ref} ({c.get('chunk_id', '')})")

    structured = final_answer.get("structured_output", {})
    if structured:
        print("\n结构化输出：")
        print(f"- 识别实体：{structured.get('entities', [])}")
        print(f"- 证据路径：{len(structured.get('evidence', []))}")
        print(f"- 推理步骤：{len(structured.get('reasoning_steps', []))}")
        for step in structured.get("reasoning_steps", []):
            print(
                f"  step={step.get('step')} relation={step.get('relation')} "
                f"conclusion={step.get('conclusion')}"
            )
        print(f"- 置信分数：{structured.get('confidence')}")

    if settings.enable_runtime_feedback:
        feedback = record_runtime_failure(
            query=selected_query,
            trace=state,
            output_root=root / "outputs" / "feedback",
            metadata={"entry": "run_demo"},
        )
        if feedback:
            print(f"- runtime failure captured: {feedback.get('jsonl_path')}")

    if os.getenv("SHOW_DEMO_LOGS", "0") == "1":
        print("\n调试日志：")
        for line in state.get("logs", []):
            print(f"- {line}")
        print("- cache_stats:", get_cache_manager().get_stats())


if __name__ == "__main__":
    main()
