from pathlib import Path
import os

from src.agents.workflow import run_agentic_graphrag


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


def main() -> None:
    root = Path(__file__).parent
    load_env_file(root / ".env")
    question = "在 Zhangsan v. Jia Corp 买卖合同纠纷中，法院如何认定违约责任，并依据哪些证据？"

    print("步骤 1/6：运行 Agentic GraphRAG 工作流...")
    state = run_agentic_graphrag(question)

    decomposition = state.get("decomposition", {})
    print("\n步骤 2/6：查询拆解结果")
    print(f"- 原始问题：{decomposition.get('original_query', question)}")
    print(f"- 拆解方式：{decomposition.get('method', 'unknown')}")
    for idx, subq in enumerate(decomposition.get("subqueries", []), start=1):
        print(f"  {idx}. {subq}")

    print("\n步骤 3/6：证据排序结果（Top-K）")
    for idx, path in enumerate(state.get("ranked_evidence", []), start=1):
        print(
            f"{idx}. score={path.get('score')} relation={path.get('relation')} "
            f"source={path.get('source')} target={path.get('target')}"
        )
        evidence = (path.get("evidence") or "").replace("\n", " ")
        if evidence:
            print(f"   证据：{evidence[:120]}")

    print("\n步骤 4/6：结构化推理轨迹")
    reasoning_trace = state.get("reasoning_trace", {})
    for step in reasoning_trace.get("structured_steps", []):
        print(
            f"step={step.get('step')} relation={step.get('relation')} "
            f"结论={step.get('conclusion')}"
        )
        print(f"  使用证据：{step.get('evidence')}")

    print("\n步骤 5/6：反思策略决策")
    verification = state.get("verification_result", {})
    policy = verification.get("policy", {})
    print(f"- 决策：{verification.get('decision')}")
    print(f"- 置信度：{policy.get('confidence_score')}")
    print(
        f"- 评分细项：entity_coverage={policy.get('entity_coverage')}, "
        f"evidence_consistency={policy.get('evidence_consistency')}, "
        f"reasoning_depth={policy.get('reasoning_depth')}"
    )

    print("\n步骤 6/6：最终答案")
    final_answer = state.get("final_answer", {})
    print(f"问题：{question}")
    print(f"短答案：{final_answer.get('short_answer')}")
    print(f"推理摘要：{final_answer.get('reasoning_summary')}")
    print(f"不确定性：{final_answer.get('uncertainty_note')}")

    if os.getenv("SHOW_DEMO_LOGS", "0") == "1":
        print("\n调试日志：")
        for line in state.get("logs", []):
            print(f"- {line}")


if __name__ == "__main__":
    main()
