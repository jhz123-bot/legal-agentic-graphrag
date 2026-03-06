# 项目名称

**Legal Agentic GraphRAG**  
基于知识图谱与多智能体的法律问答系统

---

# 项目简介

本项目实现了一个 **Agentic GraphRAG** 系统，用于法律知识问答。

系统将知识图谱检索（GraphRAG）与多智能体架构（Multi-Agent Architecture）结合，通过结构化知识推理提高回答的可靠性与可解释性。

主要特点：

- GraphRAG 知识图谱检索
- Multi-Agent 推理架构
- Evidence Ranking 证据排序
- Query Decomposition 查询分解
- Reflection Policy 自我反思机制
- Evaluation 模块用于系统评估

---

# 系统整体架构

```mermaid
flowchart LR
    A[法律文档] --> B[实体抽取]
    B --> C[知识图谱构建]
    C --> D[Graph Retrieval]
    D --> E[Evidence Ranking]
    E --> F[Reasoning Agent]
    F --> G[Reflection Policy]
    G --> H[Final Answer]
```

系统流程说明：

1. 从法律文本中抽取实体与关系，构建结构化知识图谱。  
2. 基于问题执行图谱检索，召回候选实体与证据路径。  
3. 通过证据排序模块筛选高质量证据。  
4. 由推理智能体生成结构化推理链。  
5. 反思策略判断是否需要回环（重检索/重推理）。  
6. 输出最终答案与证据摘要。

---

# 核心能力模块

## 1. GraphRAG（图谱检索增强）

- 从法律文档构建图谱（节点/边）
- 支持实体链接与关系检索
- 提供可追溯的证据来源

## 2. Multi-Agent 工作流

当前工作流基于 LangGraph 组织，主链路为：

`Planner -> Retrieval -> Evidence Ranking -> Reasoning -> Reflection -> Answer`

反思节点支持回环：

- `re-retrieve`：触发再次检索
- `re-reason`：触发再次推理
- `pass`：通过并进入答案生成

## 3. Evidence Ranking

通过轻量多因子评分对候选证据路径排序，典型因素包括：

- 查询-证据语义相似度
- 图路径距离分数
- 关系类型权重

目的：优先保留与问题更相关、结构更可靠的证据。

## 4. Query Decomposition

对复杂问题进行子问题拆解：

- 优先 LLM 拆解
- LLM 不可用时自动回退规则拆解

目的：提高检索覆盖率与规划清晰度。

## 5. Reflection Policy

通过策略评分决定是否回环，核心考虑：

- 实体覆盖率
- 证据一致性/数量
- 推理深度

目的：在不引入重型优化的前提下提升回答稳定性。

---

# 代码结构

```text
.
|- run_demo.py
|- benchmark.py
|- requirements.txt
|- README.md
|- data/
|  |- sample_legal_docs/
|  `- sample_benchmark/
|     `- legal_benchmark.json
`- src/
   |- common/
   |- graph/
   |- retrieval/
   |  |- graph_retriever.py
   |  |- evidence_formatter.py
   |  `- evidence_ranker.py
   |- reasoning/
   |  `- reasoning_trace.py
   |- llm/
   |  `- ollama_client.py
   |- agents/
   |  |- planner_agent.py
   |  |- query_decomposer.py
   |  |- retrieval_agent.py
   |  |- reasoning_agent.py
   |  |- reflection_policy.py
   |  |- reflection_agent.py
   |  |- answer_agent.py
   |  `- workflow.py
   `- evaluation/
      |- metrics.py
      `- benchmark_runner.py
```

---

# 运行环境

推荐环境：

- Python 3.10+
- Windows / Linux / macOS

安装依赖：

```bash
pip install -r requirements.txt
```

如需启用 LLM 拆解与生成能力，可配置本地 Ollama：

- `OLLAMA_BASE_URL`（默认 `http://localhost:11434`）
- `OLLAMA_MODEL`（如 `llama3.1:8b`）

---

# 快速开始

## 1) 运行演示

```bash
python run_demo.py
```

演示输出包含：

- 查询分解结果
- 排序后证据路径
- 结构化推理轨迹
- 反思策略决策
- 最终答案

## 2) 运行评估

```bash
python benchmark.py
```

将输出聚合评估结果与样本级指标。

---

# 评估模块（Evaluation）

评估入口：`benchmark.py`  
数据集：`data/sample_benchmark/legal_benchmark.json`

当前指标：

- `entity_hit_rate`：期望实体在链接实体中的命中率
- `evidence_path_hit_rate`：期望证据路径在检索证据中的命中率
- `answer_keyword_match_rate`：答案关键词命中率
- `reflection_trigger_rate`：反思触发重检索/重推理比例
- `average_latency`：平均查询耗时

---

# 设计原则

本项目强调：

- **可解释**：推理过程与证据路径可查看
- **可扩展**：模块边界清晰，便于替换策略
- **轻量化**：避免复杂基础设施，突出算法思路
- **面向工程面试**：代码结构清晰，方便讲解与演示

---

# 路线与可扩展方向

可在当前基础上继续扩展：

- 更细粒度的法律关系抽取
- 证据冲突检测与一致性校验
- 多跳路径检索与路径解释
- 更严格的基准集与自动化回归评测

---

# 许可证

可根据团队需求补充（如 MIT / Apache-2.0）。

