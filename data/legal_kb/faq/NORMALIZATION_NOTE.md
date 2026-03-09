# FAQ Layer 标准化说明

FAQ/QA 层用于快速召回与多轮对话承接，采用统一字段：

- question
- answer
- topic
- related_laws

## 编写原则

1. `question` 面向真实用户表达，尽量口语化但保持法律语义清晰。
2. `answer` 保持简洁、可执行、法律依据明确。
3. `topic` 统一使用高层主题标签（如：盗窃罪、合同违约、劳动争议）。
4. `related_laws` 使用标准法条名称，便于实体链接与图谱检索。
5. 每条 FAQ 可独立用于快速检索，也可作为多轮追问的上下文支撑。
