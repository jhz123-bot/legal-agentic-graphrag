# Law Layer 标准化说明

为便于 GraphRAG 构图与实体链接，law layer 采用以下标准化规则：

1. `article_no` 统一使用“法名简称 + 条号”格式，例如：
   - 刑法第二百六十四条
   - 民法典第五百七十七条

2. `doc_id` 使用稳定英文键，便于程序索引，例如：
   - law_criminal_264
   - law_civil_577

3. `text` 保持短句化、要点化，避免超长原文；要求可直接用于实体抽取与关系构建。

4. `keywords` 必须包含：
   - 主题词（如“盗窃罪”“违约责任”）
   - 标准法条名（如“刑法第二百六十四条”）

5. 日期字段 `effective_date` 统一 `YYYY-MM-DD`。
