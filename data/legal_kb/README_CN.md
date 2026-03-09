# 分层法律知识库设计说明

本项目新增三层法律知识库（Layered Legal KB），用于将法条、案例、问答样本分层管理，提升数据可维护性与后续检索可控性。

## 目录结构

- `data/legal_kb/laws/`：法条层
- `data/legal_kb/cases/`：案例层
- `data/legal_kb/faq/`：FAQ/QA 层

## 三层字段规范

### 1) 法条层（law layer）

- `doc_id`
- `law_name`
- `article_no`
- `chapter`
- `effective_date`
- `text`
- `keywords`

### 2) 案例层（case layer）

- `case_id`
- `case_type`
- `court`
- `facts`
- `issues`
- `holding`
- `cited_laws`
- `keywords`

### 3) FAQ/QA 层（faq layer）

- `question`
- `answer`
- `topic`
- `related_laws`

## 当前状态

每一层已提供：

1. 一个 `schema.json` 字段说明文件
2. 一个最小 `*_example.json` 示例文件

该结构仅用于先完成 schema 与组织设计，后续可逐步扩充数据规模。
