# Minimal Legal GraphRAG QA Demo

This repository contains a minimal, runnable GraphRAG-style legal QA project.

It includes:
- graph schema
- entity extraction
- graph builder
- in-memory graph store
- entity linker
- graph retriever
- evidence formatter
- Ollama client wrapper

## Project layout

```text
.
├─ .env.example
├─ requirements.txt
├─ run_demo.py
├─ data/
│  └─ sample_legal_docs/
│     ├─ contract_dispute.txt
│     ├─ negligence_case.txt
│     └─ tenant_landlord.txt
└─ src/
   ├─ common/
   │  ├─ __init__.py
   │  └─ models.py
   ├─ graph/
   │  ├─ __init__.py
   │  ├─ entity_extraction.py
   │  ├─ entity_linker.py
   │  ├─ graph_builder.py
   │  ├─ schema.py
   │  └─ store.py
   ├─ retrieval/
   │  ├─ __init__.py
   │  ├─ evidence_formatter.py
   │  └─ graph_retriever.py
   └─ llm/
      ├─ __init__.py
      └─ ollama_client.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies (minimal demo has no external runtime deps, but keep this for extension):

```bash
pip install -r requirements.txt
```

3. Copy env file:

```bash
cp .env.example .env
```

(On Windows PowerShell: `Copy-Item .env.example .env`)

4. Ensure Ollama is running (optional but recommended):
- Default endpoint: `http://localhost:11434`
- Pull a model, for example: `ollama pull llama3.1:8b`

## Run

```bash
python run_demo.py
```

The demo performs:
1. load sample legal documents
2. build the knowledge graph
3. run entity linking
4. retrieve graph evidence
5. generate an answer

If Ollama is not reachable, the demo still runs and returns a fallback answer from evidence.
