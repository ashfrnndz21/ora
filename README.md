<div align="center">

# ora

**The NL2SQL Agentic Runtime**

*Not a chatbot wrapper. A fully agentic system that reasons, self-corrects, and learns.*

[![PyPI](https://img.shields.io/pypi/v/ora-sql?color=4f7df9&label=pip%20install%20ora-sql)](https://pypi.org/project/ora-sql)
[![CI](https://github.com/ashfrnndz21/ora/actions/workflows/ci.yml/badge.svg)](https://github.com/ashfrnndz21/ora/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

</div>

---

```python
import ora

db = ora.connect("sqlite:///retail.db")
result = db.query("top 10 stores by revenue")
print(result.dataframe)
```

```
      store_name  total_revenue
 Lotus Sukhumvit       133500.0
     Lotus Korat        35700.0
Lotus Chiang Mai        31000.0
```

One call. Works with any database. Works with any LLM.

---

## What makes ora different

Every other NL2SQL tool stops at "generate SQL." ora is the **agentic runtime** that makes it production-worthy:

| Challenge | Others | ora |
|---|---|---|
| Complex schemas (500+ cols) | Hallucinate or fail | CHESS LSH schema pruning → 8 relevant columns |
| SQL fails on first try | Crash or retry blindly | 3-stage ReFoRCE self-correction |
| Multiple databases | Impossible | Decompose → parallel agents → DuckDB in-memory JOIN |
| No learning over time | Starts fresh every query | 3-tier memory + SOUL user mental model |
| Black box decisions | No visibility | Full execution trace (10-node LangGraph pipeline) |
| No governance | None | PolicyGateway: DDL block, PII protection, cost ceiling |

---

## Architecture

Built on **LangGraph** for orchestration, **OpenTelemetry** for observability, **LiteLLM** for any LLM provider.

```
User asks question
  ↓
LangGraph StateGraph (12 nodes, conditional routing)
  ↓
understand → prune → retrieve → plan → generate → execute → respond → learn
                                           ↓ (on error)
                                        correct → execute (retry, up to 3x)
                                           ↓ (cross-source)
                              decompose → fan_out → synthesize → respond
  ↓
Execution Trace (Beam.ai-style node timeline) + NL Response + SQL + Data
```

Every node emits OpenTelemetry spans. Every query is audited. Every decision is traceable.

---

## Install

```bash
pip install ora-sql
```

Works with any LLM. Works offline with Ollama. No API key required for local models.

---

## Quickstart

### One query

```python
import ora

# Connect to any database
db = ora.connect("sqlite:///northwind.db")
db = ora.connect("postgresql://user:pass@host/mydb")
db = ora.connect("data/sales.csv")

result = db.query("top 10 customers by revenue")
print(result.dataframe)   # pandas DataFrame
print(result.sql)          # generated SQL
print(result.nl_response)  # natural language summary
print(result.trace)        # full execution trace
```

### Cross-source queries

```python
import ora

with ora.connect(
    sales="postgresql://prod/sales",
    inventory="snowflake://warehouse/inventory",
    staff="headcount.csv",
) as db:
    result = db.query("which stores have the highest revenue per employee this quarter?")
    print(result.dataframe)
    # → One DataFrame. Three databases. Zero manual join keys.
```

### One-liner

```python
import ora

result = ora.ask("top 10 customers by revenue", db="sqlite:///northwind.db")
print(result.dataframe)
```

### Choose your LLM

```python
import ora

# Anthropic Claude
db = ora.connect("sqlite:///retail.db")
result = db.query("top stores", model="claude-sonnet-4-5")

# OpenAI GPT-4o
result = db.query("top stores", model="gpt-4o")

# Local (free, no API key)
result = db.query("top stores", model="ollama/llama3")
```

### Workspace UI

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # or OPENAI_API_KEY
ora serve --db sqlite:///retail.db --model claude-sonnet-4-5 --port 8080
```

Open http://localhost:8080 → Beam.ai-inspired workspace with:
- Chat interface (Ask/Execute modes)
- Execution trace timeline (node-by-node with green/red status dots)
- SQL + data table + NL response + follow-up suggestions
- Schema browser (Data view)
- Task history (Tasks view)
- QueryHub training packs (Learn view)
- 👍/👎 feedback → auto-trains the system

---

## The 10-Step Agentic Pipeline

Every query runs through a **LangGraph StateGraph** with 12 nodes and conditional routing:

| Step | What happens | Real? |
|---|---|---|
| **1. Routing** | Detect single vs cross-source, assess complexity | Yes |
| **2. Schema Pruning** | CHESS LSH: 500 cols → 8 relevant via embeddings | Yes |
| **3. Example Retrieval** | Vector search for similar NL→SQL pairs (Qdrant) | Yes |
| **4. Query Planning** | LLM plans JOIN strategy, filters, aggregations | Yes |
| **5. SQL Generation** | 3 generators run in parallel (asyncio.gather) | Yes |
| **6. Execution** | Run SQL + policy check (DDL block, PII, budget) | Yes |
| **7. Self-Correction** | ReFoRCE 3-stage: error-aware → schema-aware → DB-confirmed | Yes |
| **8. Response** | NL summary with bold highlights + follow-ups + chart | Yes |
| **9. Learning** | Write to episodic memory, update SOUL profile | Yes |
| **10. Audit** | OTel span, Prometheus metric, SQLite audit log | Yes |

Nothing is mocked. Nothing is faked. Every step is a real LLM call + real database operation.

---

## Supported databases

| Database | Status |
|---|---|
| SQLite | ✅ |
| PostgreSQL | ✅ |
| MySQL | ✅ |
| DuckDB | ✅ |
| Snowflake | ✅ |
| BigQuery | ✅ |
| Redshift | ✅ |
| CSV / XLSX / Parquet / JSON | ✅ (via DuckDB) |

## Supported LLMs

Any model via [LiteLLM](https://github.com/BerriAI/litellm):

`claude-sonnet-4-5` · `gpt-4o` · `gpt-4o-mini` · `bedrock/anthropic.claude-sonnet-4-5` · `ollama/llama3` · `azure/gpt-4o` · `vertex_ai/gemini-pro`

---

## Observability

Built-in from day one, not bolted on:

- **OpenTelemetry** — span tree per query, exportable to Jaeger/Datadog/etc.
- **Langfuse** — optional deep LLM tracing (set `LANGFUSE_PUBLIC_KEY`)
- **Prometheus** — 6 metrics: queries, latency, corrections, cost, training pairs, sessions
- **Audit log** — immutable SQLite record per query
- **Execution traces** — full node-by-node trace persisted and viewable in UI

---

## Run the demo

```bash
git clone https://github.com/ashfrnndz21/ora
cd ora
pip install -e ".[dev]"
export ANTHROPIC_API_KEY="sk-ant-..."  # or OPENAI_API_KEY
python demo.py
```

---

## Project structure (28 files)

```
sqlagent/          ← core implementation (internal package)
ora/               ← public API shim (import ora)
sqlagent/
├── agent.py              # SQLAgent + ask() + connect()
├── agents.py             # SchemaAgent, SetupAgent, DecomposeAgent, SynthesisAgent
├── auth.py               # JWT + Google OAuth + magic link
├── config.py             # AgentConfig (40+ fields)
├── exceptions.py         # 20+ typed exceptions
├── execution.py          # SQLExecutor + 3-stage ReFoRCE correction
├── generators.py         # Fewshot, Plan, Decompose + parallel Ensemble
├── hub.py                # QueryHub community training packs
├── llm.py                # LiteLLM provider + FastEmbed embedder
├── models.py             # All data models (schema, KG, trace, events)
├── retrieval.py          # Qdrant vector store + ExampleStore
├── runtime.py            # PolicyGateway + Sessions + 3-tier Memory
├── schema.py             # CHESS LSH pruning + M-Schema serializer
├── server.py             # FastAPI (40+ endpoints, SSE streaming)
├── soul.py               # SOUL user mental model (learns over 20 queries)
├── telemetry.py          # OTel + Prometheus + audit + Langfuse
├── trace.py              # TraceCollector + TraceStore
├── workspace.py          # Multi-workspace persistence
├── graph/
│   ├── state.py          # QueryState TypedDict (LangGraph)
│   ├── nodes.py          # 12 graph node functions
│   └── builder.py        # compile_query_graph() → StateGraph
├── connectors/
│   ├── sql_connectors.py      # SQLite, Postgres, MySQL, Redshift
│   ├── warehouse_connectors.py # Snowflake, BigQuery
│   └── file_connector.py      # DuckDB + CSV/XLSX/Parquet
└── ui/
    └── app.html          # Beam.ai-inspired workspace (React 18 UMD)
```

---

## Contributing

```bash
git clone https://github.com/ora-ai/ora
pip install -e ".[dev]"
pytest tests/ -q    # 194+ tests, <2s
```

---

## License

MIT — use freely, modify as needed, contribute back if you can.

---

<div align="center">

**pip install ora-sql**

</div>
