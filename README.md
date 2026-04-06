<div align="center">

# ora

**The NL2SQL Agentic Runtime**

*Not a chatbot wrapper. A fully agentic system that reasons, self-corrects, and learns from every query.*

[![PyPI](https://img.shields.io/pypi/v/ora-sql?color=4f7df9&label=pip%20install%20ora-sql)](https://pypi.org/project/ora-sql)
[![CI](https://github.com/ashfrnndz21/ora/actions/workflows/ci.yml/badge.svg)](https://github.com/ashfrnndz21/ora/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![Docs](https://img.shields.io/badge/docs-v2.0-4f7df9)](https://ashfrnndz21.github.io/ora/v2.html)

</div>

<div align="center">

An on-machine AI agent that turns natural language into SQL across any database — PostgreSQL, Snowflake, BigQuery, DuckDB, Shopify, Salesforce, or plain CSV files. Ora decomposes complex questions, resolves entity semantics, generates and self-corrects SQL, and builds a living semantic layer that gets smarter with every query.

**[Quickstart](#quickstart)** · **[v2.0 Docs](https://ashfrnndz21.github.io/ora/v2.html)** · **[Architecture](#architecture)** · **[What's New](#whats-new-in-v20)** · **[Changelog](CHANGELOG.md)**

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

### v2.0 Architecture (ReAct)

```
User asks question
  |
Ora ReAct Orchestrator (single node — does ALL thinking)
  |
  +-- Phase 1: Decompose (intent, entities, query structure, cross-source detection)
  +-- Phase 2: Semantic Resolve (alias pre-check, pattern injection, failure reflection)
  +-- Phase 3: Schema Context (CHESS LSH pruning, column validation against actual schema)
  +-- Phase 4: SQL Generation (3 candidates, parallel, self-correct up to 3x)
  +-- Phase 5: Semantic Fitness Check (LLM reviews "does this answer the question?")
  +-- Phase 6: Validate & Learn (evolve semantic layer, record rule outcomes)
  |
  +-> respond (NL summary + chart + follow-ups)
  +-> learn (aliases, patterns, enrichments, rules)
```

3-node LangGraph: `ora -> respond -> learn`. Ora does everything in one ReAct loop.

---

## What's New in v2.0

### Ora ReAct Orchestrator

The v1.0 multi-node pipeline (12 nodes, conditional routing) is replaced by a **single ReAct orchestrator** that thinks, delegates, validates, and re-routes — all in one node. This eliminates routing errors and gives Ora full context at every decision point.

- **Semantic fitness check** — after SQL executes, Ora asks: "Does this result actually answer the question?" Not just "did rows come back?" but "are the right columns, entities, and dimensions present?" If not, Ora re-routes with specific fix instructions.
- **Entity coverage validation** — if a query asks for "ALL ASEAN countries" but the semantic resolution only produced 2 filter values, Ora catches it before SQL generation. Known group sizes: ASEAN (10), G7 (7), EU (27), BRICS (5), OECD (38).
- **Schema gap detection** — before SQL generation, Ora checks if the query requires dimensions (time trends, geographic breakdown) that don't exist in the target tables. Flags data limitations honestly.
- **Source-aware decomposition** — the decomposition prompt now sees all available data sources and assigns each query part to a target domain. Cross-source queries are detected and flagged.
- **Column validation** — semantic agent output is validated against the actual pruned schema. Non-existent columns are stripped with a WARNING to the SQL agent.

### Semantic Layer Evolution

The semantic layer is no longer static. It evolves with every query through a 4-layer taxonomy:

**Foundation** (connect time) — `analyze_source()` discovers domain, column meanings, abbreviation maps, dimension/measure classification, filter tips. `build_initial_taxonomy()` detects cross-source join candidates and entity types with a single LLM call.

**Inferred** (from data patterns) — entity aliases, value mappings, cross-source joins detected via column name/type matching.

**Confirmed** (from successful queries) — `evolve_semantic_layer()` runs after every successful query:
- New aliases saved with confidence scores (strengthened +3% per confirmation, cap 0.99)
- Table relationships confirmed with query count
- Filter patterns saved (auto-injected as defaults after 3+ uses)
- Column enrichments tracked (what each column was used for)
- Semantic manifest incremented with iteration ID + history

**Corrected** (from user feedback) — structured rules created from corrections with confidence lifecycle:
- Created at 0.9 confidence from user corrections
- Applied during SQL generation (top 5 rules injected)
- Confirmed (+0.05) when query succeeds with rule applied
- Weakened (-0.10) when query fails with rule applied
- Expired when confidence drops below 0.30

Each evolution step is tracked in `semantic_manifest.json` with iteration ID, timestamp, trigger, and exact items learned.

### Semantic Agent Reasoning Loop

The semantic agent now uses a multi-pass iterative reasoning approach:

1. **Pre-check** — high-confidence aliases (>=0.93) resolved deterministically before the LLM call. Known patterns and column enrichments injected into prompt. Past failures loaded as anti-patterns.
2. **LLM reasoning** — entity mapping with full schema context, column meanings, and learned vocabulary.
3. **Schema search** — for unresolved entities, targeted DB lookups across text columns.
4. **Refinement** — merge findings, update confidence, save resolution log entry.
5. **Failure reflection** — when queries fail, negative signals recorded so the agent avoids repeating mistakes.

### Structured Rule Engine

`sqlagent/rules.py` — learned rules with confidence, scope, and lifecycle:

```python
from sqlagent.rules import load_rules, create_rule, record_rule_outcome

# Rules are created from corrections, applied during SQL generation,
# confirmed/weakened based on outcomes, and expired when stale.
rules = load_rules(workspace_id)  # sorted by confidence * hit_count
```

### Observable Learning in Traces

Every query trace now shows an "Applied learned context" node:
```
Applied learned context
  3 past examples (best: 'top selling product' sim:0.94)
  2 rules: [sex='Total' default; DuckDB no YEAR()]
  4 context notes
  12 pre-resolved aliases
  semantic layer v8
```

### REST API Connector Framework

New `RestConnector` base class for SaaS/POS/ERP/CRM integrations:

```python
from sqlagent.connectors.catalog.shopify import ShopifyConnector

conn = ShopifyConnector(
    source_id="shopify_store",
    store_name="mystore",
    api_key="shpat_xxx",
)
await conn.connect()  # pulls data into DuckDB for SQL querying
```

Built-in connectors:
| Connector | Tables | Connection URL |
|---|---|---|
| Shopify | orders, customers, products, inventory, collections | `shopify://store?api_key=xxx` |
| Salesforce | accounts, contacts, opportunities, leads, cases | `salesforce://instance?access_token=xxx` |
| Stripe | charges, customers, subscriptions, invoices, payouts | `stripe://?api_key=xxx` |
| HubSpot | contacts, companies, deals, tickets | `hubspot://?api_key=xxx` |
| Google Analytics 4 | sessions, users, events | `ga4://property_id?access_token=xxx` |
| Airbyte | 300+ sources via PyAirbyte or managed instance | `airbyte://?source=source-name` |

All REST connectors handle auth (OAuth2, API key, Bearer), pagination (cursor, offset, link-header), rate limiting (token bucket with backoff), and schema inference automatically.

### Knowledge Page (Semantic Taxonomy)

The Knowledge page is the Semantic Agent's working memory — a live, observable taxonomy showing how the agent's understanding evolves:

- **Graph tab** — force-directed semantic graph with table nodes, relationship edges, confidence halos, and learned term annotations
- **Taxonomy tab** — 4-layer knowledge feed (Foundation, Inferred, Confirmed, Corrected) with expandable entries showing exact items learned per query
- **Agent tab** — conversational interface to the Semantic Agent with full semantic layer context (aliases, patterns, rules, relationships, evolution history)

### Learn Page Improvements

- **Impact metrics** — accuracy trend (early vs recent), rule inventory with hit count + success rate
- **Persistent training pairs** — Qdrant stored on disk per workspace (survives server restarts)
- **Re-execute button** — edit corrected SQL and re-run before saving to training
- **CTE-aware rewriting** — Learn Agent's rewrite prompt prevents invalid CTE references

### UI Updates

- Landing page updated for v2.0 architecture (Ora phases, Semantic Agent passes)
- Sources page: SaaS connector cards with branded letter icons (Shopify, Salesforce, Stripe, etc.)
- Chart axis selection by LLM (analyzes varying vs constant columns)
- Token arrow tooltips (hover to see "input tokens" vs "output tokens")
- Setup chat: persistent messages across reloads, streamed greeting, workspace naming before data
- Error handling: "Fetch is aborted" shows helpful message, SSE stream catches exceptions

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

## Supported databases & integrations

| Source | Status | Type |
|---|---|---|
| SQLite | ✅ | Database |
| PostgreSQL | ✅ | Database |
| MySQL | ✅ | Database |
| DuckDB | ✅ | Database |
| Snowflake | ✅ | Data Warehouse |
| BigQuery | ✅ | Data Warehouse |
| Redshift | ✅ | Data Warehouse |
| CSV / XLSX / Parquet / JSON | ✅ | File (via DuckDB) |
| Shopify | ✅ v2.0 | SaaS (REST API) |
| Salesforce | ✅ v2.0 | SaaS (REST API) |
| Stripe | ✅ v2.0 | SaaS (REST API) |
| HubSpot | ✅ v2.0 | SaaS (REST API) |
| Google Analytics 4 | ✅ v2.0 | SaaS (REST API) |
| Airbyte (300+ sources) | ✅ v2.0 | Integration Platform |

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

## Project structure

```
sqlagent/                          ← core implementation
ora/                               ← public API shim (import ora)
sqlagent/
├── agent.py                       # SQLAgent + ask() + connect()
├── agents.py                      # SchemaAgent (full), SetupAgent, DecomposeAgent, SynthesisAgent
├── auth.py                        # JWT + Google OAuth + magic link
├── config.py                      # AgentConfig (40+ fields, bootstrap_aliases flag)
├── exceptions.py                  # 20+ typed exceptions
├── execution.py                   # SQLExecutor + 3-stage ReFoRCE correction
├── generators.py                  # Fewshot, Plan, Decompose + parallel Ensemble
├── hub.py                         # QueryHub community training packs
├── llm.py                         # LiteLLM provider + FastEmbed embedder
├── models.py                      # All data models (schema, KG, trace, events)
├── retrieval.py                   # Qdrant vector store + ExampleStore
├── rules.py                       # v2.0: Structured rule engine (confidence lifecycle)
├── runtime.py                     # PolicyGateway + Sessions + 3-tier Memory
├── schema.py                      # CHESS LSH pruning + M-Schema serializer
├── schema_agent.py                # v2.0: Bridge to full SchemaAgent with analyze()
├── semantic_agent.py              # v2.0: Semantic layer evolution + taxonomy + manifest
├── server.py                      # FastAPI (60+ endpoints, SSE streaming)
├── soul.py                        # SOUL user mental model (learns over 20 queries)
├── telemetry.py                   # OTel + Prometheus + audit + Langfuse
├── trace.py                       # TraceCollector + TraceStore
├── visualization.py               # LLM-powered chart generation (Vega-Lite)
├── confidence.py                  # LLM-assessed query confidence scoring
├── workspace.py                   # Multi-workspace persistence
├── agents/                        # v2.0: Multi-agent package
│   ├── orchestrator.py            # Ora ReAct orchestrator (fitness check, coverage, gaps)
│   ├── semantic.py                # SemanticAgent (iterative reasoning wrapper)
│   ├── schema.py                  # SchemaAgent (lightweight, for orchestrator)
│   ├── sql.py                     # SQLAgent (generation + validation)
│   ├── learn.py                   # LearnAgent (semantic layer evolution)
│   ├── setup.py                   # SetupAgent (conversational workspace creation)
│   ├── learning_loop.py           # LearningLoop (trace-aware training)
│   └── protocol.py                # AgentRequest/Response/ValidationResult/QueryDecomposition
├── graph/
│   ├── state.py                   # QueryState TypedDict (LangGraph)
│   ├── nodes.py                   # Graph node functions (respond, learn)
│   ├── builder.py                 # v2.0: 3-node graph (ora → respond → learn)
│   ├── ora_react.py               # v2.0: Ora ReAct entry point + AgentTrace
│   └── learn_graph.py             # v2.0: Learn Agent 5-node correction pipeline
├── connectors/
│   ├── __init__.py                # Connector protocol + ConnectorRegistry
│   ├── sql_connectors.py          # SQLite, Postgres, MySQL, Redshift
│   ├── warehouse_connectors.py    # Snowflake, BigQuery
│   ├── file_connector.py          # DuckDB + CSV/XLSX/Parquet/JSON
│   ├── rest_connector.py          # v2.0: REST API base class (auth, pagination, rate limit)
│   ├── airbyte_connector.py       # v2.0: Airbyte adapter (embedded + managed)
│   └── catalog/                   # v2.0: SaaS connector definitions
│       ├── shopify.py             # Shopify Admin API
│       ├── salesforce.py          # Salesforce REST API
│       ├── stripe.py              # Stripe API
│       ├── hubspot.py             # HubSpot CRM API
│       └── google_analytics.py    # Google Analytics 4 Data API
└── ui/
    ├── app.html                   # Workspace app (React 18 UMD, single file)
    └── preview_bp.html            # Landing page with architecture visualization
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

### Get started

```bash
pip install ora-sql
```

**[Quickstart](#quickstart)** · **[v2.0 Docs](https://ashfrnndz21.github.io/ora/v2.html)** · **[Changelog](CHANGELOG.md)** · **[Contributing](CONTRIBUTING.md)**

Built by [@ashfrnndz21](https://github.com/ashfrnndz21)

</div>
