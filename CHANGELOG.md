# Changelog

All notable changes to **ora** (`pip install ora-sql`) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
ora uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] — 2026-04-06 — Agentic Intelligence Release

### Ora ReAct Orchestrator
- Replaced 12-node conditional LangGraph with 3-node ReAct loop (`ora -> respond -> learn`)
- Ora does ALL thinking in one node: decompose, semantic resolve, schema context, SQL generation, validation, learning
- **Semantic fitness check** — LLM reviews its own SQL output: "does this result answer the question?" Catches missing columns, wrong entities, incomplete decomposition coverage
- **Entity coverage validation** — detects when "ALL ASEAN" resolves to only 2 of 10 countries. Known groups: ASEAN(10), G7(7), EU(27), BRICS(5), OECD(38), GCC(6), APEC(21)
- **Schema gap detection** — flags when queries need time dimensions, geographic breakdowns, or numeric measures that don't exist in the target tables
- **Source-aware decomposition** — decomposition prompt sees all available sources and assigns each query part to a target_domain. Cross-source queries detected automatically
- **Column validation** — semantic agent output validated against actual pruned schema. Non-existent columns stripped with WARNING to SQL agent
- **Observable learning trace** — every query shows "Applied learned context" node with example count, rule count, pre-resolved aliases, semantic layer version

### Semantic Layer Evolution
- **Patterns auto-injection** — `query_patterns.json` loaded during `reason_about_query()`. Filters with 3+ confirmed uses auto-injected as defaults (e.g., `sex='Total'`)
- **Column enrichments** — `column_enrichments.json` fed into LLM reasoning with past usage context
- **Alias pre-check** — aliases with confidence >= 0.93 resolved deterministically BEFORE LLM call (zero token cost for known entities)
- **Resolution logging** — every resolution attempt tracked in `resolution_log.json` (question, filters, confidence, succeeded)
- **Failure reflection** — negative signals recorded after failed queries. Past failures loaded as anti-patterns: "Vietnam doesn't work, use Viet Nam"
- **Lightweight taxonomy** — `build_initial_taxonomy()` replaces exhaustive alias bootstrap at connect time. Cross-source join detection (no LLM) + entity type classification (single LLM call)
- **Semantic manifest** — `semantic_manifest.json` tracks iteration_id, history of every semantic layer update with timestamp, trigger, and exact items learned
- **Detail tracking** — `evolve_semantic_layer()` saves actual items: which relationship confirmed, which pattern learned, which column enriched

### Structured Rule Engine (new: `sqlagent/rules.py`)
- Learned rules with confidence, scope, hit count, success rate, and lifecycle
- Created from user corrections (0.9 confidence) or pattern detection (0.7)
- Applied during SQL generation — top 5 rules injected into prompt
- Confirmed (+0.05) on success, weakened (-0.10) on failure, expired below 0.30
- Persisted to `rules.json` per workspace
- Deduplicated: same rule text boosts confidence instead of creating duplicate

### REST API Connector Framework (new: `sqlagent/connectors/rest_connector.py`)
- Base class for all SaaS/POS/ERP/CRM integrations
- Auth: OAuth2 (with token refresh), API key (custom header/prefix), Bearer token, Basic auth
- Pagination: cursor-based, offset, page number, link-header (Shopify/GitHub style)
- Rate limiting: token bucket with exponential backoff, respects Retry-After header
- Schema inference: first pull -> DataFrame -> column types auto-detected
- Data flow: REST API -> JSON -> pd.json_normalize -> DuckDB in-memory -> SQL queryable

### SaaS Connectors (new: `sqlagent/connectors/catalog/`)
- **Shopify** — orders, customers, products, inventory, collections (`shopify://store?api_key=xxx`)
- **Salesforce** — accounts, contacts, opportunities, leads, cases (`salesforce://instance?access_token=xxx`)
- **Stripe** — charges, customers, subscriptions, invoices, payouts, balance_transactions (`stripe://?api_key=xxx`)
- **HubSpot** — contacts, companies, deals, tickets (`hubspot://?api_key=xxx`)
- **Google Analytics 4** — sessions, users, events (`ga4://property_id?access_token=xxx`)
- All registered in `ConnectorRegistry.from_url()` with URL-based routing

### Airbyte Integration (new: `sqlagent/connectors/airbyte_connector.py`)
- Embedded mode: PyAirbyte runs source connectors in-process
- Managed mode: reads from existing Airbyte instance's destination DB
- Data lands in DuckDB for SQL querying (`airbyte://?source=source-name`)

### Learn Agent Improvements
- **Persistent training pairs** — Qdrant stored on disk per workspace (survives server restarts)
- **Per-workspace vectorstore** — each workspace gets its own Qdrant path (no lock conflicts)
- **_ensure_ready()** called before accessing _learn_graph (fixes 501 on first correction)
- **CTE-aware rewriting** — rewrite prompt: "CTEs are scoped to their WITH block. Never reference a CTE as a standalone table"
- **Re-execute endpoint** — `POST /learn/re-execute` tests edited SQL before saving to training
- **Re-execute button** in correction UI
- **Learn page impact metrics** — accuracy trend (early vs recent), active rules with hit count + success rate
- **LearningLoop** re-exported from agents package (fixes import shadowing)

### Knowledge Page (redesigned)
- **Graph tab** — force-directed semantic graph: table nodes with confidence halos, relationship edges, learned term annotations, drag support
- **Taxonomy tab** — 4-layer knowledge feed:
  - Foundation (connect time): domain, tables, all column meanings
  - Inferred (data patterns): aliases, cross-source joins with ON columns
  - Confirmed (queries): each iteration expandable to see exact items learned
  - Corrected (user feedback): rules with confidence lifecycle
- **Agent tab** — conversational chat with full semantic layer context (aliases, patterns, rules, relationships, evolution history). Stateful conversation history. Typewriter streaming effect.
- Clean header: just "Knowledge" + tabs. No fake metrics.

### UI Updates
- Landing page (`preview_bp.html`) updated: Ora phases, Semantic Agent passes, LearnAgent, SQL Agent labels
- Sidebar agents: Semantic Agent elevated, descriptions reflect v2.0 capabilities
- Agent Hub cards: Ora v2.0 with 6 capabilities, Semantic Agent with pattern memory, LearnAgent with semantic evolution
- Sources page: SaaS connector cards with branded letter icons (S, SF, St, HS, GA, Ab, BQ, SF)
- Setup chat: persistent messages via localStorage, streamed greeting typewriter, workspace naming before data, SaaS connector guidance
- Chart axis selection: respond_node analyzes varying vs constant columns, LLM picks x_column/y_column
- Token tooltips: hover ↑/↓ arrows for "input tokens sent to LLM" / "output tokens received from LLM"
- kpi_cards and hide_columns in respond_node output for smarter result display
- Error handling: "Fetch is aborted" shows actionable message, SSE stream catches exceptions with query.error event

### Observability
- Token tracking accurate via `LiteLLMProvider._session_tokens_input/output` accumulators
- FastAPI instrumented with `FastAPIInstrumentor` (OpenTelemetry)
- `@traced_node` decorator applied in learn_graph (5 nodes)
- Prometheus `/metrics` endpoint active with 6 metrics
- Semantic fitness check tokens tracked in session totals

### API Endpoints (new)
- `POST /learn/re-execute` — execute edited SQL from correction flow
- `GET /api/semantic/history` — semantic layer evolution history with iteration details
- `GET /api/semantic/graph` — graph-ready semantic model (nodes, edges, synonyms, patterns, rules)
- `GET /api/learn/impact` — accuracy trend, rule inventory, learning impact metrics

### Bug Fixes
- Fixed `test_graph.py`: removed old `route_after_ora` imports for v2.0 3-node architecture
- Fixed `SetupAgent` import: created `sqlagent/agents/setup.py` re-export wrapper
- Fixed `SchemaAgent` import: created `sqlagent/schema_agent.py` bridge module for full `analyze()` method
- Fixed `LearningLoop` import: created `sqlagent/agents/learning_loop.py` re-export wrapper
- Fixed Qdrant lock conflict: per-workspace vectorstore paths (`~/.sqlagent/vectorstore/{workspace_id}/`)
- Fixed `unhashable type: 'list'` in entity coverage validation (filter values can be lists for IN operators)
- Fixed React error #310: all `useState` hooks moved before conditional returns in KnowledgeView
- Fixed SQL correction loop: actual column names listed explicitly on retry so SQL agent uses correct names
- Fixed schema truncation: correction round includes full column reference (3000 chars, up from 2000)

---

## [Unreleased]

### Added
- **Conversation Lineage on Home** — pipeline execution traces embedded directly on the Home dashboard as expandable tiles; click any turn to see the full OTel-style node-by-node trace (Schema Agent → Query Pipeline → Response), replacing the static Pipeline tab
- **Column value sampling in schema introspection** — `FileConnector.introspect()` now samples up to 20 distinct values for low-cardinality columns (≤ 50 distinct) and stores them in `SchemaColumn.examples`; dramatically improves CHESS LSH pruning for code columns (e.g. `geo: ['MYS', 'VNM', 'IDN', ...]`)
- **Examples in LSH embedding text** — `SchemaSelector.prune()` now includes `col.examples` when building the column embedding, so queries referencing country codes, category values, or enum-like columns resolve correctly without any manual schema hints
- **Multi-turn conversation history** — query context flows through the pipeline as structured `conversation_history` state; `understand_node` uses prior turns for multi-source routing decisions
- **ISO code expansion in schema routing** — `understand_node` expands ISO-2/3 country codes (e.g. `VN` → `Vietnam`, `MYS` → `Malaysia`) before LSH so geographic queries match schema columns even when column names are terse
- **Schema exploration fallback** — when CHESS LSH returns zero relevant columns (exploratory queries), `prune_node` falls back to the full schema so the LLM always has something to reason over; `schema_exploration_mode` flag is passed to `generate_node` for context-aware SQL generation
- **Animated network graph background** on workspace picker (data topology canvas)
- `--no-auth` / `auth_enabled=False` local mode — skip login screen entirely for OSS use
- `GET /health` now returns `auth_required` field for frontend routing
- `app.html` served with `no-cache, no-store` headers — browser always gets fresh version
- `SECURITY.md` — vulnerability disclosure policy
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `.github/workflows/ci.yml` — automated test, lint, security scan, Docker build on every PR
- `.github/dependabot.yml` — weekly automated dependency updates
- GitHub issue and PR templates

### Fixed
- **`history_str` NameError** in `understand_node` multi-source routing prompt — variable now correctly derived from `conversation_history` state
- **CI lint failures** — 58 ruff errors resolved (unused imports, ambiguous variable names, late module-level imports, multi-statement lines, undefined names); codebase now fully lint-clean
- 45+ bare `except Exception: pass` blocks replaced with `logger.debug(...)` calls
- `print()` statements in production code replaced with `structlog` calls
- Dockerfile now runs as non-root user and includes `HEALTHCHECK`
- `MagicLinkRequest` Pydantic forward-ref error causing `/openapi.json` 500
- Greeting endpoint used invalid model name `claude-haiku-3-5` — fixed to `claude-3-haiku-20240307`
- Greeting now always uses fastest available model regardless of query model setting

### Security
- Fixed CORS wildcard (`*`) + `allow_credentials=True` combination — now defaults to deny-all; set `SQLAGENT_CORS_ORIGINS` to opt in
- Fixed magic-link code returned in HTTP response — code no longer exposed to requester
- Added `chmod 0600` to JWT secret file to prevent other-user reads
- Added `.env` to `.gitignore` to prevent accidental credential commits

### Changed
- **Pipeline tab removed** — execution traces now live on the Home dashboard as conversation lineage tiles (richer and more interactive)
- CORS default changed from `["*"]` to `[]` (deny-all) — **breaking for cross-origin setups**
- Dockerfile base installs `postgres,mysql` extras; use `--build-arg EXTRAS=all` for full suite
- JWT secret file permissions hardened to `0600`

## [2.0.0] — 2026-03-01

### Added
- Complete rewrite as agentic NL2SQL runtime
- LangGraph 7-stage pipeline (routing → schema pruning → retrieval → generation → execution → selection → audit)
- CHESS LSH schema pruning — 500 cols → ~8 relevant
- M-Schema serialization (XiYan format)
- CHASE-SQL parallel multi-generator with pairwise LLM selection
- ReFoRCE 3-stage self-correction loop
- SOUL user mental model inference
- Cross-source DuckDB synthesis for multi-DB queries
- PolicyGateway — deterministic SQL allowlist, zero LLM calls
- Beam.ai-inspired workspace UI (single HTML file, no build step)
- FastAPI server with 20+ endpoints + SSE streaming
- SQLite, PostgreSQL, DuckDB, MySQL, Snowflake connectors
- In-process Qdrant vector store for few-shot retrieval
- 3-tier memory: working → episodic → semantic
- QueryHub with 28 bundled NL→SQL packs
- AWS AgentCore adapter stubs
