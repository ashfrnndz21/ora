# sqlagent v2 — Definitive Build Spec

> This is the single source of truth for rebuilding sqlagent from scratch.
> The backend exists and is 90% production-ready (253 tests, 61 endpoints).
> This spec covers: agentic runtime architecture, Beam.ai-inspired UI/UX, schema pipeline, and every feature.

---

## 1. PRODUCT VISION

sqlagent is a **data reasoning workspace** — not a chat bot, not a SQL translator. An AI agent workspace that:
- Discovers and understands your data automatically
- Reasons through complex questions step by step
- Shows its work as **task execution traces** (node-by-node, like Beam.ai)
- Learns from every interaction (trains on thumbs-up, evolves user model)
- Works with any data source (files, databases, warehouses)
- Provides full visibility into every agent decision (no black boxes)

**Design north star: Beam.ai** — the same deep navy workspace aesthetic, task execution traces,
sidebar navigation, glass card system, and workspace-level organization.

---

## 2. AGENTIC RUNTIME ARCHITECTURE

### 2.1 Supervisor + Specialist Agents

```
                    ┌─────────────────┐
                    │   Supervisor    │  Decides what to do next
                    │   (LangGraph)   │  based on current state
                    └────────┬────────┘
                             │
         ┌───────────┬───────┼───────┬───────────┬──────────┐
         ▼           ▼       ▼       ▼           ▼          ▼
    ┌─────────┐┌─────────┐┌──────┐┌─────────┐┌─────────┐┌──────┐
    │ Schema  ││ Query   ││ SQL  ││ Execute ││ Respond ││ Learn│
    │ Agent   ││ Planner ││Writer││ Agent   ││ Agent   ││ Agent│
    └─────────┘└─────────┘└──────┘└─────────┘└─────────┘└──────┘
```

**Supervisor Agent** (LangGraph StateGraph):
- Evaluates current state after each node
- Routes conditionally: budget exhausted? → skip execution. Validation failed? → correction loop.
- Emits events to EventBus → SSE streaming to frontend → live execution trace updates

**Schema Agent** — Runs on connect + on demand:
- Introspects tables/columns/types/PKs/FKs from live database
- Samples 5 rows per table for entity resolution
- Runs LLM analysis: infer relationships, entity groups, column semantics
- Produces knowledge graph (nodes, edges, layers)

**Query Planner** — The strategist:
- Assesses complexity: simple (1 table), moderate (2-3 joins), complex (decompose)
- For complex: breaks into numbered sub-steps with individual SQL targets
- Outputs a plan with reasoning, not raw SQL

**SQL Writer** — Parallel generation:
- Runs multiple generators concurrently (fewshot, plan-based, decompose, RL model)
- Each captures full chain-of-thought reasoning
- Best candidate selected by LLM pairwise comparison

**Execute Agent** — Run + validate + correct:
- Executes SQL against connector with timeout + row limit
- Validates output: row count reasonable? columns match intent?
- If error → correction loop (up to 3 rounds, ReFoRCE algorithm)
- Emits query subgraph: which tables/joins/columns were actually used

**Respond Agent** — Human-readable output:
- NL summary with bold highlights
- Follow-up suggestions based on what the data revealed (not generic)
- Auto-detect chart type

**Learn Agent** — Background, non-blocking:
- Record to episodic memory
- Update SOUL profile (every 20 queries)
- Auto-train on positive feedback

### 2.2 LangGraph Conditional Routing

```
understand → plan → generate → [budget check] → execute → validate
                                                     ↓         ↓
                                               (exhausted)   (failed)
                                                   ↓            ↓
                                                respond    correct → execute (retry)
                                                               ↓
                                                          (max retries)
                                                               ↓
                                                            respond → learn → END
```

Three conditional edge functions:
- `route_after_generate`: if budget_exhausted → respond (skip execution)
- `route_after_validate`: if needs_correction AND round < max → correct, else → respond
- All nodes emit typed events via EventBus → SSE streaming → live execution trace

### 2.3 Memory Stack (5 Tiers + SOUL)

| Tier | What | Persistence | Use |
|------|------|-------------|-----|
| L1: Working | Current session turns | In-memory | Multi-turn context |
| L2: Episodic | Per-user query history | SQLite per workspace | Pattern recognition |
| L3: Semantic | Trained NL→SQL pairs | Qdrant vectors per workspace | Few-shot examples |
| L4: Summary | Compressed conversation | LLM-generated | Long sessions |
| L5: Reflection | Cross-episode patterns | JSON per workspace | Behavioral patterns |
| SOUL | User mental model | JSON per user | Vocabulary, preferences |

**SOUL evolves automatically:**
- Every 20 queries: LLM analyzes patterns → updates profile
- Temporal decay: instinct confidence decays over 90 days
- Convergence detection: if profile stabilizes, reduce evolution frequency

### 2.4 Guardrails + Budget

| Guard | What | Implementation |
|-------|------|----------------|
| Token ceiling | Max tokens per session/query | `session.token_budget.remaining` |
| SQL validation | No DROP, DELETE, ALTER, TRUNCATE | Regex check before execution |
| Row limit | Max rows returned (default 10,000) | `LIMIT` injected into SQL |
| PII masking | Auto-detect PII columns | `semantic_type: email/phone/ssn` → mask |
| Policy engine | Custom rules | `policy.check(sql, session)` before execution |
| Cost alerts | Warn approaching budget | Event → UI warning badge |

### 2.5 Observability Stack

| Layer | Tool | What It Captures |
|-------|------|------------------|
| Tracing | OpenTelemetry + Langfuse | Span tree per query, token counts, latency, cost |
| Metrics | Prometheus | queries_total, latency, corrections, cost, training_pairs, sessions |
| Audit | SQLite | Immutable per-query record |
| Request tracing | FastAPI middleware | x-trace-id, x-latency-ms headers |
| In-memory buffer | Ring buffer (200) | Recent trace events for /debug/traces |

### 2.6 Connectors

| Connector | Status | FK Detection |
|-----------|--------|--------------|
| PostgreSQL | ✅ | information_schema |
| MySQL | ✅ | information_schema |
| SQLite | ✅ | PRAGMA foreign_key_list |
| DuckDB | ✅ | LLM inference |
| Snowflake | ✅ | SHOW IMPORTED KEYS |
| BigQuery | ✅ | Heuristic (_id naming) |
| Redshift | ✅ | pg_constraint |
| CSV/XLSX/Parquet/JSON | ✅ | LLM inference via DuckDB |

### 2.7 Workspaces

Per-connection isolated state at `~/.sqlagent/connections/{connection_hash}/`:
```
config.json              # Connection metadata, display name, table count
training.db              # Qdrant vectors + trained NL→SQL pairs
history.db               # Chat messages + query history + stats
episodic.db              # Per-user query memory
soul/                    # User SOUL profiles (JSON)
schema_cache.json        # Last introspected schema snapshot
schema_analysis.json     # LLM analysis
uploads/                 # Uploaded CSV/XLSX files
hub_installed.json       # Installed QueryHub packs
```

### 2.8 Stateful Sessions

```json
{
  "session_id": "uuid",
  "conversation_turns": [
    {"role": "user", "text": "top customers"},
    {"role": "assistant", "sql": "SELECT...", "nl_response": "..."}
  ],
  "active_schema_context": ["customers", "orders"],
  "token_budget_remaining": 45000,
  "started_at": "...",
  "last_active": "..."
}
```

---

## 3. SCHEMA PIPELINE

### 3.1 Three Paths Into the System

```
Database URL  ──→  Live introspection (progressive)
DDL file      ──→  DDL parser → SchemaSnapshot
CSV/XLSX      ──→  DuckDB ingestion → introspection
dbt YAML      ──→  dbt parser → SchemaSnapshot + descriptions
```

All paths produce `SchemaSnapshot` → fed to Schema Agent.

### 3.2 Progressive Schema Loading

**Layer 1 (instant):** Schema names + table names + row count estimates
**Layer 2 (2-5s):** Column metadata for selected schemas/tables
**Layer 3 (5-15s):** FK relationships, constraints, indexes
**Layer 4 (10-30s):** LLM analysis — inferred relationships, entity groups, column semantics

For 500+ table databases: present schema picker UI.

### 3.3 Schema Analysis (LLM-powered)

Input: SchemaSnapshot + 5 sample rows per table
Output:
- `InferredRelationship[]` — semantic FKs (confidence 0-1)
- `EntityGrouping[]` — business domain clusters
- `ColumnDescription[]` — semantic type + description
- `DataQualityObservation[]` — anomalies
- `SuggestedJoin[]` — join paths with reasoning

### 3.4 Two-Layer Schema Visualization

**Layer A: Full Knowledge Map** (Data view)
- ALL tables, columns, relationships
- Entity group layers as colored backgrounds
- Solid teal = declared FK, dashed purple = inferred

**Layer B: Query Subgraph** (inline per query result)
- ONLY tables/columns used in this query
- Shows join path and pruning stats

### 3.5 Schema Drift Detection

On every query, compare current schema hash vs cached → notify if changed.

### 3.6 `/schema/graph` Endpoint

```json
{
  "version": "1.0",
  "project": {"name": "...", "dialect": "...", "table_count": N},
  "nodes": [
    {"id": "table:customers", "type": "table", "name": "customers", "row_count": 91, "column_count": 5},
    {"id": "column:customers.id", "type": "column", "parent": "table:customers", "data_type": "INTEGER", "is_primary_key": true}
  ],
  "edges": [
    {"source": "table:orders", "target": "table:customers", "type": "foreign_key", "weight": 0.95},
    {"source": "table:orders", "target": "table:products", "type": "inferred", "weight": 0.7}
  ],
  "layers": [
    {"id": "layer:Sales", "name": "Sales", "tables": ["orders", "order_items", "customers"]}
  ]
}
```

---

## 4. FRONTEND UI/UX — BEAM.AI-INSPIRED WORKSPACE

### 4.1 Tech Stack

- React 18 UMD (CDN, no build step)
- Chart.js 4 (CDN)
- Single HTML file: `sqlagent/ui/app.html` (~1500-2000 lines)
- `React.createElement` (no JSX, no bundler)
- Vanilla SVG for schema graph

### 4.2 Design System — Beam.ai Aesthetic

**Reference:** The Beam.ai platform screenshots define every visual decision.
Deep navy workspace, glass morphism cards, task execution traces, status badges.

#### Color Palette

```
/* Canvas */
--canvas:           #0a0f1e;      /* Deep navy — THE signature color */
--canvas-deep:      #060a14;      /* Sidebar, deepest surfaces */
--canvas-elevated:  #0e1425;      /* Slightly elevated surfaces */

/* Glass Surfaces */
--glass:            rgba(255,255,255,.04);    /* Card background */
--glass-border:     rgba(255,255,255,.06);    /* Card border */
--glass-hover:      rgba(255,255,255,.08);    /* Hover state */
--glass-active:     rgba(255,255,255,.12);    /* Active/pressed */

/* Text Hierarchy */
--text:             #e8eaf0;      /* Primary — slightly warm white */
--text-secondary:   #8b92a8;      /* Labels, descriptions */
--text-muted:       #4a5068;      /* Section headers, hints */
--text-disabled:    #2d3348;      /* Disabled elements */

/* Primary — Beam.ai Blue */
--blue:             #4f7df9;      /* Primary accent */
--blue-glow:        rgba(79,125,249,.15);   /* Ambient glow */
--blue-soft:        rgba(79,125,249,.10);   /* Active nav bg */
--blue-text:        #7ba0ff;      /* Blue on dark backgrounds */

/* Secondary — Indigo */
--indigo:           #6366f1;      /* Secondary accent */
--indigo-glow:      rgba(99,102,241,.08);   /* Atmospheric gradient */

/* Status Colors */
--green:            #22c55e;      /* Success, completed */
--green-soft:       rgba(34,197,94,.10);    /* Success bg */
--green-text:       #4ade80;      /* Green on dark */

--amber:            #f59e0b;      /* Warning, pending */
--amber-soft:       rgba(245,158,11,.10);   /* Warning bg */
--amber-text:       #fbbf24;      /* Amber on dark */

--red:              #ef4444;      /* Error, failed */
--red-soft:         rgba(239,68,68,.10);    /* Error bg */
--red-text:         #f87171;      /* Red on dark */

/* Atmospheric Effects */
--gradient-glow:    radial-gradient(ellipse at 30% 0%, rgba(79,125,249,.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 100%, rgba(99,102,241,.04) 0%, transparent 40%);
--card-shadow:      0 1px 3px rgba(0,0,0,.3), 0 0 0 1px rgba(255,255,255,.04);
--modal-shadow:     0 8px 32px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.06);
--ambient-glow:     0 0 60px rgba(79,125,249,.06);
```

#### Typography

```
/* Font Stack */
--font-ui:       'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
--font-mono:     'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
--font-display:  'Inter', sans-serif;

/* Scale */
--text-xs:   11px / 1.4;   /* Badges, tiny labels */
--text-sm:   12px / 1.5;   /* Secondary labels, table cells */
--text-base: 13px / 1.5;   /* Body text, nav items */
--text-md:   14px / 1.5;   /* Primary content */
--text-lg:   16px / 1.4;   /* Section headers */
--text-xl:   20px / 1.3;   /* Page titles */
--text-2xl:  28px / 1.2;   /* KPI numbers */
--text-3xl:  36px / 1.1;   /* Hero numbers */

/* Weights */
--font-regular:  400;
--font-medium:   500;
--font-semibold: 600;
--font-bold:     700;

/* Rendering */
-webkit-font-smoothing: antialiased;
-moz-osx-font-smoothing: grayscale;
```

#### Spacing & Layout

```
/* Spacing Scale (4px base) */
--space-1:  4px;
--space-2:  8px;
--space-3:  12px;
--space-4:  16px;
--space-5:  20px;
--space-6:  24px;
--space-8:  32px;
--space-10: 40px;
--space-12: 48px;
--space-16: 64px;

/* Border Radius */
--radius-sm:  6px;    /* Small buttons, badges */
--radius-md:  8px;    /* Buttons, inputs */
--radius-lg:  12px;   /* Cards */
--radius-xl:  16px;   /* Modals, large panels */

/* Layout */
--sidebar-width:  200px;
--context-width:  300px;
--max-content:    1200px;
```

#### Component Patterns

**Glass Card:**
```css
.card {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: var(--card-shadow);
}
.card:hover {
  background: var(--glass-hover);
  border-color: rgba(255,255,255,.10);
}
```

**Status Badge:**
```css
.badge {
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  padding: 2px 8px;
  border-radius: 100px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.badge-success {
  background: var(--green-soft);
  color: var(--green-text);
}
.badge-warning {
  background: var(--amber-soft);
  color: var(--amber-text);
}
.badge-error {
  background: var(--red-soft);
  color: var(--red-text);
}
.badge-info {
  background: var(--blue-soft);
  color: var(--blue-text);
}
```

**Sidebar Nav Item:**
```css
.nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-4);
  color: var(--text-secondary);
  font-size: var(--text-base);
  border-radius: var(--radius-md);
  cursor: pointer;
  position: relative;
  transition: all 150ms ease;
}
.nav-item:hover {
  color: var(--text);
  background: var(--glass-hover);
}
.nav-item.active {
  color: var(--text);
  background: var(--blue-soft);
}
.nav-item.active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 6px;
  bottom: 6px;
  width: 3px;
  background: var(--blue);
  border-radius: 0 2px 2px 0;
}
```

**Execution Trace Node:**
```css
.trace-node {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-3) 0;
  position: relative;
}
.trace-node::before {
  /* Vertical connector line */
  content: '';
  position: absolute;
  left: 9px;
  top: 28px;
  bottom: -12px;
  width: 1px;
  background: rgba(255,255,255,.08);
}
.trace-node:last-child::before {
  display: none;
}
.trace-dot {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
.trace-dot.completed {
  background: var(--green-soft);
  color: var(--green);
}
.trace-dot.running {
  background: var(--blue-soft);
  color: var(--blue);
  animation: pulse 1.2s ease-in-out infinite;
}
.trace-dot.failed {
  background: var(--red-soft);
  color: var(--red);
}
.trace-dot.queued {
  background: var(--glass);
  color: var(--text-muted);
}
```

#### Animations

```css
@keyframes fadeUp {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes nodeReveal {
  from { transform: translateX(-4px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
@keyframes slideIn {
  from { transform: translateX(16px); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
@keyframes badgePop {
  from { transform: scale(0.85); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 20px rgba(79,125,249,.08); }
  50% { box-shadow: 0 0 40px rgba(79,125,249,.15); }
}
```

### 4.3 Layout Structure

```
┌─ Sidebar ──┬──────────── Main Canvas ──────────────┬─ Context Panel ─┐
│  (200px)   │                                        │   (300px)       │
│            │  Currently showing one of:             │                 │
│  Logo      │  • Home (dashboard)                    │  Contextual     │
│  Nav       │  • Ask (chat + execution traces)       │  intelligence   │
│  Agents    │  • Data (knowledge map)                │  for current    │
│  Status    │  • Tasks (execution history)           │  view           │
│            │  • Learn (training progress)            │                 │
│            │  • Settings                             │                 │
└────────────┴────────────────────────────────────────┴─────────────────┘
```

### 4.4 Sidebar (200px, always visible)

```
┌────────────────────┐
│ ◆ sqlagent         │  Logo (blue gradient square)
│ retail_analytics   │  Workspace name (from connection URL)
│ 12 tables · PG     │  Table count + dialect badge
│────────────────────│
│ WORKSPACE          │  Section label (uppercase, --text-muted, 10px)
│ ▌△ Home            │  Active = blue bar + blue-soft bg
│   ◎ Ask            │
│   ⬡ Data           │
│   ☰ Tasks          │  NEW: Beam.ai task list
│   📈 Learn         │
│   ⚙ Settings       │
│                    │
│ YOUR AGENTS        │  Section label
│   ● SQL Agent      │  Green dot = active
│   ● Schema Agent   │
│   ○ Learn Agent    │  Gray dot = idle
│                    │
│                    │
│ ● Agent Ready      │  Green dot + text at bottom
│ v1.0.0             │  Version string, muted
└────────────────────┘
```

**Workspace name derivation:**
- `postgresql://host/mydb` → "mydb"
- `GenAI_Spend_Analysis.xlsx` → "GenAI Spend Analysis"
- `sqlite:///northwind.db` → "northwind"

Data: `GET /api/status` → source_id, table_count, dialect, agent_ready, version

### 4.5 Home View (Dashboard)

**Data sources (6 parallel API calls):**
- `/api/status` → table_count, column_count, model, query_count
- `/api/history/stats` → total_queries, succeeded, accuracy, total_cost_usd, avg_latency_ms
- `/api/learning/stats` → trained_pairs, recent_accuracy, recommendations
- `/api/connections` → [{dialect, display_name, table_count, is_active}]
- `/api/chat/sessions` → [{session_id, first_query, message_count}]
- `/health` → version, uptime_s

**Layout:**
```
┌─ Workspace Header ──────────────────────────────────────┐
│ retail_analytics                                         │
│ 12 tables · 87 columns · claude-sonnet-4 · v1.0.0      │
│                                        ● Agent Ready    │
└─────────────────────────────────────────────────────────┘

┌── KPI ────┐ ┌── KPI ────┐ ┌── KPI ────┐ ┌── KPI ─────┐
│    47     │ │   92%     │ │  $1.24    │ │   0.8s     │
│  queries  │ │ accuracy  │ │   cost    │ │  latency   │
└───────────┘ └───────────┘ └───────────┘ └────────────┘

┌─ Quick Query ──────────────────────── [▶ Run] ──┐
│ Ask anything about your data...                  │
└──────────────────────────────────────────────────┘

┌─ Recent Tasks ────────────────┐ ┌─ Data Sources ──────────┐
│ QRY-47  "top customers..."  ✓ │ │ 🐘 retail_prod     [ACT]│
│ QRY-46  "by region..."     ✓ │ │ 🦆 headcount.csv        │
│ QRY-45  "describe data"    ✓ │ │                          │
│ View all tasks →             │ │ + Add connection         │
└──────────────────────────────┘ └──────────────────────────┘
```

**KPI cards:** Glass cards. Large number (--text-2xl, --font-bold). Label below (--text-sm, --text-secondary). Subtle colored underline matching meaning (blue=queries, green=accuracy, amber=cost, blue=latency).

**All numbers from REAL API data. Zero mocked values.**
**Empty states:** "Run your first query →", "Connect a database →"

### 4.6 Ask View (Query Studio — Chat + Execution Traces)

**THE signature view.** This is where Beam.ai's design language matters most.

**Split layout:** Thread (flexible) | Context Panel (300px)

**Chat input bar (Beam.ai pattern — bottom-anchored):**
```
┌─[📎]──[ Ask | Execute ]──────────────────────────[⚙]──[➤]─┐
│ Ask anything about your data...                              │
└──────────────────────────────────────────────────────────────┘
```
- Tab toggle: **Ask** (NL query) / **Execute** (raw SQL)
- 📎 = attach file for ad-hoc analysis
- ⚙ = query settings dropdown (model, generators, budget cap)
- ➤ = send (also ⌘Enter)
- Glass card background, 1px border, subtle inner shadow

**Empty state (centered, inviting):**
```
        ◎
  Ask about your data
  Natural language to SQL — powered by AI agents

  [How many rows in customers?]  [Describe the data]  [What patterns exist?]
```

Starter chips: glass cards, clickable, fill the input on click.

**After query — the execution trace unfolds:**

```
┌─── User ──────────────────────────────────────────┐
│ "which stores have highest revenue per employee?" │
└────────────────────────────────────────────────────┘

┌─ Task QRY-48 ──────────────────────────── 1.2s · $0.004 ─┐
│                                                            │
│  ▸ 6 steps completed                          [▶ Replay]  │
│                                                            │
│  ● ① Schema Pruning              ✓ Completed    5ms      │
│  │   87 cols → 8 relevant (91% pruned)                    │
│  │                                                         │
│  ● ② Example Retrieval           ✓ Completed    3ms      │
│  │   2 similar queries found                               │
│  │                                                         │
│  ● ③ Query Planning              ✓ Completed    12ms     │
│  │   Strategy: JOIN stores + sales + staff                 │
│  │   ▸ Show reasoning                                      │
│  │                                                         │
│  ● ④ SQL Generation              ✓ Completed    890ms    │
│  │   3 candidates · winner: fewshot · 2.1k tokens         │
│  │   ▸ Show all candidates                                 │
│  │                                                         │
│  ● ⑤ Execution                   ✓ Completed    45ms     │
│  │   10 rows returned · 0 corrections                      │
│  │                                                         │
│  ● ⑥ Response                    ✓ Completed    200ms    │
│      NL summary + follow-ups generated                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Execution trace behavior:**
- Default: **collapsed** to single line "6 steps · 1.2s · $0.004"
- Click to expand full trace
- During execution: nodes reveal one-by-one with `nodeReveal` animation
- Running node: blue pulsing dot + "Running..." label
- Completed: green dot + checkmark + timing
- Failed: red dot + ✕ + error message + "Retry from here" button
- Each node expandable to show full detail (input/output/reasoning)

**Replay:** Click "▶ Replay" → nodes reveal sequentially with 600ms delay, previous nodes dim slightly.

**Result block (appears after trace):**

```
┌─ Query Subgraph ─────────────────────────────────┐
│ [stores] ──store_id──→ [sales]                    │
│ [stores] ──store_id──→ [staff]                    │
│ Pruned: 87 cols → 8 relevant                      │
└───────────────────────────────────────────────────┘

"Lotus Korat leads with **$2,284 revenue per employee**,
 followed by Lotus Chiang Mai ($2,207) and
 Lotus Sukhumvit ($2,114)."

┌─ [ SQL ] ─── [ Table·10 ] ─── [ Chart ] ──── Copy ⬇CSV ✎Edit ─┐
│                                                                   │
│  [Active tab content — SQL / Table / Chart]                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

[Break down by region?]  [Compare vs last quarter?]  [Which are underperforming?]

👍 👎
```

**Tab content:**
- **SQL tab:** syntax highlighted (blue keywords, green strings, white identifiers), JetBrains Mono
- **Table tab:** sortable columns, alternating subtle row tint, sticky headers, formatted numbers (1,234,567.89), max 100 rows
- **Chart tab:** auto-detected (bar if categories+values ≤20 rows, line if time series)

**Actions:**
- **Copy:** clipboard + toast "SQL Copied"
- **CSV:** download file + toast "CSV Exported"
- **Edit:** textarea mode → re-run → save as training pair
- **👍:** calls `/train/sql` + toast "Training pair saved"
- **👎:** records negative signal + optional correction form

**Follow-up suggestions:** Glass chip buttons, contextual (not generic), click to query.

**Cross-source execution trace:**
```
┌─ Task QRY-49 (Cross-Source) ─────────── 1.8s · $0.012 ─┐
│                                                          │
│  ● ① Decomposition              ✓ Completed    50ms    │
│  │   Split into 2 sub-queries across 2 sources          │
│  │                                                       │
│  ├── ● Sub-task A: postgres     ✓ Completed    300ms   │
│  │   │ SELECT store_id, SUM(total_amount)...             │
│  │   │ 284 rows                                          │
│  │                                                       │
│  ├── ● Sub-task B: staff.csv    ✓ Completed    100ms   │
│  │   │ SELECT store_id, SUM(headcount)...                │
│  │   │ 284 rows                                          │
│  │                                                       │
│  ● ② DuckDB Synthesis           ✓ Completed    40ms    │
│  │   JOIN on store_id → 284 rows                         │
│  │                                                       │
│  ● ③ Response                   ✓ Completed    200ms   │
│      Summary generated                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Sub-tasks indented under parent with branching connector lines.

**Context Panel (300px right side):**
- **Schema Reference:** expandable tree (table → columns with types), click column to insert
- **Session Info:** token count, cost so far, query count in session
- **SOUL Badge:** "store · weekly · 47q" (inferred mental model)
- **Suggested Tables:** highlighted based on current query
- **QueryHub:** installed packs + "Browse packs"

**SSE Event → UI Mapping:**
| SSE Event | UI Action |
|---|---|
| `query.started` | Create task card, show node ① as running |
| `schema.pruned` | Complete ①, show pruning stats, start ② |
| `examples.retrieved` | Complete ② with example count |
| `reasoning.emitted` | Complete ③ with plan text |
| `candidate.generated` | Update ④ with generator info per candidate |
| `execution.result` | Complete ⑤ with row count |
| `query.final` | Complete ⑥, render full result block |
| `query.error` | Mark current node as failed, show error |

### 4.7 Tasks View (Beam.ai Task List)

**This is the Beam.ai "Tasks" view.** Every query is a task with full trace.

```
┌─ Tasks ────────────────────────────────────────────────────────────┐
│ Overview of all queries in this workspace               Create task│
│                                                                     │
│ [Status ▾]  [Agent ▾]              Search tasks...          [View] │
│                                                                     │
│ ☐  QRY-48  top stores by revenue per..  ● 92%  ✓ Completed       │
│            fewshot generator · SQL Agent           2m ago    →     │
│                                                                     │
│ ☐  QRY-47  which country has most ord.. ● 88%  ✓ Completed       │
│            plan generator · SQL Agent              5m ago    →     │
│                                                                     │
│ ☐  QRY-46  describe the data structure  ● 95%  ✓ Completed       │
│            fewshot generator · Schema Agent         8m ago    →     │
│                                                                     │
│ ☐  QRY-45  cross-source revenue analys  ○ —    ✕ Failed          │
│            decompose · Orchestrator · timeout      12m ago    →     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Table columns:** checkbox, task ID, query text (truncated), confidence %, status badge, agent badge, timestamp, arrow to detail.

**Filters:**
- Status: All / Completed / Failed / Running
- Agent: SQL Agent / Schema Agent / Orchestrator
- Search: fuzzy match on query text

**Click task → Task Detail View (Beam.ai pattern):**

Split into three sections:
```
┌─ Task Activity ──┬─ Execution Trace ─────────────────┬─ Output ──────┐
│                  │                                     │               │
│ Task created     │ ● ① Schema Pruning     ✓  5ms     │ SQL           │
│ 2m ago           │ │   87→8 cols                      │ SELECT...     │
│                  │ │                                   │               │
│ Task execution   │ ● ② Example Retrieval  ✓  3ms     │ Results       │
│ started          │ │   2 found                         │ 10 rows       │
│                  │ │                                   │               │
│ Completed        │ ● ③ Planning           ✓  12ms    │ NL Response   │
│ 1.2s · $0.004    │ │   JOIN strategy                   │ "Lotus Korat  │
│                  │ │                                   │  leads..."    │
│                  │ ● ④ Generation         ✓  890ms   │               │
│                  │ │   3 candidates                    │ Decision      │
│                  │ │                                   │ ✓ Accepted    │
│                  │ ● ⑤ Execution          ✓  45ms    │               │
│                  │ │   10 rows                         │ [👍] [👎]    │
│                  │ │                                   │               │
│                  │ ● ⑥ Response           ✓  200ms   │ [Rerun]       │
│                  │                                     │ [Edit SQL]    │
└──────────────────┴─────────────────────────────────────┴───────────────┘
```

Click any trace node → expands to show full context:
- Input (the data that entered this node)
- Output (what it produced)
- Error details (if failed) with full stack trace
- "Retry from this step" button

### 4.8 Data View (Knowledge Map)

**Data source:** `GET /schema/graph`

**Canvas (dark, dot grid background):**
```
┌──────────── Canvas ──────────────────────────────┬─ Detail ──┐
│  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·    │           │
│                                                   │ Selected  │
│  ┌── Sales ──────────────────────────┐           │ table     │
│  │  ┌──────────┐    ┌──────────┐    │           │ details   │
│  │  │ orders   │───→│customers │    │           │           │
│  │  │ 1.2M · 8c│    │ 42k · 5c │    │           │ Columns   │
│  │  └──────────┘    └──────────┘    │           │ FKs       │
│  └───────────────────────────────────┘           │ Quality   │
│                    │ (Industry)                    │           │
│  ┌── Analytics ────────────────────┐             │ [Query]   │
│  │  ┌──────────┐                   │             │ [Annotate]│
│  │  │benchmarks│                   │             │           │
│  │  │ 41 · 7c  │                   │             │           │
│  │  └──────────┘                   │             │           │
│  └─────────────────────────────────┘             │           │
│                                                   │           │
│  [Auto Layout]  [Refresh]  [Run Analysis]        │           │
└───────────────────────────────────────────────────┴───────────┘
```

**Table nodes:** Glass cards (230px)
- Header: table name + row/col count
- Column list: compact, max 12 visible
- PK: key icon + blue text
- FK: arrow icon + teal text
- Selection: blue glow ring (--ambient-glow)

**Edges:**
- Declared FK: solid teal, opacity 0.6, arrow marker
- Inferred: dashed indigo, opacity 0.4
- Cross-source: dotted blue, opacity 0.3
- All: cubic bezier curves

**Entity group layers:** Translucent colored backgrounds with group name

**Interactions:**
- Drag nodes to reposition
- Pan canvas (drag background)
- Click node → detail panel slides in (glass, 300px)
- Click edge → shows join reasoning
- Auto Layout → 3-column grid
- Refresh → re-introspect
- Run Analysis → LLM schema analysis → inferred edges

**Detail panel (on node click):**
- Table name + row/col count
- Full column list with data types (mono font)
- FK relationships (clickable)
- Data quality observations
- Annotate button → glossary form
- "Query this table" button → navigates to Ask

### 4.9 Learn View (Training Progress)

**Data source:** `/api/learning/stats`, `/api/history/stats`, `/api/history`

```
┌─ Training Progress ─────────────────────────────────────┐
│                                                          │
│  Trained Pairs        Accuracy Trend       Status        │
│  ████████░░ 12       67% → 78% → 92%      ✓ Improving  │
│                                                          │
│  Recent improvements:                                    │
│  ● "top customers" → uses revenue (you corrected)        │
│  ● "by region" → maps to Country (learned ×3)            │
│                                                          │
│  Recommendations:                                        │
│  ⚠ 3 queries have low confidence — consider training     │
│  ⚠ No examples for JOIN queries — add some               │
└──────────────────────────────────────────────────────────┘

┌─ Query History (sortable table) ────────────────────────┐
│  Task ID  │  Query            │ ✓/✗ │ Cost  │ Time     │
│  QRY-48   │  "top stores..."  │  ✓  │$0.004│ 1.2s     │
│  QRY-47   │  "which country"  │  ✓  │$0.008│ 2.1s     │
│                                                          │
│  [Train]  [Retrain All]  [Export Training Data]          │
└──────────────────────────────────────────────────────────┘
```

### 4.10 Settings View

Glass cards with sections:
- **API Keys:** OpenAI + Anthropic password inputs + Validate (✓ or ✕)
- **Model:** dropdown (Claude Sonnet 4, GPT-4o, GPT-4o Mini, Llama 3)
- **Database:** connection URL + file upload zone
- **Pipeline:** Cache toggle, Auto-learn toggle, Row Limit number
- **Budget:** Token ceiling per session, cost alert threshold
- **Save** → toast "Settings Saved"

### 4.11 Setup View (Onboarding)

Centered wizard (max 520px), glass card:
1. API Key input + Validate button
2. Database URL input OR file upload (drag & drop)
3. Per-file status badges during upload
4. "Save & Discover Schema →" → connects → analyzes → Home

### 4.12 Global Features

**Toast Notifications:**
- Fixed bottom-right, glass cards
- Colored left accent (green=success, red=error, blue=info, amber=warning)
- Auto-dismiss 4s
- Triggers: Copy SQL, Export CSV, Train, Settings saved, Upload, Error

**Cmd+K Search Overlay:**
- Glass modal, centered (480px), dark
- Search: tables, columns, sessions, navigation
- Keyboard: ESC close, arrows navigate, Enter select
- Results: icon + label + subtitle

**Error Boundary:**
- React class component wrapping app
- On crash: "Something went wrong" card + Reload button

**Session Persistence:**
- `localStorage` for sessionId + active view
- Multi-turn context survives reload
- Session restore from Home → loads from `/api/chat/messages`

**AbortController:**
- Cancel in-flight SSE on new query or navigation

---

## 5. API ENDPOINTS (Complete Reference)

### Core Query
| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Execute NL query (non-streaming) |
| POST | `/query/stream` | Execute NL query (SSE streaming) |

### Schema
| Method | Path | Description |
|--------|------|-------------|
| GET | `/schema` | Basic schema (tables + columns) |
| GET | `/schema/graph` | Knowledge graph (nodes, edges, layers) |
| GET | `/schema/topology` | Cross-source topology |
| GET | `/schema/diff` | Schema changes since last introspect |
| POST | `/schema/refresh` | Force re-introspect |
| POST | `/schema/annotate` | Add glossary term to column |
| POST | `/schema/validate-link` | Confirm/reject inferred relationship |

### Training
| Method | Path | Description |
|--------|------|-------------|
| POST | `/train/sql` | Add NL→SQL training pair |
| POST | `/train/docs` | Add documentation text |
| GET | `/hub/packs` | List QueryHub packs |
| POST | `/hub/install` | Install a training pack |

### Sessions + History
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/chat/sessions` | List sessions with preview |
| GET | `/api/chat/messages` | Get messages for a session |
| POST | `/api/chat/messages` | Save a chat message |
| GET | `/api/history` | Query history (recent N) |
| GET | `/api/history/stats` | Aggregate stats |
| DELETE | `/session/{id}` | Delete a session |

### Tasks (NEW — Beam.ai-inspired)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/tasks` | List all tasks with status/timing |
| GET | `/api/tasks/{id}` | Single task with full execution trace |
| POST | `/api/tasks/{id}/retry` | Retry from a specific trace node |

### Configuration
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | System status |
| GET | `/api/config` | Full config |
| GET | `/api/settings` | Current settings |
| POST | `/api/settings` | Save settings |
| POST | `/api/settings/validate-key` | Validate API key |
| GET | `/api/connections` | List workspaces |
| POST | `/api/upload` | Upload file |

### Learning
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/learning-progress` | Training + accuracy stats |
| GET | `/api/learning/stats` | Detailed learning metrics |
| GET | `/api/schema/analysis` | Cached schema analysis |

### Observability
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + telemetry |
| GET | `/ready` | Readiness check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/debug/traces` | Recent trace events |
| GET | `/debug/audit` | Recent audit log |

### SOUL
| Method | Path | Description |
|--------|------|-------------|
| GET | `/soul/{user_id}` | SOUL profile |
| GET | `/soul/{user_id}/md` | Export as markdown |
| POST | `/soul/{user_id}/evolve` | Trigger evolution |

### Feedback
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/feedback` | Submit thumbs up/down |

---

## 6. FILE STRUCTURE

```
sqlagent/
├── agent.py                    # SQLAgent main class
├── config.py                   # AgentConfig dataclass
├── connectors/                 # Database connectors
│   ├── base.py                 # Connector protocol
│   ├── registry.py             # Auto-detect from URL
│   ├── postgres.py
│   ├── mysql.py
│   ├── sqlite.py
│   ├── duckdb.py               # + file ingestion
│   ├── snowflake.py
│   ├── bigquery.py
│   └── redshift.py
├── core/
│   ├── pipeline.py             # 7-stage pipeline
│   ├── events.py               # Typed events
│   └── hooks.py                # EventBus
├── graph/
│   ├── state.py                # QueryState TypedDict
│   ├── nodes.py                # Agent nodes
│   ├── graph.py                # StateGraph
│   └── tracing.py              # @traced_node
├── generators/
│   ├── base.py                 # Generator protocol
│   ├── fewshot_generator.py
│   ├── decompose_generator.py
│   ├── rl_model_generator.py
│   └── ensemble.py             # Parallel + selection
├── agents/
│   ├── schema_analyzer.py
│   ├── response_generator.py
│   ├── output_validator.py
│   ├── file_analyzer.py
│   ├── learning_loop.py
│   ├── decompose.py
│   ├── synthesis.py
│   └── orchestrator.py
├── runtime/
│   ├── memory/
│   │   └── working.py          # 5-tier memory
│   ├── session.py
│   ├── workspace.py
│   ├── query_history.py
│   ├── soul.py                 # SOUL runtime
│   └── query_cache.py
├── schema/
│   └── models.py               # SchemaSnapshot
├── retrieval/
│   └── qdrant_store.py
├── telemetry/
│   ├── otel.py
│   ├── metrics.py
│   └── audit_log.py
├── server.py                   # FastAPI (61+ endpoints)
├── settings_store.py
├── soul.py                     # SOUL model
└── ui/
    └── app.html                # Single-file workspace UI
```

---

## 7. VERIFICATION CHECKLIST

After rebuild, verify:

- [ ] `pytest tests/ -q` → 253+ passed
- [ ] Home: real KPIs from API (numbers update after queries)
- [ ] Home: click task → opens task detail with execution trace
- [ ] Ask: query → execution trace animates node-by-node → result
- [ ] Ask: trace nodes show green dots + timing as they complete
- [ ] Ask: failed node shows red + error + "Retry from here"
- [ ] Ask: thumbs up → toast + Learn count increments
- [ ] Ask: Copy SQL → toast. CSV → download. Edit → re-run.
- [ ] Ask: multi-turn context works
- [ ] Tasks: table view with status badges, agent labels, timing
- [ ] Tasks: click → three-panel detail (activity, trace, output)
- [ ] Tasks: filter by status and agent
- [ ] Data: knowledge map loads from /schema/graph
- [ ] Data: table nodes draggable, FK edges visible
- [ ] Data: click table → detail panel
- [ ] Data: Run Analysis → inferred edges appear
- [ ] Learn: training pairs count, accuracy trend, recommendations
- [ ] Settings: save → toast
- [ ] Setup: upload CSV → schema refresh → Home
- [ ] Setup: validate API key → ✓ or ✕
- [ ] Cmd+K: search tables, sessions, navigate
- [ ] All sidebar nav items have text labels
- [ ] Active nav item has blue left bar indicator
- [ ] Workspace name derived from connection URL
- [ ] Toasts appear for all actions (glass cards, colored accent)
- [ ] Error boundary catches crashes
- [ ] Session survives page reload (localStorage)
- [ ] Deep navy canvas (#0a0f1e), not black or gray
- [ ] Glass cards throughout (rgba white surfaces, blur)
- [ ] Status badges use correct colors (green/amber/red/blue)
- [ ] Execution trace uses vertical timeline with connected nodes
