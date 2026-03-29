"""
End-to-end demo of sqlagent's fully agentic pipeline.

This script:
1. Creates a real SQLite database with retail data
2. Initializes the SQLAgent with all services (LLM, embedders, generators, policy, memory)
3. Runs a natural language query through the full LangGraph orchestrator
4. Shows the complete execution trace — every node, every LLM call, every SQL execution

Requirements: OPENAI_API_KEY environment variable set (or use ollama/llama3 for local)
"""

import asyncio
import json
import os
import sqlite3
import sys
import tempfile

# ── Step 1: Create a real database ────────────────────────────────────────────

def create_demo_db() -> str:
    """Create a realistic retail database with actual data."""
    db_path = os.path.join(tempfile.gettempdir(), "sqlagent_demo.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS stores;

        CREATE TABLE stores (
            store_id TEXT PRIMARY KEY,
            store_name TEXT NOT NULL,
            region TEXT NOT NULL,
            country TEXT DEFAULT 'Thailand',
            employee_count INTEGER DEFAULT 0
        );

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            contact_name TEXT,
            country TEXT,
            email TEXT
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            unit_price REAL DEFAULT 0
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT REFERENCES customers(customer_id),
            store_id TEXT REFERENCES stores(store_id),
            order_date TEXT NOT NULL,
            total_amount REAL DEFAULT 0
        );

        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER DEFAULT 1,
            unit_price REAL DEFAULT 0
        );

        -- Stores
        INSERT INTO stores VALUES ('ST-001', 'Lotus Korat', 'Northeast', 'Thailand', 74);
        INSERT INTO stores VALUES ('ST-002', 'Lotus Chiang Mai', 'North', 'Thailand', 87);
        INSERT INTO stores VALUES ('ST-003', 'Lotus Sukhumvit', 'Bangkok', 'Thailand', 185);
        INSERT INTO stores VALUES ('ST-004', 'Lotus Phuket', 'South', 'Thailand', 52);
        INSERT INTO stores VALUES ('ST-005', 'Lotus Khon Kaen', 'Northeast', 'Thailand', 63);

        -- Customers
        INSERT INTO customers VALUES ('C001', 'Thai Foods Co', 'Somchai P.', 'Thailand', 'somchai@thaifoods.co.th');
        INSERT INTO customers VALUES ('C002', 'Bangkok Trading', 'Napat K.', 'Thailand', 'napat@bkktrading.com');
        INSERT INTO customers VALUES ('C003', 'Northern Goods', 'Apinya S.', 'Thailand', 'apinya@northerngoods.th');
        INSERT INTO customers VALUES ('C004', 'Southern Supply', 'Krit M.', 'Thailand', 'krit@southernsupply.th');
        INSERT INTO customers VALUES ('C005', 'Central Corp', 'Ploy T.', 'Thailand', 'ploy@centralcorp.co.th');

        -- Products
        INSERT INTO products VALUES (1, 'Jasmine Rice 5kg', 'Rice', 89.00);
        INSERT INTO products VALUES (2, 'Fish Sauce 700ml', 'Condiments', 45.00);
        INSERT INTO products VALUES (3, 'Coconut Milk 400ml', 'Canned', 32.00);
        INSERT INTO products VALUES (4, 'Palm Sugar 500g', 'Sweeteners', 55.00);
        INSERT INTO products VALUES (5, 'Tapioca Starch 1kg', 'Baking', 28.00);

        -- Orders (realistic spread across stores)
        INSERT INTO orders VALUES (1, 'C001', 'ST-001', '2024-07-01', 12500.00);
        INSERT INTO orders VALUES (2, 'C002', 'ST-001', '2024-07-02', 8900.00);
        INSERT INTO orders VALUES (3, 'C003', 'ST-002', '2024-07-03', 15200.00);
        INSERT INTO orders VALUES (4, 'C001', 'ST-002', '2024-07-04', 6700.00);
        INSERT INTO orders VALUES (5, 'C004', 'ST-003', '2024-07-05', 45000.00);
        INSERT INTO orders VALUES (6, 'C005', 'ST-003', '2024-07-06', 38000.00);
        INSERT INTO orders VALUES (7, 'C002', 'ST-003', '2024-07-07', 22000.00);
        INSERT INTO orders VALUES (8, 'C003', 'ST-004', '2024-07-08', 9800.00);
        INSERT INTO orders VALUES (9, 'C001', 'ST-004', '2024-07-09', 5600.00);
        INSERT INTO orders VALUES (10, 'C004', 'ST-005', '2024-07-10', 11200.00);
        INSERT INTO orders VALUES (11, 'C005', 'ST-005', '2024-07-11', 7800.00);
        INSERT INTO orders VALUES (12, 'C002', 'ST-001', '2024-07-12', 14300.00);
        INSERT INTO orders VALUES (13, 'C003', 'ST-003', '2024-07-13', 28500.00);
        INSERT INTO orders VALUES (14, 'C001', 'ST-002', '2024-07-14', 9100.00);
        INSERT INTO orders VALUES (15, 'C004', 'ST-004', '2024-07-15', 6200.00);
    """)
    conn.close()
    print(f"  Database created: {db_path}")
    print(f"  Tables: stores (5), customers (5), products (5), orders (15), order_items")
    return db_path


# ── Step 2: Run the full agentic pipeline ─────────────────────────────────────

async def run_demo():
    print("\n" + "="*70)
    print("  sqlagent v2 — End-to-End Agentic Pipeline Demo")
    print("="*70)

    # Check for API key — supports both OpenAI and Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = "claude-sonnet-4-6"
    elif os.environ.get("OPENAI_API_KEY"):
        model = "gpt-4o"
    else:
        print("\n  ⚠  No API key found.")
        print("  Set: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  Or:  export OPENAI_API_KEY='sk-...'")
        print("\n  Running in DRY RUN mode...\n")
        await run_dry_demo()
        return

    print(f"\n  Model: {model}")
    print(f"  API Key: {api_key[:12]}...{api_key[-4:]}\n")

    # Create database
    print("── Step 1: Create Database ──────────────────────────────")
    db_path = create_demo_db()

    # Initialize agent
    print("\n── Step 2: Initialize SQLAgent ──────────────────────────")
    from sqlagent.agent import SQLAgent
    agent = SQLAgent(db=f"sqlite:///{db_path}", llm_model=model)
    print("  Agent created (lazy init — services build on first query)")

    # Run query
    print("\n── Step 3: Run Query Through LangGraph ─────────────────")
    query = "which stores have the highest total revenue?"
    print(f"  Query: \"{query}\"")
    print(f"  Pipeline: understand → prune → retrieve → plan → generate → execute → respond → learn")
    print()

    result = await agent.query(query, user_id="demo_user")

    # Show results
    print("\n── Step 4: Results ─────────────────────────────────────")
    print(f"  Succeeded: {result.succeeded}")
    print(f"  SQL: {result.sql}")
    print(f"  Rows: {result.row_count}")
    print(f"  Winner: {result.winner_generator}")
    print(f"  Tokens: {result.total_tokens}")
    print(f"  Cost: ${result.total_cost_usd:.4f}")
    print(f"  Latency: {result.latency_ms}ms")
    print(f"  Corrections: {result.correction_rounds}")

    if result.nl_response:
        print(f"\n  NL Response: {result.nl_response}")

    if result.follow_ups:
        print(f"\n  Follow-ups:")
        for f in result.follow_ups:
            print(f"    → {f}")

    if result.rows:
        print(f"\n  Data ({result.row_count} rows):")
        for row in result.rows[:10]:
            print(f"    {row}")

    # Show trace
    print("\n── Step 5: Execution Trace ─────────────────────────────")
    if result.trace and result.trace.root:
        for node in result.trace.root.children:
            status = "✓" if node.status.value == "completed" else "✕"
            print(f"  {status} {node.name:25s} {node.latency_ms:>6d}ms  {node.summary}")
            if node.children:
                for child in node.children:
                    print(f"    └─ {child.name:23s}  {child.summary}")

    print(f"\n  Total: {result.trace.total_latency_ms}ms · ${result.trace.total_cost_usd:.4f} · {result.trace.total_tokens} tokens")

    # Show the full trace JSON
    print("\n── Step 6: Full Trace JSON (what the UI receives) ─────")
    if result.trace:
        trace_json = result.trace.to_dict()
        # Print compact
        print(json.dumps({
            "trace_id": trace_json["trace_id"],
            "succeeded": trace_json["succeeded"],
            "total_latency_ms": trace_json["total_latency_ms"],
            "total_cost_usd": trace_json["total_cost_usd"],
            "winner_generator": trace_json["winner_generator"],
            "nodes": [
                {"name": n["name"], "status": n["status"], "latency_ms": n["latency_ms"], "summary": n["summary"]}
                for n in (trace_json.get("root", {}).get("children", []))
            ],
        }, indent=2))

    print("\n" + "="*70)
    print("  Demo complete. This is what the frontend renders.")
    print("  Every step above was a REAL LLM call + REAL SQL execution.")
    print("="*70 + "\n")


async def run_dry_demo():
    """Show what the pipeline does without an API key."""
    print("── Dry Run: Pipeline Architecture ──────────────────────")
    print()
    print("  When you run with an API key, here's exactly what happens:\n")

    db_path = create_demo_db()

    # Show connector introspection (this is real, no LLM needed)
    print("\n── Real: Database Introspection (no LLM) ───────────────")
    from sqlagent.connectors.sql_connectors import SQLiteConnector
    conn = SQLiteConnector(source_id="demo", db_path=db_path)
    snap = await conn.introspect()
    print(f"  Dialect: {snap.dialect}")
    print(f"  Tables: {snap.table_count}")
    print(f"  Columns: {snap.column_count}")
    print(f"  Foreign keys: {len(snap.foreign_keys)}")
    for t in snap.tables:
        cols = ", ".join(c.name for c in t.columns)
        print(f"    {t.name} ({t.row_count_estimate} rows): {cols}")

    # Show real SQL execution (no LLM needed)
    print("\n── Real: SQL Execution (no LLM) ────────────────────────")
    df = await conn.execute("""
        SELECT s.store_name, SUM(o.total_amount) AS revenue, s.employee_count,
               ROUND(SUM(o.total_amount) / s.employee_count, 2) AS revenue_per_employee
        FROM orders o
        JOIN stores s ON o.store_id = s.store_id
        GROUP BY s.store_name, s.employee_count
        ORDER BY revenue DESC
    """)
    print(f"  Rows returned: {len(df)}")
    print(df.to_string(index=False))

    # Show policy check (real, deterministic)
    print("\n── Real: Policy Gateway Check (no LLM) ─────────────────")
    from sqlagent.runtime import PolicyGateway
    from sqlagent.config import AgentConfig
    policy = PolicyGateway(AgentConfig())

    good = policy.check("SELECT store_name, SUM(total_amount) FROM orders GROUP BY store_name")
    print(f"  SELECT query: passed={good.passed}" + (f", modified={good.modified_sql[:50]}..." if good.modified_sql else ""))

    bad = policy.check("DROP TABLE orders")
    print(f"  DROP query: passed={bad.passed}, reason={bad.reason}")

    pii = policy.check("SELECT email FROM customers", {})
    print(f"  PII query: passed={pii.passed}" + (f", reason={pii.reason}" if not pii.passed else ""))

    # Show LangGraph compilation (real)
    print("\n── Real: LangGraph Compilation ──────────────────────────")
    from sqlagent.graph.builder import compile_query_graph
    from sqlagent.agent import PipelineServices
    services = PipelineServices(config=AgentConfig())
    services.connectors = {"demo": conn}
    graph = compile_query_graph(services)
    print(f"  Graph compiled successfully")
    print(f"  Entry: understand")
    print(f"  Nodes: understand → prune → retrieve → plan → generate → execute → respond → learn")
    print(f"  Conditional: understand → decompose (cross-source)")
    print(f"  Conditional: execute → correct (on error, up to 3 rounds)")

    # Show trace structure
    print("\n── Real: Trace Structure (what UI renders) ──────────────")
    print("""  {
    "trace_id": "qry_abc123",
    "succeeded": true,
    "total_latency_ms": 1234,
    "total_cost_usd": 0.0042,
    "nodes": [
      {"name": "Routing",          "status": "completed", "latency_ms": 50,  "summary": "Single source → demo"},
      {"name": "Schema Pruning",   "status": "completed", "latency_ms": 5,   "summary": "18 cols → 8 relevant"},
      {"name": "Example Retrieval","status": "completed", "latency_ms": 3,   "summary": "0 similar examples"},
      {"name": "Query Planning",   "status": "completed", "latency_ms": 400, "summary": "Strategy: join_and_aggregate"},
      {"name": "SQL Generation",   "status": "completed", "latency_ms": 890, "summary": "3 candidates · winner: fewshot"},
      {"name": "Execution",        "status": "completed", "latency_ms": 12,  "summary": "5 rows returned"},
      {"name": "Response",         "status": "completed", "latency_ms": 200, "summary": "NL summary + 3 follow-ups"},
      {"name": "Learning",         "status": "completed", "latency_ms": 5,   "summary": "Memory updated"}
    ]
  }""")

    print("\n── What you need to run the REAL pipeline ───────────────")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  python demo.py")
    print()
    print("  Or with a local model (no API key):")
    print("  # First: ollama pull llama3")
    print("  # Then modify demo.py: model='ollama/llama3'")
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
