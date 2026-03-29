"""
ora quickstart — runs against the bundled Northwind SQLite database.

Requirements:
    pip install ora-sql
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

Run:
    python examples/quickstart.py
"""

import ora

# ── 1. Connect ────────────────────────────────────────────────────────────────
print("Connecting to Northwind database…")
db = ora.connect("sqlite:///tests/fixtures/northwind.db")

# ── 2. Query ──────────────────────────────────────────────────────────────────
print("\nQuery: top 10 customers by total revenue\n")
result = db.query("top 10 customers by total revenue")

if result.succeeded:
    print(result.nl_response)
    print()
    print(result.dataframe.to_string(index=False))
    print(f"\n✓ {result.row_count} rows · {result.latency_ms:.0f}ms · ${result.cost_usd:.4f}")
else:
    print(f"✗ {result.error}")

# ── 3. Inspect the generated SQL ─────────────────────────────────────────────
print("\n── Generated SQL ────────────────────────────────────────────────────────")
print(result.sql)

# ── 4. Follow-up queries ─────────────────────────────────────────────────────
print("\n── Follow-up suggestions ────────────────────────────────────────────────")
for suggestion in (result.follow_ups or []):
    print(f"  • {suggestion}")
