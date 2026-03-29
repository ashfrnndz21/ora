"""
ora multi-source demo — queries across two SQLite databases simultaneously.
ora decomposes the question into sub-queries, runs them in parallel, then
joins the results in-memory with DuckDB — zero manual SQL.

Requirements:
    pip install ora-sql
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

Run:
    python examples/multi_source.py
"""

import ora

# ── Cross-source query ────────────────────────────────────────────────────────
# Imagine: orders live in one database, customer info in another.
# ora handles the decomposition and synthesis automatically.
print("Running cross-source query…\n")

with ora.connect(
    orders="sqlite:///tests/fixtures/northwind.db",
    # Add more sources:
    # customers="postgresql://user:pass@host/crm",
    # headcount="staff_data.csv",
) as db:
    result = db.query("which customers placed the most orders this year, and from which countries?")

if result.succeeded:
    print(result.nl_response)
    print()
    print(result.dataframe.to_string(index=False))
    print(f"\n✓ {result.row_count} rows · {result.latency_ms:.0f}ms")
    print(f"  Sources used: {', '.join(result.sources_used or [])}")
else:
    print(f"✗ {result.error}")

# ── Inspect the execution trace ───────────────────────────────────────────────
if result.trace:
    print("\n── Execution trace ──────────────────────────────────────────────────────")
    for node in result.trace:
        status = "✓" if getattr(node, "success", True) else "✗"
        duration = getattr(node, "duration_ms", 0)
        print(f"  {status} {node.name:<20} {duration:>6.0f}ms")
