"""
ora async usage — for FastAPI, Django async views, Jupyter, and any async context.

Requirements:
    pip install ora-sql
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

Run:
    python examples/async_usage.py
"""

import asyncio
import ora


async def main():
    # Connect synchronously, query asynchronously
    db = ora.connect("sqlite:///tests/fixtures/northwind.db")

    print("Running async query…\n")
    result = await db.async_query("monthly revenue trend for the last 12 months")

    if result.succeeded:
        print(result.nl_response)
        print()
        print(result.dataframe.to_string(index=False))
        print(f"\n✓ {result.row_count} rows · {result.latency_ms:.0f}ms")
    else:
        print(f"✗ {result.error}")

    # Run multiple queries in parallel — 3x faster than sequential
    print("\n── Parallel queries ─────────────────────────────────────────────────────")
    queries = [
        "total revenue by product category",
        "top 5 countries by order count",
        "average order value by month",
    ]

    results = await asyncio.gather(*[db.async_query(q) for q in queries])

    for q, r in zip(queries, results):
        status = f"✓ {r.row_count} rows" if r.succeeded else f"✗ {r.error}"
        print(f"  [{status}] {q}")


if __name__ == "__main__":
    asyncio.run(main())
