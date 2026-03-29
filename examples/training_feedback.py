"""
ora training feedback — teach ora from your corrections.
Every thumbs-up or manual correction trains the vector store and improves future queries.

Requirements:
    pip install ora-sql
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

Run:
    python examples/training_feedback.py
"""

import ora
from sqlagent.agent import SQLAgent

# ── Direct training — add known-good NL→SQL pairs ─────────────────────────────
print("Training ora with domain-specific examples…\n")

agent = SQLAgent(db="sqlite:///tests/fixtures/northwind.db")

# Add curated training pairs for your domain
training_pairs = [
    {
        "nl": "show me revenue by store",
        "sql": "SELECT CompanyName, SUM(UnitPrice * Quantity) AS revenue FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID JOIN [Order Details] od ON o.OrderID = od.OrderID GROUP BY c.CompanyName ORDER BY revenue DESC",
        "note": "Always use CompanyName, not CustomerID in output",
    },
    {
        "nl": "which products are out of stock",
        "sql": "SELECT ProductName, UnitsInStock FROM Products WHERE UnitsInStock = 0 ORDER BY ProductName",
        "note": "Out of stock = UnitsInStock exactly 0",
    },
]

import asyncio

async def run():
    for pair in training_pairs:
        await agent.train_sql(
            nl_query=pair["nl"],
            sql=pair["sql"],
        )
        print(f"  ✓ Trained: \"{pair['nl']}\"")

    # ── Query with trained examples ───────────────────────────────────────────
    print("\nQuerying with trained examples…")
    result = await agent.query("which products are out of stock?")

    if result.succeeded:
        print(f"\n{result.nl_response}")
        print(result.dataframe.to_string(index=False))
    else:
        print(f"✗ {result.error}")

    # ── Export training data ──────────────────────────────────────────────────
    print("\n── Training summary ─────────────────────────────────────────────────────")
    print("Training pairs are stored in the Qdrant vector store.")
    print("Run `ora serve` to manage them via the Learn view in the UI.")

asyncio.run(run())
