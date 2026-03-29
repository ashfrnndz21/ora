"""
ora quickstart — runs against the bundled Northwind SQLite database.

Requirements:
    pip install ora-sql
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY

Run:
    python examples/quickstart.py
"""

import os, sqlite3, tempfile
import ora

# ── Check for API key ─────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    print("⚠ Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running this example.")
    print("  export ANTHROPIC_API_KEY='sk-ant-...'")
    raise SystemExit(1)

# ── Create a sample SQLite database ──────────────────────────────────────────
_db_file = tempfile.mktemp(suffix=".db")
_conn = sqlite3.connect(_db_file)
_conn.executescript("""
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, revenue REAL);
INSERT INTO customers VALUES (1,'Acme Corp','USA',125000),(2,'Globex','UK',98000),
  (3,'Initech','USA',74000),(4,'Umbrella','Japan',210000),(5,'Hooli','USA',188000);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, order_date TEXT);
INSERT INTO orders VALUES (1,1,12500,'2024-01-15'),(2,2,9800,'2024-01-20'),
  (3,4,21000,'2024-02-01'),(4,5,18800,'2024-02-10'),(5,1,11500,'2024-03-01');
""")
_conn.close()

# ── 1. Connect ────────────────────────────────────────────────────────────────
print("Connecting to sample database…")
db = ora.connect(f"sqlite:///{_db_file}")

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
