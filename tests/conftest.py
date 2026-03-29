"""Shared test fixtures for sqlagent."""

from __future__ import annotations

import tempfile
import sqlite3
from pathlib import Path

import pytest

from sqlagent.config import AgentConfig, DataSourceConfig
from sqlagent.models import (
    SchemaColumn, SchemaTable, SchemaSnapshot, ForeignKey,
    KnowledgeGraph, KGNode, KGEdge, KGLayer, EdgeType,
    Workspace, User, TrainingExample,
)


# ── Northwind-like test database ──────────────────────────────────────────────

NORTHWIND_DDL = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    contact_name TEXT,
    country TEXT,
    email TEXT
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    total_amount REAL DEFAULT 0
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category TEXT,
    unit_price REAL DEFAULT 0
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER DEFAULT 1,
    unit_price REAL DEFAULT 0
);

INSERT INTO customers VALUES ('ALFKI', 'Alfreds Futterkiste', 'Maria Anders', 'Germany', 'maria@alfreds.de');
INSERT INTO customers VALUES ('BERGS', 'Berglunds snabbköp', 'Christina Berglund', 'Sweden', 'christina@berglunds.se');
INSERT INTO customers VALUES ('CHOPS', 'Chop-suey Chinese', 'Yang Wang', 'Switzerland', 'yang@chopsuey.ch');

INSERT INTO products VALUES (1, 'Chai', 'Beverages', 18.00);
INSERT INTO products VALUES (2, 'Chang', 'Beverages', 19.00);
INSERT INTO products VALUES (3, 'Aniseed Syrup', 'Condiments', 10.00);

INSERT INTO orders VALUES (1, 'ALFKI', '2024-07-04', 440.00);
INSERT INTO orders VALUES (2, 'BERGS', '2024-07-05', 1863.40);
INSERT INTO orders VALUES (3, 'CHOPS', '2024-07-06', 100.00);

INSERT INTO order_items VALUES (1, 1, 1, 10, 18.00);
INSERT INTO order_items VALUES (2, 1, 2, 5, 19.00);
INSERT INTO order_items VALUES (3, 2, 3, 20, 10.00);
"""


@pytest.fixture
def northwind_db(tmp_path: Path) -> str:
    """Create a minimal Northwind SQLite database. Returns the file path."""
    db_path = tmp_path / "northwind.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(NORTHWIND_DDL)
    conn.close()
    return str(db_path)


@pytest.fixture
def northwind_snapshot() -> SchemaSnapshot:
    """Pre-built SchemaSnapshot matching the northwind_db fixture."""
    return SchemaSnapshot(
        source_id="test_northwind",
        dialect="sqlite",
        tables=[
            SchemaTable(
                name="customers",
                columns=[
                    SchemaColumn(name="customer_id", data_type="TEXT", is_primary_key=True, nullable=False),
                    SchemaColumn(name="company_name", data_type="TEXT", nullable=False),
                    SchemaColumn(name="contact_name", data_type="TEXT"),
                    SchemaColumn(name="country", data_type="TEXT"),
                    SchemaColumn(name="email", data_type="TEXT", semantic_type="email", is_pii=True),
                ],
                row_count_estimate=3,
            ),
            SchemaTable(
                name="orders",
                columns=[
                    SchemaColumn(name="order_id", data_type="INTEGER", is_primary_key=True, nullable=False),
                    SchemaColumn(name="customer_id", data_type="TEXT", is_foreign_key=True,
                                 foreign_key_ref={"table": "customers", "column": "customer_id"}),
                    SchemaColumn(name="order_date", data_type="TEXT", semantic_type="timestamp"),
                    SchemaColumn(name="total_amount", data_type="REAL", semantic_type="currency",
                                 aliases=["revenue", "sales"]),
                ],
                row_count_estimate=3,
            ),
            SchemaTable(
                name="products",
                columns=[
                    SchemaColumn(name="product_id", data_type="INTEGER", is_primary_key=True, nullable=False),
                    SchemaColumn(name="product_name", data_type="TEXT", nullable=False),
                    SchemaColumn(name="category", data_type="TEXT"),
                    SchemaColumn(name="unit_price", data_type="REAL", semantic_type="currency"),
                ],
                row_count_estimate=3,
            ),
            SchemaTable(
                name="order_items",
                columns=[
                    SchemaColumn(name="item_id", data_type="INTEGER", is_primary_key=True, nullable=False),
                    SchemaColumn(name="order_id", data_type="INTEGER", is_foreign_key=True),
                    SchemaColumn(name="product_id", data_type="INTEGER", is_foreign_key=True),
                    SchemaColumn(name="quantity", data_type="INTEGER"),
                    SchemaColumn(name="unit_price", data_type="REAL", semantic_type="currency"),
                ],
                row_count_estimate=3,
            ),
        ],
        foreign_keys=[
            ForeignKey(from_table="orders", from_column="customer_id", to_table="customers", to_column="customer_id"),
            ForeignKey(from_table="order_items", from_column="order_id", to_table="orders", to_column="order_id"),
            ForeignKey(from_table="order_items", from_column="product_id", to_table="products", to_column="product_id"),
        ],
    )


@pytest.fixture
def default_config() -> AgentConfig:
    """Default AgentConfig for tests."""
    return AgentConfig(
        llm_model="gpt-4o",
        auth_enabled=False,
        otel_enabled=False,
    )


@pytest.fixture
def test_user() -> User:
    return User(
        user_id="test_user_1",
        email="test@example.com",
        display_name="Test User",
        provider="email",
    )


@pytest.fixture
def test_workspace() -> Workspace:
    return Workspace(
        workspace_id="ws_test_1",
        name="Test Analytics",
        owner_id="test_user_1",
    )
