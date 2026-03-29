"""Test connectors — SQLite roundtrip (real DB, no mocks)."""

import pytest
from sqlagent.connectors import ConnectorRegistry
from sqlagent.connectors.sql_connectors import SQLiteConnector
from sqlagent.connectors.file_connector import FileConnector, DuckDBConnector


@pytest.mark.asyncio
async def test_sqlite_introspect(northwind_db):
    conn = SQLiteConnector(source_id="test", db_path=northwind_db)
    snap = await conn.introspect()
    assert snap.dialect == "sqlite"
    assert snap.table_count == 4
    assert snap.column_count == 18
    assert len(snap.foreign_keys) == 3
    # Check customers table
    customers = snap.get_table("customers")
    assert customers is not None
    assert customers.row_count_estimate == 3
    pk_cols = [c for c in customers.columns if c.is_primary_key]
    assert len(pk_cols) == 1
    assert pk_cols[0].name == "customer_id"


@pytest.mark.asyncio
async def test_sqlite_execute(northwind_db):
    conn = SQLiteConnector(source_id="test", db_path=northwind_db)
    df = await conn.execute("SELECT customer_id, company_name FROM customers ORDER BY customer_id")
    assert len(df) == 3
    assert list(df.columns) == ["customer_id", "company_name"]
    assert df.iloc[0]["customer_id"] == "ALFKI"


@pytest.mark.asyncio
async def test_sqlite_sample(northwind_db):
    conn = SQLiteConnector(source_id="test", db_path=northwind_db)
    sample = await conn.sample("orders", n=2)
    assert sample.table == "orders"
    assert len(sample.sample_rows) == 2


@pytest.mark.asyncio
async def test_sqlite_health_check(northwind_db):
    conn = SQLiteConnector(source_id="test", db_path=northwind_db)
    assert await conn.health_check() is True


@pytest.mark.asyncio
async def test_sqlite_execute_error(northwind_db):
    conn = SQLiteConnector(source_id="test", db_path=northwind_db)
    from sqlagent.exceptions import SQLExecutionFailed
    with pytest.raises(SQLExecutionFailed):
        await conn.execute("SELECT nonexistent FROM customers")


@pytest.mark.asyncio
async def test_registry_sqlite(northwind_db):
    conn = ConnectorRegistry.from_url("test", f"sqlite:///{northwind_db}")
    assert isinstance(conn, SQLiteConnector)
    assert conn.dialect == "sqlite"


@pytest.mark.asyncio
async def test_registry_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")
    conn = ConnectorRegistry.from_url("test", str(csv_path))
    assert isinstance(conn, FileConnector)


def test_registry_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        ConnectorRegistry.from_url("test", "ftp://somewhere/data")


@pytest.mark.asyncio
async def test_duckdb_connector():
    conn = DuckDBConnector(source_id="test")
    await conn.connect()
    await conn.execute("CREATE TABLE t1 (id INT, name TEXT)")
    await conn.execute("INSERT INTO t1 VALUES (1, 'Alice'), (2, 'Bob')")
    df = await conn.execute("SELECT * FROM t1 ORDER BY id")
    assert len(df) == 2
    snap = await conn.introspect()
    assert snap.table_count == 1
    assert snap.tables[0].name == "t1"


@pytest.mark.asyncio
async def test_file_connector_csv(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("store_id,revenue\nST-001,1000\nST-002,2000\nST-003,1500\n")
    conn = FileConnector(source_id="test", file_path=str(csv_path))
    await conn.connect()
    snap = await conn.introspect()
    assert snap.table_count == 1
    assert snap.tables[0].name == "sales"
    df = await conn.execute("SELECT * FROM sales ORDER BY revenue DESC")
    assert len(df) == 3
    assert df.iloc[0]["revenue"] == 2000
