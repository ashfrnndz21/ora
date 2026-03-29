"""Database connectors — protocol + registry.

All connectors implement the same protocol:
  connect(), execute(), introspect(), sample(), health_check()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
import pandas as pd

from sqlagent.models import SchemaSnapshot, SampleData


@runtime_checkable
class Connector(Protocol):
    """Protocol all database connectors must implement."""

    @property
    def dialect(self) -> str: ...

    @property
    def source_id(self) -> str: ...

    async def connect(self) -> None: ...

    async def execute(self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000) -> pd.DataFrame: ...

    async def introspect(self) -> SchemaSnapshot: ...

    async def sample(self, table: str, n: int = 5) -> SampleData: ...

    async def health_check(self) -> bool: ...


class ConnectorRegistry:
    """Factory that creates the right connector from a connection URL or config."""

    @staticmethod
    def from_url(source_id: str, url: str) -> Connector:
        """Create a connector from a connection URL string.

        Supports:
          sqlite:///path/to/db.db
          postgresql://user:pass@host/db
          mysql://user:pass@host:3306/db
          snowflake://user:pass@account/db/schema
          bigquery://project/dataset
          redshift+psycopg2://...
          *.csv, *.xlsx, *.parquet, *.json (file paths → DuckDB)
        """
        url_lower = url.lower()

        if url_lower.startswith("sqlite"):
            from sqlagent.connectors.sql_connectors import SQLiteConnector
            path = url.replace("sqlite:///", "").replace("sqlite://", "")
            return SQLiteConnector(source_id=source_id, db_path=path)

        if url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
            from sqlagent.connectors.sql_connectors import PostgresConnector
            return PostgresConnector(source_id=source_id, connection_string=url)

        if url_lower.startswith("mysql"):
            from sqlagent.connectors.sql_connectors import MySQLConnector
            return MySQLConnector(source_id=source_id, connection_string=url)

        if url_lower.startswith("redshift"):
            from sqlagent.connectors.sql_connectors import RedshiftConnector
            return RedshiftConnector(source_id=source_id, connection_string=url)

        if url_lower.startswith("snowflake"):
            from sqlagent.connectors.warehouse_connectors import SnowflakeConnector
            return SnowflakeConnector(source_id=source_id, connection_string=url)

        if url_lower.startswith("bigquery"):
            from sqlagent.connectors.warehouse_connectors import BigQueryConnector
            return BigQueryConnector(source_id=source_id, connection_string=url)

        # File-based sources → DuckDB
        if any(url_lower.endswith(ext) for ext in (".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv", ".pptx", ".ppt")):
            from sqlagent.connectors.file_connector import FileConnector
            return FileConnector(source_id=source_id, file_path=url)

        if url_lower.startswith("duckdb"):
            from sqlagent.connectors.file_connector import DuckDBConnector
            path = url.replace("duckdb:///", "").replace("duckdb://", "")
            return DuckDBConnector(source_id=source_id, db_path=path)

        raise ValueError(f"Unknown connection URL scheme: {url}")
