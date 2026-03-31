"""Warehouse connectors: Snowflake, BigQuery.

These use sync SDKs wrapped in asyncio.to_thread for async compatibility.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import structlog

from sqlagent.models import SchemaSnapshot, SchemaTable, SchemaColumn, SampleData

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# SNOWFLAKE
# ═══════════════════════════════════════════════════════════════════════════════


class SnowflakeConnector:
    """Snowflake connector — sync SDK wrapped in asyncio.to_thread."""

    def __init__(self, source_id: str, connection_string: str):
        self._source_id = source_id
        self._conn_str = connection_string
        self._conn = None

    @property
    def dialect(self) -> str:
        return "snowflake"

    @property
    def source_id(self) -> str:
        return self._source_id

    def _parse_url(self) -> dict:
        """Parse snowflake://user:pass@account/database/schema?warehouse=WH"""
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self._conn_str)
        parts = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        return {
            "user": parsed.username or "",
            "password": parsed.password or "",
            "account": parsed.hostname or "",
            "database": parts[0] if parts else "",
            "schema": parts[1] if len(parts) > 1 else "PUBLIC",
            "warehouse": qs.get("warehouse", [""])[0],
        }

    async def connect(self) -> None:
        params = self._parse_url()
        import snowflake.connector

        self._conn = await asyncio.to_thread(snowflake.connector.connect, **params)

    async def _ensure_conn(self):
        if self._conn is None:
            await self.connect()

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        await self._ensure_conn()

        def _run():
            cursor = self._conn.cursor()
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
            return df.head(max_rows)

        return await asyncio.to_thread(_run)

    async def introspect(self) -> SchemaSnapshot:
        await self._ensure_conn()
        params = self._parse_url()

        def _introspect():
            cursor = self._conn.cursor()
            cursor.execute(f"""
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = '{params["schema"]}'
                AND TABLE_TYPE = 'BASE TABLE'
            """)
            table_names = [r[0] for r in cursor.fetchall()]

            tables = []
            for tname in table_names:
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, ORDINAL_POSITION
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{params["schema"]}' AND TABLE_NAME = '{tname}'
                    ORDER BY ORDINAL_POSITION
                """)
                columns = [
                    SchemaColumn(
                        name=r[0],
                        data_type=r[1],
                        nullable=r[2] == "YES",
                        default_value=r[3],
                        column_position=r[4],
                    )
                    for r in cursor.fetchall()
                ]
                tables.append(SchemaTable(name=tname, columns=columns))

            return SchemaSnapshot(
                source_id=self._source_id,
                dialect="snowflake",
                tables=tables,
            )

        return await asyncio.to_thread(_introspect)

    async def sample(self, table: str, n: int = 5) -> SampleData:
        df = await self.execute(f"SELECT * FROM {table} LIMIT {n}")
        return SampleData(table=table, sample_rows=df.to_dict("records"))

    async def health_check(self) -> bool:
        try:
            await self.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.debug("connector.warehouse.operation_failed", error=str(exc))
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# BIGQUERY
# ═══════════════════════════════════════════════════════════════════════════════


class BigQueryConnector:
    """BigQuery connector — google-cloud-bigquery wrapped in asyncio.to_thread."""

    def __init__(self, source_id: str, connection_string: str):
        self._source_id = source_id
        self._conn_str = connection_string
        self._client = None
        self._project = ""
        self._dataset = ""

    @property
    def dialect(self) -> str:
        return "bigquery"

    @property
    def source_id(self) -> str:
        return self._source_id

    def _parse_url(self):
        """Parse bigquery://project/dataset"""
        parts = self._conn_str.replace("bigquery://", "").strip("/").split("/")
        self._project = parts[0] if parts else ""
        self._dataset = parts[1] if len(parts) > 1 else ""

    async def connect(self) -> None:
        self._parse_url()
        from google.cloud import bigquery

        self._client = await asyncio.to_thread(bigquery.Client, project=self._project)

    async def _ensure_client(self):
        if self._client is None:
            await self.connect()

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        await self._ensure_client()

        def _run():
            job = self._client.query(sql)
            return job.to_dataframe().head(max_rows)

        return await asyncio.to_thread(_run)

    async def introspect(self) -> SchemaSnapshot:
        await self._ensure_client()

        def _introspect():
            tables = []
            dataset_ref = f"{self._project}.{self._dataset}"
            for table_item in self._client.list_tables(dataset_ref):
                table = self._client.get_table(table_item.reference)
                columns = [
                    SchemaColumn(
                        name=field.name,
                        data_type=field.field_type,
                        nullable=field.mode != "REQUIRED",
                        description=field.description or "",
                    )
                    for field in table.schema
                ]
                tables.append(
                    SchemaTable(
                        name=table.table_id,
                        columns=columns,
                        row_count_estimate=table.num_rows or 0,
                    )
                )
            return SchemaSnapshot(
                source_id=self._source_id,
                dialect="bigquery",
                tables=tables,
            )

        return await asyncio.to_thread(_introspect)

    async def sample(self, table: str, n: int = 5) -> SampleData:
        df = await self.execute(f"SELECT * FROM `{self._dataset}.{table}` LIMIT {n}")
        return SampleData(table=table, sample_rows=df.to_dict("records"))

    async def health_check(self) -> bool:
        try:
            await self.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.debug("connector.warehouse.operation_failed", error=str(exc))
            return False
