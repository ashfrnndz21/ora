"""Airbyte connector adapter — bridges sqlagent to Airbyte's 300+ source connectors.

Supports two modes:
1. Managed Airbyte: Point at an Airbyte instance, read synced data from its destination
2. Embedded Airbyte: Use PyAirbyte (airbyte lib) to run source connectors in-process

Data lands in DuckDB for SQL querying — same pattern as all other connectors.
"""

from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger()


class AirbyteConnector:
    """Adapter that reads data from Airbyte-synced sources.

    Mode 1 — Managed Airbyte instance:
        conn = AirbyteConnector(
            source_id="crm",
            mode="managed",
            airbyte_url="http://localhost:8000",
            connection_id="abc-123",
            destination_db="postgresql://localhost/airbyte_raw",
        )

    Mode 2 — Embedded PyAirbyte (in-process):
        conn = AirbyteConnector(
            source_id="crm",
            mode="embedded",
            source_name="source-hubspot",
            source_config={"credentials": {"api_key": "..."}},
            streams=["contacts", "companies", "deals"],
        )
    """

    def __init__(
        self,
        source_id: str,
        mode: str = "embedded",  # "managed" | "embedded"
        # Managed mode
        airbyte_url: str = "",
        connection_id: str = "",
        destination_db: str = "",
        # Embedded mode
        source_name: str = "",
        source_config: dict | None = None,
        streams: list[str] | None = None,
    ):
        self._source_id = source_id
        self._mode = mode
        self._airbyte_url = airbyte_url
        self._connection_id = connection_id
        self._destination_db = destination_db
        self._source_name = source_name
        self._source_config = source_config or {}
        self._streams = streams or []
        self._conn = None  # DuckDB
        self._tables: dict[str, pd.DataFrame] = {}

    @property
    def dialect(self) -> str:
        return "airbyte"

    @property
    def source_id(self) -> str:
        return self._source_id

    async def connect(self) -> None:
        """Connect and sync data from Airbyte source."""
        import duckdb
        self._conn = duckdb.connect(":memory:")

        if self._mode == "embedded":
            await self._connect_embedded()
        elif self._mode == "managed":
            await self._connect_managed()

    async def _connect_embedded(self) -> None:
        """Use PyAirbyte to run source connector in-process."""
        try:
            import airbyte as ab

            source = ab.get_source(
                self._source_name,
                config=self._source_config,
            )

            # Select streams
            if self._streams:
                source.select_streams(self._streams)
            else:
                source.select_all_streams()

            # Read into cache (DuckDB-backed by default)
            cache = ab.get_default_cache()
            result = source.read(cache)

            # Load each stream into our DuckDB
            for stream_name in result.streams.keys():
                try:
                    df = result[stream_name].to_pandas()
                    if not df.empty:
                        table_name = stream_name.replace("-", "_")
                        self._conn.register(table_name, df)
                        self._tables[table_name] = df
                        logger.info("airbyte.stream_loaded", stream=stream_name, rows=len(df))
                except Exception as exc:
                    logger.warning("airbyte.stream_failed", stream=stream_name, error=str(exc))

        except ImportError:
            raise ImportError(
                "PyAirbyte not installed. Run: pip install airbyte"
            )

    async def _connect_managed(self) -> None:
        """Read synced data from a managed Airbyte instance's destination."""
        if not self._destination_db:
            raise ValueError("destination_db is required for managed Airbyte mode")

        # Connect to the Airbyte destination database and read synced tables
        from sqlagent.connectors import ConnectorRegistry
        dest_conn = ConnectorRegistry.from_url(f"{self._source_id}_dest", self._destination_db)
        await dest_conn.connect()

        snap = await dest_conn.introspect()
        for table in snap.tables:
            try:
                df = await dest_conn.execute(f'SELECT * FROM "{table.name}" LIMIT 50000')
                if not df.empty:
                    self._conn.register(table.name, df)
                    self._tables[table.name] = df
                    logger.info("airbyte.managed_loaded", table=table.name, rows=len(df))
            except Exception as exc:
                logger.warning("airbyte.managed_failed", table=table.name, error=str(exc))

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        if self._conn is None:
            await self.connect()
        try:
            result = self._conn.execute(sql)
            return result.fetchdf().head(max_rows)
        except Exception as e:
            from sqlagent.exceptions import SQLExecutionFailed
            raise SQLExecutionFailed(sql=sql, error=str(e))

    async def introspect(self):
        if self._conn is None:
            await self.connect()

        from sqlagent.models import SchemaSnapshot, SchemaTable, SchemaColumn

        tables = []
        for table_name, df in self._tables.items():
            columns = [
                SchemaColumn(
                    name=col, data_type=str(df[col].dtype),
                    examples=[str(v) for v in df[col].dropna().unique()[:6]],
                )
                for col in df.columns
            ]
            tables.append(SchemaTable(
                name=table_name, columns=columns,
                row_count_estimate=len(df),
            ))
        return SchemaSnapshot(tables=tables)

    async def sample(self, table: str, n: int = 5):
        from sqlagent.models import SampleData
        if table in self._tables:
            df = self._tables[table].head(n)
            return SampleData(table=table, rows=df.to_dict("records"), columns=list(df.columns))
        return SampleData(table=table, rows=[], columns=[])

    async def health_check(self) -> bool:
        return self._conn is not None and len(self._tables) > 0
