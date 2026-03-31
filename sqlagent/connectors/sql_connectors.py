"""SQL-based connectors: SQLite, PostgreSQL, MySQL, Redshift.

All share the same introspection pattern via SQL queries against
information_schema or equivalent system tables.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd
import structlog

from sqlagent.models import (
    SchemaSnapshot,
    SchemaTable,
    SchemaColumn,
    ForeignKey,
    SampleData,
    ColumnStats,
)

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE
# ═══════════════════════════════════════════════════════════════════════════════


class SQLiteConnector:
    """SQLite connector — fully async via aiosqlite."""

    def __init__(self, source_id: str, db_path: str):
        self._source_id = source_id
        self._db_path = db_path
        self._conn = None

    @property
    def dialect(self) -> str:
        return "sqlite"

    @property
    def source_id(self) -> str:
        return self._source_id

    async def connect(self) -> None:
        import aiosqlite

        self._conn = await aiosqlite.connect(self._db_path)

    async def _ensure_conn(self):
        if self._conn is None:
            await self.connect()

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        await self._ensure_conn()
        try:
            cursor = await self._conn.execute(sql)
            rows = await cursor.fetchmany(max_rows)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            from sqlagent.exceptions import SQLExecutionFailed

            raise SQLExecutionFailed(sql=sql, error=str(e))

    async def introspect(self) -> SchemaSnapshot:
        await self._ensure_conn()
        tables = []
        foreign_keys = []

        # Get all table names
        cursor = await self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = [row[0] for row in await cursor.fetchall()]

        for tname in table_names:
            columns = []
            cursor = await self._conn.execute(f"PRAGMA table_info('{tname}')")
            for row in await cursor.fetchall():
                # row: (cid, name, type, notnull, dflt_value, pk)
                columns.append(
                    SchemaColumn(
                        name=row[1],
                        data_type=row[2] or "TEXT",
                        nullable=not bool(row[3]),
                        is_primary_key=bool(row[5]),
                        default_value=row[4],
                        column_position=row[0],
                    )
                )

            # Row count estimate
            try:
                cursor = await self._conn.execute(f"SELECT COUNT(*) FROM '{tname}'")
                row_count = (await cursor.fetchone())[0]
            except Exception as exc:
                logger.debug("connector.sql.operation_failed", error=str(exc))
                row_count = 0

            # Foreign keys
            cursor = await self._conn.execute(f"PRAGMA foreign_key_list('{tname}')")
            for fk_row in await cursor.fetchall():
                # (id, seq, table, from, to, on_update, on_delete, match)
                ref_table = fk_row[2]
                from_col = fk_row[3]
                to_col = fk_row[4]
                foreign_keys.append(
                    ForeignKey(
                        from_table=tname,
                        from_column=from_col,
                        to_table=ref_table,
                        to_column=to_col,
                    )
                )
                # Mark column as FK
                for col in columns:
                    if col.name == from_col:
                        col.is_foreign_key = True
                        col.foreign_key_ref = {"table": ref_table, "column": to_col}

            tables.append(
                SchemaTable(
                    name=tname,
                    columns=columns,
                    row_count_estimate=row_count,
                )
            )

        return SchemaSnapshot(
            source_id=self._source_id,
            dialect="sqlite",
            tables=tables,
            foreign_keys=foreign_keys,
            introspected_at=datetime.now(timezone.utc),
        )

    async def sample(self, table: str, n: int = 5) -> SampleData:
        await self._ensure_conn()
        cursor = await self._conn.execute(f"SELECT * FROM '{table}' LIMIT {n}")
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        sample_rows = [dict(zip(columns, row)) for row in rows]

        # Compute column stats: distinct values for low-cardinality columns
        col_stats: dict[str, ColumnStats] = {}
        for col in columns:
            try:
                cur2 = await self._conn.execute(f"SELECT COUNT(DISTINCT \"{col}\") FROM '{table}'")
                row = await cur2.fetchone()
                distinct = row[0] if row else 0
                samples: list = []
                if 0 < distinct <= 50:
                    cur3 = await self._conn.execute(
                        f'SELECT DISTINCT "{col}" FROM \'{table}\' WHERE "{col}" IS NOT NULL LIMIT 30'
                    )
                    samples = [r[0] for r in await cur3.fetchall()]
                col_stats[col] = ColumnStats(distinct_count=distinct, sample_values=samples)
            except Exception as exc:
                logger.debug("connector.sql.operation_failed", error=str(exc))

        return SampleData(table=table, sample_rows=sample_rows, column_stats=col_stats)

    async def health_check(self) -> bool:
        try:
            await self._ensure_conn()
            cursor = await self._conn.execute("SELECT 1")
            await cursor.fetchone()
            return True
        except Exception as exc:
            logger.debug("connector.sql.operation_failed", error=str(exc))
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL
# ═══════════════════════════════════════════════════════════════════════════════


class PostgresConnector:
    """PostgreSQL connector — async via asyncpg."""

    def __init__(self, source_id: str, connection_string: str):
        self._source_id = source_id
        self._conn_str = connection_string
        self._pool = None

    @property
    def dialect(self) -> str:
        return "postgresql"

    @property
    def source_id(self) -> str:
        return self._source_id

    async def connect(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self._conn_str, min_size=1, max_size=5)

    async def _ensure_pool(self):
        if self._pool is None:
            await self.connect()

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            try:
                rows = await asyncio.wait_for(conn.fetch(sql), timeout=timeout_s)
                if not rows:
                    return pd.DataFrame()
                columns = list(rows[0].keys())
                data = [dict(r) for r in rows[:max_rows]]
                return pd.DataFrame(data, columns=columns)
            except asyncio.TimeoutError:
                from sqlagent.exceptions import ExecutionTimeout

                raise ExecutionTimeout(f"Query timed out after {timeout_s}s")
            except Exception as e:
                from sqlagent.exceptions import SQLExecutionFailed

                raise SQLExecutionFailed(sql=sql, error=str(e))

    async def introspect(self) -> SchemaSnapshot:
        await self._ensure_pool()
        tables = []
        foreign_keys = []

        async with self._pool.acquire() as conn:
            # Get tables
            rows = await conn.fetch("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            table_names = [r["table_name"] for r in rows]

            for tname in table_names:
                # Columns
                col_rows = await conn.fetch(
                    """
                    SELECT column_name, data_type, is_nullable, column_default, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = $1
                    ORDER BY ordinal_position
                """,
                    tname,
                )

                # PKs
                pk_rows = await conn.fetch(
                    """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = $1 AND tc.constraint_type = 'PRIMARY KEY'
                """,
                    tname,
                )
                pk_cols = {r["column_name"] for r in pk_rows}

                columns = [
                    SchemaColumn(
                        name=r["column_name"],
                        data_type=r["data_type"].upper(),
                        nullable=r["is_nullable"] == "YES",
                        is_primary_key=r["column_name"] in pk_cols,
                        default_value=r["column_default"],
                        column_position=r["ordinal_position"],
                    )
                    for r in col_rows
                ]

                # Row count
                try:
                    count_row = await conn.fetchrow(f'SELECT COUNT(*) AS cnt FROM "{tname}"')
                    row_count = count_row["cnt"]
                except Exception as exc:
                    logger.debug("connector.sql.operation_failed", error=str(exc))
                    row_count = 0

                tables.append(
                    SchemaTable(name=tname, columns=columns, row_count_estimate=row_count)
                )

            # Foreign keys
            fk_rows = await conn.fetch("""
                SELECT kcu.table_name, kcu.column_name,
                       ccu.table_name AS ref_table, ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
            """)
            for r in fk_rows:
                foreign_keys.append(
                    ForeignKey(
                        from_table=r["table_name"],
                        from_column=r["column_name"],
                        to_table=r["ref_table"],
                        to_column=r["ref_column"],
                    )
                )

        return SchemaSnapshot(
            source_id=self._source_id,
            dialect="postgresql",
            tables=tables,
            foreign_keys=foreign_keys,
        )

    async def sample(self, table: str, n: int = 5) -> SampleData:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f'SELECT * FROM "{table}" LIMIT {n}')
            return SampleData(
                table=table,
                sample_rows=[dict(r) for r in rows],
            )

    async def health_check(self) -> bool:
        try:
            await self._ensure_pool()
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:
            logger.debug("connector.sql.operation_failed", error=str(exc))
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# MYSQL
# ═══════════════════════════════════════════════════════════════════════════════


class MySQLConnector:
    """MySQL connector — async via aiomysql."""

    def __init__(self, source_id: str, connection_string: str):
        self._source_id = source_id
        self._conn_str = connection_string
        self._pool = None

    @property
    def dialect(self) -> str:
        return "mysql"

    @property
    def source_id(self) -> str:
        return self._source_id

    async def connect(self) -> None:
        # Parse mysql://user:pass@host:port/dbname
        from urllib.parse import urlparse

        parsed = urlparse(self._conn_str)
        import aiomysql

        self._pool = await aiomysql.create_pool(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            db=parsed.path.lstrip("/"),
            minsize=1,
            maxsize=5,
        )

    async def _ensure_pool(self):
        if self._pool is None:
            await self.connect()

    async def execute(
        self, sql: str, timeout_s: float = 30.0, max_rows: int = 10_000
    ) -> pd.DataFrame:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql)
                rows = await cur.fetchmany(max_rows)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return pd.DataFrame(rows, columns=columns)

    async def introspect(self) -> SchemaSnapshot:
        await self._ensure_pool()
        tables = []
        foreign_keys = []

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT TABLE_NAME, TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE'
                """)
                table_info = await cur.fetchall()

                for tname, row_count in table_info:
                    await cur.execute(
                        """
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT,
                               ORDINAL_POSITION, COLUMN_KEY
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
                        ORDER BY ORDINAL_POSITION
                    """,
                        (tname,),
                    )
                    col_rows = await cur.fetchall()

                    columns = [
                        SchemaColumn(
                            name=r[0],
                            data_type=r[1].upper(),
                            nullable=r[2] == "YES",
                            is_primary_key=r[5] == "PRI",
                            default_value=r[3],
                            column_position=r[4],
                        )
                        for r in col_rows
                    ]
                    tables.append(
                        SchemaTable(name=tname, columns=columns, row_count_estimate=row_count or 0)
                    )

                # FKs
                await cur.execute("""
                    SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
                for r in await cur.fetchall():
                    foreign_keys.append(
                        ForeignKey(
                            from_table=r[0],
                            from_column=r[1],
                            to_table=r[2],
                            to_column=r[3],
                        )
                    )

        return SchemaSnapshot(
            source_id=self._source_id,
            dialect="mysql",
            tables=tables,
            foreign_keys=foreign_keys,
        )

    async def sample(self, table: str, n: int = 5) -> SampleData:
        df = await self.execute(f"SELECT * FROM `{table}` LIMIT {n}")
        return SampleData(table=table, sample_rows=df.to_dict("records"))

    async def health_check(self) -> bool:
        try:
            await self._ensure_pool()
            await self.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.debug("connector.sql.operation_failed", error=str(exc))
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# REDSHIFT (extends PostgreSQL — same wire protocol)
# ═══════════════════════════════════════════════════════════════════════════════


class RedshiftConnector(PostgresConnector):
    """Redshift connector — inherits from PostgreSQL, overrides introspection."""

    @property
    def dialect(self) -> str:
        return "redshift"

    async def introspect(self) -> SchemaSnapshot:
        # Redshift uses SVV_ views for faster introspection
        snap = await super().introspect()
        snap = SchemaSnapshot(
            source_id=self._source_id,
            dialect="redshift",
            tables=snap.tables,
            foreign_keys=snap.foreign_keys,
        )
        return snap
