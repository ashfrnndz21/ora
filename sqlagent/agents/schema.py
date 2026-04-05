"""Schema Agent — pure database functions, no LLM.

Provides factual database information to Ora and Semantic Agent.
Searches values, checks joins, verifies existence.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class SchemaAgent:
    """Pure DB operations — search, verify, introspect."""

    def __init__(self, connectors: dict):
        self._connectors = connectors

    async def search(self, column: str, term: str, table_hint: str = "") -> list[dict]:
        """Search for a value across all tables."""
        import pandas as _pd
        results = []
        for sid, conn in self._connectors.items():
            try:
                snap = await conn.introspect()
                for tbl in snap.tables:
                    if table_hint and table_hint.lower() not in tbl.name.lower():
                        continue
                    for col in tbl.columns:
                        if col.name.lower() == column.lower():
                            try:
                                res = await conn.execute(
                                    f"SELECT DISTINCT \"{col.name}\" FROM \"{tbl.name}\" "
                                    f"WHERE LOWER(\"{col.name}\") LIKE LOWER('%{term}%') LIMIT 10"
                                )
                                if isinstance(res, _pd.DataFrame) and not res.empty:
                                    for v in res.iloc[:, 0].dropna().tolist():
                                        results.append({
                                            "value": str(v), "table": tbl.name,
                                            "column": col.name, "source_id": sid,
                                        })
                            except Exception:
                                pass
            except Exception:
                pass
        return results

    async def search_all_text_columns(self, term: str) -> list[dict]:
        """Search for a term across ALL text columns in ALL tables."""
        import pandas as _pd
        results = []
        for sid, conn in self._connectors.items():
            try:
                snap = await conn.introspect()
                for tbl in snap.tables:
                    for col in tbl.columns:
                        dt = (col.data_type or "").lower()
                        if any(t in dt for t in ("varchar", "text", "string", "char")):
                            try:
                                res = await conn.execute(
                                    f"SELECT DISTINCT \"{col.name}\" FROM \"{tbl.name}\" "
                                    f"WHERE LOWER(\"{col.name}\") LIKE LOWER('%{term}%') LIMIT 5"
                                )
                                if isinstance(res, _pd.DataFrame) and not res.empty:
                                    for v in res.iloc[:, 0].dropna().tolist():
                                        results.append({
                                            "value": str(v), "table": tbl.name,
                                            "column": col.name, "source_id": sid,
                                        })
                            except Exception:
                                pass
            except Exception:
                pass
        return results

    async def check_join(self, table_a: str, table_b: str) -> dict:
        """Check if two tables can be joined and on what columns."""
        cols_a, cols_b = set(), set()
        for sid, conn in self._connectors.items():
            try:
                snap = await conn.introspect()
                for tbl in snap.tables:
                    if tbl.name.lower() == table_a.lower():
                        cols_a = {c.name.lower() for c in tbl.columns}
                    if tbl.name.lower() == table_b.lower():
                        cols_b = {c.name.lower() for c in tbl.columns}
            except Exception:
                pass
        shared = cols_a & cols_b
        if shared:
            preferred = ["country", "iso_code", "region", "id", "customer_id", "industry", "_year", "year"]
            for p in preferred:
                if p in shared:
                    return {"joinable": True, "join_columns": [p], "all_shared": list(shared)}
            return {"joinable": True, "join_columns": [next(iter(shared))], "all_shared": list(shared)}
        return {"joinable": False, "join_columns": [], "all_shared": []}

    async def verify_value(self, table: str, column: str, value: str) -> bool:
        """Verify a value exists in a specific table.column."""
        import pandas as _pd
        for sid, conn in self._connectors.items():
            try:
                res = await conn.execute(
                    f"SELECT 1 FROM \"{table}\" WHERE \"{column}\" = '{value}' LIMIT 1"
                )
                if isinstance(res, _pd.DataFrame) and not res.empty:
                    return True
            except Exception:
                pass
        return False

    async def get_relevant_tables(self, query: str, all_tables: list, selector) -> tuple:
        """Prune tables relevant to the query."""
        if selector:
            try:
                pruned = await selector.prune(query=query, tables=all_tables, soul_context="")
                if pruned:
                    return pruned, self._build_schema_dict(pruned)
            except Exception:
                pass
        return all_tables, self._build_schema_dict(all_tables)

    async def get_all_tables(self) -> list:
        """Get all tables from all connectors."""
        all_tables = []
        enriched = getattr(self, "_enriched_snaps", {})
        for sid, conn in self._connectors.items():
            try:
                snap = enriched.get(sid) or await conn.introspect()
                all_tables.extend(snap.tables)
            except Exception:
                pass
        return all_tables

    async def register_cross_tables(self, primary_source_id: str):
        """Register tables from other connectors into the primary connection (for JOINs)."""
        conn = self._connectors.get(primary_source_id)
        if not conn or not hasattr(conn, '_conn') or conn._conn is None:
            return
        for other_sid, other_conn in self._connectors.items():
            if other_sid != primary_source_id and hasattr(other_conn, '_conn') and other_conn._conn:
                try:
                    snap = await other_conn.introspect()
                    for tbl in snap.tables:
                        try:
                            conn._conn.execute(f'SELECT 1 FROM "{tbl.name}" LIMIT 0')
                        except Exception:
                            try:
                                df = other_conn._conn.execute(f'SELECT * FROM "{tbl.name}"').fetchdf()
                                conn._conn.register(tbl.name, df)
                            except Exception:
                                pass
                except Exception:
                    pass

    @staticmethod
    def _build_schema_dict(tables) -> dict:
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": [
                        {
                            "name": c.name, "data_type": c.data_type or "",
                            "is_pk": getattr(c, "is_primary_key", False),
                            "is_fk": getattr(c, "is_foreign_key", False),
                            "description": getattr(c, "description", ""),
                            "examples": getattr(c, "examples", []) or [],
                        }
                        for c in t.columns
                    ],
                }
                for t in tables
            ]
        }
