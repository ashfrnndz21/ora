"""SQL execution + 3-stage ReFoRCE correction loop.

The executor runs SQL against a connector with policy checks.
When execution fails, the correction loop escalates through 3 stages:
  Stage 1: Error-aware — feed error back to LLM, regenerate
  Stage 2: Schema-aware — re-examine schema, rewrite with correct columns
  Stage 3: DB-confirmed — query actual DB metadata, regenerate
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

from sqlagent.exceptions import SQLExecutionFailed, CorrectionExhausted, PolicyViolation

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# SQL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Result from executing SQL against a database."""
    sql: str = ""
    dataframe: pd.DataFrame | None = None
    row_count: int = 0
    latency_ms: int = 0
    succeeded: bool = False
    error: str = ""


class SQLExecutor:
    """Execute SQL against a connector with timeout and row limit."""

    def __init__(self, connector: Any, policy: Any = None, config: Any = None):
        self._conn = connector
        self._policy = policy
        self._config = config

    async def execute(self, sql: str) -> ExecutionResult:
        started = time.monotonic()

        # Policy check (deterministic, no LLM)
        if self._policy:
            result = self._policy.check(sql, {})
            if not result.passed:
                return ExecutionResult(
                    sql=sql, succeeded=False,
                    error=f"Policy blocked [{result.rule_id}]: {result.reason}",
                )
            if result.modified_sql:
                sql = result.modified_sql

        # Execute
        try:
            timeout = self._config.query_timeout_s if self._config else 30.0
            max_rows = self._config.row_limit if self._config else 10_000
            df = await self._conn.execute(sql, timeout_s=timeout, max_rows=max_rows)
            latency = int((time.monotonic() - started) * 1000)
            return ExecutionResult(
                sql=sql, dataframe=df, row_count=len(df),
                latency_ms=latency, succeeded=True,
            )
        except SQLExecutionFailed as e:
            latency = int((time.monotonic() - started) * 1000)
            return ExecutionResult(
                sql=sql, latency_ms=latency, succeeded=False, error=str(e),
            )
        except Exception as e:
            latency = int((time.monotonic() - started) * 1000)
            return ExecutionResult(
                sql=sql, latency_ms=latency, succeeded=False, error=str(e),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3-STAGE ReFoRCE CORRECTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class CorrectionLoop:
    """3-stage self-correction loop (ReFoRCE algorithm, Snowflake 2025).

    Stage 1: Error-aware — feed the error message back to the LLM
    Stage 2: Schema-aware — show the actual column names from the schema
    Stage 3: DB-confirmed — query the database for DESCRIBE TABLE output

    Each stage lowers confidence by 15%. If all stages fail, returns None.
    """

    def __init__(self, llm: Any, connector: Any, executor: SQLExecutor):
        self._llm = llm
        self._conn = connector
        self._executor = executor

    async def correct(
        self,
        original_sql: str,
        error: str,
        nl_query: str,
        schema_context: str = "",
        max_rounds: int = 3,
    ) -> ExecutionResult | None:
        """Try to fix the SQL through escalating correction stages.

        Returns ExecutionResult on success, None if all stages exhausted.
        """
        current_sql = original_sql
        current_error = error

        for round_num in range(1, max_rounds + 1):
            logger.info("correction.attempt", round=round_num, error=current_error[:100])

            if round_num == 1:
                # Stage 1: Error-aware
                new_sql = await self._error_aware_correct(current_sql, current_error, nl_query)
            elif round_num == 2:
                # Stage 2: Schema-aware
                new_sql = await self._schema_aware_correct(
                    current_sql, current_error, nl_query, schema_context
                )
            else:
                # Stage 3: DB-confirmed
                new_sql = await self._db_confirmed_correct(
                    current_sql, current_error, nl_query
                )

            if not new_sql:
                continue

            # Try executing the corrected SQL
            result = await self._executor.execute(new_sql)
            if result.succeeded:
                logger.info("correction.success", round=round_num, rows=result.row_count)
                return result

            current_sql = new_sql
            current_error = result.error

        logger.warn("correction.exhausted", rounds=max_rounds)
        return None

    async def _error_aware_correct(self, sql: str, error: str, nl_query: str) -> str:
        """Stage 1: Feed the error back, ask LLM to fix."""
        prompt = (
            f"The following SQL failed:\n```sql\n{sql}\n```\n"
            f"Error: {error}\n\n"
            f"Original question: {nl_query}\n\n"
            f"Fix the SQL error. Return ONLY the corrected SQL, no explanation."
        )
        resp = await self._llm.complete([{"role": "user", "content": prompt}])
        return _extract_sql_from_response(resp.content)

    async def _schema_aware_correct(
        self, sql: str, error: str, nl_query: str, schema_context: str
    ) -> str:
        """Stage 2: Show the actual schema, ask LLM to rewrite."""
        prompt = (
            f"The following SQL failed:\n```sql\n{sql}\n```\n"
            f"Error: {error}\n\n"
            f"The actual database schema is:\n{schema_context}\n\n"
            f"Original question: {nl_query}\n\n"
            f"Rewrite using ONLY the exact column names shown above. Return ONLY SQL."
        )
        resp = await self._llm.complete([{"role": "user", "content": prompt}])
        return _extract_sql_from_response(resp.content)

    async def _db_confirmed_correct(self, sql: str, error: str, nl_query: str) -> str:
        """Stage 3: Query the DB for actual column names, then regenerate."""
        # Get real table metadata from the database
        try:
            snap = await self._conn.introspect()
            schema_lines = []
            for table in snap.tables:
                cols = ", ".join(f"{c.name} {c.data_type}" for c in table.columns)
                schema_lines.append(f"  {table.name}({cols})")
            confirmed_schema = "\n".join(schema_lines)
        except Exception as exc:
            logger.debug("execution.operation_failed", error=str(exc))
            confirmed_schema = "(could not introspect)"

        prompt = (
            f"SQL failed after 2 attempts:\n```sql\n{sql}\n```\n"
            f"Error: {error}\n\n"
            f"CONFIRMED database schema (from live introspection):\n{confirmed_schema}\n\n"
            f"Original question: {nl_query}\n\n"
            f"Write correct SQL using ONLY these confirmed columns. Return ONLY SQL."
        )
        resp = await self._llm.complete([{"role": "user", "content": prompt}])
        return _extract_sql_from_response(resp.content)


def _extract_sql_from_response(text: str) -> str:
    """Extract SQL from LLM response."""
    text = text.strip()
    if "```sql" in text:
        return text.split("```sql")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text
