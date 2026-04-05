"""SQL Agent — translates structured query spec into SQL.

Never decides what to query. Receives a COMPLETE spec from Ora
and writes syntactically correct SQL.
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from sqlagent.agents.protocol import AgentRequest, AgentResponse
from sqlagent.graph.ora_react import AgentTrace

logger = structlog.get_logger()


class SQLAgent:
    """Generates SQL from a structured specification."""

    def __init__(self, ensemble: Any, policy: Any = None):
        self._ensemble = ensemble
        self._policy = policy

    async def handle(self, request: AgentRequest, trace: AgentTrace) -> AgentResponse:
        """Generate SQL from Ora's query spec."""
        start = time.monotonic()

        query_spec = request.context.get("query_spec", "")
        pruned_schema = request.context.get("pruned_schema", {})
        examples = request.context.get("examples", [])
        context_notes = request.context.get("context_notes", [])

        try:
            candidates = await self._ensemble.generate(
                nl_query=query_spec,
                pruned_schema=pruned_schema,
                examples=examples,
                context_notes=context_notes,
            )

            if len(candidates) > 1 and hasattr(self._ensemble, 'select'):
                winner, _ = await self._ensemble.select(candidates)
            elif candidates:
                winner = candidates[0]
            else:
                winner = None

            sql = winner.get("sql", "") if winner else ""
            generator = winner.get("generator_id", "") if winner else ""
            tokens = sum(c.get("tokens_used", 0) for c in candidates)

        except Exception as err:
            trace.record("sql_agent", "Generation failed", status="failed",
                         output=str(err)[:100],
                         latency_ms=int((time.monotonic() - start) * 1000))
            return AgentResponse(
                from_agent="sql_agent", status="failed",
                reasoning=str(err), confidence=0.0,
            )

        # Clean SQL — strip markdown, comments, notes
        sql = self._clean_sql(sql)

        latency = int((time.monotonic() - start) * 1000)

        if not sql:
            trace.record("sql_agent", "No SQL generated", status="failed",
                         latency_ms=latency)
            return AgentResponse(
                from_agent="sql_agent", status="failed",
                reasoning="No SQL output from generators", confidence=0.0,
            )

        trace.record("sql_agent", "Generated SQL",
                     output=sql[:120],
                     reasoning=f"Strategy: {generator}",
                     latency_ms=latency, tokens=tokens)

        return AgentResponse(
            from_agent="sql_agent", status="completed",
            result={"sql": sql, "generator": generator, "tokens": tokens, "candidates": candidates},
            confidence=0.8,
            reasoning=f"Generated via {generator}",
        )

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """Strip markdown, comments, and non-SQL content."""
        if "```" in sql:
            match = re.search(r'```(?:sql)?\s*(.*?)```', sql, re.DOTALL)
            if match:
                sql = match.group(1).strip()
            else:
                sql = sql.replace("```sql", "").replace("```", "").strip()
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.strip().startswith("--")).strip()
        if "**" in sql:
            sql = sql.split("**")[0].strip()
        return sql.rstrip(";").strip()
