"""SQL generators + ensemble — all generation strategies in one file.

Generators produce SQL candidates from NL queries. The ensemble runs
them in parallel via asyncio.gather and selects the best candidate
using CHASE-SQL pairwise LLM comparison.

Every generator is REAL — it calls the actual LLM to generate SQL.
No mocks, no templates, no hardcoded responses.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol, runtime_checkable

import structlog

from sqlagent.models import Candidate
from sqlagent.schema import MSchemaSerializer

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Generator(Protocol):
    @property
    def generator_id(self) -> str: ...

    async def generate(
        self,
        nl_query: str,
        schema: dict,
        examples: list[dict] | None = None,
        plan: str = "",
    ) -> Candidate: ...


# ═══════════════════════════════════════════════════════════════════════════════
# FEWSHOT GENERATOR (primary — highest accuracy on BIRD benchmark)
# ═══════════════════════════════════════════════════════════════════════════════


class FewshotGenerator:
    """RAG few-shot generator — retrieves similar examples and uses them as context.

    This is the most accurate generator on the BIRD benchmark.
    """

    def __init__(self, llm: Any):
        self._llm = llm

    @property
    def generator_id(self) -> str:
        return "fewshot"

    async def generate(
        self,
        nl_query: str,
        schema: dict,
        examples: list[dict] | None = None,
        plan: str = "",
        context_notes: list[str] | None = None,
    ) -> Candidate:
        started = time.monotonic()

        # Build M-Schema context
        from sqlagent.models import SchemaTable, SchemaColumn

        tables = []
        for t in schema.get("tables", []):
            cols = [
                SchemaColumn(
                    name=c["name"],
                    data_type=c.get("data_type", ""),
                    is_primary_key=c.get("is_pk", False),
                    is_foreign_key=c.get("is_fk", False),
                    description=c.get("description", ""),
                    examples=c.get("examples", []),
                )
                for c in t.get("columns", [])
            ]
            tables.append(SchemaTable(name=t["name"], columns=cols))

        schema_text = MSchemaSerializer.serialize(tables)

        # Build few-shot examples
        example_text = ""
        for ex in examples or []:
            example_text += f"\nQuestion: {ex.get('nl', '')}\nSQL: {ex.get('sql', '')}\n"

        # Build workspace-specific learned context (from user corrections)
        # These are MANDATORY — injected before everything else so the LLM never forgets them
        learned_context = ""
        if context_notes:
            learned_context = (
                "\nLEARNED DATA CONTEXT (apply to every query — from user corrections):\n"
                + "\n".join(f"• {note}" for note in context_notes)
                + "\n"
            )

        prompt = (
            f"You are an expert SQL writer. Given the schema below, write SQL to answer the question.\n\n"
            f"Schema:\n{schema_text}\n"
            f"{learned_context}"
            f"\nIMPORTANT RULES:\n"
            f"1. Column values shown in [values:...] are the ACTUAL data values in that column.\n"
            f"   Use them to understand the data domain and write precise, accurate SQL.\n"
            f"2. When a question asks about a specific entity type (e.g. 'countries', 'cities',\n"
            f"   'products', 'customers'), look at the actual column values. Use your knowledge\n"
            f"   to distinguish individual entities from aggregate/bucket groupings, and filter out\n"
            f"   the aggregates. Apply this reasoning generically to whatever aggregates appear\n"
            f"   in THIS dataset's values.\n"
            f"3. If a correction note appears in examples (e.g. '[learn: ...]'), apply that\n"
            f"   lesson directly to the current query.\n"
        )
        if example_text:
            prompt += f"\nSimilar examples:{example_text}\n"
        prompt += (
            f"\nQuestion: {nl_query}\n\n"
            f"Write the SQL query. Think step by step, then output ONLY the SQL.\n"
            f"Reasoning: [your reasoning]\nSQL: [your query]"
        )

        try:
            resp = await self._llm.complete([{"role": "user", "content": prompt}])
            sql = _extract_sql(resp.content)
            latency = int((time.monotonic() - started) * 1000)

            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="fewshot",
                sql=sql,
                reasoning=resp.content.split("SQL:")[0] if "SQL:" in resp.content else "",
                confidence=0.85,
                tokens_used=resp.tokens_input + resp.tokens_output,
                cost_usd=resp.cost_usd,
                latency_ms=latency,
            )
        except Exception as e:
            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="fewshot",
                error=str(e),
                latency_ms=int((time.monotonic() - started) * 1000),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATOR (CoT planning → SQL)
# ═══════════════════════════════════════════════════════════════════════════════


class PlanGenerator:
    """Chain-of-thought planning generator — plans first, then writes SQL."""

    def __init__(self, llm: Any):
        self._llm = llm

    @property
    def generator_id(self) -> str:
        return "plan"

    async def generate(
        self,
        nl_query: str,
        schema: dict,
        examples: list[dict] | None = None,
        plan: str = "",
        context_notes: list[str] | None = None,
    ) -> Candidate:
        started = time.monotonic()

        # Schema representation with column value hints for low-cardinality columns
        schema_text = ""
        for t in schema.get("tables", []):
            col_parts = []
            for c in t.get("columns", []):
                part = c["name"]
                if c.get("examples"):
                    ex = ", ".join(f'"{e}"' for e in c["examples"][:5])
                    part += f"[values:{ex}]"
                col_parts.append(part)
            schema_text += f"  {t['name']}({', '.join(col_parts)})\n"

        learned_context = ""
        if context_notes:
            learned_context = (
                "\nLEARNED DATA CONTEXT (mandatory — from user corrections):\n"
                + "\n".join(f"• {note}" for note in context_notes)
                + "\n"
            )

        prompt = (
            f"You are an expert SQL writer using a plan-then-code approach.\n\n"
            f"Schema:\n{schema_text}\n"
            f"{learned_context}"
            f"Question: {nl_query}\n\n"
            f"Step 1: Plan your approach (which tables, joins, filters, aggregations)\n"
            f"Step 2: Write the SQL\n\n"
            f"Plan: [your plan]\nSQL: [your query]"
        )

        try:
            resp = await self._llm.complete([{"role": "user", "content": prompt}])
            sql = _extract_sql(resp.content)
            latency = int((time.monotonic() - started) * 1000)

            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="plan",
                sql=sql,
                reasoning=resp.content.split("SQL:")[0] if "SQL:" in resp.content else "",
                confidence=0.80,
                tokens_used=resp.tokens_input + resp.tokens_output,
                cost_usd=resp.cost_usd,
                latency_ms=latency,
            )
        except Exception as e:
            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="plan",
                error=str(e),
                latency_ms=int((time.monotonic() - started) * 1000),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# DECOMPOSE GENERATOR (CTE decomposition)
# ═══════════════════════════════════════════════════════════════════════════════


class DecomposeGenerator:
    """CTE decomposition generator — breaks complex queries into CTEs."""

    def __init__(self, llm: Any):
        self._llm = llm

    @property
    def generator_id(self) -> str:
        return "decompose"

    async def generate(
        self,
        nl_query: str,
        schema: dict,
        examples: list[dict] | None = None,
        plan: str = "",
        context_notes: list[str] | None = None,
    ) -> Candidate:
        started = time.monotonic()

        schema_text = ""
        for t in schema.get("tables", []):
            col_parts = []
            for c in t.get("columns", []):
                part = c["name"]
                if c.get("examples"):
                    ex = ", ".join(f'"{e}"' for e in c["examples"][:5])
                    part += f"[values:{ex}]"
                col_parts.append(part)
            schema_text += f"  {t['name']}({', '.join(col_parts)})\n"

        learned_context = ""
        if context_notes:
            learned_context = (
                "\nLEARNED DATA CONTEXT (mandatory — from user corrections):\n"
                + "\n".join(f"• {note}" for note in context_notes)
                + "\n"
            )

        prompt = (
            f"You are an expert SQL writer. Use CTEs (WITH clauses) to decompose the query.\n\n"
            f"Schema:\n{schema_text}\n"
            f"{learned_context}"
            f"Question: {nl_query}\n\n"
            f"Write SQL using WITH clauses for each logical step.\n"
            f"SQL:"
        )

        try:
            resp = await self._llm.complete([{"role": "user", "content": prompt}])
            sql = _extract_sql(resp.content)
            latency = int((time.monotonic() - started) * 1000)

            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="decompose",
                sql=sql,
                confidence=0.75,
                tokens_used=resp.tokens_input + resp.tokens_output,
                cost_usd=resp.cost_usd,
                latency_ms=latency,
            )
        except Exception as e:
            return Candidate(
                candidate_id=str(uuid.uuid4())[:8],
                generator_id="decompose",
                error=str(e),
                latency_ms=int((time.monotonic() - started) * 1000),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE (parallel execution + pairwise selection)
# ═══════════════════════════════════════════════════════════════════════════════


class GeneratorEnsemble:
    """Runs all generators in parallel, selects best candidate.

    Selection: CHASE-SQL pairwise LLM comparison when 2+ candidates succeed.
    """

    def __init__(self, generators: list[Generator], llm: Any):
        self._generators = generators
        self._llm = llm

    async def generate(
        self,
        nl_query: str,
        pruned_schema: dict,
        examples: list[dict] | None = None,
        plan: str = "",
        generators_override: list | None = None,
        context_notes: list[str] | None = None,
    ) -> list[dict]:
        """Run generators in parallel. Returns list of candidate dicts."""
        gens = generators_override if generators_override is not None else self._generators

        async def _run(gen: Generator) -> dict:
            try:
                candidate = await gen.generate(
                    nl_query=nl_query,
                    schema=pruned_schema,
                    examples=examples,
                    plan=plan,
                    context_notes=context_notes,
                )
                return {
                    "candidate_id": candidate.candidate_id,
                    "generator_id": candidate.generator_id,
                    "sql": candidate.sql,
                    "reasoning": candidate.reasoning,
                    "confidence": candidate.confidence,
                    "tokens_used": candidate.tokens_used,
                    "cost_usd": candidate.cost_usd,
                    "latency_ms": candidate.latency_ms,
                    "error": candidate.error,
                    "succeeded": candidate.succeeded,
                }
            except Exception as e:
                return {
                    "generator_id": gen.generator_id,
                    "error": str(e),
                    "succeeded": False,
                }

        results = await asyncio.gather(*[_run(g) for g in gens])
        return list(results)

    async def select(self, candidates: list[dict]) -> tuple[dict | None, str]:
        """Select the best candidate. Returns (winner, reasoning)."""
        successful = [c for c in candidates if c.get("succeeded")]

        if not successful:
            return None, "No candidates succeeded"
        if len(successful) == 1:
            return successful[0], f"Single candidate: {successful[0].get('generator_id')}"

        # Pairwise LLM comparison (CHASE-SQL technique)
        a, b = successful[0], successful[1]
        prompt = (
            f"Compare these two SQL queries. Which is more correct and efficient?\n\n"
            f"Query A ({a.get('generator_id')}):\n{a.get('sql')}\n\n"
            f"Query B ({b.get('generator_id')}):\n{b.get('sql')}\n\n"
            f"Reply with just 'A' or 'B' and a brief reason."
        )

        try:
            resp = await self._llm.complete([{"role": "user", "content": prompt}])
            content = resp.content.strip().upper()
            if content.startswith("A"):
                return a, f"Pairwise: {a.get('generator_id')} won — {resp.content[:100]}"
            elif content.startswith("B"):
                return b, f"Pairwise: {b.get('generator_id')} won — {resp.content[:100]}"
        except Exception as exc:
            logger.debug("generators.operation_failed", error=str(exc))

        # Fallback: highest confidence
        successful.sort(key=lambda c: c.get("confidence", 0), reverse=True)
        winner = successful[0]
        return winner, f"Fallback: highest confidence ({winner.get('generator_id')})"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_sql(text: str) -> str:
    """Extract SQL from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Try to find SQL: marker
    if "SQL:" in text:
        sql = text.split("SQL:")[-1].strip()
    elif "```sql" in text:
        sql = text.split("```sql")[1].split("```")[0].strip()
    elif "```" in text:
        sql = text.split("```")[1].split("```")[0].strip()
    else:
        sql = text

    # Clean up
    sql = sql.strip().rstrip(";") + ";" if sql.strip() else ""
    return sql.rstrip(";")  # Remove trailing semicolon for execution
