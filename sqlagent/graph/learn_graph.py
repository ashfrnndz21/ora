"""Learn Agent — LangGraph orchestration for SQL correction and lesson extraction.

This graph runs when a user marks a query result as wrong.
It is a SEPARATE graph from the query graph, compiled once alongside it.

Node pipeline:
  understand_correction → analyze_schema → rewrite_sql → execute_corrected → extract_lesson

Each node is decorated with @traced_node so every correction produces a full
OTel span tree in Jaeger/Langfuse, identical to query spans.

The graph returns the final LearnState containing:
  - rewritten_sql      (the corrected query)
  - domain_insight     (what the agent understood about the data domain)
  - extracted_lesson   (the general rule injected into all future queries)
  - rows/columns       (execution preview so the user can review before saving)
  - learn_trace_events (per-node timing for the trace timeline in the UI)

Saving to the vector store and persisting the LessonRecord happens AFTER
the user reviews and approves — via POST /train/sql in server.py.
"""

from __future__ import annotations

import json
import time
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from sqlagent.telemetry import traced_node


# ═══════════════════════════════════════════════════════════════════════════════
# LEARN STATE
# ═══════════════════════════════════════════════════════════════════════════════


class LearnState(TypedDict, total=False):
    """State flowing through the Learn Agent correction graph."""

    # ── Input ─────────────────────────────────────────────────────────────────
    nl_query: str
    original_sql: str
    failure_type: str  # wrong_tables | wrong_columns | wrong_filter | wrong_logic
    failed_node: str  # prune | retrieve | plan | generate | execute | correct
    correction_note: str  # user's plain-English description of what was wrong
    trace_events: list[dict]  # full trace_events from the original query run
    workspace_id: str
    user_id: str
    source_id: str  # which source to execute corrected SQL against

    # ── Analysis ──────────────────────────────────────────────────────────────
    failed_stage: str  # mapped: schema | retrieval | planning | generation | filtering
    schema_context: str  # serialized schema passed to the rewrite LLM

    # ── Rewrite ───────────────────────────────────────────────────────────────
    rewritten_sql: str
    what_changed: str  # one sentence: technical explanation of the fix
    domain_insight: str  # what the agent discovered about the DATA DOMAIN

    # ── Execution ─────────────────────────────────────────────────────────────
    rows: list[dict]
    columns: list[str]
    row_count: int
    exec_error: str

    # ── Lesson ────────────────────────────────────────────────────────────────
    extracted_lesson: str  # ONE sentence general rule — injected into all future queries

    # ── Budget ────────────────────────────────────────────────────────────────
    tokens_used: int
    cost_usd: float

    # ── Trace ─────────────────────────────────────────────────────────────────
    learn_trace_events: list[dict]  # per-node timing, status, summary


# ═══════════════════════════════════════════════════════════════════════════════
# NODE FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

_FAILURE_TYPE_TO_STAGE = {
    "wrong_tables": "schema",
    "wrong_columns": "schema",
    "bad_example": "retrieval",
    "wrong_plan": "planning",
    "wrong_logic": "generation",
    "wrong_filter": "filtering",
    "wrong_aggregation": "generation",
}

_NODE_TO_STAGE = {
    "prune": "schema",
    "retrieve": "retrieval",
    "plan": "planning",
    "generate": "generation",
    "execute": "generation",
    "correct": "generation",
}


def make_understand_correction_node(services: Any):
    """Map failure_type / failed_node / trace_events → a canonical failed_stage."""

    @traced_node("learn.understand")
    async def understand_correction_node(state: LearnState) -> dict:
        started = time.monotonic()

        failed_stage = ""

        # Priority 1: explicit failed_node label
        if state.get("failed_node"):
            failed_stage = _NODE_TO_STAGE.get(state["failed_node"], "generation")

        # Priority 2: user-selected failure_type
        elif state.get("failure_type"):
            failed_stage = _FAILURE_TYPE_TO_STAGE.get(state["failure_type"], "generation")

        # Priority 3: scan trace_events for a failed node
        else:
            for evt in reversed(state.get("trace_events", [])):
                if evt.get("status") == "failed":
                    failed_stage = _NODE_TO_STAGE.get(evt.get("node", ""), "generation")
                    break

        if not failed_stage:
            failed_stage = "generation"  # safe fallback

        latency = int((time.monotonic() - started) * 1000)
        return {
            "failed_stage": failed_stage,
            "learn_trace_events": [
                {
                    "node": "learn.understand",
                    "status": "completed",
                    "latency_ms": latency,
                    "summary": f"Failure mapped → stage: {failed_stage}",
                }
            ],
        }

    return understand_correction_node


def make_analyze_schema_node(services: Any):
    """Read the live database schema so the rewrite LLM has full column context."""

    @traced_node("learn.analyze_schema")
    async def analyze_schema_node(state: LearnState) -> dict:
        started = time.monotonic()

        source_id = state.get("source_id", "")
        conn = services.connectors.get(source_id)
        if not conn and services.connectors:
            conn = next(iter(services.connectors.values()))

        schema_context = ""
        if conn:
            try:
                snap = await conn.introspect()
                lines = []
                for t in snap.tables[:25]:
                    col_list = ", ".join(f"{c.name} ({c.data_type})" for c in t.columns[:20])
                    lines.append(f"  {t.name}: [{col_list}]")
                    # Add any sample values we have (helps with aggregate row detection)
                    if hasattr(t, "sample_values") and t.sample_values:
                        for col_name, vals in list(t.sample_values.items())[:3]:
                            lines.append(f"    sample {col_name}: {vals[:5]}")
                schema_context = "\n".join(lines)
            except Exception as exc:
                schema_context = f"(schema unavailable: {exc})"

        latency = int((time.monotonic() - started) * 1000)
        return {
            "schema_context": schema_context,
            "learn_trace_events": state.get("learn_trace_events", [])
            + [
                {
                    "node": "learn.analyze_schema",
                    "status": "completed",
                    "latency_ms": latency,
                    "summary": f"Schema loaded — {len(schema_context.splitlines())} table definitions",
                }
            ],
        }

    return analyze_schema_node


def make_rewrite_sql_node(services: Any):
    """LLM call: understand the data domain, rewrite the SQL, explain what changed."""

    @traced_node("learn.rewrite_sql")
    async def rewrite_sql_node(state: LearnState) -> dict:
        started = time.monotonic()

        prompt = (
            "You are the Learn Agent for a SQL workspace. A user received a wrong result "
            "and is providing a correction. Your job is to:\n"
            "1. Deeply understand WHY the original SQL was wrong — look at the schema "
            "carefully for aggregate rows, lookup values, naming conventions, or data "
            "quality issues that caused the mistake.\n"
            "2. Write the CORRECT SQL that properly answers the original question.\n"
            "3. State what you discovered about the data domain itself.\n\n"
            f"ORIGINAL QUESTION: {state.get('nl_query', '')}\n\n"
            f"ORIGINAL SQL (WRONG):\n{state.get('original_sql', '')}\n\n"
            f"FAILURE STAGE: {state.get('failed_stage', 'generation')}\n"
            f"USER'S NOTE: {state.get('correction_note', '(none provided)')}\n\n"
            f"DATABASE SCHEMA:\n{state.get('schema_context', '(none)')}\n\n"
            "Return JSON with these exact keys:\n"
            '{"rewritten_sql": "...", '
            '"what_changed": "One sentence: exactly what SQL change fixes the problem.", '
            '"domain_insight": "One sentence: what is true about the DATA DOMAIN that caused '
            "the original mistake (e.g., column X contains aggregate rows, table Y uses a "
            'surrogate key, etc.)."}'
        )

        resp = await services.llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=1200,
        )

        try:
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            parsed = {
                "rewritten_sql": state.get("original_sql", ""),
                "what_changed": "Could not parse rewrite response",
                "domain_insight": "",
            }

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)

        return {
            "rewritten_sql": parsed.get("rewritten_sql", ""),
            "what_changed": parsed.get("what_changed", ""),
            "domain_insight": parsed.get("domain_insight", ""),
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "learn_trace_events": state.get("learn_trace_events", [])
            + [
                {
                    "node": "learn.rewrite_sql",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "summary": parsed.get("what_changed", "SQL rewritten")[:80],
                }
            ],
        }

    return rewrite_sql_node


def make_execute_corrected_node(services: Any):
    """Run the rewritten SQL so the user can see real results before approving."""

    @traced_node("learn.execute_corrected")
    async def execute_corrected_node(state: LearnState) -> dict:
        started = time.monotonic()
        sql = state.get("rewritten_sql", "")

        if not sql:
            return {
                "exec_error": "No SQL to execute",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "learn_trace_events": state.get("learn_trace_events", [])
                + [
                    {
                        "node": "learn.execute_corrected",
                        "status": "failed",
                        "latency_ms": 0,
                        "summary": "No SQL produced by rewrite step",
                    }
                ],
            }

        source_id = state.get("source_id", "")
        conn = services.connectors.get(source_id)
        if not conn and services.connectors:
            conn = next(iter(services.connectors.values()))

        if not conn:
            return {
                "exec_error": "No database connector available",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "learn_trace_events": state.get("learn_trace_events", [])
                + [
                    {
                        "node": "learn.execute_corrected",
                        "status": "failed",
                        "latency_ms": 0,
                        "summary": "No connector for source",
                    }
                ],
            }

        try:
            import pandas as _pd

            # conn.execute() returns pd.DataFrame for DuckDB/file connectors
            result = await conn.execute(sql, timeout_s=30)
            latency = int((time.monotonic() - started) * 1000)

            # Normalise: DataFrame → rows/columns/row_count
            if isinstance(result, _pd.DataFrame):
                rows = result.head(20).to_dict("records")
                columns = list(result.columns)
                row_count = len(result)
            elif hasattr(result, "rows"):
                # ExecutionResult or similar dataclass with .rows
                rows = list(result.rows)[:20]
                columns = list(getattr(result, "columns", []))
                row_count = getattr(result, "row_count", len(rows))
            else:
                rows, columns, row_count = [], [], 0

            return {
                "rows": rows,
                "columns": columns,
                "row_count": row_count,
                "exec_error": "",
                "learn_trace_events": state.get("learn_trace_events", [])
                + [
                    {
                        "node": "learn.execute_corrected",
                        "status": "completed",
                        "latency_ms": latency,
                        "summary": f"{row_count} rows returned",
                    }
                ],
            }
        except Exception as exc:
            latency = int((time.monotonic() - started) * 1000)
            return {
                "exec_error": str(exc),
                "rows": [],
                "columns": [],
                "row_count": 0,
                "learn_trace_events": state.get("learn_trace_events", [])
                + [
                    {
                        "node": "learn.execute_corrected",
                        "status": "failed",
                        "latency_ms": latency,
                        "summary": f"Execution error: {str(exc)[:60]}",
                    }
                ],
            }

    return execute_corrected_node


def make_extract_lesson_node(services: Any):
    """LLM call: distil one reusable business rule from the correction."""

    @traced_node("learn.extract_lesson")
    async def extract_lesson_node(state: LearnState) -> dict:
        started = time.monotonic()

        prompt = (
            "A SQL query was corrected. Extract ONE reusable business rule from this correction.\n\n"
            f"Question: {state.get('nl_query', '')}\n"
            f"Data domain insight: {state.get('domain_insight', '')}\n"
            f"What changed: {state.get('what_changed', '')}\n"
            f"User note: {state.get('correction_note', '')}\n\n"
            "Write ONE sentence (max 25 words) starting with an action verb. "
            "This rule will be prepended to ALL future SQL generation prompts. "
            "Make it general and reusable — not tied to this specific query.\n\n"
            "Good examples:\n"
            "- Always exclude regional aggregate rows (e.g. RestOfASEAN) when filtering by individual country.\n"
            "- Use revenue_net column for profitability calculations, not revenue_gross.\n"
            "- Filter out NULL store_id rows before joining sales to stores.\n\n"
            "Reply with ONLY the rule sentence — no JSON, no explanation, no quotes."
        )

        resp = await services.llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=False,
            max_tokens=80,
        )

        lesson = resp.content.strip().strip('"').strip("'")
        # Ensure it ends with a period
        if lesson and not lesson.endswith("."):
            lesson += "."

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)

        return {
            "extracted_lesson": lesson,
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "learn_trace_events": state.get("learn_trace_events", [])
            + [
                {
                    "node": "learn.extract_lesson",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "summary": f"Rule: {lesson[:70]}",
                }
            ],
        }

    return extract_lesson_node


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════


def compile_learn_graph(services: Any) -> Any:
    """Compile the Learn Agent correction graph.

    Pipeline:
        understand_correction
            ↓ (map failure → stage)
        analyze_schema
            ↓ (load live schema)
        rewrite_sql
            ↓ (LLM: fix SQL + domain insight)
        execute_corrected
            ↓ (run against DB, return real rows)
        extract_lesson
            ↓ (LLM: one general business rule)
        END

    The caller (server.py /learn/regenerate) invokes this graph and returns
    the full LearnState to the frontend for user review BEFORE saving.
    Saving happens via POST /train/sql (not in this graph).
    """
    graph = StateGraph(LearnState)

    graph.add_node("understand_correction", make_understand_correction_node(services))
    graph.add_node("analyze_schema", make_analyze_schema_node(services))
    graph.add_node("rewrite_sql", make_rewrite_sql_node(services))
    graph.add_node("execute_corrected", make_execute_corrected_node(services))
    graph.add_node("extract_lesson", make_extract_lesson_node(services))

    graph.set_entry_point("understand_correction")
    graph.add_edge("understand_correction", "analyze_schema")
    graph.add_edge("analyze_schema", "rewrite_sql")
    graph.add_edge("rewrite_sql", "execute_corrected")
    graph.add_edge("execute_corrected", "extract_lesson")
    graph.add_edge("extract_lesson", END)

    return graph.compile()
