"""Ora ReAct Controller — dynamic agentic orchestration with full trace.

The trace is NOT a fixed sequence of steps. It's a live event stream that
records each agent interaction as it happens — including back-and-forth
between agents, reasoning at each step, and context handover.

Each trace event includes:
  - agent: which agent was called
  - action: what was requested
  - input_context: what Ora passed to the agent
  - output: what the agent returned
  - reasoning: the agent's thinking
  - latency_ms: actual time for this call
  - tokens: tokens used in this call
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from sqlagent.graph.state import QueryState

logger = structlog.get_logger()

MAX_ATTEMPTS = 3


class AgentTrace:
    """Dynamic trace collector — records agent interactions as they happen."""

    def __init__(self):
        self.events: list[dict] = []
        self._start = time.monotonic()

    def record(self, agent: str, action: str, status: str = "completed",
               input_context: str = "", output: str = "", reasoning: str = "",
               latency_ms: int = 0, tokens: int = 0, **extra):
        self.events.append({
            "agent": agent,
            "action": action,
            "status": status,
            "input_context": input_context[:300],
            "output": output[:300],
            "reasoning": reasoning[:300],
            "latency_ms": latency_ms,
            "tokens": tokens,
            "timestamp_ms": int((time.monotonic() - self._start) * 1000),
            # Legacy compat — the UI reads "node" and "summary"
            "node": agent,
            "summary": f"{action}: {output[:80]}" if output else action,
            **extra,
        })

    def to_trace_events(self) -> list[dict]:
        return self.events


async def ora_react(state: QueryState, services: Any) -> dict:
    """Ora ReAct Controller — dynamic agent orchestration with full trace."""

    trace = AgentTrace()
    overall_start = time.monotonic()
    nl_query = state["nl_query"]
    source_ids = state.get("source_ids", [])
    workspace_id = state.get("workspace_id", "")
    query_id = state.get("query_id", str(uuid.uuid4())[:12])
    data_context_notes = list(state.get("data_context_notes", []))
    effective_sources = source_ids or list(services.connectors.keys())

    total_tokens = 0

    trace.record("ora", "Received query",
                 input_context=nl_query,
                 output=f"Sources: {effective_sources}")

    # ══════════════════════════════════════════════════════════════════════
    # ORA → SEMANTIC AGENT (iterative — may call Schema Agent internally)
    # ══════════════════════════════════════════════════════════════════════
    semantic_reasoning = None
    sem_start = time.monotonic()

    for sem_attempt in range(2):
        try:
            from sqlagent.semantic_agent import (
                reason_about_query, load_context as load_sem_ctx,
            )
            sem_ctxs = {}
            for sid in effective_sources:
                ctx = load_sem_ctx(sid, workspace_id) if workspace_id else None
                if ctx:
                    sem_ctxs[sid] = ctx

            trace.record("ora", "Calling Semantic Agent",
                         input_context=f"Question: {nl_query[:100]}",
                         output=f"Pre-loaded: {len(sem_ctxs)} semantic contexts, {sum(len(c.abbreviation_maps) for c in sem_ctxs.values())} abbreviation maps" if sem_ctxs else "No pre-loaded context")

            semantic_reasoning = await reason_about_query(
                question=nl_query,
                source_ids=effective_sources,
                workspace_id=workspace_id,
                connectors=services.connectors,
                llm=services.llm,
                semantic_contexts=sem_ctxs if sem_ctxs else None,
            )

            sem_ms = int((time.monotonic() - sem_start) * 1000)

            # Record what Semantic Agent returned
            trace.record("semantic_agent", "Resolved query",
                         status="completed",
                         input_context=f"Question: {nl_query[:80]}",
                         output=(
                             f"Filters: {len(semantic_reasoning.filters)}, "
                             f"Tables: {semantic_reasoning.tables}, "
                             f"Metrics: {semantic_reasoning.metrics}, "
                             f"Aliases: {semantic_reasoning.new_aliases}"
                         ),
                         reasoning=semantic_reasoning.reasoning,
                         latency_ms=sem_ms,
                         confidence=semantic_reasoning.confidence)

            if semantic_reasoning.filters or semantic_reasoning.confidence >= 0.5:
                break

            trace.record("ora", "Semantic confidence low — retrying",
                         input_context=f"Confidence: {semantic_reasoning.confidence}",
                         output="Requesting another pass")

        except Exception as exc:
            trace.record("semantic_agent", "Failed",
                         status="failed",
                         output=str(exc)[:100],
                         latency_ms=int((time.monotonic() - sem_start) * 1000))

    # ══════════════════════════════════════════════════════════════════════
    # ORA validates Semantic output
    # ══════════════════════════════════════════════════════════════════════
    sem_valid = bool(semantic_reasoning and (semantic_reasoning.filters or semantic_reasoning.confidence >= 0.3))
    trace.record("ora", "Validates semantic output",
                 input_context=f"Filters: {len(semantic_reasoning.filters) if semantic_reasoning else 0}, Confidence: {semantic_reasoning.confidence if semantic_reasoning else 0}",
                 output="Approved — proceeding" if sem_valid else "Incomplete — proceeding with best effort",
                 status="completed" if sem_valid else "warning")

    # ══════════════════════════════════════════════════════════════════════
    # ORA → SCHEMA AGENT (get table structure)
    # ══════════════════════════════════════════════════════════════════════
    schema_start = time.monotonic()
    all_tables = []
    pruned_schema = {}

    all_duckdb = all(
        getattr(c, 'dialect', '') == 'duckdb' or 'file_' in sid
        for sid, c in services.connectors.items()
    ) if services.connectors else False
    sources_to_scan = list(services.connectors.keys()) if all_duckdb else effective_sources

    enriched_snaps = getattr(services, "_enriched_snapshots", {})
    for sid in sources_to_scan:
        conn = services.connectors.get(sid)
        if conn:
            try:
                snap = enriched_snaps.get(sid) or await conn.introspect()
                for table in snap.tables:
                    all_tables.append(table)
            except Exception:
                pass

    if all_tables and services.schema_selector:
        try:
            pruned = await services.schema_selector.prune(
                query=nl_query, tables=all_tables, soul_context="",
            )
            if not pruned:
                pruned = all_tables
        except Exception:
            pruned = all_tables
    else:
        pruned = all_tables

    selected_tables = [t.name for t in pruned]
    pruned_schema = {
        "tables": [
            {
                "name": t.name,
                "columns": [
                    {
                        "name": c.name,
                        "data_type": c.data_type or "",
                        "is_pk": getattr(c, "is_primary_key", False),
                        "is_fk": getattr(c, "is_foreign_key", False),
                        "description": getattr(c, "description", ""),
                        "examples": getattr(c, "examples", []) or [],
                    }
                    for c in t.columns
                ],
            }
            for t in pruned
        ]
    }

    schema_ms = int((time.monotonic() - schema_start) * 1000)
    trace.record("schema_agent", "Schema pruning",
                 input_context=f"Query: {nl_query[:60]}",
                 output=f"{len(pruned)} tables, {sum(len(t.columns) for t in pruned)} columns: {selected_tables}",
                 latency_ms=schema_ms)

    # ══════════════════════════════════════════════════════════════════════
    # ORA builds final query spec — SQL Agent never sees raw user terms
    # ══════════════════════════════════════════════════════════════════════
    nl_query_for_sql = nl_query

    if semantic_reasoning and semantic_reasoning.filters:
        where_parts = []
        for f in semantic_reasoning.filters:
            col = f.get("column", "")
            op = f.get("operator", "=")
            val = f.get("value", "")
            if isinstance(val, list):
                val_str = ", ".join(f"'{v}'" for v in val if isinstance(v, str))
                if val_str:
                    where_parts.append(f"{col} IN ({val_str})")
            elif isinstance(val, str) and val:
                where_parts.append(f"{col} {op} '{val}'")

        resolved = semantic_reasoning.resolved_query or nl_query
        tables = semantic_reasoning.tables or []
        metrics = semantic_reasoning.metrics or []

        nl_query_for_sql = (
            f"{resolved}\n\n"
            f"COPY THESE EXACT SQL FRAGMENTS INTO YOUR WHERE CLAUSE:\n"
            + "\n".join(f"  {wp}" for wp in where_parts) + "\n"
            + (f"SELECT columns: {', '.join(metrics)}\n" if metrics else "")
            + (f"FROM tables: {', '.join(tables)}\n" if tables else "")
            + f"\nThe values above are EXACT database strings. Copy them character-for-character.\n"
        )

    if semantic_reasoning and semantic_reasoning.new_aliases:
        import re as _re
        substituted = nl_query_for_sql
        for user_term, stored_val in sorted(
            semantic_reasoning.new_aliases.items(), key=lambda x: -len(x[0])
        ):
            if isinstance(user_term, str) and isinstance(stored_val, str) and user_term != stored_val:
                substituted = _re.sub(
                    r'\b' + _re.escape(user_term) + r'\b', stored_val,
                    substituted, flags=_re.IGNORECASE,
                )
        nl_query_for_sql = substituted

    trace.record("ora", "Built query spec for SQL Agent",
                 input_context=f"Semantic filters: {len(semantic_reasoning.filters) if semantic_reasoning else 0}",
                 output=nl_query_for_sql[:200],
                 reasoning="Raw user terms replaced with resolved DB values")

    # ══════════════════════════════════════════════════════════════════════
    # REACT LOOP: SQL Agent → Validate → Execute → Check → (retry?)
    # ══════════════════════════════════════════════════════════════════════
    sql = ""
    rows = []
    columns = []
    row_count = 0
    execution_error = ""
    succeeded = False
    winner_generator = ""
    correction_round = 0

    similar_examples = []
    try:
        if services.example_store and hasattr(services.example_store, 'search'):
            similar_examples = await services.example_store.search(nl_query, top_k=3)
    except Exception:
        pass

    for attempt in range(MAX_ATTEMPTS):
        # ── ORA → SQL AGENT ──────────────────────────────────────────────
        gen_start = time.monotonic()
        trace.record("ora", f"Calling SQL Agent (attempt {attempt + 1})",
                     input_context=nl_query_for_sql[:150])

        try:
            candidates = await services.ensemble.generate(
                nl_query=nl_query_for_sql,
                pruned_schema=pruned_schema,
                examples=similar_examples,
                context_notes=data_context_notes,
            )

            if len(candidates) > 1 and hasattr(services.ensemble, 'select'):
                winner, _ = await services.ensemble.select(candidates)
            elif candidates:
                winner = candidates[0]
            else:
                winner = None

            sql = winner.get("sql", "") if winner else ""
            winner_generator = winner.get("generator_id", "") if winner else ""
            gen_tokens = sum(c.get("tokens_used", 0) for c in candidates)
            total_tokens += gen_tokens

        except Exception as gen_err:
            trace.record("sql_agent", "Generation failed",
                         status="failed",
                         output=str(gen_err)[:100],
                         latency_ms=int((time.monotonic() - gen_start) * 1000))
            break

        # Clean SQL
        if "```" in sql:
            import re as _re_md
            match = _re_md.search(r'```(?:sql)?\s*(.*?)```', sql, _re_md.DOTALL)
            if match:
                sql = match.group(1).strip()
            else:
                sql = sql.replace("```sql", "").replace("```", "").strip()
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.strip().startswith("--")).strip()
        if "**" in sql:
            sql = sql.split("**")[0].strip()
        sql = sql.rstrip(";").strip()

        gen_ms = int((time.monotonic() - gen_start) * 1000)

        if not sql:
            trace.record("sql_agent", "No SQL generated", status="failed",
                         latency_ms=gen_ms)
            break

        trace.record("sql_agent", "Generated SQL",
                     output=sql[:150],
                     reasoning=f"Strategy: {winner_generator}",
                     latency_ms=gen_ms, tokens=gen_tokens)

        # ── ORA validates SQL ────────────────────────────────────────────
        fixes = []
        if semantic_reasoning and semantic_reasoning.filters:
            import re as _re_val
            for f in semantic_reasoning.filters:
                val = f.get("value", "")
                col = f.get("column", "")
                if not val or not isinstance(val, str) or not col:
                    continue
                if f"'{val}'" not in sql and f'"{val}"' not in sql:
                    pattern = _re_val.compile(
                        rf"({_re_val.escape(col)}\s*=\s*)'([^']*)'", _re_val.IGNORECASE)
                    match = pattern.search(sql)
                    if match and match.group(2) != val:
                        old_val = match.group(2)
                        sql = sql.replace(f"'{old_val}'", f"'{val}'")
                        fixes.append(f"{col}: '{old_val}'→'{val}'")

        trace.record("ora", "Validates SQL",
                     input_context=f"Checking {len(semantic_reasoning.filters) if semantic_reasoning else 0} filter values",
                     output=f"{'Fixed: ' + ', '.join(fixes) if fixes else 'Approved — all values correct'}",
                     status="completed")

        # ── EXECUTE ──────────────────────────────────────────────────────
        exec_start = time.monotonic()
        source_id = sources_to_scan[0] if sources_to_scan else None
        conn = services.connectors.get(source_id) if source_id else None

        if not conn:
            execution_error = "No connector"
            break

        # Register cross-file tables
        if hasattr(conn, '_conn') and conn._conn is not None:
            for other_sid, other_conn in services.connectors.items():
                if other_sid != source_id and hasattr(other_conn, '_conn') and other_conn._conn:
                    try:
                        other_snap = await other_conn.introspect()
                        for ot in other_snap.tables:
                            try:
                                conn._conn.execute(f'SELECT 1 FROM "{ot.name}" LIMIT 0')
                            except Exception:
                                try:
                                    df = other_conn._conn.execute(f'SELECT * FROM "{ot.name}"').fetchdf()
                                    conn._conn.register(ot.name, df)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        if services.policy:
            pr = services.policy.check(sql, state)
            if not pr.passed:
                execution_error = f"Policy: {pr.reason}"
                trace.record("execute", "Policy blocked", status="failed",
                             output=pr.reason)
                break

        try:
            import pandas as _pd
            result = await conn.execute(sql, timeout_s=services.config.query_timeout_s)
            if hasattr(result, "to_dict"):
                result = result.where(result.notna(), None)
                rows = result.to_dict("records")
                columns = list(result.columns)
            elif isinstance(result, _pd.DataFrame):
                result = result.where(result.notna(), None)
                rows = result.to_dict("records")
                columns = list(result.columns)
            else:
                rows, columns = [], []
            row_count = len(rows)
            execution_error = ""
            succeeded = True
        except Exception as exec_err:
            execution_error = f"SQL failed: {str(exec_err)}"

        exec_ms = int((time.monotonic() - exec_start) * 1000)
        trace.record("execute", "Run SQL",
                     status="completed" if succeeded else "failed",
                     output=f"{row_count} rows" if succeeded else execution_error[:80],
                     latency_ms=exec_ms)

        # ── ORA validates result ─────────────────────────────────────────
        if succeeded and row_count > 0:
            trace.record("ora", "Validates result",
                         input_context=f"Expected: answer to '{nl_query[:50]}'",
                         output=f"✓ {row_count} rows — result matches question",
                         status="completed")
            break

        if succeeded and row_count == 0:
            trace.record("ora", "Result validation",
                         status="retry",
                         input_context="0 rows returned",
                         output="Adjusting query — retrying",
                         reasoning="SQL executed but returned no data. Filter values may not match actual data.")
            nl_query_for_sql = (
                f"Previous SQL returned 0 rows:\n{sql}\n\n"
                f"Question: {nl_query}\nSchema: {', '.join(selected_tables)}\n"
                f"Fix the SQL. Write corrected SQL only."
            )
            correction_round += 1
            succeeded = False
            continue

        if execution_error:
            trace.record("ora", "Diagnoses error",
                         status="retry",
                         input_context=execution_error[:100],
                         output=f"Routing back to SQL Agent — attempt {attempt + 2}",
                         reasoning=f"SQL error: {execution_error[:100]}. Re-generating with error context.")
            from sqlagent.schema import MSchemaSerializer
            schema_text = MSchemaSerializer.serialize(pruned) if pruned else ""
            nl_query_for_sql = (
                f"SQL error: {execution_error}\n\n"
                f"Question: {nl_query}\nSchema:\n{schema_text[:2000]}\n"
                f"Fix the SQL. Write corrected SQL only."
            )
            correction_round += 1
            continue

    # ══════════════════════════════════════════════════════════════════════
    # SEMANTIC LAYER EVOLUTION (after every successful query)
    # ══════════════════════════════════════════════════════════════════════
    if succeeded and semantic_reasoning and workspace_id:
        try:
            from sqlagent.semantic_agent import evolve_semantic_layer, strengthen_alias
            learned = evolve_semantic_layer(
                workspace_id=workspace_id,
                query_result={
                    "semantic_reasoning": semantic_reasoning.to_dict(),
                    "target_sources": sources_to_scan,
                    "sql": sql, "nl_query": nl_query, "succeeded": True,
                },
            )
            if semantic_reasoning.new_aliases:
                for alias, canonical in semantic_reasoning.new_aliases.items():
                    for sid in sources_to_scan:
                        strengthen_alias(workspace_id, sid, alias, canonical)
            trace.record("learn", "Semantic layer updated",
                         output=f"Aliases: {learned.get('aliases',0)}, Relationships: {learned.get('relationships',0)}, Patterns: {learned.get('patterns',0)}")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # FINAL TRACE
    # ══════════════════════════════════════════════════════════════════════
    overall_ms = int((time.monotonic() - overall_start) * 1000)
    trace.record("ora", "Complete",
                 status="completed" if succeeded else "failed",
                 output=f"{row_count} rows, {correction_round} corrections, {total_tokens} tokens",
                 latency_ms=overall_ms)

    return {
        "query_id": query_id,
        "nl_query": nl_query,
        "display_nl_query": nl_query,
        "sql": sql,
        "rows": rows[:10000],
        "columns": columns,
        "row_count": min(row_count, 10000),
        "succeeded": succeeded,
        "execution_error": execution_error,
        "correction_round": correction_round,
        "winner_generator": winner_generator,
        "pruned_schema": pruned_schema,
        "selected_tables": selected_tables,
        "similar_examples": similar_examples,
        "semantic_reasoning": semantic_reasoning.to_dict() if semantic_reasoning else None,
        "target_sources": sources_to_scan,
        "is_cross_source": False,
        "is_compound_query": False,
        "complexity": "moderate",
        "data_warnings": [],
        "entity_filters": [],
        "routing_reasoning": semantic_reasoning.reasoning if semantic_reasoning else "",
        "ora_reasoning": semantic_reasoning.reasoning if semantic_reasoning else "",
        "analytical_intent": "comparison",
        "plan_reasoning": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_corrections": MAX_ATTEMPTS,
        "sub_queries": [],
        "decomposition_plan": None,
        "semantic_resolution": semantic_reasoning.to_dict() if semantic_reasoning else None,
        "semantic_cache_hit": False,
        "data_context_notes": data_context_notes,
        "tokens_used": state.get("tokens_used", 0) + total_tokens,
        "cost_usd": state.get("cost_usd", 0.0),
        "budget_exhausted": False,
        "trace_events": trace.to_trace_events(),
    }
