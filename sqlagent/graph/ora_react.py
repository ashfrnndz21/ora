"""Ora ReAct Controller — single orchestrator that owns the full query lifecycle.

NOT a linear pipeline. Ora is a ReAct loop that:
  1. Calls Semantic Agent → validates output → retries if needed
  2. Gets schema context → validates columns exist → retries if needed
  3. Builds query spec → calls SQL Agent → validates SQL → fixes if needed
  4. Executes → validates result → diagnoses failures → routes to right agent
  5. Only returns when it has a good answer OR exhausted all attempts

Every successful query updates the semantic layer.
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


async def ora_react(state: QueryState, services: Any) -> dict:
    """Ora ReAct Controller — orchestrates all agents in a reasoning loop.

    This replaces the linear ora → generate → execute → validate chain
    with a single function that controls everything.
    """
    started = time.monotonic()
    nl_query = state["nl_query"]
    source_ids = state.get("source_ids", [])
    workspace_id = state.get("workspace_id", "")
    query_id = state.get("query_id", str(uuid.uuid4())[:12])

    trace_events = list(state.get("trace_events", []))
    total_tokens = 0
    total_cost = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: SEMANTIC REASONING (with retry)
    # ══════════════════════════════════════════════════════════════════════

    semantic_reasoning = None
    sem_trace = None

    for sem_attempt in range(2):  # max 2 semantic attempts
        try:
            from sqlagent.semantic_agent import (
                reason_about_query, load_context as load_sem_ctx,
            )

            sem_ctxs = {}
            for sid in (source_ids or list(services.connectors.keys())):
                ctx = load_sem_ctx(sid, workspace_id) if workspace_id else None
                if ctx:
                    sem_ctxs[sid] = ctx

            semantic_reasoning = await reason_about_query(
                question=nl_query,
                source_ids=source_ids,
                workspace_id=workspace_id,
                connectors=services.connectors,
                llm=services.llm,
                semantic_contexts=sem_ctxs if sem_ctxs else None,
            )

            sem_trace = {
                "node": "semantic_reasoning",
                "status": "completed",
                "latency_ms": 0,
                "summary": (
                    f"Resolved {len(semantic_reasoning.filters)} filters, "
                    f"{len(semantic_reasoning.new_aliases)} new aliases · "
                    f"Confidence: {int(semantic_reasoning.confidence * 100)}%"
                ),
                "thinking": semantic_reasoning.reasoning[:300],
                "confidence": semantic_reasoning.confidence,
            }
            trace_events.append(sem_trace)

            if semantic_reasoning.filters or semantic_reasoning.confidence >= 0.5:
                break  # Good enough — proceed

            logger.info("ora.react.semantic_retry", attempt=sem_attempt, confidence=semantic_reasoning.confidence)

        except Exception as exc:
            logger.warning("ora.react.semantic_failed", attempt=sem_attempt, error=str(exc))

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: SCHEMA CONTEXT (get all tables + columns)
    # ══════════════════════════════════════════════════════════════════════

    all_tables = []
    pruned_schema = {}

    # For DuckDB sources: include ALL tables from ALL connectors
    all_duckdb = all(
        getattr(c, 'dialect', '') == 'duckdb' or 'file_' in sid
        for sid, c in services.connectors.items()
    ) if services.connectors else False
    sources_to_scan = list(services.connectors.keys()) if all_duckdb else source_ids

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

    # Prune if we have a schema selector
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

    from sqlagent.schema import MSchemaSerializer
    schema_text = MSchemaSerializer.serialize(pruned) if pruned else ""
    selected_tables = [t.name for t in pruned]
    pruned_schema = {
        "tables": [
            {"name": t.name, "columns": [{"name": c.name, "type": c.data_type} for c in t.columns]}
            for t in pruned
        ]
    }

    trace_events.append({
        "node": "schema",
        "status": "completed",
        "summary": f"Schema: {len(pruned)} tables, {sum(len(t.columns) for t in pruned)} columns",
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: BUILD QUERY FOR SQL AGENT
    # Ora assembles the refined query — SQL Agent never sees raw user terms
    # ══════════════════════════════════════════════════════════════════════

    nl_query_for_sql = nl_query  # default fallback

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

    # Also substitute user terms with resolved values in the NL query
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

    logger.info(
        "ora.react.pre_generate",
        tables=len(pruned),
        schema_text_len=len(schema_text),
        query_len=len(nl_query_for_sql),
        connectors=list(services.connectors.keys()),
        has_ensemble=bool(services.ensemble),
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: SQL GENERATION → VALIDATION → EXECUTION (ReAct loop)
    # Ora generates, validates, fixes, executes — retries at the right level
    # ══════════════════════════════════════════════════════════════════════

    sql = ""
    rows = []
    columns = []
    row_count = 0
    execution_error = ""
    succeeded = False
    winner_generator = ""
    correction_round = 0

    # Get examples for few-shot (gracefully handle missing/broken store)
    similar_examples = []
    try:
        if services.example_store and hasattr(services.example_store, 'search'):
            similar_examples = await services.example_store.search(nl_query, top_k=3)
    except Exception:
        pass  # example store not available — generate from scratch

    # Semantic context notes
    data_context_notes = list(state.get("data_context_notes", []))

    for attempt in range(MAX_ATTEMPTS):
        # ── Generate SQL ─────────────────────────────────────────────
        try:
            logger.info("ora.react.calling_ensemble", pruned_count=len(pruned), nl_len=len(nl_query_for_sql))
            # Pass the actual table objects (not dict) — generators use MSchemaSerializer
            candidates = await services.ensemble.generate(
                nl_query=nl_query_for_sql,
                pruned_schema=pruned,  # list of SchemaTable objects
                examples=similar_examples,
                context_notes=data_context_notes,
            )

            if len(candidates) > 1 and hasattr(services.ensemble, 'select'):
                winner, selection_reasoning = await services.ensemble.select(candidates)
            elif candidates:
                winner = candidates[0]
                selection_reasoning = "Single candidate"
            else:
                winner = None
                selection_reasoning = "No candidates generated"

            sql = winner.get("sql", "") if winner else ""
            winner_generator = winner.get("generator_id", "") if winner else ""
            gen_tokens = sum(c.get("tokens_used", 0) for c in candidates)
            total_tokens += gen_tokens
            total_cost += sum(c.get("cost_usd", 0.0) for c in candidates)

        except Exception as gen_err:
            import traceback
            logger.error(
                "ora.react.generate_failed",
                error=str(gen_err),
                traceback=traceback.format_exc()[:500],
                attempt=attempt,
            )
            trace_events.append({
                "node": "generate", "status": "failed",
                "summary": f"SQL generation failed: {str(gen_err)[:80]}",
            })
            break

        if not sql:
            trace_events.append({
                "node": "generate", "status": "failed",
                "summary": "No SQL generated",
            })
            break

        # ── Ora validates SQL BEFORE execution ───────────────────────
        if semantic_reasoning and semantic_reasoning.filters:
            import re as _re_val
            for f in semantic_reasoning.filters:
                val = f.get("value", "")
                col = f.get("column", "")
                if not val or not isinstance(val, str) or not col:
                    continue
                if f"'{val}'" not in sql and f'"{val}"' not in sql:
                    # Value missing — try to fix
                    pattern = _re_val.compile(
                        rf"({_re_val.escape(col)}\s*=\s*)'([^']*)'",
                        _re_val.IGNORECASE
                    )
                    match = pattern.search(sql)
                    if match and match.group(2) != val:
                        old_val = match.group(2)
                        sql = sql.replace(f"'{old_val}'", f"'{val}'")
                        logger.info("ora.react.fixed_value", column=col, old=old_val, new=val)

        trace_events.append({
            "node": "generate",
            "status": "completed",
            "summary": f"Generated SQL via {winner_generator}",
            "tokens": gen_tokens,
        })

        # ── Execute ──────────────────────────────────────────────────
        source_id = sources_to_scan[0] if sources_to_scan else None
        conn = services.connectors.get(source_id) if source_id else None

        if not conn:
            execution_error = "No connector available"
            break

        # Register cross-file tables (DuckDB multi-table fix)
        if hasattr(conn, '_conn') and conn._conn is not None:
            for other_sid, other_conn in services.connectors.items():
                if other_sid != source_id and hasattr(other_conn, '_conn') and other_conn._conn:
                    try:
                        other_snap = await other_conn.introspect()
                        for other_table in other_snap.tables:
                            try:
                                conn._conn.execute(f'SELECT 1 FROM "{other_table.name}" LIMIT 0')
                            except Exception:
                                try:
                                    df = other_conn._conn.execute(
                                        f'SELECT * FROM "{other_table.name}"'
                                    ).fetchdf()
                                    conn._conn.register(other_table.name, df)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        # Policy check
        if services.policy:
            policy_result = services.policy.check(sql, state)
            if not policy_result.passed:
                execution_error = f"Policy blocked: {policy_result.reason}"
                trace_events.append({
                    "node": "execute", "status": "failed",
                    "summary": f"Policy: {policy_result.reason}",
                })
                break

        try:
            result = await conn.execute(sql, timeout_s=services.config.query_timeout_s)
            import pandas as _pd
            if hasattr(result, "to_dict"):
                result = result.where(result.notna(), None)
                rows = result.to_dict("records")
                columns = list(result.columns)
            elif isinstance(result, _pd.DataFrame):
                result = result.where(result.notna(), None)
                rows = result.to_dict("records")
                columns = list(result.columns)
            else:
                rows = []
                columns = []
            row_count = len(rows)
            execution_error = ""
            succeeded = True

            trace_events.append({
                "node": "execute", "status": "completed",
                "summary": f"{row_count} rows · {int((time.monotonic() - started) * 1000)}ms",
            })

        except Exception as exec_err:
            execution_error = f"SQL failed: {str(exec_err)}"
            trace_events.append({
                "node": "execute", "status": "failed",
                "summary": str(exec_err)[:100],
            })

        # ── Ora evaluates result ─────────────────────────────────────
        if succeeded and row_count > 0:
            # Good result — done
            break

        if succeeded and row_count == 0:
            # SQL ran but no rows — might be wrong filter values
            logger.info("ora.react.empty_result", attempt=attempt, sql=sql[:100])
            # Build correction prompt
            nl_query_for_sql = (
                f"The previous SQL returned 0 rows:\n{sql}\n\n"
                f"The question was: {nl_query}\n"
                f"Schema: {', '.join(selected_tables)}\n"
                f"Fix the SQL to return results. Check that filter values match actual data.\n"
                f"Write corrected SQL only."
            )
            correction_round += 1
            succeeded = False
            continue

        if execution_error:
            # SQL error — correct and retry
            logger.info("ora.react.sql_error", attempt=attempt, error=execution_error[:80])
            nl_query_for_sql = (
                f"SQL error: {execution_error}\n\n"
                f"Original question: {nl_query}\n"
                f"Schema tables: {', '.join(selected_tables)}\n"
                f"Schema:\n{schema_text[:2000]}\n\n"
                f"Fix the SQL. Write corrected SQL only."
            )
            correction_round += 1
            continue

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: EVOLVE SEMANTIC LAYER (after every attempt, successful or not)
    # ══════════════════════════════════════════════════════════════════════

    if succeeded and semantic_reasoning and workspace_id:
        try:
            from sqlagent.semantic_agent import evolve_semantic_layer, strengthen_alias
            evolve_semantic_layer(
                workspace_id=workspace_id,
                query_result={
                    "semantic_reasoning": semantic_reasoning.to_dict(),
                    "target_sources": sources_to_scan,
                    "sql": sql,
                    "nl_query": nl_query,
                    "succeeded": True,
                },
            )
            if semantic_reasoning.new_aliases:
                for alias, canonical in semantic_reasoning.new_aliases.items():
                    for sid in sources_to_scan:
                        strengthen_alias(workspace_id, sid, alias, canonical)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # RETURN STATE
    # ══════════════════════════════════════════════════════════════════════

    latency = int((time.monotonic() - started) * 1000)

    trace_events.append({
        "node": "ora",
        "status": "completed" if succeeded else "failed",
        "latency_ms": latency,
        "summary": (
            f"{'Success' if succeeded else 'Failed'}: {row_count} rows, "
            f"{correction_round} corrections, {total_tokens} tokens"
        ),
    })

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
        "cost_usd": state.get("cost_usd", 0.0) + total_cost,
        "budget_exhausted": False,
        "trace_events": trace_events,
    }
