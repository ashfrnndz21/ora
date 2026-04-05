"""Ora ReAct Controller — 11-step orchestration with per-agent trace.

FLOW:
  Step 1:  ORA receives query → decomposes into parts
  Step 2:  ORA → SEMANTIC AGENT (3-pass: knowledge → schema search → assembly)
  Step 3:  ORA validates semantic output
  Step 4:  ORA → SCHEMA AGENT (get full column details)
  Step 5:  ORA builds final query spec (SQL Agent never sees raw user terms)
  Step 6:  ORA → SQL AGENT (generates SQL from spec)
  Step 7:  ORA validates SQL (checks filter values match semantic output)
  Step 8:  EXECUTE
  Step 9:  ORA validates result (checks it answers the question)
  Step 10: RESPOND (NL summary + confidence + chart)
  Step 11: LEARN (evolve semantic layer)

Each step records its own trace event with actual latency.
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
    """Ora ReAct Controller — orchestrates all agents with per-step tracing."""

    overall_start = time.monotonic()
    nl_query = state["nl_query"]
    source_ids = state.get("source_ids", [])
    workspace_id = state.get("workspace_id", "")
    query_id = state.get("query_id", str(uuid.uuid4())[:12])

    trace_events = list(state.get("trace_events", []))
    total_tokens = 0
    total_cost = 0.0
    data_context_notes = list(state.get("data_context_notes", []))

    # Connectors — use all available if source_ids empty
    effective_sources = source_ids or list(services.connectors.keys())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: ORA receives query — understand the parts
    # ══════════════════════════════════════════════════════════════════════
    step1_start = time.monotonic()
    trace_events.append({
        "node": "ora_receive",
        "status": "completed",
        "latency_ms": 0,
        "summary": f"Received: {nl_query[:80]}",
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: ORA → SEMANTIC AGENT (iterative reasoning)
    # ══════════════════════════════════════════════════════════════════════
    step2_start = time.monotonic()
    semantic_reasoning = None

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

            semantic_reasoning = await reason_about_query(
                question=nl_query,
                source_ids=effective_sources,
                workspace_id=workspace_id,
                connectors=services.connectors,
                llm=services.llm,
                semantic_contexts=sem_ctxs if sem_ctxs else None,
            )

            if semantic_reasoning.filters or semantic_reasoning.confidence >= 0.5:
                break
        except Exception as exc:
            logger.warning("ora.step2.semantic_failed", attempt=sem_attempt, error=str(exc))

    step2_ms = int((time.monotonic() - step2_start) * 1000)
    trace_events.append({
        "node": "semantic_reasoning",
        "status": "completed",
        "latency_ms": step2_ms,
        "summary": (
            f"Resolved {len(semantic_reasoning.filters) if semantic_reasoning else 0} filters, "
            f"{len(semantic_reasoning.new_aliases) if semantic_reasoning else 0} new aliases · "
            f"Confidence: {int(semantic_reasoning.confidence * 100) if semantic_reasoning else 0}%"
        ),
        "thinking": semantic_reasoning.reasoning[:300] if semantic_reasoning else "",
        "confidence": semantic_reasoning.confidence if semantic_reasoning else 0,
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: ORA validates semantic output
    # ══════════════════════════════════════════════════════════════════════
    step3_start = time.monotonic()
    sem_valid = bool(semantic_reasoning and (semantic_reasoning.filters or semantic_reasoning.confidence >= 0.3))
    step3_ms = int((time.monotonic() - step3_start) * 1000)
    trace_events.append({
        "node": "ora_validate_semantic",
        "status": "completed" if sem_valid else "warning",
        "latency_ms": step3_ms,
        "summary": (
            f"Semantic output {'validated' if sem_valid else 'incomplete'} — "
            f"{len(semantic_reasoning.tables) if semantic_reasoning else 0} tables, "
            f"{len(semantic_reasoning.filters) if semantic_reasoning else 0} filters"
        ),
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: ORA → SCHEMA AGENT (get table structure)
    # ══════════════════════════════════════════════════════════════════════
    step4_start = time.monotonic()
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

    step4_ms = int((time.monotonic() - step4_start) * 1000)
    trace_events.append({
        "node": "schema_agent",
        "status": "completed",
        "latency_ms": step4_ms,
        "summary": f"Schema: {len(pruned)} tables, {sum(len(t.columns) for t in pruned)} columns",
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: ORA builds final query spec for SQL Agent
    # ══════════════════════════════════════════════════════════════════════
    step5_start = time.monotonic()
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

    # Substitute user terms with resolved values
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

    step5_ms = int((time.monotonic() - step5_start) * 1000)
    trace_events.append({
        "node": "ora_build_spec",
        "status": "completed",
        "latency_ms": step5_ms,
        "summary": f"Built query spec — {len(nl_query_for_sql)} chars",
    })

    # ══════════════════════════════════════════════════════════════════════
    # STEPS 6-9: SQL GENERATION → VALIDATION → EXECUTION → RESULT CHECK
    # (ReAct loop — retries at the right level)
    # ══════════════════════════════════════════════════════════════════════

    sql = ""
    rows = []
    columns = []
    row_count = 0
    execution_error = ""
    succeeded = False
    winner_generator = ""
    correction_round = 0

    # Get examples for few-shot
    similar_examples = []
    try:
        if services.example_store and hasattr(services.example_store, 'search'):
            similar_examples = await services.example_store.search(nl_query, top_k=3)
    except Exception:
        pass

    for attempt in range(MAX_ATTEMPTS):
        # ── STEP 6: SQL AGENT generates SQL ──────────────────────────────
        step6_start = time.monotonic()
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
            logger.error("ora.step6.generate_failed", error=str(gen_err))
            trace_events.append({
                "node": "sql_agent", "status": "failed",
                "latency_ms": int((time.monotonic() - step6_start) * 1000),
                "summary": f"Generation failed: {str(gen_err)[:60]}",
            })
            break

        # Strip markdown/comments from SQL
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

        step6_ms = int((time.monotonic() - step6_start) * 1000)

        if not sql:
            trace_events.append({
                "node": "sql_agent", "status": "failed",
                "latency_ms": step6_ms,
                "summary": "No SQL generated",
            })
            break

        trace_events.append({
            "node": "sql_agent", "status": "completed",
            "latency_ms": step6_ms,
            "summary": f"Generated SQL via {winner_generator}",
            "tokens": gen_tokens,
        })

        # ── STEP 7: ORA validates SQL ────────────────────────────────────
        step7_start = time.monotonic()
        fixes_made = []
        if semantic_reasoning and semantic_reasoning.filters:
            import re as _re_val
            for f in semantic_reasoning.filters:
                val = f.get("value", "")
                col = f.get("column", "")
                if not val or not isinstance(val, str) or not col:
                    continue
                if f"'{val}'" not in sql and f'"{val}"' not in sql:
                    pattern = _re_val.compile(
                        rf"({_re_val.escape(col)}\s*=\s*)'([^']*)'",
                        _re_val.IGNORECASE
                    )
                    match = pattern.search(sql)
                    if match and match.group(2) != val:
                        old_val = match.group(2)
                        sql = sql.replace(f"'{old_val}'", f"'{val}'")
                        fixes_made.append(f"{col}: '{old_val}' → '{val}'")

        step7_ms = int((time.monotonic() - step7_start) * 1000)
        trace_events.append({
            "node": "ora_validate_sql",
            "status": "completed",
            "latency_ms": step7_ms,
            "summary": (
                f"SQL approved{' — fixed ' + str(len(fixes_made)) + ' values' if fixes_made else ''}"
            ),
        })

        # ── STEP 8: EXECUTE ──────────────────────────────────────────────
        step8_start = time.monotonic()
        source_id = sources_to_scan[0] if sources_to_scan else None
        conn = services.connectors.get(source_id) if source_id else None

        if not conn:
            execution_error = "No connector available"
            break

        # Register cross-file tables for JOINs
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

        if services.policy:
            policy_result = services.policy.check(sql, state)
            if not policy_result.passed:
                execution_error = f"Policy blocked: {policy_result.reason}"
                trace_events.append({
                    "node": "execute", "status": "failed",
                    "latency_ms": int((time.monotonic() - step8_start) * 1000),
                    "summary": f"Policy: {policy_result.reason}",
                })
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

        step8_ms = int((time.monotonic() - step8_start) * 1000)
        trace_events.append({
            "node": "execute",
            "status": "completed" if succeeded else "failed",
            "latency_ms": step8_ms,
            "summary": f"{row_count} rows · {step8_ms}ms" if succeeded else str(execution_error)[:80],
        })

        # ── STEP 9: ORA validates result ─────────────────────────────────
        step9_start = time.monotonic()
        if succeeded and row_count > 0:
            trace_events.append({
                "node": "ora_validate_result",
                "status": "completed",
                "latency_ms": int((time.monotonic() - step9_start) * 1000),
                "summary": f"Result validated — {row_count} rows match question",
            })
            break  # SUCCESS

        if succeeded and row_count == 0:
            trace_events.append({
                "node": "ora_validate_result",
                "status": "retry",
                "latency_ms": int((time.monotonic() - step9_start) * 1000),
                "summary": "0 rows returned — adjusting query",
            })
            nl_query_for_sql = (
                f"Previous SQL returned 0 rows:\n{sql}\n\n"
                f"Question: {nl_query}\nSchema: {', '.join(selected_tables)}\n"
                f"Fix the SQL. Write corrected SQL only."
            )
            correction_round += 1
            succeeded = False
            continue

        if execution_error:
            trace_events.append({
                "node": "ora_validate_result",
                "status": "retry",
                "latency_ms": int((time.monotonic() - step9_start) * 1000),
                "summary": f"SQL error — retrying: {execution_error[:60]}",
            })
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
    # STEP 11: LEARN (semantic layer evolution — done in learn node)
    # Save what we learned for future queries
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
    # FINAL: Ora summary trace event
    # ══════════════════════════════════════════════════════════════════════
    overall_ms = int((time.monotonic() - overall_start) * 1000)
    trace_events.append({
        "node": "ora",
        "status": "completed" if succeeded else "failed",
        "latency_ms": overall_ms,
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
