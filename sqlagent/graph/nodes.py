"""LangGraph node functions — each takes QueryState, returns partial update.

Every node is a real, fully agentic function. No mocks, no fakes.
Each node calls real LLMs, real databases, real vector stores.

Nodes are decorated with @traced_node for OTel spans + trace events.
The services (LLM, connectors, etc.) are injected via closure at graph compile time.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from sqlagent.graph.state import QueryState

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# NODE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
# Each make_*_node function takes PipelineServices and returns
# an async function(state) -> dict. This is the LangGraph pattern:
# compile-time dependency injection via closures.


def make_understand_node(services: Any):
    """Route: decide single-source vs cross-source based on available data sources.

    No rule-based intent classification. Conversation history flows through as state
    so downstream agents (especially respond_node) can reason over it naturally.
    """

    # ISO-3166 alpha-2 → country name expansion (helps LSH find the right columns)
    _ISO2 = {
        "AF": "Afghanistan",
        "AL": "Albania",
        "DZ": "Algeria",
        "AR": "Argentina",
        "AU": "Australia",
        "AT": "Austria",
        "BD": "Bangladesh",
        "BE": "Belgium",
        "BR": "Brazil",
        "CA": "Canada",
        "CL": "Chile",
        "CN": "China",
        "CO": "Colombia",
        "HR": "Croatia",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EG": "Egypt",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "GH": "Ghana",
        "GR": "Greece",
        "HU": "Hungary",
        "IN": "India",
        "ID": "Indonesia",
        "IR": "Iran",
        "IQ": "Iraq",
        "IE": "Ireland",
        "IL": "Israel",
        "IT": "Italy",
        "JP": "Japan",
        "JO": "Jordan",
        "KE": "Kenya",
        "KR": "South Korea",
        "KW": "Kuwait",
        "LB": "Lebanon",
        "MY": "Malaysia",
        "MX": "Mexico",
        "MA": "Morocco",
        "NL": "Netherlands",
        "NZ": "New Zealand",
        "NG": "Nigeria",
        "NO": "Norway",
        "PK": "Pakistan",
        "PE": "Peru",
        "PH": "Philippines",
        "PL": "Poland",
        "PT": "Portugal",
        "QA": "Qatar",
        "RO": "Romania",
        "RU": "Russia",
        "SA": "Saudi Arabia",
        "SG": "Singapore",
        "ZA": "South Africa",
        "ES": "Spain",
        "SE": "Sweden",
        "CH": "Switzerland",
        "TH": "Thailand",
        "TN": "Tunisia",
        "TR": "Turkey",
        "UA": "Ukraine",
        "AE": "United Arab Emirates",
        "GB": "United Kingdom",
        "US": "United States",
        "VN": "Vietnam",
        "MYS": "Malaysia",
        "IDN": "Indonesia",
        "SGP": "Singapore",
        "THA": "Thailand",
        "PHL": "Philippines",
        "VNM": "Vietnam",
        "MMR": "Myanmar",
        "KHM": "Cambodia",
        "LAO": "Laos",
        "BRN": "Brunei",
    }

    def _expand_query(q: str) -> str:
        """Expand ISO codes and known abbreviations so LSH can match schema columns."""
        import re

        tokens = re.findall(r"\b[A-Z]{2,3}\b", q)
        additions = []
        for tok in tokens:
            if tok in _ISO2:
                additions.append(_ISO2[tok])
        if additions:
            return q + " " + " ".join(additions)
        return q

    async def understand_node(state: QueryState) -> dict:
        nl_query = state["nl_query"]
        # Expand ISO codes / abbreviations so downstream embedding finds the right columns
        nl_query_expanded = _expand_query(nl_query)
        source_ids = state.get("source_ids", [])
        query_id = state.get("query_id", str(uuid.uuid4())[:12])
        started = time.monotonic()

        import json as _json

        def _trace(summary, tokens=0, latency=None):
            return state.get("trace_events", []) + [
                {
                    "node": "understand",
                    "status": "completed",
                    "latency_ms": latency or int((time.monotonic() - started) * 1000),
                    "summary": summary,
                    "tokens": tokens,
                }
            ]

        # ── Single source: fast path, no LLM needed ──────────────────────────
        if len(source_ids) <= 1:
            return {
                "query_id": query_id,
                "is_cross_source": False,
                "target_sources": source_ids,
                "complexity": "simple",
                "routing_reasoning": "Single source — direct pipeline",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "correction_round": 0,
                "max_corrections": services.config.max_corrections,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "budget_exhausted": False,
                # Pass expanded query so prune_node gets better embedding signal
                "nl_query_for_pruning": nl_query_expanded,
                "trace_events": _trace(f"Using {source_ids[0] if source_ids else 'no'} source"),
            }

        # ── Multi-source: LLM decides which sources and routing ───────────────
        source_descriptions = []
        for sid in source_ids:
            conn = services.connectors.get(sid)
            if conn:
                snap = await conn.introspect()
                tables = [t.name for t in snap.tables[:10]]
                source_descriptions.append(f"Source '{sid}' ({conn.dialect}): tables = {tables}")

        _hist = state.get("conversation_history", [])
        history_str = (
            "\n".join(
                f"{'User' if t.get('role') == 'user' else 'Agent'}: {t.get('text') or t.get('nl_response') or ''}"
                for t in _hist[-4:]
            )
            if _hist
            else ""
        )

        prompt = (
            "Available data sources:\n"
            + "\n".join(source_descriptions)
            + (f"\n\nConversation context:\n{history_str}" if history_str else "")
            + f"\n\nQuestion: {nl_query}\n\n"
            "Which sources are needed? Is this cross-source?\n"
            'Return JSON: {"target_sources": [...], "is_cross_source": bool, '
            '"complexity": "simple|moderate|complex", "reasoning": "..."}'
        )

        resp = await services.llm.complete([{"role": "user", "content": prompt}], json_mode=True)

        raw = resp.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            parsed = {
                "target_sources": source_ids,
                "is_cross_source": len(source_ids) > 1,
                "complexity": "moderate",
                "reasoning": "Fallback: parse error",
            }

        latency = int((time.monotonic() - started) * 1000)
        return {
            "query_id": query_id,
            "intent": "data_query",
            "is_cross_source": parsed.get("is_cross_source", False),
            "target_sources": parsed.get("target_sources", source_ids),
            "complexity": parsed.get("complexity", "moderate"),
            "routing_reasoning": parsed.get("reasoning", ""),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "correction_round": 0,
            "max_corrections": services.config.max_corrections,
            "tokens_used": resp.tokens_input + resp.tokens_output,
            "cost_usd": resp.cost_usd,
            "budget_exhausted": False,
            "trace_events": _trace(
                (
                    f"This spans {len(parsed.get('target_sources', source_ids))} datasets — "
                    f"I'll pull from {' and '.join(s.replace('file_', '').replace('_', ' ') for s in parsed.get('target_sources', source_ids)[:3])}"
                )
                if parsed.get("is_cross_source")
                else (f"Querying {(parsed.get('target_sources') or source_ids or ['?'])[0]}"),
                tokens=resp.tokens_input + resp.tokens_output,
                latency=latency,
            ),
        }

    return understand_node


def make_prune_node(services: Any):
    """Schema pruning via CHESS LSH — reduce 500 cols to ~8 relevant ones."""

    async def prune_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]
        # Use ISO-expanded query for embedding similarity (set by understand_node)
        nl_query_for_pruning = state.get("nl_query_for_pruning") or nl_query
        target_sources = state.get("target_sources", [])
        soul_context = state.get("soul_context", "")

        # Get schema from target sources — prefer enriched snapshots (with examples) if available
        enriched_snaps = getattr(services, "_enriched_snapshots", {})
        all_tables = []
        total_columns = 0
        for sid in target_sources:
            conn = services.connectors.get(sid)
            if conn:
                snap = enriched_snaps.get(sid) or await conn.introspect()
                for table in snap.tables:
                    total_columns += len(table.columns)
                    all_tables.append((sid, table))

        fallback_used = False

        # Schema pruning via embedder similarity (uses expanded query for better ISO/entity matching)
        if services.schema_selector:
            pruned = await services.schema_selector.prune(
                query=nl_query_for_pruning,
                tables=[t for _, t in all_tables],
                soul_context=soul_context,
            )
            # ── Zero-result fallback: LSH found nothing relevant ──────────────
            # This happens for exploratory queries ("tell me about MY") where the
            # query vocabulary doesn't overlap with column names.
            # Fall back to full schema so the LLM can reason over real structure.
            if not pruned and all_tables:
                pruned = [t for _, t in all_tables]
                fallback_used = True

            selected_tables = [t.name for t in pruned]
            columns_after = sum(len(t.columns) for t in pruned)
        else:
            # No embedder: send everything unfiltered
            pruned = [t for _, t in all_tables]
            selected_tables = [t.name for t in pruned]
            columns_after = total_columns

        latency = int((time.monotonic() - started) * 1000)

        if fallback_used:
            prune_summary = f"Schema exploration mode — showing all {columns_after} columns ({total_columns} total)"
            prune_reasoning = (
                "LSH found 0 relevant columns; using full schema for exploratory query"
            )
        elif selected_tables:
            prune_summary = (
                f"Focusing on {', '.join(selected_tables[:3])}{'…' if len(selected_tables) > 3 else ''} "
                f"— {columns_after} of {total_columns} columns are relevant"
            )
            prune_reasoning = f"Pruned {total_columns} → {columns_after} columns"
        else:
            prune_summary = f"Scanned {total_columns} columns"
            prune_reasoning = "No columns selected"

        return {
            "columns_before": total_columns,
            "columns_after": columns_after,
            "selected_tables": selected_tables,
            "pruning_reasoning": prune_reasoning,
            "schema_exploration_mode": fallback_used,
            "pruned_schema": {
                "tables": [
                    {
                        "name": t.name,
                        "columns": [
                            {
                                "name": c.name,
                                "data_type": c.data_type,
                                "is_pk": c.is_primary_key,
                                "is_fk": c.is_foreign_key,
                                "description": c.description,
                                "examples": c.examples or [],
                            }
                            for c in t.columns
                        ],
                    }
                    for t in pruned
                ]
            },
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "prune",
                    "status": "completed",
                    "latency_ms": latency,
                    "columns_after": columns_after,
                    "selected_tables": selected_tables,
                    "summary": prune_summary,
                }
            ],
        }

    return prune_node


def make_retrieve_node(services: Any):
    """Retrieve similar NL→SQL examples from the vector store."""

    async def retrieve_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]

        examples = []
        if services.example_store:
            results = await services.example_store.search(
                nl_query, top_k=services.config.example_retrieval_top_k
            )
            examples = [
                {
                    "nl": r.example.nl_query,
                    "sql": r.example.sql,
                    "similarity": r.similarity,
                    "source": r.example.generator or "trained",
                }
                for r in results
            ]

        latency = int((time.monotonic() - started) * 1000)

        top_sim = examples[0]["similarity"] if examples else 0
        if not examples:
            ex_summary = "No close matches in memory — reasoning fresh"
        elif top_sim > 0.9:
            ex_summary = f"Found a nearly identical past query (similarity {top_sim:.0%})"
        elif len(examples) == 1:
            ex_summary = f"Found 1 related example ({top_sim:.0%} match)"
        else:
            ex_summary = f"Found {len(examples)} related examples — top match {top_sim:.0%}"

        return {
            "similar_examples": examples,
            "example_count": len(examples),
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "retrieve",
                    "status": "completed",
                    "latency_ms": latency,
                    "example_count": len(examples),
                    "summary": ex_summary,
                }
            ],
        }

    return retrieve_node


_DESCRIBE_PATTERNS = (
    "what is this",
    "what are",
    "describe",
    "explain",
    "tell me about",
    "what data",
    "what tables",
    "what columns",
    "what schema",
    "summarize",
    "overview",
    "what does this",
    "list all",
    "show me all",
    "how many tables",
    "what's in",
    "what is in",
    "show me the data",
    "give me an overview",
    "what kind of data",
    "how are",
    "how is the data",
    "what can i",
    "what can you",
    "what information",
    "show me what",
)


def _is_describe_query(nl: str) -> bool:
    """Returns True for schema/describe questions that don't need SQL generation."""
    nl_lower = nl.lower().strip().rstrip("?!")
    return any(nl_lower.startswith(p) or nl_lower == p for p in _DESCRIBE_PATTERNS)


def make_plan_node(services: Any):
    """LLM plans the query approach before generating SQL. Skipped for simple/describe queries."""

    async def plan_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]
        pruned_schema = state.get("pruned_schema", {})
        complexity = state.get("complexity", "moderate")

        # Skip expensive planning LLM call for simple/describe queries
        if complexity == "simple" or complexity == "describe" or _is_describe_query(nl_query):
            return {
                "plan_strategy": "direct",
                "plan_reasoning": "Simple query — direct execution, no planning needed",
                "planned_tables": [],
                "planned_joins": [],
                "planned_filters": [],
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "plan",
                        "status": "completed",
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "tokens": 0,
                        "summary": "Strategy: direct (skipped for simple query)",
                    }
                ],
            }

        # Build schema context for the LLM — include column value hints
        schema_text = ""
        for table in pruned_schema.get("tables", []):
            col_parts = []
            for c in table.get("columns", []):
                part = f"{c['name']} {c.get('data_type', '')}"
                if c.get("is_pk"):
                    part += " PK"
                if c.get("is_fk"):
                    part += " FK"
                if c.get("examples"):
                    ex = ", ".join(f'"{e}"' for e in c["examples"][:4])
                    part += f" [values: {ex}]"
                col_parts.append(part)
            schema_text += f"  {table['name']}({', '.join(col_parts)})\n"

        prompt = (
            f"You are a SQL query planner. Given this schema:\n{schema_text}\n"
            f"Question: {nl_query}\n\n"
            f"IMPORTANT: Column values shown in [values: ...] are actual data values.\n"
            f"When the question asks about specific entity types (countries, cities, products),\n"
            f"use your knowledge to identify which column values are individual entities vs.\n"
            f"aggregate/bucket groupings, then plan to filter out the aggregates.\n\n"
            f"Plan how to write the SQL. Return JSON:\n"
            f'{{"strategy": "direct|join_and_aggregate|subquery|window",'
            f' "reasoning": "step by step plan",'
            f' "tables": ["..."], "joins": ["..."], "filters": ["..."]}}'
        )

        resp = await services.llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )

        import json

        try:
            parsed = json.loads(resp.content)
        except json.JSONDecodeError:
            parsed = {
                "strategy": "direct",
                "reasoning": resp.content,
                "tables": [],
                "joins": [],
                "filters": [],
            }

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)

        return {
            "plan_strategy": parsed.get("strategy", "direct"),
            "plan_reasoning": parsed.get("reasoning", ""),
            "planned_tables": parsed.get("tables", []),
            "planned_joins": parsed.get("joins", []),
            "planned_filters": parsed.get("filters", []),
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "plan",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "strategy": parsed.get("strategy", "direct"),
                    "summary": (
                        f"Joining {', '.join(parsed.get('tables', [])[:3])}"
                        if parsed.get("joins") and parsed.get("tables")
                        else f"Direct approach on {', '.join(parsed.get('tables', [])[:3])}"
                        if parsed.get("tables")
                        else parsed.get("reasoning", "")[:120]
                    ),
                    "thinking": parsed.get("reasoning", "")[:300],
                }
            ],
        }

    return plan_node


def make_generate_node(services: Any):
    """Parallel SQL generation — fires generators based on query complexity."""

    async def generate_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]
        pruned_schema = state.get("pruned_schema", {})
        examples = state.get("similar_examples", [])
        plan = state.get("plan_reasoning", "")
        complexity = state.get("complexity", "moderate")

        # In schema exploration mode the prune fallback returned the full schema.
        # Augment the query so the LLM knows to infer what's available and map
        # the entity (e.g. "MY" / "Malaysia") to the correct column filter.
        schema_exploration = state.get("schema_exploration_mode", False)
        nl_query_for_gen = nl_query
        if schema_exploration:
            # Pull distinct values hint from schema to guide entity mapping
            tables_hint = ", ".join(t["name"] for t in (pruned_schema.get("tables") or []))
            nl_query_for_gen = (
                f"{nl_query}\n\n"
                f"[Context: The schema contains tables: {tables_hint}. "
                f"Infer the correct column filter for any country/entity codes in the question "
                f"(e.g. 'MY' = Malaysia = iso_code filter). "
                f"Generate a comprehensive statistics query using available columns.]"
            )

        # Scale generators by complexity:
        #   simple/describe → 1 generator (fewshot only, no pairwise)
        #   moderate        → 2 generators (fewshot + decompose, lightweight pairwise)
        #   complex         → all 3 generators + full pairwise
        all_gens = services.ensemble._generators
        if complexity in ("simple", "describe") or _is_describe_query(nl_query):
            generators_to_use = all_gens[:1]  # fewshot only
        elif complexity == "moderate":
            generators_to_use = all_gens[:2]  # fewshot + decompose
        else:
            generators_to_use = all_gens  # all 3

        # Run selected generators in parallel
        # Inject workspace-specific learned context (from user corrections)
        data_context_notes = state.get("data_context_notes") or []
        candidates = await services.ensemble.generate(
            nl_query=nl_query_for_gen,
            pruned_schema=pruned_schema,
            examples=examples,
            plan=plan,
            generators_override=generators_to_use,
            context_notes=data_context_notes if data_context_notes else None,
        )

        # Select winner — skip pairwise for single-generator runs
        if len([c for c in candidates if c.get("succeeded")]) <= 1:
            winner = next(
                (c for c in candidates if c.get("succeeded")), candidates[0] if candidates else None
            )
            selection_reasoning = (
                f"Single generator: {winner.get('generator_id', '?') if winner else 'none'}"
            )
        else:
            winner, selection_reasoning = await services.ensemble.select(candidates)

        total_tokens = sum(c.get("tokens_used", 0) for c in candidates)
        total_cost = sum(c.get("cost_usd", 0.0) for c in candidates)
        latency = int((time.monotonic() - started) * 1000)

        return {
            "candidates": candidates,
            "winner": winner,
            "winner_generator": winner.get("generator_id", "") if winner else "",
            "sql": winner.get("sql", "") if winner else "",
            "selection_reasoning": selection_reasoning,
            "generation_tokens": total_tokens,
            "generation_cost_usd": total_cost,
            "tokens_used": state.get("tokens_used", 0) + total_tokens,
            "cost_usd": state.get("cost_usd", 0.0) + total_cost,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "generate",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": total_tokens,
                    "summary": (
                        f"Tried {len(candidates)} approaches — "
                        + (
                            f"went with {winner.get('generator_id', '?').replace('_', ' ')} strategy"
                            if winner
                            else "no winner selected"
                        )
                    )
                    if len(candidates) > 1
                    else (
                        f"Wrote the query using {winner.get('generator_id', '?').replace('_', ' ')} approach"
                        if winner
                        else "SQL generation failed"
                    ),
                }
            ],
        }

    return generate_node


def make_execute_node(services: Any):
    """Execute SQL against the database with policy check."""

    async def execute_node(state: QueryState) -> dict:
        started = time.monotonic()
        sql = state.get("sql", "")
        target_sources = state.get("target_sources", [])

        if not sql:
            return {
                "execution_error": "No SQL to execute",
                "succeeded": False,
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "execute",
                        "status": "failed",
                        "summary": "No SQL to execute",
                    }
                ],
            }

        # Policy check
        if services.policy:
            policy_result = services.policy.check(sql, state)
            if not policy_result.passed:
                return {
                    "execution_error": f"Policy blocked: {policy_result.reason}",
                    "succeeded": False,
                    "trace_events": state.get("trace_events", [])
                    + [
                        {
                            "node": "execute",
                            "status": "failed",
                            "summary": f"Policy blocked: {policy_result.rule_id}",
                        }
                    ],
                }
            if policy_result.modified_sql:
                sql = policy_result.modified_sql

        # Execute against first target source (fall back to any available connector)
        source_id = target_sources[0] if target_sources else None
        if not source_id:
            # No target source set — use first available connector
            available = list(services.connectors.keys())
            source_id = available[0] if available else None
        conn = services.connectors.get(source_id) if source_id else None

        if not conn:
            return {
                "execution_error": f"No connector available (tried: {source_id})",
                "succeeded": False,
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "execute",
                        "status": "failed",
                        "summary": "No connector available",
                    }
                ],
            }

        try:
            result = await conn.execute(sql, timeout_s=services.config.query_timeout_s)
            if hasattr(result, "to_dict"):
                # Replace NaN/Inf with None so result rows are JSON-serializable
                import pandas as _pd

                if isinstance(result, _pd.DataFrame):
                    result = result.where(result.notna(), other=None)
                rows = result.to_dict("records")
            else:
                rows = []
            columns = list(result.columns) if hasattr(result, "columns") else []
            row_count = len(rows)

            latency = int((time.monotonic() - started) * 1000)

            return {
                "sql": sql,
                "rows": rows,
                "columns": columns,
                "row_count": row_count,
                "execution_latency_ms": latency,
                "execution_error": "",
                "succeeded": True,
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "execute",
                        "status": "completed",
                        "latency_ms": latency,
                        "row_count": row_count,
                        "summary": (
                            f"{row_count:,} rows · {latency}ms"
                            if row_count > 0
                            else f"Ran in {latency}ms — no rows matched"
                        ),
                    }
                ],
            }
        except Exception as e:
            latency = int((time.monotonic() - started) * 1000)
            return {
                "execution_error": str(e),
                "succeeded": False,
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "execute",
                        "status": "failed",
                        "latency_ms": latency,
                        "summary": f"Error: {str(e)[:100]}",
                    }
                ],
            }

    return execute_node


def make_correct_node(services: Any):
    """ReFoRCE 3-stage correction loop."""

    async def correct_node(state: QueryState) -> dict:
        started = time.monotonic()
        correction_round = state.get("correction_round", 0) + 1
        error = state.get("execution_error", "")
        original_sql = state.get("sql", "")
        pruned_schema = state.get("pruned_schema", {})

        # Determine correction stage
        if correction_round == 1:
            stage = "error_aware"
            prompt = (
                f"The following SQL failed:\n```sql\n{original_sql}\n```\n"
                f"Error: {error}\n\n"
                f"Fix the SQL. Return only the corrected SQL, no explanation."
            )
        elif correction_round == 2:
            stage = "schema_aware"
            schema_text = ""
            for table in pruned_schema.get("tables", []):
                cols = ", ".join(c["name"] for c in table.get("columns", []))
                schema_text += f"  {table['name']}({cols})\n"
            prompt = (
                f"The following SQL failed:\n```sql\n{original_sql}\n```\n"
                f"Error: {error}\n\n"
                f"The actual schema is:\n{schema_text}\n"
                f"Rewrite the SQL using only these exact column names. Return only SQL."
            )
        else:
            stage = "db_confirmed"
            # In a real implementation, this would query DESCRIBE TABLE
            prompt = (
                f"SQL failed after 2 attempts:\n```sql\n{original_sql}\n```\n"
                f"Error: {error}\n\n"
                f"Rewrite completely. Return only SQL."
            )

        resp = await services.llm.complete([{"role": "user", "content": prompt}])

        # Extract SQL from response
        new_sql = resp.content.strip()
        if "```sql" in new_sql:
            new_sql = new_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in new_sql:
            new_sql = new_sql.split("```")[1].split("```")[0].strip()

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)

        return {
            "sql": new_sql,
            "correction_round": correction_round,
            "correction_stage": stage,
            "execution_error": "",  # Clear error for retry
            "succeeded": False,  # Will be set by execute_node on retry
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "correct",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "stage": stage,
                    "summary": (
                        f"Round {correction_round} — looked at the error more carefully"
                        if stage == "error_aware"
                        else f"Round {correction_round} — cross-checked against the full schema"
                        if stage == "schema_aware"
                        else f"Round {correction_round} — confirmed against the database"
                    ),
                }
            ],
        }

    return correct_node


def make_respond_node(services: Any):
    """Generate NL summary + follow-ups + chart config from results."""

    async def respond_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]
        sql = state.get("sql", "")
        rows = state.get("rows", [])
        row_count = state.get("row_count", 0)
        columns = state.get("columns", [])
        error = state.get("execution_error", "")

        # Detect trivial/fallback SQL that can't answer schema/describe questions
        _sql_upper = sql.strip().upper()
        _is_trivial_sql = _sql_upper in ("SELECT 1", 'SELECT 1 AS "1"', "SELECT 1 AS `1`", "")
        _is_trivial_result = (
            row_count <= 1
            and bool(rows)
            and len(columns) <= 1
            and all(str(v) in ("1", "0", "") for r in rows for v in r.values())
        )

        # For schema/describe queries with trivial SQL: bypass SQL results,
        # use pruned schema to generate a direct NL description.
        if _is_describe_query(nl_query) and (_is_trivial_sql or _is_trivial_result or not rows):
            pruned = state.get("pruned_schema") or state.get("full_schema") or {}
            # Serialise the schema compactly for the LLM
            import json as _json

            schema_lines = []
            if isinstance(pruned, dict):
                for src_id, snap in (
                    pruned.items()
                    if isinstance(list(pruned.values() or [None])[0], dict)
                    else [("data", pruned)]
                ):
                    tables = snap.get("tables", []) if isinstance(snap, dict) else []
                    for t in tables[:20]:
                        cols = ", ".join(
                            f"{c['name']} ({c.get('data_type', '?')})"
                            for c in t.get("columns", [])[:15]
                        )
                        schema_lines.append(
                            f"  {t['name']}: [{cols}]  — {t.get('row_count', 0)} rows"
                        )
            schema_text = "\n".join(schema_lines) if schema_lines else "(schema not available)"
            describe_prompt = (
                f'A user asked: "{nl_query}"\n\n'
                f"You are a data analyst. Based on the schema below, give a clear and useful answer "
                f"describing what the data contains, how the tables relate, and what kinds of analysis are possible.\n\n"
                f"SCHEMA:\n{schema_text}\n\n"
                f"Guidelines:\n"
                f"- Use **bold** for key table and column names\n"
                f"- 3-5 sentences max\n"
                f"- End with 3 follow-up questions the user could ask\n\n"
                f'Return JSON: {{"summary": "...", "follow_ups": ["...", "...", "..."]}}'
            )
            resp = await services.llm.complete(
                [{"role": "user", "content": describe_prompt}],
                json_mode=True,
            )
            try:
                raw = resp.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                parsed = _json.loads(raw)
            except Exception as exc:
                logger.debug("pipeline.node.operation_failed", error=str(exc))
                parsed = {"summary": resp.content, "follow_ups": []}
            tokens = resp.tokens_input + resp.tokens_output
            latency = int((time.monotonic() - started) * 1000)
            return {
                "nl_response": parsed.get("summary", resp.content),
                "follow_ups": parsed.get("follow_ups", []),
                "chart_config": None,
                "sql": "",  # hide the SELECT 1
                "rows": [],
                "columns": [],
                "row_count": 0,
                "succeeded": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "tokens_used": state.get("tokens_used", 0) + tokens,
                "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "respond",
                        "status": "completed",
                        "latency_ms": latency,
                        "tokens": tokens,
                        "summary": "Schema description (no SQL needed)",
                    }
                ],
            }

        if error or not rows:
            # Give the agent the full conversation history so it can answer
            # conversational follow-ups ("what countries did you use?") naturally —
            # even when no new SQL ran successfully.
            history = state.get("conversation_history", [])
            if history:
                history_block = "\n".join(
                    f"{'User' if t.get('role') == 'user' else 'Agent'}: "
                    + (t.get("text") or t.get("nl_response") or "")[:400]
                    + (f"\n  (SQL: {t['sql'][:200]})" if t.get("sql") else "")
                    for t in history[-6:]
                )
                context_prompt = (
                    f"Conversation history:\n{history_block}\n\n"
                    f"Current question: {nl_query}\n\n"
                    "Answer the question using the conversation context above. "
                    "If the question refers to something from a previous query, explain it clearly. "
                    "If no new data was retrieved and the question can't be answered from context, "
                    "say so briefly and suggest what the user could ask instead.\n"
                    "Also suggest 2-3 follow-up questions.\n\n"
                    'Return JSON: {"summary": "...", "follow_ups": ["...", "..."]}'
                )
                ctx_resp = await services.llm.complete(
                    [{"role": "user", "content": context_prompt}],
                    json_mode=True,
                )
                import json as _json2

                try:
                    raw_ctx = ctx_resp.content.strip()
                    if raw_ctx.startswith("```"):
                        raw_ctx = raw_ctx.split("```")[1]
                        if raw_ctx.startswith("json"):
                            raw_ctx = raw_ctx[4:]
                        raw_ctx = raw_ctx.strip()
                    ctx_parsed = _json2.loads(raw_ctx)
                except Exception:
                    ctx_parsed = {"summary": ctx_resp.content, "follow_ups": []}
                ctx_tokens = ctx_resp.tokens_input + ctx_resp.tokens_output
                return {
                    "nl_response": ctx_parsed.get("summary", ctx_resp.content),
                    "follow_ups": ctx_parsed.get("follow_ups", []),
                    "chart_config": None,
                    "succeeded": True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "tokens_used": state.get("tokens_used", 0) + ctx_tokens,
                    "cost_usd": state.get("cost_usd", 0.0) + ctx_resp.cost_usd,
                    "trace_events": state.get("trace_events", [])
                    + [
                        {
                            "node": "respond",
                            "status": "completed",
                            "latency_ms": int((time.monotonic() - started) * 1000),
                            "tokens": ctx_tokens,
                            "summary": "Answered from conversation context",
                        }
                    ],
                }
            # No history and no results — generic fallback
            return {
                "nl_response": "The query returned no results."
                + (f" Error: {error}" if error else ""),
                "follow_ups": [],
                "chart_config": None,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "trace_events": state.get("trace_events", [])
                + [
                    {
                        "node": "respond",
                        "status": "completed",
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "summary": "Error response" if error else "Empty results",
                    }
                ],
            }

        # Build conversation context prefix for the LLM
        history = state.get("conversation_history", [])
        history_prefix = ""
        if history:
            lines = []
            for t in history[-4:]:
                role = "User" if t.get("role") == "user" else "Agent"
                text = (t.get("text") or t.get("nl_response") or "")[:200]
                lines.append(f"{role}: {text}")
            history_prefix = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        # Sample data for the LLM
        sample = rows[:5]
        prompt = (
            f"{history_prefix}"
            f"Question: {nl_query}\n"
            f"SQL: {sql}\n"
            f"Results ({row_count} rows, showing first 5):\n{sample}\n\n"
            f"1. Write a concise natural language answer (2-3 sentences, use **bold** for key numbers).\n"
            f"2. Suggest 3 follow-up questions.\n"
            f"3. Suggest a chart type (bar, line, pie, table, or none).\n\n"
            f'Return JSON: {{"summary": "...", "follow_ups": ["..."], '
            f'"chart_type": "bar|line|pie|table|none"}}'
        )

        resp = await services.llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )

        import json

        try:
            raw = resp.content.strip()
            # Claude often wraps JSON in markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            parsed = {"summary": resp.content, "follow_ups": [], "chart_type": "table"}

        chart_config = None
        chart_type = parsed.get("chart_type", "table")
        if chart_type in ("bar", "line", "pie") and len(columns) >= 2:
            chart_config = {
                "type": chart_type,
                "x_column": columns[0],
                "y_column": columns[-1],
            }

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)

        return {
            "nl_response": parsed.get("summary", resp.content),
            "follow_ups": parsed.get("follow_ups", []),
            "chart_config": chart_config,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "respond",
                    "status": "completed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "summary": (
                        f"Found the answer — {row_count:,} data points, shown as {chart_type}"
                        if chart_type not in ("none", "table", "") and row_count
                        else f"Summarised {row_count:,} rows into a clear answer"
                        if row_count
                        else "Wrote up the findings"
                    ),
                }
            ],
        }

    return respond_node


def make_learn_node(services: Any):
    """Write to episodic memory, update SOUL observation. Non-blocking."""

    async def learn_node(state: QueryState) -> dict:
        started = time.monotonic()

        # Record to episodic memory
        if services.memory_manager and state.get("succeeded"):
            try:
                await services.memory_manager.record_query(
                    user_id=state.get("user_id", ""),
                    nl_query=state["nl_query"],
                    sql=state.get("sql", ""),
                    tables=state.get("selected_tables", []),
                    generator=state.get("winner_generator", ""),
                    succeeded=state.get("succeeded", False),
                )
            except Exception as exc:
                logger.debug(
                    "pipeline.node.operation_failed", error=str(exc)
                )  # SOUL/memory never blocks a query

        # SOUL observation
        if services.soul and state.get("succeeded"):
            try:
                await services.soul.observe(
                    user_id=state.get("user_id", ""),
                    query=state["nl_query"],
                    tables=state.get("selected_tables", []),
                    generator=state.get("winner_generator", ""),
                )
            except Exception as exc:
                logger.debug(
                    "pipeline.node.operation_failed", error=str(exc)
                )  # SOUL never blocks a query

        latency = int((time.monotonic() - started) * 1000)

        return {
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "learn",
                    "status": "completed",
                    "latency_ms": latency,
                    "summary": (
                        "Saved this Q&A pair and updated my understanding of your data"
                        if state.get("succeeded")
                        else "Query didn't succeed — nothing saved"
                    ),
                }
            ],
        }

    return learn_node


def make_decompose_node(services: Any):
    """Split a cross-source query into sub-queries targeting different sources."""

    async def decompose_node(state: QueryState) -> dict:
        started = time.monotonic()
        nl_query = state["nl_query"]
        target_sources = state.get("target_sources", [])

        # Resolve connectors — target_sources may not match all connector keys, so
        # build a superset: (1) explicit target_sources, (2) ALL registered connectors
        all_connector_ids = list(services.connectors.keys())
        candidate_ids = list(
            dict.fromkeys(target_sources + all_connector_ids)
        )  # dedup, preserve order

        source_info = []
        resolved_ids = []
        for sid in candidate_ids:
            conn = services.connectors.get(sid)
            if conn:
                try:
                    snap = await conn.introspect()
                    col_lines = []
                    for t in snap.tables:
                        cols = ", ".join(c.name for c in t.columns[:20])
                        col_lines.append(f"    table '{t.name}': [{cols}]")
                    table_block = "\n".join(col_lines) or "    (no tables)"
                    source_info.append(f"Source id='{sid}' dialect={conn.dialect}:\n{table_block}")
                    resolved_ids.append(sid)
                except Exception as exc:
                    source_info.append(f"Source id='{sid}': (schema unavailable: {exc})")

        prompt = (
            "You are a query decomposer. Your job is to split a question into independent "
            "sub-queries, one per data source, so they can be run in parallel and later joined.\n\n"
            "Available data sources and their schemas:\n" + "\n\n".join(source_info) + "\n\n"
            f"Question: {nl_query}\n\n"
            "Instructions:\n"
            "- Create exactly one sub-query per source that contains relevant data.\n"
            "- Each sub-query 'nl' field must be a plain-English description of what to SELECT from that source.\n"
            "- The sub-query 'source_id' MUST exactly match one of the source ids listed above.\n"
            "- In 'synthesis.join_keys', identify the column(s) shared across sources for joining.\n\n"
            "Return JSON with this exact structure:\n"
            '{"sub_queries": [{"id": "sq_a", "source_id": "<exact source id>", "nl": "...", "expected_columns": ["col1", "col2"]}, ...], '
            '"synthesis": {"join_keys": [{"left": "sq_a.col", "right": "sq_b.col"}], "ordering": "col DESC", "limit": 100}}'
        )

        resp = await services.llm.complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )

        import json

        raw = resp.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Hard fallback: build one sub-query per resolved source automatically
            parsed = {
                "sub_queries": [
                    {"id": f"sq_{i}", "source_id": sid, "nl": nl_query, "expected_columns": []}
                    for i, sid in enumerate(resolved_ids)
                ],
                "synthesis": {},
            }

        # Validate source_ids in sub_queries — replace unknown IDs with nearest match
        valid_ids = set(all_connector_ids)
        for sq in parsed.get("sub_queries", []):
            if sq.get("source_id") not in valid_ids:
                # Try to find a connector ID that contains the sub-query source_id as substring
                sq_sid = sq.get("source_id", "")
                match = next(
                    (cid for cid in all_connector_ids if sq_sid in cid or cid in sq_sid),
                    resolved_ids[0] if resolved_ids else "",
                )
                sq["source_id"] = match

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)
        n = len(parsed.get("sub_queries", []))

        return {
            "decomposition_plan": parsed,
            "sub_queries": parsed.get("sub_queries", []),
            "tokens_used": state.get("tokens_used", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + resp.cost_usd,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "decompose",
                    "status": "completed" if n > 0 else "failed",
                    "latency_ms": latency,
                    "tokens": tokens,
                    "sub_query_count": n,
                    "summary": (
                        f"Your question needs data from {len(resolved_ids)} sources — "
                        f"splitting into {n} queries to run in parallel"
                    )
                    if n > 1
                    else "Mapped to a single source query",
                }
            ],
        }

    return decompose_node


def make_fan_out_node(services: Any):
    """Execute sub-queries in parallel — each has its own generate→execute→reflect loop."""

    async def fan_out_node(state: QueryState) -> dict:
        started = time.monotonic()
        sub_queries = state.get("sub_queries", [])

        async def run_sub(sq: dict) -> dict:
            source_id = sq.get("source_id", "")
            nl = sq.get("nl", "")
            conn = services.connectors.get(source_id)
            if not conn:
                return {
                    "sub_query_id": sq["id"],
                    "succeeded": False,
                    "error": f"No connector: {source_id}",
                }

            snap = await conn.introspect()
            schema_text = "\n".join(
                f"  {t.name}({', '.join(c.name + ' ' + c.data_type for c in t.columns)})"
                for t in snap.tables
            )

            def _extract_sql(text: str) -> str:
                text = text.strip()
                if "```sql" in text:
                    return text.split("```sql")[1].split("```")[0].strip()
                if "```" in text:
                    return text.split("```")[1].split("```")[0].strip()
                return text

            def _exec_result(result):
                import pandas as _pd

                if isinstance(result, _pd.DataFrame):
                    result = result.where(result.notna(), other=None)
                    return result.to_dict("records"), list(result.columns), len(result)
                rows = list(getattr(result, "rows", []))
                cols = list(getattr(result, "columns", []))
                return rows, cols, len(rows)

            SYSTEM = (
                "You are a SQL agent. You reason step-by-step before writing SQL.\n\n"
                "If you are unsure about column values, table structure, or how terms in "
                "the question map to the schema, write an EXPLORATORY query first "
                "(e.g. SELECT DISTINCT col FROM table LIMIT 20, or DESCRIBE table). "
                "You will see the result and can then write the final answer SQL.\n\n"
                "When you have enough context, write the final SQL that directly answers "
                "the question. Return ONLY valid SQL — no explanation, no markdown."
            )

            # Agentic conversation — agent reasons, explores, then answers
            # Each round: agent writes SQL → we execute → feed result back → agent decides next step
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Schema:\n{schema_text}\n\nQuestion: {nl}"},
            ]
            sql = ""
            last_error = ""
            _final_rows, _final_cols, _final_count = [], [], 0

            for attempt in range(5):  # up to 5 turns (explore + answer + 3 corrections)
                resp = await services.llm.complete(messages)
                sql = _extract_sql(resp.content)
                messages.append({"role": "assistant", "content": resp.content})

                try:
                    result = await conn.execute(sql)
                    rows, columns, row_count = _exec_result(result)

                    # If this looks exploratory (few rows, single column) AND we haven't
                    # answered yet, feed the result back so the agent can write the real SQL
                    is_exploratory = (
                        row_count <= 30
                        and len(columns) <= 3
                        and attempt < 4
                        and any(
                            kw in sql.upper()
                            for kw in (
                                "DISTINCT",
                                "DESCRIBE",
                                "SHOW",
                                "PRAGMA",
                                "INFORMATION_SCHEMA",
                            )
                        )
                    )

                    if is_exploratory:
                        import json as _json

                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Exploration result ({row_count} rows):\n"
                                    f"{_json.dumps(rows[:20], default=str)}\n\n"
                                    "Now write the final SQL that answers the original question."
                                ),
                            }
                        )
                        continue  # let agent use this to write the real SQL

                    # Non-exploratory result — this is the answer
                    return {
                        "sub_query_id": sq["id"],
                        "source_id": source_id,
                        "sql": sql,
                        "rows": rows,
                        "columns": columns,
                        "row_count": row_count,
                        "succeeded": True,
                        "error": "",
                        "attempts": attempt + 1,
                    }
                except Exception as e:
                    last_error = str(e)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"That SQL failed with: {last_error}\n\n"
                                "Reason through what went wrong and write corrected SQL."
                            ),
                        }
                    )

            return {
                "sub_query_id": sq["id"],
                "source_id": source_id,
                "sql": sql,
                "succeeded": False,
                "error": f"Failed after {len(messages) // 2} attempts. Last error: {last_error}",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "attempts": 5,
            }

        # Run all sub-queries in parallel
        sub_results = await asyncio.gather(*[run_sub(sq) for sq in sub_queries])
        latency = int((time.monotonic() - started) * 1000)

        return {
            "sub_results": list(sub_results),
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "fan_out",
                    "status": "completed",
                    "latency_ms": latency,
                    "sub_query_count": len(sub_results),
                    "summary": (
                        "All " + str(len(sub_results)) + " sources responded"
                        if all(r.get("succeeded") for r in sub_results)
                        else f"{sum(1 for r in sub_results if r.get('succeeded'))} of {len(sub_results)} sources responded"
                    ),
                    "children": [
                        {
                            "node": f"sub_{r['sub_query_id']}",
                            "status": "completed" if r.get("succeeded") else "failed",
                            "summary": (
                                f"{r.get('row_count', 0):,} rows from {r.get('source_id', '?')}"
                                + (
                                    f" (needed {r.get('attempts', 1)} tries)"
                                    if r.get("attempts", 1) > 1
                                    else ""
                                )
                            ),
                        }
                        for r in sub_results
                    ],
                }
            ],
        }

    return fan_out_node


def make_synthesize_node(services: Any):
    """LLM generates synthesis SQL over DuckDB in-memory tables, self-corrects on error."""

    async def synthesize_node(state: QueryState) -> dict:
        started = time.monotonic()
        sub_results = state.get("sub_results", [])
        nl_query = state.get("nl_query", "")
        plan = state.get("decomposition_plan", {})
        synthesis = plan.get("synthesis", {})

        import duckdb
        import pandas as pd

        con = duckdb.connect(":memory:")

        # Load each sub-result as a named DuckDB table
        table_schemas = []
        for sr in sub_results:
            if sr.get("succeeded") and sr.get("rows"):
                df = pd.DataFrame(sr["rows"])
                con.register(sr["sub_query_id"], df)
                col_desc = ", ".join(f"{c} ({str(df[c].dtype)})" for c in df.columns)
                table_schemas.append(
                    f"  {sr['sub_query_id']}({col_desc})  — {len(df)} rows "
                    f"[source: {sr.get('source_id', '?')}]"
                )

        schema_hint = "\n".join(table_schemas) or "(no tables)"
        join_hint = ""
        if synthesis.get("join_keys"):
            jk = synthesis["join_keys"][0]
            join_hint = f"\nSuggested join key: {jk.get('left', '')} = {jk.get('right', '')}"

        sql = ""
        last_error = ""
        # Generate → execute → reflect loop (up to 3 rounds)
        for attempt in range(3):
            if attempt == 0:
                prompt = (
                    f"You have these in-memory DuckDB tables from parallel sub-queries:\n{schema_hint}"
                    f"{join_hint}\n\n"
                    f"Original question: {nl_query}\n\n"
                    "Write a single DuckDB SQL query that combines these tables to answer the question. "
                    "Use only the table names listed above (do NOT reference original source tables). "
                    "Return ONLY valid DuckDB SQL, no explanation, no markdown."
                )
            else:
                prompt = (
                    f"DuckDB tables available:\n{schema_hint}\n\n"
                    f"Original question: {nl_query}\n\n"
                    f"Previous SQL (attempt {attempt}):\n```sql\n{sql}\n```\n"
                    f"DuckDB error: {last_error}\n\n"
                    "Reason step by step about why the SQL failed, then write a corrected DuckDB SQL query. "
                    "Return ONLY valid DuckDB SQL, no explanation, no markdown."
                )

            resp = await services.llm.complete([{"role": "user", "content": prompt}])
            raw = resp.content.strip()
            if "```sql" in raw:
                sql = raw.split("```sql")[1].split("```")[0].strip()
            elif "```" in raw:
                sql = raw.split("```")[1].split("```")[0].strip()
            else:
                sql = raw

            # Strip trailing semicolons — DuckDB execute() returns None for the
            # empty statement after a semicolon, causing 'NoneType'.fetchdf() error.
            sql = sql.rstrip(";").strip()

            try:
                if not sql:
                    raise ValueError("LLM returned empty SQL — cannot execute")
                cursor = con.execute(sql)
                if cursor is None:
                    raise ValueError(
                        "DuckDB execute() returned None — SQL may be non-SELECT or malformed"
                    )
                result = cursor.fetchdf()
                # Replace NaN/Inf with None so the result is JSON-serializable.
                # Pandas NaN is not valid JSON; json.dumps outputs bare 'NaN' which
                # browsers reject, causing the SSE event to be silently dropped.
                result = result.where(result.notna(), other=None)
                rows = result.to_dict("records")
                columns = list(result.columns)
                row_count = len(rows)
                succeeded = True
                error = ""
                break
            except Exception as e:
                last_error = str(e)
                rows, columns, row_count = [], [], 0
                succeeded = False
                error = last_error
        else:
            # All attempts exhausted
            error = f"Synthesis failed after 3 attempts. Last error: {last_error}"

        con.close()
        latency = int((time.monotonic() - started) * 1000)

        return {
            "synthesis_sql": sql,
            "rows": rows,
            "columns": columns,
            "row_count": row_count,
            "sql": sql,
            "succeeded": succeeded,
            "execution_error": error,
            "trace_events": state.get("trace_events", [])
            + [
                {
                    "node": "synthesize",
                    "status": "completed" if succeeded else "failed",
                    "latency_ms": latency,
                    "row_count": row_count,
                    "summary": (
                        f"Joined everything together — {row_count:,} rows ready to answer your question"
                        if succeeded and row_count
                        else "Merged the results"
                        if succeeded
                        else f"Couldn't combine the results: {error[:80]}"
                    ),
                }
            ],
        }

    return synthesize_node
