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

    async def understand_node(state: QueryState) -> dict:
        nl_query = state["nl_query"]
        source_ids = state.get("source_ids", [])

        # ── Resolve anaphora using conversation history ───────────────────────
        # "these 2", "them", "same countries", "the trend" etc. must be rewritten
        # to explicit entities before routing or pruning — otherwise the agent
        # queries the wrong data and trains on wrong results.
        _hist = state.get("conversation_history", [])
        # Always rewrite when there is history — short follow-ups like "the employment vs
        # unemployment" or "what about trends?" lose their subject without rewriting even
        # when they contain no classic pronoun. Only skip if the query is already long and
        # self-contained (>80 chars with both a subject and a verb — unlikely to be a fragment).
        _needs_resolution = bool(_hist)
        if _needs_resolution:
            try:
                history_context = "\n".join(
                    f"{'User' if t.get('role') == 'user' else 'Agent'}: {t.get('text') or t.get('nl_response') or ''}"
                    for t in _hist[-6:]
                )
                rewrite_resp = await services.llm.complete([
                    {"role": "user", "content": (
                        f"Conversation so far:\n{history_context}\n\n"
                        f"Latest question: {nl_query}\n\n"
                        "Rewrite the latest question to be fully self-contained — replace all pronouns and "
                        "references (e.g. 'these 2', 'them', 'the same', 'the trend') with the explicit "
                        "entities from the conversation. Return ONLY the rewritten question, nothing else."
                    )}
                ])
                rewritten = rewrite_resp.content.strip().strip('"').strip("'")
                if rewritten and len(rewritten) > 5:
                    nl_query = rewritten
            except Exception:
                pass

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

        # ── Single source: LLM classifies intent and compound structure ─────────
        if len(source_ids) <= 1:
            import json as _json

            _hist_ctx = ""
            if _hist:
                _hist_ctx = "\n".join(
                    f"{'User' if t.get('role') == 'user' else 'Agent'}: "
                    + (t.get("text") or t.get("nl_response") or "")[:200]
                    for t in _hist[-4:]
                )

            _route_prompt = (
                f"You are a query router. Classify this question.\n\n"
                + (f"Conversation context:\n{_hist_ctx}\n\n" if _hist_ctx else "")
                + f"Question: {nl_query}\n\n"
                "A question is COMPOUND when it contains two or more DISTINCT analytical questions "
                "that require separate SQL queries to answer — each with its own entities, metrics, "
                "or time frames. Examples:\n"
                "  COMPOUND: 'How did Malaysia perform vs ASEAN, and also show PHP vs VNT currency trends'\n"
                "  COMPOUND: 'Compare store revenue alongside employee headcount growth'\n"
                "  COMPOUND: 'Breakdown sales by region, then show top 5 products by margin'\n"
                "  NOT COMPOUND: 'Compare revenue and profit by store'  ← one query, two metrics\n"
                "  NOT COMPOUND: 'Show top 10 countries by GDP'  ← one question\n"
                "  NOT COMPOUND: 'What is the trend for Malaysia?'  ← one question\n\n"
                "Also extract any specific entity names (countries, products, regions, codes) "
                "the question wants to filter for. If it asks for aggregates like 'top 5' with no "
                "specific names, return empty list.\n\n"
                'Return JSON: {"is_compound": true|false, "reasoning": "...", '
                '"complexity": "simple|moderate|complex", '
                '"entity_filters": ["EntityA", "EntityB"]}'
            )

            try:
                _route_resp = await services.llm.complete(
                    [{"role": "user", "content": _route_prompt}],
                    json_mode=True,
                )
                _raw = _route_resp.content.strip()
                if _raw.startswith("```"):
                    _raw = _raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
                _route_parsed = _json.loads(_raw)
            except Exception:
                _route_parsed = {"is_compound": False, "reasoning": "parse error", "complexity": "moderate", "entity_filters": []}

            _is_compound = bool(_route_parsed.get("is_compound", False))
            _complexity = _route_parsed.get("complexity", "moderate")
            _entity_filters = _route_parsed.get("entity_filters", [])
            if not isinstance(_entity_filters, list):
                _entity_filters = []
            _route_tokens = getattr(_route_resp, "tokens_input", 0) + getattr(_route_resp, "tokens_output", 0)
            _route_cost = getattr(_route_resp, "cost_usd", 0.0)

            return {
                "query_id": query_id,
                "nl_query": nl_query,
                "is_cross_source": False,
                "is_compound_query": _is_compound,
                "target_sources": source_ids,
                "complexity": _complexity,
                "routing_reasoning": _route_parsed.get("reasoning", ""),
                "entity_filters": _entity_filters,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "correction_round": 0,
                "max_corrections": services.config.max_corrections,
                "tokens_used": _route_tokens,
                "cost_usd": _route_cost,
                "budget_exhausted": False,
                "nl_query_for_pruning": nl_query,
                "trace_events": _trace(
                    "Compound query — splitting into parallel sub-questions"
                    if _is_compound
                    else f"Using {source_ids[0] if source_ids else 'no'} source",
                    tokens=_route_tokens,
                ),
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
            "Answer:\n"
            "1. Which sources are needed and is this cross-source?\n"
            "2. What specific entity names (country names, company names, dates, etc.) does the\n"
            "   question want to filter for? Copy them exactly as written. If none (e.g. 'top 5\n"
            "   countries', 'all data'), return empty list.\n"
            '   Examples: "compare Malaysia vs Vietnam" → ["Malaysia", "Vietnam"]\n'
            '             "Indonesia and Thailand rates" → ["Indonesia", "Thailand"]\n'
            '             "Narnia trends" → ["Narnia"]\n'
            '             "top 5 countries" → []\n'
            'Return JSON: {"target_sources": [...], "is_cross_source": bool, '
            '"complexity": "simple|moderate|complex", "reasoning": "...", '
            '"entity_filters": ["EntityA", "EntityB"]}'
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
                "entity_filters": [],
            }

        entity_filters = parsed.get("entity_filters", [])
        if not isinstance(entity_filters, list):
            entity_filters = []

        latency = int((time.monotonic() - started) * 1000)
        return {
            "query_id": query_id,
            "nl_query": nl_query,  # ISO codes expanded to full names for SQL generation
            "nl_query_for_pruning": nl_query,
            "intent": "data_query",
            "is_cross_source": parsed.get("is_cross_source", False),
            "target_sources": parsed.get("target_sources") or source_ids,
            "complexity": parsed.get("complexity", "moderate"),
            "routing_reasoning": parsed.get("reasoning", ""),
            "entity_filters": entity_filters,
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

        # Include ALL connectors' tables when they're all DuckDB (same workspace)
        # This ensures JOINs between occupazione + disoccupazione work
        all_duckdb = all(
            getattr(c, 'dialect', '') == 'duckdb' or 'file_' in sid
            for sid, c in services.connectors.items()
        ) if services.connectors else False

        sources_to_scan = (
            list(services.connectors.keys()) if all_duckdb
            else target_sources
        )

        for sid in sources_to_scan:
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

        # ── Semantic entity substitution ─────────────────────────────────────
        # Replace user abbreviations/typos in the NL query with the actual
        # stored values resolved by the semantic agent (e.g. PHP→PHL, VNT→VNM).
        # This prevents the SQL generator from deciding "PHP doesn't exist"
        # before it even reads the semantic context hints.
        sem_res = state.get("semantic_resolution")
        # Flatten entity_map — may be nested by source_id or flat at top level
        entity_map = {}
        if sem_res:
            if isinstance(sem_res, dict):
                if "entity_map" in sem_res:
                    entity_map = sem_res["entity_map"]
                else:
                    # Nested by source_id: {source_id: {entity_map: {...}}}
                    for _sid_val in sem_res.values():
                        if isinstance(_sid_val, dict) and "entity_map" in _sid_val:
                            entity_map.update(_sid_val["entity_map"])
        if entity_map:
            import re as _re
            substituted = nl_query_for_gen
            # Sort by length descending so longer keys are replaced first
            for user_term, stored_val in sorted(entity_map.items(), key=lambda x: -len(x[0])):
                if user_term != stored_val:
                    # Word-boundary replacement — avoid replacing inside longer words
                    substituted = _re.sub(
                        r'\b' + _re.escape(user_term) + r'\b',
                        stored_val,
                        substituted,
                        flags=_re.IGNORECASE,
                    )
            nl_query_for_gen = substituted

            # Also inject structured resolution hint into the generation prompt
            resolution_hints = []
            for user_term, stored_val in entity_map.items():
                if user_term != stored_val:
                    resolution_hints.append(f'  "{user_term}" → use value \'{stored_val}\' in SQL WHERE clause')
            if resolution_hints:
                nl_query_for_gen += (
                    "\n\n[SEMANTIC RESOLUTION — use these exact values in your SQL:\n"
                    + "\n".join(resolution_hints)
                    + "\nDo NOT use the original user terms — use the resolved values above.]"
                )

        # ── Build entity_map from Semantic Reasoning Agent ─────────────────
        # Use BOTH new_aliases AND filter values to substitute in the NL query.
        # This ensures the SQL Agent sees actual DB values, not user abbreviations.
        sem_reasoning = state.get("semantic_reasoning")
        if sem_reasoning and not entity_map:
            import re as _re
            # Collect all mappings: from new_aliases + from filter values
            all_mappings = dict(sem_reasoning.get("new_aliases", {}))
            # Also extract entity names from filters
            for f in sem_reasoning.get("filters", []):
                reasoning = f.get("reasoning", "")
                val = f.get("value", "")
                # The filter reasoning often says "X means Y" — extract the user term
                # But we can also use the Ora node's entity_filters

            if all_mappings:
                entity_map = all_mappings
                substituted = nl_query_for_gen
                for user_term, stored_val in sorted(all_mappings.items(), key=lambda x: -len(x[0])):
                    if user_term != stored_val and isinstance(user_term, str) and isinstance(stored_val, str):
                        substituted = _re.sub(
                            r'\b' + _re.escape(user_term) + r'\b',
                            stored_val,
                            substituted,
                            flags=_re.IGNORECASE,
                        )
                nl_query_for_gen = substituted

        # ── Ora assembles the FINAL refined query for the SQL Agent ────────
        # The SQL Agent should ONLY see the refined query with exact DB values.
        # It should NEVER see the raw user terms (CP group, TH, VNT, etc.)
        # Ora is the orchestrator — it passes only the resolved output downstream.
        if sem_reasoning and sem_reasoning.get("filters"):
            where_parts = []
            for f in sem_reasoning["filters"]:
                col = f.get("column", "")
                op = f.get("operator", "=")
                val = f.get("value", "")
                if isinstance(val, list):
                    val_str = ", ".join(f"'{v}'" for v in val)
                    where_parts.append(f"{col} IN ({val_str})")
                elif isinstance(val, str) and val:
                    where_parts.append(f"{col} {op} '{val}'")

            if where_parts:
                resolved = sem_reasoning.get("resolved_query", nl_query_for_gen)
                tables = sem_reasoning.get("tables", [])
                metrics = sem_reasoning.get("metrics", [])

                # REPLACE the query entirely — SQL Agent never sees raw user terms
                nl_query_for_gen = (
                    f"{resolved}\n\n"
                    f"SQL WHERE clause MUST include: {' AND '.join(where_parts)}\n"
                    + (f"Include these columns: {', '.join(metrics)}\n" if metrics else "")
                    + (f"Query from: {', '.join(tables)}\n" if tables else "")
                )

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
            available = list(services.connectors.keys())
            source_id = available[0] if available else None
        conn = services.connectors.get(source_id) if source_id else None

        # ── Multi-table DuckDB fix: register all file tables in one connection ──
        # When SQL references tables from multiple file sources (e.g. JOIN
        # occupazione o ON ... JOIN disoccupazione d ON ...), each file source
        # has its own DuckDB connection with only one table. We need to register
        # ALL tables into the executing connector's DuckDB instance.
        if conn and hasattr(conn, '_conn') and conn._conn is not None:
            for other_sid, other_conn in services.connectors.items():
                if other_sid != source_id and hasattr(other_conn, '_conn') and other_conn._conn is not None:
                    try:
                        # Check what tables the other connector has
                        other_snap = await other_conn.introspect()
                        for other_table in other_snap.tables:
                            # Check if this table exists in our connection
                            try:
                                conn._conn.execute(f'SELECT 1 FROM "{other_table.name}" LIMIT 0')
                            except Exception:
                                # Table doesn't exist — register it from the other connection
                                try:
                                    df = other_conn._conn.execute(
                                        f'SELECT * FROM "{other_table.name}"'
                                    ).fetchdf()
                                    conn._conn.register(other_table.name, df)
                                    logger.info(
                                        "execute.registered_cross_table",
                                        table=other_table.name,
                                        from_source=other_sid,
                                        into_source=source_id,
                                    )
                                except Exception as _reg_err:
                                    logger.debug(
                                        "execute.register_failed",
                                        table=other_table.name,
                                        error=str(_reg_err),
                                    )
                    except Exception:
                        pass

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


def make_validate_node(services: Any):
    """Ora validation — checks if the result actually answers the user's question.

    If the result is empty or doesn't match the question intent, Ora can
    re-route back to the Semantic Agent or SQL Agent with feedback.
    """

    async def validate_node(state: QueryState) -> dict:
        nl_query = state["nl_query"]
        sql = state.get("sql", "")
        rows = state.get("rows", [])
        row_count = state.get("row_count", 0)
        error = state.get("execution_error", "")
        semantic_reasoning = state.get("semantic_reasoning")
        correction_round = state.get("correction_round", 0)

        # If execution succeeded with data, validate
        if not error and row_count > 0:
            # Quick structural check — if semantic reasoning specified filters,
            # verify the result actually contains the expected entities
            if semantic_reasoning and semantic_reasoning.get("filters"):
                expected_values = []
                for f in semantic_reasoning["filters"]:
                    val = f.get("value", "")
                    if isinstance(val, str) and val:
                        expected_values.append(val.lower())
                    elif isinstance(val, list):
                        expected_values.extend(v.lower() for v in val if isinstance(v, str))

                # Check if any expected entity appears in the result rows
                if expected_values and rows:
                    found_any = False
                    for row in rows[:20]:
                        for v in row.values():
                            if isinstance(v, str) and v.lower() in expected_values:
                                found_any = True
                                break
                        if found_any:
                            break

                    if not found_any and correction_round < 2:
                        # Result doesn't contain expected entities — flag for retry
                        logger.warning(
                            "validate.mismatch",
                            expected=expected_values[:5],
                            got_rows=row_count,
                        )
                        return {
                            "execution_error": (
                                f"Validation: result has {row_count} rows but none contain "
                                f"the expected entities ({', '.join(expected_values[:3])}). "
                                f"The SQL WHERE clause may be using wrong values. "
                                f"Use the EXACT values from the SEMANTIC REASONING section."
                            ),
                            "correction_round": correction_round,
                            "trace_events": state.get("trace_events", []) + [{
                                "node": "validate",
                                "status": "retry",
                                "summary": f"Result doesn't contain expected entities — retrying",
                            }],
                        }

            # Passed validation
            return {
                "trace_events": state.get("trace_events", []) + [{
                    "node": "validate",
                    "status": "completed",
                    "summary": f"Validated {row_count} rows — result matches question",
                }],
            }

        # Empty result or error — pass through (respond node will handle)
        return {
            "trace_events": state.get("trace_events", []) + [{
                "node": "validate",
                "status": "completed",
                "summary": f"{'Empty result' if not error else 'Error'} — passing to respond",
            }],
        }

    return validate_node


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
                # Include schema so the LLM can reason about what went wrong
                # (e.g. "country column stores 'Malaysia' not 'MY'")
                _pruned = state.get("pruned_schema") or state.get("full_schema") or {}
                _schema_lines = []
                if isinstance(_pruned, dict):
                    for _src_id, _snap in (
                        _pruned.items()
                        if _pruned and isinstance(list(_pruned.values())[0], dict)
                        else [("data", _pruned)]
                    ):
                        _tables = _snap.get("tables", []) if isinstance(_snap, dict) else []
                        for _t in _tables[:10]:
                            _cols = ", ".join(
                                c["name"] for c in _t.get("columns", [])[:12]
                            )
                            _schema_lines.append(f"  {_t['name']}: [{_cols}]")
                _schema_text = "\n".join(_schema_lines) if _schema_lines else ""

                _error_block = f"\nExecution error: {error}\n" if error else ""

                context_prompt = (
                    f"Conversation history:\n{history_block}\n\n"
                    + (_schema_text and f"Available schema:\n{_schema_text}\n\n" or "")
                    + (_error_block or "")
                    + f"Current question: {nl_query}\n\n"
                    "You are a data analyst with access to the schema above. "
                    "Answer the question using conversation context. "
                    "If the query failed, explain WHY based on the error and schema "
                    "(e.g. 'The country column stores full names like Malaysia, not ISO codes like MY — "
                    "try rephrasing with the full name'). "
                    "Never say you don't have schema access — you do.\n"
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

        # Sample data for the LLM — show enough rows to cover all entities
        # Previously only showed 5 rows, causing the LLM to miss data
        # (e.g. Vietnam data in rows 7-12 was invisible)
        if row_count <= 20:
            sample = rows  # Show all rows if small enough
        else:
            # Show first 10 + a summary of unique values per text column
            sample = rows[:10]

        # ── Compound query: structure the answer by sub-question part ──────────
        # When synthesize_node used UNION ALL, rows include a 'sub_question' column
        # (values like 'sq_a', 'sq_b'). Tell the LLM to narrate each part separately.
        is_compound = state.get("is_compound_query", False)
        sub_question_col = "sub_question" if "sub_question" in columns else None

        # Surface data_warnings from Ora in the response
        data_warnings = state.get("data_warnings") or []
        warnings_prefix = ""
        if data_warnings:
            warnings_prefix = (
                "DATA WARNINGS (share these with the user upfront):\n"
                + "\n".join(f"  ⚠ {w}" for w in data_warnings[:3])
                + "\n\n"
            )

        if is_compound and sub_question_col:
            # Build a condensed per-part sample so the LLM can narrate each part
            import collections as _collections
            parts_data: dict = _collections.defaultdict(list)
            for r in rows:
                label = r.get(sub_question_col, "part")
                parts_data[label].append(r)
            parts_block = ""
            for label, part_rows in parts_data.items():
                parts_block += f"\n[{label}] — {len(part_rows)} rows, sample: {part_rows[:3]}\n"

            # Map sub_query IDs to the sub-query NL descriptions so the LLM
            # can say "Part 1: Malaysia vs ASEAN" instead of "sq_a"
            sub_queries = state.get("sub_queries", [])
            sq_label_map = {sq.get("id", ""): sq.get("nl", "")[:120] for sq in sub_queries}

            sq_legend = "\n".join(
                f"  {label}: {sq_label_map.get(label, label)}"
                for label in parts_data
            )

            prompt = (
                f"{history_prefix}"
                f"{warnings_prefix}"
                f"Question: {nl_query}\n\n"
                f"This was a multi-part question. Results are split by sub-question:\n"
                f"{sq_legend}\n\n"
                f"Data per part:{parts_block}\n"
                f"Instructions:\n"
                f"1. Answer each part separately with a clear heading (e.g. '**Part 1: Malaysia vs ASEAN**').\n"
                f"2. Use **bold** for key numbers. 2-3 sentences per part.\n"
                f"3. End with 3 follow-up questions that span both parts.\n"
                f"4. Suggest a chart type (bar, line, pie, table, or none).\n\n"
                f'Return JSON: {{"summary": "...", "follow_ups": ["..."], "chart_type": "bar|line|pie|table|none"}}'
            )
        else:
            prompt = (
                f"{history_prefix}"
                f"{warnings_prefix}"
                f"Question: {nl_query}\n"
                f"SQL: {sql}\n"
                f"Results ({row_count} rows{', ALL shown' if row_count <= 20 else ', showing first 10'}):\n{sample}\n\n"
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

        # ── Confidence scoring (LLM-assessed) ────────────────────────────
        confidence_data = None
        try:
            from sqlagent.confidence import score_confidence
            confidence_result = await score_confidence(
                question=nl_query,
                sql=sql,
                row_count=row_count,
                corrections=state.get("correction_round", 0),
                error=error,
                semantic_reasoning=state.get("semantic_reasoning"),
                llm=services.llm,
            )
            confidence_data = confidence_result.to_dict()
        except Exception as _conf_err:
            logger.warning("confidence.failed", error=str(_conf_err))

        # ── Auto-visualization (LLM-selected chart) ──────────────────────
        viz_data = None
        if row_count > 0 and columns:
            try:
                from sqlagent.visualization import generate_chart
                viz_result = await generate_chart(
                    question=nl_query, sql=sql,
                    columns=columns, rows=rows[:10],
                    row_count=row_count, llm=services.llm,
                )
                if viz_result.chart_type != "table":
                    viz_data = viz_result.to_dict()
                    # Update chart_config with LLM-selected type
                    chart_config = {
                        "type": viz_result.chart_type,
                        "vega_lite": viz_result.vega_lite,
                        "title": viz_result.title,
                        "description": viz_result.description,
                    }
            except Exception as _viz_err:
                logger.warning("visualization.failed", error=str(_viz_err))

        return {
            "nl_response": parsed.get("summary", resp.content),
            "follow_ups": parsed.get("follow_ups", []),
            "chart_config": chart_config,
            "confidence": confidence_data,
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


def make_semantic_resolve_node(services: Any):
    """Semantic reasoning node — runs between understand and prune/decompose.

    Cache hit (cosine >= 0.85): injects SemanticResolution from cache, no LLM call.
    Cache miss: fires SemanticReasoningAgent (one LLM call) against actual schema+samples.
    Resolution is injected into generate_node as context_notes.
    Only saved to cache after a successful, non-thumbs-down query (in learn_node).
    """

    async def semantic_resolve_node(state: QueryState) -> dict:
        import time as _time
        started = _time.monotonic()

        nl_query = state.get("nl_query", "")
        target_sources = state.get("target_sources") or state.get("source_ids", [])
        # Use first source — for cross-source queries each sub-query gets its own resolution
        source_id = target_sources[0] if target_sources else ""

        _trace_events = state.get("trace_events", [])

        # ── Try cache first ───────────────────────────────────────────────────
        cache_hit = False
        resolution = None

        if hasattr(services, "_semantic_cache") and services._semantic_cache:
            resolution = await services._semantic_cache.get(nl_query, source_id)
            if resolution:
                cache_hit = True
                logger.info(
                    "semantic_resolve.cache_hit",
                    nl_query=nl_query[:60],
                    resolution_id=resolution.resolution_id,
                )

        # ── Cache miss: run the reasoning agent ───────────────────────────────
        if not resolution:
            # Get the schema for this source (with sample values)
            schema = {}
            conn = services.connectors.get(source_id)
            if conn:
                try:
                    snap = await conn.introspect()
                    schema = snap.to_dict() if hasattr(snap, "to_dict") else {}
                    # Ensure sample values are present (use first source's pruned or full schema)
                    if not schema:
                        schema = state.get("full_schema", {}).get(source_id, {})
                except Exception:
                    schema = state.get("full_schema", {}).get(source_id, {})

            from sqlagent.semantic_layer import run_semantic_agent
            resolution = await run_semantic_agent(
                nl_query=nl_query,
                source_id=source_id,
                schema=schema,
                llm=services.llm,
            )

        latency = int((_time.monotonic() - started) * 1000)

        # ── Build context block to inject into generation ─────────────────────
        context_block = resolution.to_context_block() if resolution else ""
        existing_notes = list(state.get("data_context_notes") or [])
        if context_block and context_block not in existing_notes:
            existing_notes.insert(0, context_block)  # semantic context goes first

        # ── Rewrite nl_query with resolved entity substitutions ───────────────
        # Apply entity_map to nl_query globally so ALL downstream nodes
        # (decompose, fan_out, synthesize) see "MYS" not "MY", "PHL" not "PHP".
        import re as _re
        nl_query_resolved = state.get("nl_query", "")
        entity_map = (resolution.entity_map if resolution else {}) or {}

        # ── Novel entity fallback: scan live schema for unresolved short tokens ──
        # Any 2-4 char uppercase token NOT in entity_map may be an unknown ISO/code.
        # Try a live LIKE scan against the first available connector so future nodes
        # receive a resolved value (e.g. "GH" → "Ghana") rather than the raw code.
        if source_id and hasattr(services, "connectors"):
            _conn_live = services.connectors.get(source_id)
            if _conn_live:
                try:
                    _snap_live = await _conn_live.introspect()
                    # Find text columns that look like entity name or code columns
                    _entity_cols = []
                    for _t in _snap_live.tables:
                        for _c in _t.columns:
                            if any(kw in _c.name.lower() for kw in ("country", "name", "code", "entity", "region", "iso")):
                                _entity_cols.append((_t.name, _c.name))
                    # Extract short uppercase tokens from the query that are not mapped yet
                    _unmapped = [
                        tok for tok in _re.findall(r'\b[A-Z]{2,4}\b', nl_query_resolved)
                        if tok not in entity_map and tok.lower() not in {k.lower() for k in entity_map}
                        and tok not in ("AND", "OR", "NOT", "IN", "BY", "FROM", "SQL", "ISO")
                    ]
                    for _tok in _unmapped[:6]:  # limit live queries
                        for _tname, _cname in _entity_cols[:4]:
                            try:
                                _r = await _conn_live.execute(
                                    f'SELECT DISTINCT "{_cname}" FROM "{_tname}" '
                                    f'WHERE UPPER("{_cname}") = {repr(_tok.upper())} '
                                    f'OR LOWER("{_cname}") LIKE {repr("%" + _tok.lower() + "%")} LIMIT 3'
                                )
                                _rrows = _r.to_dict("records") if hasattr(_r, "to_dict") else []
                                if _rrows:
                                    _found = str(list(_rrows[0].values())[0])
                                    # Only trust if it's a close match (token appears in found value)
                                    if _tok.lower()[:3] in _found.lower() or _found.upper() == _tok.upper():
                                        entity_map[_tok] = _found
                                        break
                            except Exception:
                                pass
                except Exception:
                    pass

        if entity_map:
            for user_term, stored_val in sorted(entity_map.items(), key=lambda x: -len(x[0])):
                if user_term != stored_val:
                    nl_query_resolved = _re.sub(
                        r'\b' + _re.escape(user_term) + r'\b',
                        stored_val,
                        nl_query_resolved,
                        flags=_re.IGNORECASE,
                    )

        return {
            "nl_query": nl_query_resolved,
            "semantic_resolution": resolution.to_dict() if resolution else None,
            "semantic_cache_hit": cache_hit,
            "data_context_notes": existing_notes,
            "trace_events": _trace_events + [{
                "node": "semantic_resolve",
                "status": "completed",
                "latency_ms": latency,
                "summary": (
                    f"Cache hit — reused known mappings"
                    if cache_hit else
                    f"Reasoned: {len(resolution.entity_map)} entity maps, "
                    f"{len(resolution.synonyms)} synonyms"
                    if resolution else "No semantic context"
                ),
                "cache_hit": cache_hit,
                "entity_map": resolution.entity_map if resolution else {},
                "synonyms": resolution.synonyms if resolution else {},
            }],
        }

    return semantic_resolve_node


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

        # Save semantic resolution to cache — only on success (no thumbs-down checked server-side)
        if (
            state.get("succeeded")
            and state.get("semantic_resolution")
            and not state.get("semantic_cache_hit")  # don't re-save what was already cached
            and hasattr(services, "_semantic_cache")
            and services._semantic_cache
        ):
            try:
                from sqlagent.semantic_layer import SemanticResolution
                resolution = SemanticResolution.from_dict(state["semantic_resolution"])
                await services._semantic_cache.save(resolution)
            except Exception as exc:
                logger.debug("pipeline.node.operation_failed", error=str(exc))

        # Persist entity discoveries to SemanticMemory for cross-session learning.
        # After a successful query we know which entities exist in which sources.
        semantic_memory = getattr(services, "_semantic_memory", None)
        if state.get("succeeded") and semantic_memory:
            # Save entity values discovered during fan_out sub-queries
            sub_results = state.get("sub_results", [])
            for sr in sub_results:
                if sr.get("succeeded") and sr.get("rows") and sr.get("source_id"):
                    sid = sr["source_id"]
                    # Check entity_filters used for this sub-query
                    sub_queries = state.get("sub_queries", [])
                    sq_match = next(
                        (sq for sq in sub_queries if sq.get("id") == sr.get("sub_query_id")), {}
                    )
                    used_filters = sq_match.get("entity_filters", [])
                    if used_filters:
                        try:
                            # Save column from first column in result rows
                            cols = sr.get("columns", [])
                            if cols:
                                await semantic_memory.save_discovered_entities(
                                    sid, "", cols[0], used_filters
                                )
                        except Exception as exc:
                            logger.debug("learn.semantic_memory_save_failed", error=str(exc))

            # Save concept→column mappings from the resolved semantic context
            sem_res_dict = state.get("semantic_resolution")
            if sem_res_dict and sem_res_dict.get("synonyms"):
                target_sources = state.get("target_sources", [])
                source_id = target_sources[0] if target_sources else ""
                if source_id:
                    for concept, col in sem_res_dict["synonyms"].items():
                        try:
                            await semantic_memory.save_concept_column(source_id, concept, col)
                        except Exception as exc:
                            logger.debug("learn.semantic_memory_concept_failed", error=str(exc))

        # ── Evolve semantic layer from successful queries ──────────────────
        # Persists ALL discoveries: aliases, relationships, patterns, enrichments.
        # This is how the semantic layer gets smarter over time.
        if state.get("succeeded") and state.get("workspace_id"):
            try:
                from sqlagent.semantic_agent import evolve_semantic_layer, strengthen_alias

                # Evolve — saves relationships, patterns, enrichments
                evolve_semantic_layer(
                    workspace_id=state["workspace_id"],
                    query_result={
                        "semantic_reasoning": state.get("semantic_reasoning"),
                        "target_sources": state.get("target_sources", []),
                        "sql": state.get("sql", ""),
                        "nl_query": state["nl_query"],
                        "succeeded": state.get("succeeded", False),
                    },
                )

                # Also strengthen individual aliases
                sem_r = state.get("semantic_reasoning") or {}
                if sem_r.get("new_aliases"):
                    target_sources = state.get("target_sources", [])
                    for alias, canonical in sem_r["new_aliases"].items():
                        for sid in target_sources:
                            strengthen_alias(state["workspace_id"], sid, alias, canonical)
            except Exception as _evolve_err:
                logger.debug("learn.evolve_failed", error=str(_evolve_err))

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


# ── Known group-name → member expansion ──────────────────────────────────────
# Group expansions removed — NO hardcoded dictionaries.
# The Semantic Reasoning Agent handles group names (ASEAN, G7, EU, etc.)
# by looking at the actual data values in the database and reasoning about
# which entities belong to the group. This is fully agentic.
_GROUP_EXPANSIONS: dict[str, list[str]] = {}


def _expand_group_entities(filters: list[str]) -> list[str]:
    """Expand any group-name filters to individual member names.

    Passes through non-group filters unchanged.
    e.g. ["ASEAN", "India"] → ["Malaysia", "Indonesia", ..., "India"]
    """
    expanded = []
    for f in filters:
        members = _GROUP_EXPANSIONS.get(f.lower())
        if members:
            expanded.extend(members)
        else:
            expanded.append(f)
    return expanded




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
        # col_name → {sid: (distinct_count, total_rows)} for cardinality analysis
        col_cardinality: dict = {}

        for sid in candidate_ids:
            conn = services.connectors.get(sid)
            if conn:
                try:
                    snap = await conn.introspect()
                    col_lines = []
                    for t in snap.tables:
                        total_rows = t.row_count_estimate or 0
                        col_parts = []
                        for c in t.columns[:20]:
                            # Sample cardinality for columns that might be join keys
                            distinct = None
                            try:
                                res = await conn.execute(
                                    f'SELECT COUNT(DISTINCT "{c.name}") FROM "{t.name}"'
                                )
                                rows_r = getattr(res, "rows", None) or (
                                    res.to_dict("records") if hasattr(res, "to_dict") else []
                                )
                                if rows_r:
                                    distinct = list(rows_r[0].values())[0]
                            except Exception:
                                pass

                            cardinality_note = ""
                            sample_values_note = ""
                            if distinct is not None and total_rows > 0:
                                ratio = int(distinct) / total_rows
                                if ratio < 0.05:
                                    cardinality_note = f" [dimension: {distinct} distinct / {total_rows} rows — NOT a unique key, must GROUP BY before joining]"
                                elif ratio >= 0.95:
                                    cardinality_note = f" [unique-ish: {distinct}/{total_rows}]"
                                else:
                                    cardinality_note = f" [{distinct} distinct / {total_rows} rows]"
                                if c.name not in col_cardinality:
                                    col_cardinality[c.name] = {}
                                col_cardinality[c.name][sid] = (int(distinct), total_rows)

                                # Sample values are NOT shown in decompose schema —
                                # they cause LLM to use schema values instead of question entities.
                                # Fan_out handles entity resolution via its own discovery loop.

                            col_parts.append(f"{c.name} {c.data_type}{cardinality_note}{sample_values_note}")

                        col_lines.append(f"    table '{t.name}' ({total_rows} rows):\n      " + "\n      ".join(col_parts))
                    table_block = "\n".join(col_lines) or "    (no tables)"
                    source_info.append(f"Source id='{sid}' dialect={conn.dialect}:\n{table_block}")
                    resolved_ids.append(sid)
                except Exception as exc:
                    source_info.append(f"Source id='{sid}': (schema unavailable: {exc})")

        # Identify shared columns across sources (candidates for join keys)
        shared_cols = [
            col for col, sids in col_cardinality.items() if len(sids) > 1
        ]
        shared_note = ""
        if shared_cols:
            shared_note = (
                f"\nShared columns across sources (potential join keys): {', '.join(shared_cols)}\n"
                "IMPORTANT: If a shared column has cardinality << total rows (marked 'dimension'), "
                "it is NOT a unique join key. The sub-queries MUST aggregate (GROUP BY) to produce "
                "one row per unique combination of join dimensions before synthesis can JOIN them. "
                "Failure to aggregate results in a cartesian explosion.\n"
            )

        is_compound = state.get("is_compound_query", False) and not state.get("is_cross_source", False)

        is_compound = state.get("is_compound_query", False) and not state.get("is_cross_source", False)

        _sub_query_schema = (
            '{"id": "sq_a", "source_id": "<exact source id>", "nl": "description", '
            '"entity_filters": ["entities FOR THIS sub-question only"], '
            '"expected_columns": ["col1", "col2"]}'
        )

        if is_compound:
            prompt = (
                "You are a query planner. This is a COMPOUND question — multiple distinct analytical\n"
                "questions asked in a single sentence, all against the SAME data source.\n\n"
                "Available data source with schema and cardinality:\n" + "\n\n".join(source_info) + "\n\n"
                f"Question: {nl_query}\n\n"
                "Instructions:\n"
                "- Identify each distinct analytical sub-question within the compound query.\n"
                "  (e.g. 'compare MY vs ASEAN' is one question; 'deep dive PHP vs VNT trends' is another)\n"
                "- Create one sub-query per distinct analytical question.\n"
                "- ALL sub-queries use the SAME source_id (the single data source above).\n"
                "- CRITICAL: each sub-query has its OWN 'entity_filters' — ONLY the entities relevant\n"
                "  to THAT specific sub-question. Do NOT put all entities in every sub-query.\n"
                "  Example: 'MY vs ASEAN and then PHP vs VNT trends'\n"
                "    sq_a entity_filters: ['MY', 'ASEAN countries']\n"
                "    sq_b entity_filters: ['PHP', 'VNT']\n"
                "- In 'synthesis.join_keys': list common columns to join on, or [] if independent.\n"
                "- In 'synthesis.strategy': 'join' if results share a key, 'independent' if not.\n\n"
                "Return JSON:\n"
                '{"entity_filters": ["all entities from full question"], '
                f'"sub_queries": [{_sub_query_schema}, ...], '
                '"synthesis": {"join_keys": [], "strategy": "independent|join", "ordering": "", "limit": 100}}'
            )
        else:
            prompt = (
                "You are a query planner. Split a cross-source question into sub-queries.\n\n"
                "Available data sources with schema and cardinality:\n" + "\n\n".join(source_info) + "\n\n"
                f"Question: {nl_query}\n"
                f"{shared_note}\n"
                "Instructions:\n"
                "- Create exactly one sub-query per source that contains relevant data.\n"
                "- The sub-query 'nl' describes what data to retrieve from that source (plain English).\n"
                "- The sub-query 'source_id' MUST exactly match one of the source ids listed above.\n"
                "- CRITICAL: each sub-query has its OWN 'entity_filters' — ONLY the entities relevant\n"
                "  to THAT specific sub-query source. Do NOT put all entities in every sub-query.\n"
                "- In 'synthesis.join_keys', list the columns the sub-results will be joined on.\n"
                "- In 'synthesis.strategy': 'join' if results share a key, 'independent' if not.\n\n"
                "  entity_filters examples per sub-query:\n"
                "  Sub-query on sales table: entity_filters: ['Malaysia', 'Vietnam']\n"
                "  Sub-query on staff table: entity_filters: []\n\n"
                "Return JSON:\n"
                '{"entity_filters": ["all entities from full question"], '
                f'"sub_queries": [{_sub_query_schema}, ...], '
                '"synthesis": {"join_keys": [{"left": "sq_a.col", "right": "sq_b.col"}], "strategy": "join", "ordering": "col DESC", "limit": 100}}'
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

        # entity_filters come from understand_node (extracted from the original question)
        entity_filters = state.get("entity_filters", [])

        # Apply semantic resolution entity_map — translate user terms to actual stored values.
        # e.g. ["IND", "SG", "VNT"] → ["IND", "SGP", "VNM"] if the semantic agent resolved them.
        semantic_res = state.get("semantic_resolution")
        if semantic_res and semantic_res.get("entity_map"):
            entity_map = semantic_res["entity_map"]
            # Case-insensitive lookup + fuzzy prefix fallback (handles LLM key phrasing variations)
            entity_map_lower = {k.lower(): v for k, v in entity_map.items()}
            resolved_filters = []
            for e in entity_filters:
                if not isinstance(e, str):
                    resolved_filters.append(str(e) if e else "")
                    continue
                if e in entity_map:
                    resolved_filters.append(entity_map[e])
                elif e.lower() in entity_map_lower:
                    resolved_filters.append(entity_map_lower[e.lower()])
                else:
                    # Fuzzy: match by first 3 chars (handles "PHN" → "Philippines": "PHL")
                    match = next(
                        (v for k, v in entity_map.items() if
                         len(e) >= 2 and len(k) >= 2 and e[:2].lower() == k[:2].lower()),
                        None,
                    )
                    resolved_filters.append(match if match else e)
            entity_filters = resolved_filters
            logger.info("decompose.entity_filters_resolved", raw=state.get("entity_filters", []), resolved=entity_filters)

        logger.info("decompose.entity_filters", filters=entity_filters, nl_query=nl_query[:100])

        # Validate source_ids in sub_queries — replace unknown IDs with nearest match
        valid_ids = set(all_connector_ids)
        for sq in parsed.get("sub_queries", []):
            if sq.get("source_id") not in valid_ids:
                sq_sid = sq.get("source_id", "")
                match = next(
                    (cid for cid in all_connector_ids if sq_sid in cid or cid in sq_sid),
                    resolved_ids[0] if resolved_ids else "",
                )
                sq["source_id"] = match

            # Per-sub-query entity_filters: use what the LLM assigned to this sub-query
            # if it provided them; otherwise fall back to the global resolved list.
            # This prevents blasting ALL entities into every sub-query (e.g. in a compound
            # query "MY vs ASEAN & PHP vs VNT", sq_a should only filter MY/ASEAN not PHP/VNT).
            sq_local_filters = sq.get("entity_filters") or []
            if sq_local_filters:
                # Resolve abbreviations in per-sub-query filters using the same entity_map
                if semantic_res and semantic_res.get("entity_map"):
                    entity_map = semantic_res["entity_map"]
                    entity_map_lower = {k.lower(): v for k, v in entity_map.items()}
                    resolved_sq_filters = []
                    for e in sq_local_filters:
                        if e in entity_map:
                            resolved_sq_filters.append(entity_map[e])
                        elif e.lower() in entity_map_lower:
                            resolved_sq_filters.append(entity_map_lower[e.lower()])
                        else:
                            match_f = next(
                                (v for k, v in entity_map.items() if
                                 len(e) >= 2 and len(k) >= 2 and e[:2].lower() == k[:2].lower()),
                                None,
                            )
                            resolved_sq_filters.append(match_f if match_f else e)
                    sq["entity_filters"] = resolved_sq_filters
                # else keep sq_local_filters as-is
            else:
                # LLM didn't provide per-sub-query filters — fall back to global list
                sq["entity_filters"] = entity_filters

            # ── Expand group names to individual members ──────────────────────
            # "ASEAN" → ["Malaysia", "Indonesia", ...] so fan_out can find them
            sq["entity_filters"] = _expand_group_entities(sq["entity_filters"])

            if semantic_res:
                sq["semantic_entity_map"] = semantic_res.get("entity_map", {})
                sq["semantic_sql_fragments"] = semantic_res.get("sql_fragments", [])
            sub_nl = sq.get("nl", "") or nl_query
            sq["nl"] = sub_nl

        tokens = resp.tokens_input + resp.tokens_output
        latency = int((time.monotonic() - started) * 1000)
        n = len(parsed.get("sub_queries", []))

        for _sq in parsed.get("sub_queries", []):
            logger.info("decompose.sub_query_nl", id=_sq.get("id"), nl=_sq.get("nl","")[:300])

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
            import json as _json

            source_id = sq.get("source_id", "")
            nl = sq.get("nl", "")
            # entity_filters: resolved by decompose (semantic entity_map applied)
            entity_filters: list = sq.get("entity_filters", [])
            semantic_entity_map: dict = sq.get("semantic_entity_map", {})
            semantic_sql_fragments: list = sq.get("semantic_sql_fragments", [])
            conn = services.connectors.get(source_id)
            if not conn:
                return {
                    "sub_query_id": sq["id"],
                    "succeeded": False,
                    "error": f"No connector: {source_id}",
                }

            snap = await conn.introspect()

            def _exec_result(result):
                import pandas as _pd
                if isinstance(result, _pd.DataFrame):
                    result = result.where(result.notna(), other=None)
                    return result.to_dict("records"), list(result.columns), len(result)
                rows = list(getattr(result, "rows", []))
                cols = list(getattr(result, "columns", []))
                return rows, cols, len(rows)

            # ── PHASE 1: Deterministic entity discovery ────────────────────────
            # For each entity in entity_filters, find its exact stored spelling
            # using LIKE queries. This is deterministic — no LLM needed.
            resolved_entities: list[str] = []
            not_found: list[str] = []

            if entity_filters:
                # Find which column holds entity names (look for 'country' or text col)
                entity_col = None
                entity_table = None
                for t in snap.tables:
                    for c in t.columns:
                        if c.name.lower() in ("country", "name", "entity", "region"):
                            entity_col = c.name
                            entity_table = t.name
                            break
                    if entity_col:
                        break
                # Fallback: first low-cardinality text column
                if not entity_col and snap.tables:
                    t = snap.tables[0]
                    for c in t.columns:
                        if any(kw in c.data_type.upper() for kw in ("VARCHAR", "TEXT", "STRING")):
                            entity_col = c.name
                            entity_table = t.name
                            break

                if entity_col and entity_table:
                    for entity in entity_filters:
                        # Try exact match first
                        try:
                            exact = await conn.execute(
                                f'SELECT DISTINCT "{entity_col}" FROM "{entity_table}" '
                                f'WHERE LOWER("{entity_col}") = LOWER({repr(entity)}) LIMIT 1'
                            )
                            exact_rows, _, exact_count = _exec_result(exact)
                            if exact_count > 0:
                                resolved_entities.append(str(list(exact_rows[0].values())[0]))
                                continue
                        except Exception:
                            pass
                        # Fuzzy match: use first 5 chars as LIKE keyword
                        keyword = entity.lower()[:5]
                        resolved_this = False
                        try:
                            fuzzy = await conn.execute(
                                f'SELECT DISTINCT "{entity_col}" FROM "{entity_table}" '
                                f'WHERE LOWER("{entity_col}") LIKE {repr("%" + keyword + "%")} LIMIT 5'
                            )
                            fuzzy_rows, _, fuzzy_count = _exec_result(fuzzy)
                            if fuzzy_count > 0:
                                # Pick best match: prefer exact substring match
                                matches = [str(list(r.values())[0]) for r in fuzzy_rows]
                                best = next(
                                    (m for m in matches if entity.lower() in m.lower()),
                                    matches[0]
                                )
                                resolved_entities.append(best)
                                resolved_this = True
                        except Exception:
                            resolved_entities.append(entity)  # Use original if error
                            resolved_this = True

                        # Fallback 1: check if semantic_entity_map has a mapped value for this entity
                        # (catches cases where entity_map keys didn't match entity_filters exactly)
                        if not resolved_this and semantic_entity_map:
                            sem_map_lower = {k.lower(): v for k, v in semantic_entity_map.items()}
                            mapped_val = (
                                semantic_entity_map.get(entity)
                                or sem_map_lower.get(entity.lower())
                            )
                            if mapped_val:
                                # Search iso_code / other columns for the mapped value
                                for t in snap.tables:
                                    if resolved_this:
                                        break
                                    for c in t.columns:
                                        if not any(kw in c.data_type.upper() for kw in ("VARCHAR", "TEXT", "STRING")):
                                            continue
                                        try:
                                            r3 = await conn.execute(
                                                f'SELECT DISTINCT "{c.name}" FROM "{t.name}" '
                                                f'WHERE LOWER("{c.name}") = LOWER({repr(mapped_val)}) LIMIT 1'
                                            )
                                            rows3, _, cnt3 = _exec_result(r3)
                                            if cnt3 > 0:
                                                entity_col = c.name
                                                entity_table = t.name
                                                resolved_entities.append(str(list(rows3[0].values())[0]))
                                                resolved_this = True
                                                break
                                        except Exception:
                                            pass

                        # Fallback 2: search other text columns (e.g. iso_code) for the original entity
                        if not resolved_this:
                            for t in snap.tables:
                                if resolved_this:
                                    break
                                for c in t.columns:
                                    if c.name == entity_col:
                                        continue  # already tried this
                                    if not any(kw in c.data_type.upper() for kw in ("VARCHAR", "TEXT", "STRING")):
                                        continue
                                    try:
                                        exact2 = await conn.execute(
                                            f'SELECT DISTINCT "{c.name}" FROM "{t.name}" '
                                            f'WHERE LOWER("{c.name}") = LOWER({repr(entity)}) LIMIT 1'
                                        )
                                        r2, _, cnt2 = _exec_result(exact2)
                                        if cnt2 > 0:
                                            # Found in a code column — switch entity_col to this
                                            entity_col = c.name
                                            entity_table = t.name
                                            resolved_entities.append(str(list(r2[0].values())[0]))
                                            resolved_this = True
                                            break
                                    except Exception:
                                        pass
                            if not resolved_this:
                                not_found.append(entity)

                if not_found and not resolved_entities:
                    # All entities not found — honest failure so trace shows ✗
                    # and synthesize_node knows this sub-query contributed nothing.
                    return {
                        "sub_query_id": sq["id"],
                        "source_id": source_id,
                        "sql": "",
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                        "succeeded": False,
                        "error": (
                            f"Could not find {not_found} in '{source_id}'. "
                            f"Check that the dataset contains these entities, or rephrase with "
                            f"the exact names as stored in the data."
                        ),
                        "attempts": 1,
                    }

            # ── PHASE 2: Build schema text for LLM (no entity values — just structure) ──
            schema_lines = []
            for t in snap.tables:
                col_names = [c.name for c in t.columns]
                schema_lines.append(f"  {t.name}({', '.join(col_names)})")
            schema_text = "\n".join(schema_lines)

            # ── PHASE 3: Build WHERE clause from resolved entities ─────────────
            where_clause = ""
            if resolved_entities and entity_col:
                vals = ", ".join(repr(e) for e in resolved_entities)
                where_clause = f'WHERE "{entity_col}" IN ({vals})'
            elif entity_filters and not resolved_entities and entity_col:
                # Entities named but none found — use originals (will return 0 rows)
                vals = ", ".join(repr(e) for e in entity_filters)
                where_clause = f'WHERE "{entity_col}" IN ({vals})'

            # ── PHASE 4: LLM generates SQL structure (knows WHERE is given) ────
            where_note = (
                f"\nWHERE clause to use: {where_clause}\n"
                "Include this WHERE clause verbatim in your SQL (do NOT add or change it)."
                if where_clause else
                "\nNo entity filter — return all rows."
            )
            # Inject semantic sql_fragments as additional hints when entity resolution
            # produced a WHERE clause using a different column (e.g. iso_code)
            if semantic_sql_fragments and not where_clause:
                where_note += (
                    "\nSemantic hints (from data analysis — prefer these if no WHERE clause above): "
                    + "; ".join(semantic_sql_fragments)
                )

            def _extract_sql(text: str) -> str:
                text = text.strip()
                if "```sql" in text:
                    return text.split("```sql")[1].split("```")[0].strip()
                if "```" in text:
                    return text.split("```")[1].split("```")[0].strip()
                return text

            # Inject workspace-level learned context rules (from user corrections)
            context_notes = state.get("data_context_notes") or []
            context_hint = (
                "\n\nWORKSPACE RULES (learned from corrections — apply these):\n"
                + "\n".join(f"• {n}" for n in context_notes[:5])
                if context_notes else ""
            )

            SYSTEM = (
                "You are a SQL agent. Write a single SQL SELECT statement.\n"
                "Follow the schema and WHERE clause provided exactly.\n"
                "Always GROUP BY the join keys.\n"
                "Return ONLY valid SQL — no explanation, no markdown."
            )

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": (
                    f"SCHEMA:\n{schema_text}\n\n"
                    f"TASK: {nl}"
                    f"{where_note}"
                    f"{context_hint}"
                )},
            ]

            sql = ""
            last_error = ""

            for attempt in range(4):  # write SQL + up to 3 corrections
                resp = await services.llm.complete(messages)
                sql = _extract_sql(resp.content)
                logger.info("fan_out.attempt", sub_id=sq.get("id"), attempt=attempt,
                            sql=sql[:250], entities=resolved_entities or entity_filters)
                messages.append({"role": "assistant", "content": resp.content})

                try:
                    result = await conn.execute(sql)
                    rows, columns, row_count = _exec_result(result)
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
                    messages.append({"role": "user", "content": (
                        f"SQL error: {last_error}\n\nFix the SQL. Keep the WHERE clause unchanged."
                    )})

            return {
                "sub_query_id": sq["id"],
                "source_id": source_id,
                "sql": sql,
                "succeeded": False,
                "error": f"Failed after {attempt + 1} attempts. Last error: {last_error}",
                "rows": [],
                "columns": [],
                "row_count": 0,
                "attempts": 4,
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
        sub_result_diagnostics = []
        for sr in sub_results:
            if sr.get("succeeded") and sr.get("rows"):
                df = pd.DataFrame(sr["rows"])
                con.register(sr["sub_query_id"], df)
                col_desc = ", ".join(f"{c} ({str(df[c].dtype)})" for c in df.columns)
                # Include a sample row so the LLM can see actual values (helps detect wrong filters)
                sample = df.head(2).to_dict("records") if len(df) > 0 else []
                table_schemas.append(
                    f"  {sr['sub_query_id']}({col_desc})  — {len(df)} rows "
                    f"[source: {sr.get('source_id', '?')}]\n"
                    f"  Sample rows: {sample}"
                )
            else:
                sub_result_diagnostics.append(
                    f"  {sr.get('sub_query_id', '?')} — EMPTY or FAILED "
                    f"(error: {sr.get('error', 'unknown')}) [source: {sr.get('source_id', '?')}]"
                )

        schema_hint = "\n".join(table_schemas) if table_schemas else ""
        empty_hint = "\n".join(sub_result_diagnostics) if sub_result_diagnostics else ""
        join_hint = ""
        if synthesis.get("join_keys"):
            jk = synthesis["join_keys"][0]
            join_hint = f"\nSuggested join key: {jk.get('left', '')} = {jk.get('right', '')}"

        # Inject workspace-level learned context rules into synthesis
        synth_context_notes = state.get("data_context_notes") or []
        synth_context_hint = (
            "\nWORKSPACE RULES (learned from corrections — apply these):\n"
            + "\n".join(f"• {n}" for n in synth_context_notes[:5])
            + "\n"
            if synth_context_notes else ""
        )

        # Analytical intent from Ora — drives synthesis SQL strategy
        analytical_intent = state.get("analytical_intent", "comparison")

        # Determine synthesis strategy: join vs independent (compound with no shared key)
        synth_strategy = synthesis.get("strategy", "join")
        has_join_keys = bool(synthesis.get("join_keys"))
        is_independent = synth_strategy == "independent" or (
            state.get("is_compound_query") and not has_join_keys
        )

        # For independent sub-questions (compound query with incompatible result shapes),
        # build a UNION ALL with a sub_question label rather than attempting a JOIN.
        # This avoids cartesian products and INSUFFICIENT_DATA from incompatible grains.
        if is_independent and len(table_schemas) > 1:
            # Find common columns across all sub-result tables
            available_tables = [sr for sr in sub_results if sr.get("succeeded") and sr.get("rows")]
            all_col_sets = [set(sr.get("columns", [])) for sr in available_tables]
            common_cols = set.intersection(*all_col_sets) if all_col_sets else set()

            if common_cols:
                col_list = ", ".join(f'"{c}"' for c in sorted(common_cols))
                union_parts = []
                for sr in available_tables:
                    label = sr.get("sub_query_id", sr.get("source_id", "result"))
                    union_parts.append(
                        f"SELECT {col_list}, '{label}' AS sub_question FROM {sr['sub_query_id']}"
                    )
                auto_sql = "\nUNION ALL\n".join(union_parts)
            else:
                # No common columns — stack all columns, pad missing with NULL
                all_cols_ordered = []
                seen = set()
                for sr in available_tables:
                    for c in sr.get("columns", []):
                        if c not in seen:
                            all_cols_ordered.append(c)
                            seen.add(c)
                union_parts = []
                for sr in available_tables:
                    sr_cols = set(sr.get("columns", []))
                    select_parts = []
                    for c in all_cols_ordered:
                        select_parts.append(f'"{c}"' if c in sr_cols else f'NULL AS "{c}"')
                    label = sr.get("sub_query_id", sr.get("source_id", "result"))
                    select_parts.append(f"'{label}' AS sub_question")
                    union_parts.append(
                        f"SELECT {', '.join(select_parts)} FROM {sr['sub_query_id']}"
                    )
                auto_sql = "\nUNION ALL\n".join(union_parts)

            # Execute the auto-built UNION ALL immediately (no LLM needed)
            try:
                result_df = con.execute(auto_sql).fetchdf()
                result_df = result_df.where(result_df.notna(), None)
                rows = result_df.to_dict("records")
                columns = list(result_df.columns)
                row_count = len(rows)
                sql = auto_sql
                succeeded = True
                error = ""
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
                    "trace_events": state.get("trace_events", []) + [{
                        "node": "synthesize",
                        "status": "completed",
                        "latency_ms": latency,
                        "row_count": row_count,
                        "summary": (
                            f"Combined {len(available_tables)} independent sub-results — "
                            f"{row_count:,} rows across all parts"
                        ),
                    }],
                }
            except Exception as exc:
                # Fall through to LLM-based synthesis if UNION ALL failed
                last_error = str(exc)

        sql = ""
        last_error = ""  # always initialize before the generation loop
        # Generate → execute → reflect loop (up to 3 rounds)
        for attempt in range(3):
            if attempt == 0:
                # Build analytical intent guidance for the synthesis prompt
                if analytical_intent == "correlation" and not is_independent:
                    intent_note = (
                        "The user wants to find a CORRELATION between two metrics across these sub-results. "
                        "After joining, compute CORR(metric_a, metric_b) using DuckDB's built-in CORR() function. "
                        "Also include CORR() alongside individual metric values so the user can see both the "
                        "raw data and the correlation coefficient.\n"
                    )
                elif analytical_intent == "trend":
                    intent_note = (
                        "The user wants to see TRENDS over time. "
                        "Ensure the result is ordered by the time/date column ascending.\n"
                    )
                elif analytical_intent == "ranking":
                    intent_note = (
                        "The user wants a RANKED list. "
                        "Order results by the primary metric descending.\n"
                    )
                else:
                    intent_note = ""

                strategy_note = (
                    "These sub-results are from INDEPENDENT sub-questions — do NOT join them. "
                    "Use UNION ALL to combine them, adding a label column to identify each part.\n"
                    if is_independent else
                    f"Write DuckDB SQL that joins the tables at the correct grain "
                    f"(GROUP BY before joining if keys are not 1:1).\n{intent_note}"
                )
                prompt = (
                    f"You are a synthesis agent combining results from parallel sub-queries.\n\n"
                    f"Original question: {nl_query}\n"
                    f"Analytical intent: {analytical_intent}\n\n"
                    + synth_context_hint
                    + (f"Available DuckDB tables:\n{schema_hint}{join_hint}\n\n" if schema_hint else "")
                    + (f"Sub-queries that returned NO data:\n{empty_hint}\n\n" if empty_hint else "")
                    + "REFLECT before writing SQL:\n"
                    "- Do the available tables actually contain data relevant to the question?\n"
                    "- If sub-queries returned empty results, the upstream filters may have been wrong "
                    "(e.g. wrong column name, wrong value format). Reason about what went wrong.\n"
                    "- If you have no usable data, respond with exactly: INSUFFICIENT_DATA: <reason>\n"
                    f"- If you have data: {strategy_note}"
                    "Use only the table names listed above.\n"
                    "Return ONLY valid DuckDB SQL or INSUFFICIENT_DATA: <reason>."
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

            # Pure agentic signal: LLM reflected and determined sub-results are
            # insufficient to answer the question. Surface this clearly — no retry.
            if sql.upper().startswith("INSUFFICIENT_DATA:"):
                reason = sql[len("INSUFFICIENT_DATA:"):].strip()
                rows, columns, row_count = [], [], 0
                succeeded = False
                error = f"Agent reflection: {reason}"
                break

            try:
                if not sql:
                    raise ValueError("LLM returned empty SQL — cannot execute")
                cursor = con.execute(sql)
                if cursor is None:
                    raise ValueError(
                        "DuckDB execute() returned None — SQL may be non-SELECT or malformed"
                    )
                result = cursor.fetchdf()
                # Hard safety valve: if the LLM generated a cartesian join and produced
                # an explosion (>50k rows), truncate and re-run with an explicit LIMIT.
                # This prevents MB-scale payloads from crashing the SSE stream and localStorage.
                _MAX_ROWS = 50_000
                if len(result) > _MAX_ROWS:
                    try:
                        limited = con.execute(f"SELECT * FROM ({sql}) __q LIMIT {_MAX_ROWS}").fetchdf()
                        result = limited
                    except Exception:
                        result = result.head(_MAX_ROWS)
                # Replace NaN/Inf with None so the result is JSON-serializable.
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


# ═══════════════════════════════════════════════════════════════════════════════
# ORA — Unified orchestrator (understand + semantic_resolve + decompose in one)
# ═══════════════════════════════════════════════════════════════════════════════


def make_ora_node(services: Any):
    """Ora: the single orchestrator that plans every query end-to-end.

    Phases (all in one node call):
      0. Anaphora resolution — rewrite pronouns/references using conversation history
      1. Intent + entity extraction (LLM) — classify compound/cross-source, extract
         entity names, detect analytical_intent (correlation, comparison, etc.)
      2. Group expansion — ASEAN → 10 member countries (deterministic, no LLM)
      3. Data coverage analysis (Python/SQL) — for each source, check SemanticMemory
         first (fast path), then live SQL scan for exact entity column + values
      4. Semantic resolution (SemanticAgent) — per source, entity_map + synonyms
         (uses SemanticCache for quick re-use)
      5. Planning (LLM) — final sub-query plan with all context verified

    Outputs into state:
      - nl_query (anaphora-resolved)
      - entity_filters (expanded + code-resolved)
      - analytical_intent
      - is_cross_source, is_compound_query, target_sources
      - sub_queries (non-empty → graph routes to fan_out)
      - decomposition_plan
      - semantic_resolution, data_context_notes
      - data_warnings (entities not found — surfaced to UI)
      - ora_reasoning (full chain-of-thought for trace)
    """

    async def ora_node(state: QueryState) -> dict:
        import json as _json
        import re as _re

        started = time.monotonic()
        nl_query = state["nl_query"]
        source_ids = state.get("source_ids", [])
        query_id = state.get("query_id", str(uuid.uuid4())[:12])

        # ── Phase 0: Anaphora resolution ──────────────────────────────────────
        _hist = state.get("conversation_history", [])
        if _hist:
            try:
                history_context = "\n".join(
                    f"{'User' if t.get('role') == 'user' else 'Agent'}: "
                    + (t.get("text") or t.get("nl_response") or "")[:200]
                    for t in _hist[-6:]
                )
                rewrite_resp = await services.llm.complete([{
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{history_context}\n\n"
                        f"Latest question: {nl_query}\n\n"
                        "Rewrite the latest question to be fully self-contained — replace all "
                        "pronouns and references (e.g. 'these 2', 'them', 'the same', 'the trend') "
                        "with the explicit entities from the conversation. "
                        "Return ONLY the rewritten question, nothing else."
                    ),
                }])
                rewritten = rewrite_resp.content.strip().strip('"').strip("'")
                if rewritten and len(rewritten) > 5:
                    nl_query = rewritten
            except Exception:
                pass

        # ── Phase 1: Intent + entity extraction ───────────────────────────────
        _hist_ctx = ""
        if _hist:
            _hist_ctx = "\n".join(
                f"{'User' if t.get('role') == 'user' else 'Agent'}: "
                + (t.get("text") or t.get("nl_response") or "")[:200]
                for t in _hist[-4:]
            )

        source_summary = ""
        for sid in source_ids[:5]:
            conn = services.connectors.get(sid)
            if conn:
                try:
                    snap = await conn.introspect()
                    tables = [t.name for t in snap.tables[:8]]
                    source_summary += f"\n  Source '{sid}' ({conn.dialect}): {tables}"
                except Exception:
                    source_summary += f"\n  Source '{sid}': (unavailable)"

        intent_prompt = (
            "You are Ora, a query orchestration agent. Analyse this question and plan how to answer it.\n\n"
            + (f"Available data sources:{source_summary}\n\n" if source_summary else "")
            + (f"Conversation context:\n{_hist_ctx}\n\n" if _hist_ctx else "")
            + f"Question: {nl_query}\n\n"
            "Classify the question:\n\n"
            "COMPOUND: two or more DISTINCT analytical questions requiring separate SQL queries.\n"
            "  e.g. 'Compare Malaysia vs ASEAN employment AND show GenAI spend trends' → COMPOUND\n"
            "  e.g. 'Compare revenue and profit by store' → NOT COMPOUND (one query, two metrics)\n\n"
            "CROSS-SOURCE: needs data from more than one data source above.\n\n"
            "ANALYTICAL_INTENT — pick the primary intent:\n"
            "  correlation: asks about relationship/correlation between two metrics\n"
            "  comparison: compares values across groups/entities\n"
            "  trend: asks about change over time\n"
            "  ranking: top N, best/worst, ranked list\n"
            "  aggregate: sum/count/average with no comparison\n"
            "  describe: schema/data description question\n\n"
            "ENTITY_FILTERS: specific named entities the question filters for.\n"
            "  Include group names (ASEAN, G7) as-is — they will be expanded later.\n"
            "  If asking for 'top 5' with no specific names, return [].\n\n"
            "Return JSON:\n"
            '{\n'
            '  "is_compound": true|false,\n'
            '  "is_cross_source": true|false,\n'
            '  "analytical_intent": "correlation|comparison|trend|ranking|aggregate|describe",\n'
            '  "entity_filters": ["EntityA", "GroupName"],\n'
            '  "target_sources": ["source_id", ...],\n'
            '  "complexity": "simple|moderate|complex",\n'
            '  "reasoning": "step by step rationale"\n'
            '}'
        )

        try:
            intent_resp = await services.llm.complete(
                [{"role": "user", "content": intent_prompt}],
                json_mode=True,
            )
            raw = intent_resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
            intent_parsed = _json.loads(raw)
        except Exception:
            intent_parsed = {
                "is_compound": False,
                "is_cross_source": len(source_ids) > 1,
                "analytical_intent": "comparison",
                "entity_filters": [],
                "target_sources": source_ids,
                "complexity": "moderate",
                "reasoning": "parse error — defaults applied",
            }

        is_compound = bool(intent_parsed.get("is_compound", False))
        is_cross_source = bool(intent_parsed.get("is_cross_source", len(source_ids) > 1))
        analytical_intent = intent_parsed.get("analytical_intent", "comparison")
        entity_filters: list[str] = intent_parsed.get("entity_filters") or []
        if not isinstance(entity_filters, list):
            entity_filters = []
        target_sources: list[str] = intent_parsed.get("target_sources") or source_ids
        complexity: str = intent_parsed.get("complexity", "moderate")
        ora_reasoning: str = intent_parsed.get("reasoning", "")

        intent_tokens = (
            getattr(intent_resp, "tokens_input", 0) + getattr(intent_resp, "tokens_output", 0)
        )
        intent_cost = getattr(intent_resp, "cost_usd", 0.0)

        # ── Phase 2: Group name expansion ────────────────────────────────────
        raw_entity_filters = list(entity_filters)
        entity_filters = _expand_group_entities(entity_filters)

        # ── Phase 2.5: Semantic Reasoning Agent ─────────────────────────────
        # Single LLM call that reasons about the ENTIRE query in data context.
        # Maps user terms to exact columns, values, tables, and metrics.
        # This is the primary intelligence — everything below is secondary.
        semantic_reasoning = None
        _sem_trace = None
        try:
            from sqlagent.semantic_agent import (
                reason_about_query, load_context as load_sem_ctx,
                SemanticReasoning,
            )
            ws_id = state.get("workspace_id", "")

            # Load connect-time semantic contexts for all sources
            sem_ctxs = {}
            for sid in target_sources:
                ctx = load_sem_ctx(sid, ws_id) if ws_id else None
                if ctx:
                    sem_ctxs[sid] = ctx

            semantic_reasoning = await reason_about_query(
                question=nl_query,
                source_ids=target_sources,
                workspace_id=ws_id,
                connectors=services.connectors,
                llm=services.llm,
                semantic_contexts=sem_ctxs if sem_ctxs else None,
            )

            if semantic_reasoning and semantic_reasoning.filters:
                # Apply reasoning results to entity_filters for downstream
                for f in semantic_reasoning.filters:
                    val = f.get("value", "")
                    # LLM may return value as a list — flatten
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, str) and v and v not in entity_filters:
                                entity_filters.append(v)
                    elif isinstance(val, str) and val and val not in entity_filters:
                        entity_filters.append(val)

                ora_reasoning += (
                    f"\nSemantic Reasoning: {semantic_reasoning.reasoning}"
                    f"\n  Filters: {semantic_reasoning.filters}"
                    f"\n  Tables: {semantic_reasoning.tables}"
                    f"\n  Metrics: {semantic_reasoning.metrics}"
                    f"\n  Confidence: {semantic_reasoning.confidence}"
                )

                # ── Override cross-source if all tables are in the same backend ──
                # The Semantic Reasoning Agent knows which tables are needed.
                # If all needed tables are DuckDB file sources in the same workspace,
                # they can be JOINed directly — no cross-source decomposition needed.
                if semantic_reasoning.tables and is_cross_source:
                    reasoning_tables = set(semantic_reasoning.tables)
                    # Check if all connectors are DuckDB (file sources)
                    all_duckdb = all(
                        getattr(services.connectors.get(sid), 'dialect', '') == 'duckdb'
                        or 'file_' in sid
                        for sid in target_sources
                    )
                    if all_duckdb:
                        # All tables are in DuckDB — treat as single source with JOIN
                        is_cross_source = False
                        is_compound = False
                        # Route to the first source that has connectors
                        if target_sources:
                            target_sources = [target_sources[0]]
                        ora_reasoning += (
                            "\n  Routing override: all tables in DuckDB — using single-source "
                            "JOIN instead of cross-source decomposition"
                        )

                # Save new aliases learned from this query
                if semantic_reasoning.new_aliases:
                    ora_reasoning += (
                        f"\n  New aliases: "
                        + ", ".join(f"{k}→{v}" for k, v in semantic_reasoning.new_aliases.items())
                    )

                # Add trace event for Semantic Reasoning (visible in lineage)
                _sem_trace = {
                    "node": "semantic_reasoning",
                    "status": "completed",
                    "latency_ms": 0,  # included in ora latency
                    "summary": (
                        f"Resolved {len(semantic_reasoning.filters)} filters, "
                        f"{len(semantic_reasoning.new_aliases)} new aliases · "
                        f"Confidence: {int(semantic_reasoning.confidence * 100)}%"
                    ),
                    "thinking": semantic_reasoning.reasoning[:300],
                    "filters": semantic_reasoning.filters,
                    "tables": semantic_reasoning.tables,
                    "metrics": semantic_reasoning.metrics,
                    "new_aliases": semantic_reasoning.new_aliases,
                    "confidence": semantic_reasoning.confidence,
                }

                # ── Disambiguation check ─────────────────────────────────
                # If reasoning confidence is low, ask user before proceeding
                if semantic_reasoning.confidence < 0.6:
                    try:
                        from sqlagent.disambiguation import detect_disambiguation
                        schema_summary = "\n".join(
                            f"{t['name']}: {[d['name'] for d in t.get('dimensions',[])]}"
                            for t in (await list(services.connectors.values())[0].introspect()).tables[:5]
                        ) if services.connectors else ""
                        clarification = await detect_disambiguation(
                            question=nl_query,
                            semantic_reasoning=semantic_reasoning.to_dict(),
                            schema_context=schema_summary,
                            llm=services.llm,
                            threshold=0.6,
                        )
                        if clarification:
                            # Store clarification in state — UI will display dialog
                            ora_reasoning += f"\n  ⚠ Disambiguation needed: {clarification.question}"
                    except Exception as _disamb_err:
                        logger.warning("disambiguation.failed_in_ora", error=str(_disamb_err))

        except Exception as _reason_err:
            logger.warning("semantic.reasoning.failed_in_ora", error=str(_reason_err))

        # ── Phase 3: Data coverage analysis ──────────────────────────────────
        # For each source: check SemanticMemory first, then live SQL scan.
        # Produces: source_coverage (source_id → list of found entities)
        #           data_warnings (entities not found in ANY source)
        source_coverage: dict[str, list[str]] = {}
        source_entity_cols: dict[str, tuple[str, str]] = {}  # source_id → (table, col)
        data_warnings: list[str] = []
        semantic_memory = getattr(services, "_semantic_memory", None)

        if entity_filters:
            entities_lower = {e.lower(): e for e in entity_filters}

            for sid in target_sources:
                conn = services.connectors.get(sid)
                if not conn:
                    continue
                found_entities: list[str] = []

                # Fast path: check SemanticMemory
                mem_entity_col: tuple[str, str] | None = None
                if semantic_memory:
                    try:
                        coverage_check = await semantic_memory.entities_covered(sid, entity_filters)
                        known_found = [e for e, found in coverage_check.items() if found]
                        if known_found:
                            found_entities = known_found
                            mem_entity_col = await semantic_memory.get_entity_column(sid)
                        # If none known, don't set found_entities yet — will do live scan
                    except Exception:
                        pass

                # ── Semantic Reasoning Agent (v2.0) ──────────────────────────
                # Single LLM call with full schema context. Reasons about
                # what the user means — maps terms to exact columns/values.
                # Replaces lookup-based entity resolution entirely.
                pass  # coverage scan above is now secondary — reasoning is primary

                if mem_entity_col:
                    source_entity_cols[sid] = mem_entity_col
                source_coverage[sid] = found_entities

            # Identify entities not found in ANY source
            all_found_lower = {
                e.lower() for found in source_coverage.values() for e in found
            }
            missing = [e for e in entity_filters if e.lower() not in all_found_lower]
            if missing:
                for grp_name, members in _GROUP_EXPANSIONS.items():
                    members_lower = {m.lower() for m in members}
                    if any(m.lower() in all_found_lower for m in members):
                        # Group partially found — remove from missing if it was a group name
                        missing = [
                            m for m in missing
                            if m.lower() not in members_lower
                        ]
                if missing:
                    data_warnings.append(
                        f"Could not find '{', '.join(missing)}' in any connected source. "
                        "Results may be incomplete — try rephrasing with the exact name as stored."
                    )

        # ── Phase 4: Semantic resolution per relevant source ──────────────────
        semantic_cache = getattr(services, "_semantic_cache", None)
        combined_entity_map: dict[str, str] = {}
        combined_synonyms: dict[str, str] = {}
        combined_filter_hints: list[str] = []
        combined_sql_frags: list[str] = []
        primary_resolution = None

        for sid in target_sources:
            conn = services.connectors.get(sid)
            if not conn:
                continue
            try:
                # Check SemanticCache first
                resolution = None
                if semantic_cache:
                    resolution = await semantic_cache.get(nl_query, sid)

                if not resolution:
                    snap = await conn.introspect()
                    schema_dict = snap.to_dict() if hasattr(snap, "to_dict") else {}
                    from sqlagent.semantic_layer import run_semantic_agent
                    resolution = await run_semantic_agent(
                        nl_query=nl_query,
                        source_id=sid,
                        schema=schema_dict,
                        llm=services.llm,
                    )

                if resolution:
                    combined_entity_map.update(resolution.entity_map)
                    combined_synonyms.update(resolution.synonyms)
                    combined_filter_hints.extend(resolution.filter_hints)
                    combined_sql_frags.extend(resolution.sql_fragments)
                    if primary_resolution is None:
                        primary_resolution = resolution

                    # Persist concept→column mappings
                    if semantic_memory and resolution.synonyms:
                        for concept, col in resolution.synonyms.items():
                            try:
                                await semantic_memory.save_concept_column(sid, concept, col)
                            except Exception:
                                pass
            except Exception as exc:
                logger.debug("ora.semantic_resolve_failed", source_id=sid, error=str(exc))

        # Build context block from combined resolution
        existing_notes = list(state.get("data_context_notes") or [])
        if combined_entity_map or combined_synonyms or combined_filter_hints:
            from sqlagent.semantic_layer import SemanticResolution as _SemRes
            combined_res = _SemRes(
                nl_query=nl_query,
                source_id=target_sources[0] if target_sources else "",
                entity_map=combined_entity_map,
                synonyms=combined_synonyms,
                filter_hints=combined_filter_hints,
                sql_fragments=combined_sql_frags,
                reasoning="Combined across sources by Ora",
                confidence=0.85,
            )
            ctx_block = combined_res.to_context_block()
            if ctx_block and ctx_block not in existing_notes:
                existing_notes.insert(0, ctx_block)

        # Apply entity_map to nl_query (resolve user terms → stored values)
        nl_query_resolved = nl_query
        if combined_entity_map:
            for user_term, stored_val in sorted(
                combined_entity_map.items(), key=lambda x: -len(x[0])
            ):
                if user_term != stored_val:
                    nl_query_resolved = _re.sub(
                        r'\b' + _re.escape(user_term) + r'\b',
                        stored_val,
                        nl_query_resolved,
                        flags=_re.IGNORECASE,
                    )

        # Resolve entity_filters through entity_map
        entity_map_lower = {k.lower(): v for k, v in combined_entity_map.items()}
        resolved_filters = []
        for e in entity_filters:
            resolved = (
                combined_entity_map.get(e)
                or entity_map_lower.get(e.lower())
                or e
            )
            resolved_filters.append(resolved)

        # ── Phase 5: Decomposition planning (only if cross-source or compound) ─
        sub_queries: list[dict] = []
        decomposition_plan: dict | None = None
        plan_tokens = 0
        plan_cost = 0.0

        needs_decompose = is_cross_source or is_compound

        if needs_decompose:
            # Build source info for the planner
            source_info_lines = []
            all_connector_ids = list(services.connectors.keys())
            for sid in all_connector_ids:
                conn = services.connectors.get(sid)
                if not conn:
                    continue
                try:
                    snap = await conn.introspect()
                    col_lines = []
                    for t in snap.tables:
                        col_parts = [f"{c.name} {c.data_type}" for c in t.columns[:15]]
                        coverage_info = ""
                        if sid in source_coverage:
                            found_there = source_coverage[sid]
                            if found_there:
                                coverage_info = f" [HAS DATA for: {', '.join(found_there[:5])}]"
                        col_lines.append(
                            f"  table '{t.name}' ({t.row_count_estimate or 0} rows)"
                            f"{coverage_info}: {', '.join(col_parts)}"
                        )
                    source_info_lines.append(
                        f"Source id='{sid}' ({conn.dialect}):\n" + "\n".join(col_lines)
                    )
                except Exception:
                    source_info_lines.append(f"Source id='{sid}': (schema unavailable)")

            is_compound_single = is_compound and not is_cross_source
            _sub_q_schema = (
                '{"id": "sq_a", "source_id": "<exact source id>", '
                '"nl": "sub-question description", '
                '"entity_filters": ["entities FOR THIS sub-question only"], '
                '"expected_columns": ["col1", "col2"]}'
            )

            if is_compound_single:
                plan_prompt = (
                    "You are Ora, a query planner. This is a COMPOUND question — multiple distinct\n"
                    "analytical sub-questions against the SAME data source.\n\n"
                    "Available source:\n" + "\n\n".join(source_info_lines) + "\n\n"
                    f"Question: {nl_query}\n"
                    f"Analytical intent: {analytical_intent}\n\n"
                    "Instructions:\n"
                    "- Create one sub-query per distinct analytical sub-question.\n"
                    "- ALL sub-queries use the SAME source_id.\n"
                    "- Each sub-query has its OWN entity_filters (only its relevant entities).\n"
                    "- synthesis.strategy: 'independent' if sub-questions have incompatible grains.\n\n"
                    "Return JSON:\n"
                    '{"entity_filters": [...], '
                    f'"sub_queries": [{_sub_q_schema}, ...], '
                    '"synthesis": {"join_keys": [], "strategy": "independent|join", "ordering": "", "limit": 100}}'
                )
            else:
                plan_prompt = (
                    "You are Ora, a query planner. Split this cross-source question into sub-queries.\n\n"
                    "Available sources with data coverage pre-verified:\n"
                    + "\n\n".join(source_info_lines) + "\n\n"
                    f"Question: {nl_query}\n"
                    f"Analytical intent: {analytical_intent}\n"
                    + (f"Data warnings: {data_warnings}\n" if data_warnings else "")
                    + "\nInstructions:\n"
                    "- Create one sub-query per source that contains relevant data.\n"
                    "- The source_id MUST exactly match one of the source ids listed.\n"
                    "- Each sub-query has its OWN entity_filters (only its relevant entities).\n"
                    "- synthesis.join_keys: columns to join on (left=sq_a.col, right=sq_b.col).\n"
                    "- synthesis.strategy: 'join' if results share a key, 'independent' if not.\n\n"
                    "Return JSON:\n"
                    '{"entity_filters": [...], '
                    f'"sub_queries": [{_sub_q_schema}, ...], '
                    '"synthesis": {"join_keys": [{"left": "sq_a.col", "right": "sq_b.col"}], '
                    '"strategy": "join", "ordering": "col DESC", "limit": 100}}'
                )

            try:
                plan_resp = await services.llm.complete(
                    [{"role": "user", "content": plan_prompt}],
                    json_mode=True,
                )
                plan_tokens = plan_resp.tokens_input + plan_resp.tokens_output
                plan_cost = plan_resp.cost_usd
                raw_plan = plan_resp.content.strip()
                if raw_plan.startswith("```"):
                    raw_plan = raw_plan.split("```")[1].lstrip("json").strip().rstrip("```").strip()
                decomposition_plan = _json.loads(raw_plan)
            except Exception:
                # Hard fallback: one sub-query per source
                decomposition_plan = {
                    "sub_queries": [
                        {"id": f"sq_{i}", "source_id": sid, "nl": nl_query, "entity_filters": [], "expected_columns": []}
                        for i, sid in enumerate(target_sources or list(services.connectors.keys()))
                    ],
                    "synthesis": {"join_keys": [], "strategy": "independent", "ordering": "", "limit": 100},
                }

            # Post-process sub_queries: validate source IDs, apply entity resolution
            valid_ids = set(services.connectors.keys())
            for sq in decomposition_plan.get("sub_queries", []):
                # Fix unknown source IDs
                if sq.get("source_id") not in valid_ids:
                    sq_sid = sq.get("source_id", "")
                    match = next(
                        (cid for cid in valid_ids if sq_sid in cid or cid in sq_sid),
                        list(valid_ids)[0] if valid_ids else "",
                    )
                    sq["source_id"] = match

                # Resolve per-sub-query entity_filters
                sq_filters = sq.get("entity_filters") or resolved_filters or []
                if sq_filters and combined_entity_map:
                    sq_filters = [
                        combined_entity_map.get(e, entity_map_lower.get(e.lower(), e))
                        for e in sq_filters
                    ]
                # Expand group names
                sq["entity_filters"] = _expand_group_entities(sq_filters)

                # Attach semantic resolution hints
                if combined_entity_map:
                    sq["semantic_entity_map"] = combined_entity_map
                if combined_sql_frags:
                    sq["semantic_sql_fragments"] = combined_sql_frags

                sub_nl = sq.get("nl", "") or nl_query
                sq["nl"] = sub_nl

            sub_queries = decomposition_plan.get("sub_queries", [])

        # ── Build trace event ─────────────────────────────────────────────────
        latency = int((time.monotonic() - started) * 1000)
        total_tokens = intent_tokens + plan_tokens
        total_cost = intent_cost + plan_cost

        route_summary = (
            f"Cross-source: {len(sub_queries)} sub-queries → {', '.join(t[:20] for t in target_sources[:3])}"
            if is_cross_source and sub_queries
            else f"Compound: {len(sub_queries)} sub-questions"
            if is_compound and sub_queries
            else f"Single query → {(target_sources or source_ids or ['?'])[0]}"
        )
        if data_warnings:
            route_summary += f" ⚠ {data_warnings[0][:80]}"

        ora_reasoning_full = (
            f"Intent: {analytical_intent} | Compound: {is_compound} | Cross-source: {is_cross_source}\n"
            f"Entities extracted: {raw_entity_filters} → expanded to {entity_filters}\n"
            f"Coverage: {source_coverage}\n"
            f"Semantic: {len(combined_entity_map)} entity maps, {len(combined_synonyms)} synonyms\n"
            f"Reasoning: {ora_reasoning}"
        )

        return {
            "query_id": query_id,
            "nl_query": nl_query_resolved,
            "nl_query_for_pruning": nl_query_resolved,
            "display_nl_query": nl_query,  # preserve original for UI display
            "is_cross_source": is_cross_source,
            "is_compound_query": is_compound,
            "target_sources": target_sources or source_ids,
            "complexity": complexity,
            "routing_reasoning": ora_reasoning,
            "entity_filters": resolved_filters,
            "analytical_intent": analytical_intent,
            "data_warnings": data_warnings,
            "ora_reasoning": ora_reasoning_full,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "correction_round": 0,
            "max_corrections": services.config.max_corrections,
            "sub_queries": sub_queries,
            "decomposition_plan": decomposition_plan,
            "semantic_resolution": primary_resolution.to_dict() if primary_resolution else (
                semantic_reasoning.to_dict() if semantic_reasoning and semantic_reasoning.filters else None
            ),
            "semantic_reasoning": semantic_reasoning.to_dict() if semantic_reasoning else None,
            "semantic_cache_hit": False,
            "data_context_notes": existing_notes,
            "tokens_used": state.get("tokens_used", 0) + total_tokens,
            "cost_usd": state.get("cost_usd", 0.0) + total_cost,
            "budget_exhausted": False,
            "trace_events": state.get("trace_events", [])
            + ([_sem_trace] if _sem_trace else [])
            + [{
                "node": "ora",
                "status": "completed",
                "latency_ms": latency,
                "tokens": total_tokens,
                "analytical_intent": analytical_intent,
                "entity_map": combined_entity_map,
                "data_warnings": data_warnings,
                "summary": route_summary,
                "thinking": ora_reasoning_full[:300],
            }],
        }

    return ora_node

