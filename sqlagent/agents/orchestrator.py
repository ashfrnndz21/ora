"""Ora Orchestrator — thinks, decomposes, delegates, validates.

The brain of the multi-agent system. Ora never generates SQL or
resolves entities — it delegates to specialized agents and validates
every handoff. Dynamic re-routing when validation fails.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from sqlagent.agents.protocol import (
    AgentRequest, AgentResponse, ValidationResult, QueryDecomposition,
)
from sqlagent.agents.schema import SchemaAgent
from sqlagent.agents.semantic import SemanticAgent
from sqlagent.agents.sql import SQLAgent
from sqlagent.agents.learn import LearnAgent
from sqlagent.graph.ora_react import AgentTrace
from sqlagent.graph.state import QueryState

logger = structlog.get_logger()

MAX_ATTEMPTS = 3


class OraOrchestrator:
    """Ora — thinks, decomposes, delegates, validates, re-routes."""

    def __init__(self, services: Any, trace: AgentTrace):
        self._services = services
        self._trace = trace
        self._semantic = SemanticAgent(services)
        self._schema = SchemaAgent(services.connectors)
        self._sql = SQLAgent(services.ensemble, services.policy)
        self._learn = LearnAgent()

    async def run(self, state: QueryState) -> dict:
        """Full orchestration loop."""
        overall_start = time.monotonic()
        nl_query = state["nl_query"]
        source_ids = state.get("source_ids", [])
        workspace_id = state.get("workspace_id", "")
        query_id = state.get("query_id", str(uuid.uuid4())[:12])
        data_context_notes = list(state.get("data_context_notes", []))
        effective_sources = source_ids or list(self._services.connectors.keys())

        total_tokens = 0

        self._trace.record("ora", "Received query",
                           input_context=nl_query[:120],
                           output=f"Sources: {effective_sources}")

        # ══════════════════════════════════════════════════════════════
        # PHASE 1: DECOMPOSE — Ora thinks about the query structure
        # ══════════════════════════════════════════════════════════════
        decomp = await self._decompose(nl_query, effective_sources)
        total_tokens += getattr(decomp, "_tokens", 0)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: SEMANTIC RESOLUTION — call Semantic Agent
        # ══════════════════════════════════════════════════════════════
        sem_response = await self._resolve(nl_query, decomp, effective_sources, workspace_id)

        # Validate semantic output
        sem_valid = self._validate_semantic(decomp, sem_response, workspace_id)
        checks_detail = "\n".join(
            f"  {'✓' if c.get('passed') else '✗'} {c.get('check','')}"
            for c in sem_valid.checks
        )
        self._trace.record("ora", "Validates semantic output",
                           input_context=f"Confidence: {sem_response.confidence}, Filters: {len(sem_response.result.get('filters',[]))}",
                           output=f"{'Approved' if sem_valid.passed else 'ISSUES FOUND'}:\n{checks_detail}",
                           reasoning=f"Issues: {sem_valid.issues}" if sem_valid.issues else "All checks passed",
                           status="completed" if sem_valid.passed else "warning",
                           validation={"checks": sem_valid.checks, "approved": sem_valid.passed})

        # If semantic failed, retry once with explicit gaps
        if not sem_valid.passed and sem_response.confidence < 0.3:
            sem_response = await self._resolve(
                nl_query, decomp, effective_sources, workspace_id,
                feedback=f"Previous resolution incomplete. Missing: {sem_valid.issues}"
            )

        # ══════════════════════════════════════════════════════════════
        # PHASE 3: SCHEMA CONTEXT — get table structure
        # ══════════════════════════════════════════════════════════════
        all_tables = await self._schema.get_all_tables()
        pruned, pruned_schema = await self._schema.get_relevant_tables(
            nl_query, all_tables, self._services.schema_selector
        )
        selected_tables = [t.name for t in pruned]

        self._trace.record("schema_agent", "Schema context",
                           output=f"{len(pruned)} tables, {sum(len(t.columns) for t in pruned)} columns: {selected_tables}")

        # ── Schema gap detection ────────────────────────────────────
        schema_gaps = self._detect_schema_gaps(decomp, pruned)
        if schema_gaps:
            decomp.data_gaps = schema_gaps
            gap_detail = "; ".join(g["detail"][:80] for g in schema_gaps)
            self._trace.record("ora", "Data gaps detected",
                               status="warning",
                               output=f"{len(schema_gaps)} gap(s): {gap_detail}",
                               reasoning="Some decomposition parts cannot be fully answered with available schema")

        # ══════════════════════════════════════════════════════════════
        # PHASE 4: BUILD QUERY SPEC — Ora assembles for SQL Agent
        # ══════════════════════════════════════════════════════════════
        # Validate semantic column names against actual schema before building spec
        actual_columns = set()
        for t in pruned:
            for c in t.columns:
                actual_columns.add(c.name.lower())

        query_spec = self._build_spec(nl_query, decomp, sem_response, actual_columns)

        # Inject schema gaps into spec so SQL Agent and Response Writer know
        if schema_gaps:
            gap_notes = "\n".join(f"  - {g['detail']}" for g in schema_gaps)
            query_spec += (
                f"\n\nDATA LIMITATIONS (communicate these honestly in the response):\n"
                f"{gap_notes}\n"
                f"Generate the best SQL possible with available columns. "
                f"Do NOT invent columns that don't exist.\n"
            )

        self._trace.record("ora", "Built query spec",
                           input_context=f"Decomposition: {len(decomp.parts)} parts, Semantic: {len(sem_response.result.get('filters',[]))} filters",
                           output=query_spec[:150],
                           reasoning="Raw user terms replaced with resolved DB values"
                           + (f" | {len(schema_gaps)} data gaps flagged" if schema_gaps else ""))

        # ══════════════════════════════════════════════════════════════
        # PHASE 5: SQL → VALIDATE → EXECUTE → CHECK (ReAct loop)
        # ══════════════════════════════════════════════════════════════
        sql = ""
        rows, columns = [], []
        row_count = 0
        execution_error = ""
        succeeded = False
        winner_generator = ""
        correction_round = 0

        # Get examples
        similar_examples = []
        try:
            if self._services.example_store and hasattr(self._services.example_store, 'search'):
                similar_examples = await self._services.example_store.search(nl_query, top_k=3)
        except Exception:
            pass

        # ── Load learned rules ───────────────────────────────────
        active_rules = []
        try:
            from sqlagent.rules import load_rules
            active_rules = load_rules(workspace_id)
            if active_rules:
                # Inject top rules into data_context_notes for SQL Agent
                for rule in active_rules[:5]:
                    if rule.get("text") and rule["text"] not in data_context_notes:
                        data_context_notes.append(rule["text"])
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════════
        # PHASE 4b: APPLY LEARNING — show what learned knowledge Ora is using
        # ══════════════════════════════════════════════════════════════
        # Load semantic manifest for iteration info
        manifest_iter = 0
        try:
            from sqlagent.semantic_agent import _load_manifest
            manifest = _load_manifest(workspace_id)
            manifest_iter = manifest.get("iteration_id", 0)
        except Exception:
            pass

        # Count pre-resolved aliases from semantic agent
        pre_resolved_count = len(sem_response.result.get("new_aliases", {}))

        learning_parts = []
        if similar_examples:
            best = similar_examples[0]
            best_nl = getattr(best, 'nl_query', best.get('nl_query', ''))[:60] if isinstance(best, dict) else str(best)[:60]
            best_score = getattr(best, 'score', getattr(best, 'similarity', 0))
            learning_parts.append(f"{len(similar_examples)} past examples (best: '{best_nl}' sim:{best_score:.2f})")
        if active_rules:
            rule_names = [r.get("text", "")[:40] for r in active_rules[:3]]
            learning_parts.append(f"{len(active_rules)} rules: {'; '.join(rule_names)}")
        if data_context_notes:
            learning_parts.append(f"{len(data_context_notes)} context notes")
        if pre_resolved_count:
            learning_parts.append(f"{pre_resolved_count} pre-resolved aliases")
        if manifest_iter:
            learning_parts.append(f"semantic layer v{manifest_iter}")

        if learning_parts:
            self._trace.record("learn", "Applied learned context",
                               output=" | ".join(learning_parts),
                               reasoning=(
                                   f"Examples: {len(similar_examples)}, "
                                   f"Rules: {len(active_rules)}, "
                                   f"Context notes: {len(data_context_notes)}, "
                                   f"Pre-resolved: {pre_resolved_count}, "
                                   f"Semantic iteration: {manifest_iter}"
                               ),
                               status="completed")
        else:
            self._trace.record("learn", "No learned context available",
                               output="First query — no prior learning to apply",
                               status="completed")

        for attempt in range(MAX_ATTEMPTS):
            # ── SQL Agent ────────────────────────────────────────
            self._trace.record("ora", f"Calling SQL Agent (attempt {attempt + 1})",
                               input_context=query_spec[:100])

            sql_response = await self._sql.handle(
                AgentRequest(to_agent="sql_agent", action="generate",
                             context={"query_spec": query_spec, "pruned_schema": pruned_schema,
                                      "examples": similar_examples, "context_notes": data_context_notes}),
                self._trace,
            )

            if sql_response.status == "failed":
                break

            sql = sql_response.result.get("sql", "")
            winner_generator = sql_response.result.get("generator", "")
            total_tokens += sql_response.result.get("tokens", 0)

            if not sql:
                break

            # ── Ora validates SQL ────────────────────────────────
            sql, sql_valid = self._validate_sql(sql, sem_response)
            sql_checks_detail = "\n".join(
                f"  {'✓' if c.get('passed') else '✗'} {c.get('check','')}" + (f" [FIXED]" if c.get('fixed') else "")
                for c in sql_valid.checks
            )
            self._trace.record("ora", "Validates SQL",
                               input_context=f"Checking {len(sql_valid.checks)} filter values + table presence",
                               output=f"{'Approved' if not sql_valid.issues else 'Fixed issues'}:\n{sql_checks_detail}",
                               reasoning=f"Fixes applied: {sql_valid.issues}" if sql_valid.issues else "All values correct",
                               status="completed",
                               validation={"checks": sql_valid.checks, "approved": sql_valid.passed})

            # ── Execute ──────────────────────────────────────────
            exec_start = time.monotonic()
            source_id = list(self._services.connectors.keys())[0] if self._services.connectors else None
            conn = self._services.connectors.get(source_id)

            if not conn:
                execution_error = "No connector available"
                break

            # Register cross-file tables
            await self._schema.register_cross_tables(source_id)

            # Policy check
            if self._services.policy:
                pr = self._services.policy.check(sql, state)
                if not pr.passed:
                    execution_error = f"Policy: {pr.reason}"
                    self._trace.record("execute", "Policy blocked", status="failed", output=pr.reason)
                    break

            try:
                import pandas as _pd
                result = await conn.execute(sql, timeout_s=self._services.config.query_timeout_s)
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
            self._trace.record("execute", "Run SQL",
                               status="completed" if succeeded else "failed",
                               output=f"{row_count} rows" if succeeded else execution_error[:80],
                               latency_ms=exec_ms)

            # ── Ora validates result ─────────────────────────────
            result_valid = self._validate_result(rows, row_count, execution_error, decomp, sql)
            self._trace.record("ora", "Validates result",
                               status="completed" if result_valid.passed else "retry",
                               output=f"✓ {row_count} rows — result matches question" if result_valid.passed
                                      else f"Issues: {result_valid.issues[:2]}",
                               validation={"checks": result_valid.checks, "approved": result_valid.passed})

            if result_valid.passed:
                # ── Semantic fitness check ──────────────────────────
                # Only on first attempt — skip on correction rounds to avoid timeout.
                # Corrections already have targeted fix instructions.
                if correction_round > 0:
                    self._trace.record("ora", "Semantic fitness check",
                                       status="completed",
                                       output="Skipped (correction round — fix was targeted)")
                    break

                fitness = await self._check_semantic_fitness(
                    nl_query, sql, columns, rows[:5], decomp,
                )
                if fitness["fit"]:
                    self._trace.record("ora", "Semantic fitness check",
                                       status="completed",
                                       output=f"✓ {fitness['reasoning'][:80]}")
                    break
                else:
                    self._trace.record("ora", "Semantic fitness check",
                                       status="retry",
                                       output=f"✗ {fitness['reasoning'][:80]}",
                                       reasoning=fitness.get("fix_hint", ""))
                    # Build a corrected spec using the fitness feedback
                    query_spec = (
                        f"PREVIOUS SQL DOES NOT ANSWER THE QUESTION.\n"
                        f"Question: {nl_query}\n\n"
                        f"Previous SQL:\n{sql}\n\n"
                        f"Problem: {fitness['reasoning']}\n"
                        f"Fix: {fitness.get('fix_hint', 'Rewrite the SQL to properly answer the question.')}\n\n"
                        f"Schema: {', '.join(selected_tables)}\n"
                        f"Available columns: {', '.join(c.name for t in pruned for c in t.columns[:20])}\n\n"
                        f"Write corrected SQL. Output ONLY SQL."
                    )
                    correction_round += 1
                    succeeded = False
                    continue

            # ── Re-route based on failure type ───────────────────
            if row_count == 0 and succeeded:
                self._trace.record("ora", "Diagnoses: 0 rows",
                                   reasoning="SQL ran but returned no data. Adjusting query.",
                                   status="retry")
                query_spec = (
                    f"Previous SQL returned 0 rows:\n{sql}\n\n"
                    f"Question: {nl_query}\nSchema: {', '.join(selected_tables)}\n"
                    f"Fix the SQL to return results. Check filter values against actual data."
                )
                correction_round += 1
                succeeded = False
                continue

            if execution_error:
                self._trace.record("ora", "Diagnoses: SQL error",
                                   reasoning=execution_error[:100],
                                   status="retry")
                # Build a complete column reference so the SQL Agent uses correct names
                from sqlagent.schema import MSchemaSerializer
                schema_text = MSchemaSerializer.serialize(pruned) if pruned else ""
                # Also list columns explicitly to prevent wrong column name errors
                col_ref = ""
                for t in pruned:
                    cols = [f"{c.name} ({c.data_type})" for c in t.columns[:30]]
                    col_ref += f"\nTable {t.name} columns: {', '.join(cols)}"
                query_spec = (
                    f"SQL error: {execution_error}\n\n"
                    f"Question: {nl_query}\n\n"
                    f"ACTUAL SCHEMA (use ONLY these column names):{col_ref}\n\n"
                    f"Full schema:\n{schema_text[:3000]}\n"
                    f"Fix the SQL using ONLY the column names listed above. Output ONLY SQL."
                )
                correction_round += 1
                continue

        # ══════════════════════════════════════════════════════════════
        # PHASE 6: LEARN — evolve semantic layer
        # ══════════════════════════════════════════════════════════════
        semantic_reasoning_dict = sem_response.result.get("reasoning_obj")
        if succeeded and workspace_id and semantic_reasoning_dict:
            sr_dict = semantic_reasoning_dict.to_dict() if hasattr(semantic_reasoning_dict, 'to_dict') else semantic_reasoning_dict
            await self._learn.handle(
                AgentRequest(to_agent="learn", action="evolve",
                             context={"workspace_id": workspace_id,
                                      "semantic_reasoning": sr_dict,
                                      "sources": effective_sources,
                                      "sql": sql, "nl_query": nl_query}),
                self._trace,
            )

        # ── Record rule outcomes ─────────────────────────────────────
        if active_rules and workspace_id:
            try:
                from sqlagent.rules import record_rule_outcome
                applied_ids = [r["rule_id"] for r in active_rules[:5] if r.get("rule_id")]
                record_rule_outcome(workspace_id, applied_ids, succeeded)
            except Exception:
                pass

        # ── Record failure for semantic reflection ───────────────────
        if not succeeded and workspace_id and semantic_reasoning_dict:
            try:
                from sqlagent.semantic_agent import record_resolution_failure
                sr = semantic_reasoning_dict
                sr_dict_fail = sr.to_dict() if hasattr(sr, 'to_dict') else sr
                record_resolution_failure(
                    workspace_id=workspace_id,
                    question=nl_query,
                    failed_filters=sr_dict_fail.get("filters", []) if isinstance(sr_dict_fail, dict) else [],
                    error=execution_error or "empty result",
                )
            except Exception:
                pass  # Never break the pipeline over logging

        # ══════════════════════════════════════════════════════════════
        # FINAL — build return state
        # ══════════════════════════════════════════════════════════════
        overall_ms = int((time.monotonic() - overall_start) * 1000)
        self._trace.record("ora", "Complete",
                           status="completed" if succeeded else "failed",
                           output=f"{row_count} rows, {correction_round} corrections, {total_tokens} tokens",
                           latency_ms=overall_ms)

        sr = sem_response.result.get("reasoning_obj")
        sr_dict = sr.to_dict() if hasattr(sr, 'to_dict') else (sr if isinstance(sr, dict) else None)

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
            "semantic_reasoning": sr_dict,
            "target_sources": effective_sources,
            "is_cross_source": decomp.is_cross_source,
            "data_gaps": [g["detail"] for g in decomp.data_gaps] if decomp.data_gaps else [],
            "is_compound_query": len(decomp.parts) > 1,
            "complexity": "complex" if len(decomp.parts) > 2 else "moderate",
            "data_warnings": [],
            "entity_filters": [],
            "routing_reasoning": sem_response.reasoning,
            "ora_reasoning": sem_response.reasoning,
            "analytical_intent": decomp.comparison_type or "comparison",
            "plan_reasoning": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "max_corrections": MAX_ATTEMPTS,
            "sub_queries": [],
            "decomposition_plan": None,
            "semantic_resolution": sr_dict,
            "semantic_cache_hit": False,
            "data_context_notes": data_context_notes,
            "tokens_used": state.get("tokens_used", 0) + total_tokens,
            "cost_usd": state.get("cost_usd", 0.0),
            "budget_exhausted": False,
            "trace_events": self._trace.to_trace_events(),
        }

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ══════════════════════════════════════════════════════════════════

    async def _decompose(self, query: str, source_ids: list[str] | None = None) -> QueryDecomposition:
        """Ora THINKS — decomposes query into logical parts via LLM.

        Source-aware: the LLM sees available data sources and assigns each
        part to a target_domain. Ora then matches domains to sources to detect
        cross-source queries.
        """
        start = time.monotonic()
        decomp = QueryDecomposition(raw_query=query)

        # Build source descriptions for the LLM
        source_desc = ""
        source_table_map: dict[str, list[str]] = {}  # source_id → table names
        if source_ids and self._services.connectors:
            for sid in source_ids:
                conn = self._services.connectors.get(sid)
                if conn:
                    try:
                        snap = await conn.introspect()
                        tables = [t.name for t in snap.tables[:8]]
                        source_table_map[sid] = tables
                        cols_sample = []
                        for t in snap.tables[:4]:
                            cols_sample.append(
                                f"    {t.name}: {', '.join(c.name for c in t.columns[:8])}"
                            )
                        source_desc += f"\n  Source '{sid}':\n" + "\n".join(cols_sample)
                    except Exception:
                        source_table_map[sid] = []

        try:
            prompt = (
                f"Decompose this question into logical parts. Identify:\n"
                f"1. What sub-questions does it contain?\n"
                f"2. What entities need to be resolved to database values?\n"
                f"3. What calculations are needed (ratio, average, percentage, ranking)?\n"
                f"4. What comparisons are requested (A vs B, trend, ranking)?\n"
                f"5. Which data domain does each part need?\n\n"
                f"Question: {query}\n\n"
            )
            if source_desc:
                prompt += (
                    f"AVAILABLE DATA SOURCES:{source_desc}\n\n"
                    f"For each part, specify target_domain: a short label for the type of data "
                    f"it needs (e.g. 'revenue/spend data', 'employment data', 'customer data').\n\n"
                )
            prompt += (
                f"Return JSON:\n"
                f'{{"parts": [{{"id": "A", "description": "what this part asks", '
                f'"target_domain": "what data domain it needs"}}], '
                f'"entities_to_resolve": ["list of user terms needing DB resolution"], '
                f'"calculations_needed": ["describe derived metrics"], '
                f'"comparison_type": "how parts relate"}}\n'
                f"Return ONLY valid JSON."
            )

            resp = await self._services.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512, json_mode=True,
            )

            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()

            parsed = json.loads(raw)
            decomp.parts = parsed.get("parts", [])
            decomp.entities_to_resolve = parsed.get("entities_to_resolve", [])
            decomp.calculations_needed = parsed.get("calculations_needed", [])
            decomp.comparison_type = parsed.get("comparison_type", "")
            decomp._tokens = getattr(resp, "tokens_input", 0) + getattr(resp, "tokens_output", 0)

            # ── Cross-source detection ─────────────────────────────
            # Match each part's target_domain against source tables
            if source_table_map and len(decomp.parts) > 1:
                domains = set()
                for part in decomp.parts:
                    domain = part.get("target_domain", "").lower()
                    if domain:
                        domains.add(domain)
                        # Try to match domain to a specific source
                        for sid, tables in source_table_map.items():
                            tables_lower = " ".join(t.lower() for t in tables)
                            # Simple keyword matching: does any table name overlap with domain?
                            domain_words = set(domain.replace("/", " ").split())
                            if any(w in tables_lower for w in domain_words if len(w) > 3):
                                part["target_source"] = sid
                                break

                # If parts target different sources → cross-source
                targeted_sources = set(
                    p.get("target_source", "") for p in decomp.parts
                    if p.get("target_source")
                )
                if len(targeted_sources) > 1:
                    decomp.is_cross_source = True

        except Exception as exc:
            logger.warning("ora.decompose_failed", error=str(exc))
            decomp.parts = [{"id": "A", "description": query}]
            decomp.entities_to_resolve = []
            decomp._tokens = 0

        latency = int((time.monotonic() - start) * 1000)
        parts_detail = " | ".join(
            f"Part {p.get('id','?')}: {p.get('description','')[:50]}"
            + (f" [{p.get('target_source','')}]" if p.get('target_source') else "")
            for p in decomp.parts
        )
        self._trace.record("ora", "Decomposed query",
                           input_context=query[:120],
                           output=f"Parts: {parts_detail}" + (" [CROSS-SOURCE]" if decomp.is_cross_source else ""),
                           reasoning=(
                               f"Entities to resolve: {decomp.entities_to_resolve}\n"
                               f"Calculations: {decomp.calculations_needed}\n"
                               f"Comparison: {decomp.comparison_type}\n"
                               f"Cross-source: {decomp.is_cross_source}"
                           ),
                           latency_ms=latency)

        return decomp

    async def _resolve(self, query, decomp, sources, workspace_id, feedback=""):
        """Call Semantic Agent with decomposition context."""
        return await self._semantic.handle(
            AgentRequest(to_agent="semantic_agent", action="resolve_entities",
                         context={"query": query, "decomposition": {
                             "parts": decomp.parts,
                             "entities": decomp.entities_to_resolve,
                             "calculations": decomp.calculations_needed,
                             "comparison_type": decomp.comparison_type,
                         }, "source_ids": sources, "workspace_id": workspace_id,
                         "feedback": feedback}),
            self._trace,
        )

    # Fallback group sizes — used ONLY when the semantic manifest has no
    # group_memberships data. Once build_initial_taxonomy() runs, the
    # manifest's detected groups take priority over this static list.
    _FALLBACK_GROUP_SIZES = {
        "asean": 10, "g7": 7, "g20": 20, "eu": 27, "brics": 5,
        "apec": 21, "oecd": 38, "gcc": 6, "nafta": 3, "mercosur": 5,
    }

    def _validate_semantic(self, decomp, response, workspace_id="") -> ValidationResult:
        """Check semantic output covers all decomposition parts.

        Validates STRUCTURE (tables, filters, metrics exist) AND COVERAGE
        (resolved filters are sufficient for the decomposition's scope).
        """
        issues = []
        checks = []

        # Check confidence
        conf = response.confidence
        checks.append({"check": "confidence >= 0.5", "passed": conf >= 0.5})
        if conf < 0.5:
            issues.append(f"Low confidence: {conf}")

        # Check tables identified
        tables = response.result.get("tables", [])
        checks.append({"check": "tables identified", "passed": len(tables) > 0})
        if not tables:
            issues.append("No tables identified")

        # Check filters exist
        filters = response.result.get("filters", [])
        checks.append({"check": "filters resolved", "passed": len(filters) > 0})
        if not filters:
            issues.append("No filters resolved")

        # Check calculations coverage
        if decomp.calculations_needed:
            metrics = response.result.get("metrics", [])
            checks.append({"check": "metrics for calculations", "passed": len(metrics) > 0})
            if not metrics:
                issues.append("Calculations needed but no metrics identified")

        # ── Entity coverage check ────────────────────────────────────
        # Cross-reference decomposition scope against resolved filters.
        # If decomp says "ALL ASEAN" but filters only have 2 countries,
        # Ora catches it and sends back for re-resolution.
        decomp_text = " ".join(
            p.get("description", "") for p in decomp.parts
        ).lower() + " " + " ".join(
            e.lower() for e in decomp.entities_to_resolve
        )

        # Detect quantifier + group patterns
        quantifiers = r'\b(all|every|each|entire|the rest|compared? to (?:the )?rest)\b'
        has_quantifier = bool(re.search(quantifiers, decomp_text))

        if has_quantifier and filters:
            # Count unique filter values (how many entities resolved)
            filter_values = set()
            for f in filters:
                v = f.get("value", "")
                if isinstance(v, list):
                    for item in v:
                        if item:
                            filter_values.add(str(item))
                elif v:
                    filter_values.add(str(v))

            # Build group sizes: prefer manifest (dynamic) over fallback (static)
            group_sizes = dict(self._FALLBACK_GROUP_SIZES)
            try:
                from sqlagent.semantic_agent import _load_manifest
                if workspace_id:
                    manifest = _load_manifest(workspace_id)
                    taxonomy = manifest.get("taxonomy", {})
                    for col, col_groups in taxonomy.get("group_memberships", {}).items():
                        for gname, members in col_groups.items():
                            if isinstance(members, list) and len(members) > 1:
                                group_sizes[gname.lower()] = len(members)
            except Exception:
                pass  # fall back to static sizes

            for group_name, expected_size in group_sizes.items():
                if group_name in decomp_text:
                    coverage_ok = len(filter_values) >= expected_size // 2
                    checks.append({
                        "check": f"entity coverage for {group_name.upper()}",
                        "passed": coverage_ok,
                        "detail": f"{len(filter_values)} filter values vs ~{expected_size} expected",
                    })
                    if not coverage_ok:
                        issues.append(
                            f"Decomposition requires ALL {group_name.upper()} "
                            f"(~{expected_size} members) but semantic resolution "
                            f"only resolved {len(filter_values)} filter values. "
                            f"The query asks for the full group, not a subset."
                        )
                    break  # only check the first matching group

        return ValidationResult(passed=len(issues) == 0, issues=issues, checks=checks)

    def _build_spec(self, query, decomp, sem_response, actual_columns: set | None = None) -> str:
        """Build the COMPLETE query spec for SQL Agent.

        Validates semantic column names against actual_columns from the pruned
        schema. Strips columns that don't exist and warns the SQL Agent.
        """
        filters = sem_response.result.get("filters", [])
        resolved = sem_response.result.get("resolved_query", query)
        tables = sem_response.result.get("tables", [])
        metrics = sem_response.result.get("metrics", [])
        calculations = sem_response.result.get("calculations", [])

        # Validate filter columns against actual schema
        where_parts = []
        invalid_cols = []
        for f in filters:
            col = f.get("column", "")
            op = f.get("operator", "=")
            val = f.get("value", "")

            # Check if column exists in actual schema
            if actual_columns and col and col.lower() not in actual_columns:
                invalid_cols.append(col)
                continue  # skip filters with non-existent columns

            if isinstance(val, list):
                val_str = ", ".join(f"'{v}'" for v in val if isinstance(v, str))
                if val_str:
                    where_parts.append(f"{col} IN ({val_str})")
            elif isinstance(val, str) and val:
                where_parts.append(f"{col} {op} '{val}'")

        # Validate metrics against actual schema
        if actual_columns and metrics:
            valid_metrics = [m for m in metrics if m.lower() in actual_columns]
            bad_metrics = [m for m in metrics if m.lower() not in actual_columns]
            if bad_metrics:
                invalid_cols.extend(bad_metrics)
            metrics = valid_metrics

        spec = f"{resolved}\n\n"
        if where_parts:
            spec += "COPY THESE EXACT SQL FRAGMENTS INTO YOUR WHERE CLAUSE:\n"
            spec += "\n".join(f"  {wp}" for wp in where_parts) + "\n"
        if metrics:
            spec += f"SELECT columns: {', '.join(metrics)}\n"
        if tables:
            spec += f"FROM tables: {', '.join(tables)}\n"
        if calculations:
            spec += f"CALCULATIONS: {'; '.join(calculations)}\n"
        if invalid_cols:
            spec += (
                f"\nWARNING: These columns do NOT exist in the schema: {', '.join(invalid_cols)}. "
                f"Use the actual column names from the schema provided.\n"
            )
        spec += "\nThe values above are EXACT database strings. Copy them character-for-character.\n"

        # Substitute user terms with resolved values
        new_aliases = sem_response.result.get("new_aliases", {})
        if new_aliases:
            for user_term, stored_val in sorted(new_aliases.items(), key=lambda x: -len(x[0])):
                if isinstance(user_term, str) and isinstance(stored_val, str) and user_term != stored_val:
                    spec = re.sub(r'\b' + re.escape(user_term) + r'\b', stored_val, spec, flags=re.IGNORECASE)

        return spec

    async def _check_semantic_fitness(
        self, question: str, sql: str, columns: list, sample_rows: list, decomp,
    ) -> dict:
        """Ora reviews its own work — does the SQL + result actually answer the question?

        This is the analyst double-check: not "did it run?" but "does it make sense?"
        One LLM call. Returns {"fit": bool, "reasoning": str, "fix_hint": str}.
        """
        # Build a compact view of what came back
        cols_str = ", ".join(columns[:15]) if columns else "(no columns)"
        sample_str = ""
        if sample_rows:
            for row in sample_rows[:3]:
                vals = [f"{k}={v}" for k, v in (row.items() if isinstance(row, dict) else [])]
                sample_str += "  " + ", ".join(vals[:6]) + "\n"

        parts_str = " | ".join(
            f"Part {p.get('id','?')}: {p.get('description','')[:60]}"
            for p in decomp.parts
        )

        prompt = (
            "You are reviewing a SQL query and its results to check if they actually "
            "answer the user's question. Think like a senior data analyst checking "
            "a junior's work before presenting to the client.\n\n"
            f"ORIGINAL QUESTION: {question}\n\n"
            f"DECOMPOSITION: {parts_str}\n\n"
            f"SQL:\n{sql}\n\n"
            f"RESULT COLUMNS: {cols_str}\n"
            f"SAMPLE ROWS:\n{sample_str or '(empty)'}\n\n"
            "Check ALL of these:\n"
            "1. Does the SQL query ALL entities the user mentioned? (not just some)\n"
            "2. For trend/time queries: does the result include a time column for each row?\n"
            "3. For comparison queries: does the result include an identifier column "
            "(country, category, etc.) so rows can be distinguished?\n"
            "4. For correlation queries: are the entities being correlated in the same "
            "result set with aligned dimensions?\n"
            "5. Does the SQL match the decomposition parts, or were some parts dropped?\n\n"
            "Return JSON:\n"
            '{"fit": true/false, "reasoning": "one sentence why it fits or not", '
            '"fix_hint": "if not fit: what specific change to the SQL would fix it"}\n'
            "Return ONLY valid JSON."
        )

        try:
            resp = await self._services.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=256, json_mode=True,
            )
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()
            parsed = json.loads(raw)
            return {
                "fit": parsed.get("fit", True),
                "reasoning": parsed.get("reasoning", ""),
                "fix_hint": parsed.get("fix_hint", ""),
            }
        except Exception as exc:
            logger.warning("ora.fitness_check_failed", error=str(exc))
            return {"fit": True, "reasoning": "Fitness check failed — proceeding", "fix_hint": ""}

    def _detect_schema_gaps(self, decomp, pruned) -> list[dict]:
        """Check if pruned schema can support each decomposition part's requirements.

        Detects when a query asks for temporal trends but the table has no date column,
        or asks for geographic breakdown but no country column exists.
        """
        gaps = []
        time_keywords = {"trend", "uptrend", "downtrend", "over time", "cycle", "year",
                         "yearly", "annual", "quarterly", "monthly", "period"}
        geo_keywords = {"country", "region", "by country", "per country", "geographic",
                        "by region", "per region"}

        # Build column type index from pruned schema
        has_time_col = False
        has_geo_col = False
        table_cols: dict[str, set] = {}
        for t in pruned:
            cols = set()
            for c in t.columns:
                cols.add(c.name.lower())
                dt = (c.data_type or "").lower()
                if any(kw in c.name.lower() for kw in ("year", "date", "time", "period", "month", "quarter")):
                    has_time_col = True
                if any(kw in dt for kw in ("date", "time", "timestamp")):
                    has_time_col = True
                if any(kw in c.name.lower() for kw in ("country", "region", "geo", "iso", "state", "city")):
                    has_geo_col = True
            table_cols[t.name] = cols

        for part in decomp.parts:
            desc = part.get("description", "").lower()

            # Check time dimension
            if any(kw in desc for kw in time_keywords) and not has_time_col:
                gaps.append({
                    "part_id": part.get("id", "?"),
                    "gap": "no time dimension",
                    "detail": (
                        f"Part {part.get('id','?')} asks about temporal patterns "
                        f"('{next(kw for kw in time_keywords if kw in desc)}') "
                        f"but no date/year/time column found in pruned schema. "
                        f"Tables: {list(table_cols.keys())}"
                    ),
                })

            # Check geographic dimension
            if any(kw in desc for kw in geo_keywords) and not has_geo_col:
                gaps.append({
                    "part_id": part.get("id", "?"),
                    "gap": "no geography dimension",
                    "detail": (
                        f"Part {part.get('id','?')} asks about geographic breakdown "
                        f"but no country/region column found in pruned schema."
                    ),
                })

        return gaps

    def _validate_sql(self, sql, sem_response) -> tuple[str, ValidationResult]:
        """Validate SQL structure and fix wrong filter values."""
        issues = []
        checks = []

        filters = sem_response.result.get("filters", [])
        tables = sem_response.result.get("tables", [])

        # Check all tables present
        for t in tables:
            present = t.lower() in sql.lower()
            checks.append({"check": f"Table '{t}' in SQL", "passed": present})
            if not present:
                issues.append(f"Missing table: {t}")

        # Check and fix filter values
        for f in filters:
            val = f.get("value", "")
            col = f.get("column", "")
            if not val or not isinstance(val, str) or not col:
                continue
            if f"'{val}'" in sql or f'"{val}"' in sql:
                checks.append({"check": f"{col}='{val}'", "passed": True})
            else:
                # Try to fix
                pattern = re.compile(rf"({re.escape(col)}\s*=\s*)'([^']*)'", re.IGNORECASE)
                match = pattern.search(sql)
                if match and match.group(2) != val:
                    old = match.group(2)
                    sql = sql.replace(f"'{old}'", f"'{val}'")
                    issues.append(f"Fixed {col}: '{old}'→'{val}'")
                    checks.append({"check": f"{col}='{val}'", "passed": True, "fixed": True})
                else:
                    checks.append({"check": f"{col}='{val}'", "passed": False})

        return sql, ValidationResult(passed=True, issues=issues, checks=checks)  # always pass after fixes

    def _validate_result(self, rows, row_count, error, decomp, sql="") -> ValidationResult:
        """Validate execution result answers the question — checks against decomposition."""
        issues = []
        checks = []

        if error:
            issues.append(f"Execution error: {error[:80]}")
            checks.append({"check": "no errors", "passed": False})
            return ValidationResult(passed=False, issues=issues, checks=checks)

        # Check row count
        checks.append({"check": "rows > 0", "passed": row_count > 0})
        if row_count == 0:
            issues.append("Zero rows returned")
            return ValidationResult(passed=False, issues=issues, checks=checks)

        # Check against decomposition: does result have enough data for all parts?
        num_parts = len(decomp.parts)
        if num_parts > 1 and row_count < num_parts:
            issues.append(f"Question has {num_parts} parts but only {row_count} rows — may be incomplete")
            checks.append({"check": f"rows >= parts ({num_parts})", "passed": row_count >= num_parts})

        # Check if result columns cover expected metrics
        if rows and decomp.calculations_needed:
            has_numeric = any(
                isinstance(v, (int, float)) for v in (rows[0].values() if rows else [])
            )
            checks.append({"check": "has numeric columns for calculations", "passed": has_numeric})
            if not has_numeric:
                issues.append("No numeric columns in result — calculations may be missing")

        # ── Decomposition coverage check ─────────────────────────────
        # If cross-source: verify SQL touches tables from each target source
        if sql and decomp.is_cross_source and len(decomp.parts) > 1:
            sql_lower = sql.lower()
            for part in decomp.parts:
                target = part.get("target_source", "")
                if not target:
                    continue
                # Check if any table from this source appears in the SQL
                conn = self._services.connectors.get(target)
                if conn:
                    try:
                        snap_tables = [t.name.lower() for t in (getattr(conn, '_last_snap', None) or type('', (), {'tables': []})()).tables]
                    except Exception:
                        snap_tables = []
                    if snap_tables and not any(t in sql_lower for t in snap_tables):
                        part_desc = part.get("description", "")[:60]
                        issues.append(
                            f"Part {part.get('id','?')} ('{part_desc}') targets source '{target}' "
                            f"but no tables from that source appear in the SQL"
                        )
                        checks.append({
                            "check": f"Part {part.get('id','?')} source coverage",
                            "passed": False,
                        })

        # Check for data gaps flagged during schema analysis
        if decomp.data_gaps:
            for gap in decomp.data_gaps:
                checks.append({
                    "check": f"data gap: {gap.get('gap', '')}",
                    "passed": False,
                    "detail": gap.get("detail", ""),
                })
            # Data gaps are informational — don't fail the query, but note them
            # The Response Writer should explain the limitation

        passed = row_count > 0 and not any(
            "Execution error" in i or "Zero rows" in i for i in issues
        )
        return ValidationResult(passed=passed, issues=issues, checks=checks)
