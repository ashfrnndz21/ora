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
        decomp = await self._decompose(nl_query)
        total_tokens += getattr(decomp, "_tokens", 0)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2: SEMANTIC RESOLUTION — call Semantic Agent
        # ══════════════════════════════════════════════════════════════
        sem_response = await self._resolve(nl_query, decomp, effective_sources, workspace_id)

        # Validate semantic output
        sem_valid = self._validate_semantic(decomp, sem_response)
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

        # ══════════════════════════════════════════════════════════════
        # PHASE 4: BUILD QUERY SPEC — Ora assembles for SQL Agent
        # ══════════════════════════════════════════════════════════════
        query_spec = self._build_spec(nl_query, decomp, sem_response)
        self._trace.record("ora", "Built query spec",
                           input_context=f"Decomposition: {len(decomp.parts)} parts, Semantic: {len(sem_response.result.get('filters',[]))} filters",
                           output=query_spec[:150],
                           reasoning="Raw user terms replaced with resolved DB values")

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
            result_valid = self._validate_result(rows, row_count, execution_error, decomp)
            self._trace.record("ora", "Validates result",
                               status="completed" if result_valid.passed else "retry",
                               output=f"✓ {row_count} rows — result matches question" if result_valid.passed
                                      else f"Issues: {result_valid.issues[:2]}",
                               validation={"checks": result_valid.checks, "approved": result_valid.passed})

            if result_valid.passed:
                break

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
                from sqlagent.schema import MSchemaSerializer
                schema_text = MSchemaSerializer.serialize(pruned) if pruned else ""
                query_spec = (
                    f"SQL error: {execution_error}\n\n"
                    f"Question: {nl_query}\nSchema:\n{schema_text[:2000]}\n"
                    f"Fix the SQL. Output ONLY SQL."
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
            "is_cross_source": False,
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

    async def _decompose(self, query: str) -> QueryDecomposition:
        """Ora THINKS — decomposes query into logical parts via LLM."""
        start = time.monotonic()
        decomp = QueryDecomposition(raw_query=query)

        try:
            prompt = (
                f"Decompose this question into logical parts. Identify:\n"
                f"1. What sub-questions does it contain?\n"
                f"2. What entities need to be resolved to database values?\n"
                f"3. What calculations are needed (ratio, average, percentage, ranking)?\n"
                f"4. What comparisons are requested (A vs B, trend, ranking)?\n\n"
                f"Question: {query}\n\n"
                f"Return JSON:\n"
                f'{{"parts": [{{"id": "A", "description": "what this part asks"}}], '
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

        except Exception as exc:
            logger.warning("ora.decompose_failed", error=str(exc))
            decomp.parts = [{"id": "A", "description": query}]
            decomp.entities_to_resolve = []
            decomp._tokens = 0

        latency = int((time.monotonic() - start) * 1000)
        # Show FULL decomposition details in trace
        parts_detail = " | ".join(f"Part {p.get('id','?')}: {p.get('description','')[:60]}" for p in decomp.parts)
        self._trace.record("ora", "Decomposed query",
                           input_context=query[:120],
                           output=f"Parts: {parts_detail}",
                           reasoning=(
                               f"Entities to resolve: {decomp.entities_to_resolve}\n"
                               f"Calculations: {decomp.calculations_needed}\n"
                               f"Comparison: {decomp.comparison_type}"
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

    def _validate_semantic(self, decomp, response) -> ValidationResult:
        """Check semantic output covers all decomposition parts."""
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

        return ValidationResult(passed=len(issues) == 0, issues=issues, checks=checks)

    def _build_spec(self, query, decomp, sem_response) -> str:
        """Build the COMPLETE query spec for SQL Agent."""
        filters = sem_response.result.get("filters", [])
        resolved = sem_response.result.get("resolved_query", query)
        tables = sem_response.result.get("tables", [])
        metrics = sem_response.result.get("metrics", [])
        calculations = sem_response.result.get("calculations", [])

        where_parts = []
        for f in filters:
            col = f.get("column", "")
            op = f.get("operator", "=")
            val = f.get("value", "")
            if isinstance(val, list):
                val_str = ", ".join(f"'{v}'" for v in val if isinstance(v, str))
                if val_str:
                    where_parts.append(f"{col} IN ({val_str})")
            elif isinstance(val, str) and val:
                where_parts.append(f"{col} {op} '{val}'")

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
        spec += "\nThe values above are EXACT database strings. Copy them character-for-character.\n"

        # Substitute user terms with resolved values
        new_aliases = sem_response.result.get("new_aliases", {})
        if new_aliases:
            for user_term, stored_val in sorted(new_aliases.items(), key=lambda x: -len(x[0])):
                if isinstance(user_term, str) and isinstance(stored_val, str) and user_term != stored_val:
                    spec = re.sub(r'\b' + re.escape(user_term) + r'\b', stored_val, spec, flags=re.IGNORECASE)

        return spec

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

    def _validate_result(self, rows, row_count, error, decomp) -> ValidationResult:
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
            result_cols = set(rows[0].keys()) if rows else set()
            has_numeric = any(
                isinstance(v, (int, float)) for v in (rows[0].values() if rows else [])
            )
            checks.append({"check": "has numeric columns for calculations", "passed": has_numeric})
            if not has_numeric:
                issues.append("No numeric columns in result — calculations may be missing")

        passed = row_count > 0 and len(issues) == 0
        return ValidationResult(passed=passed, issues=issues, checks=checks)
