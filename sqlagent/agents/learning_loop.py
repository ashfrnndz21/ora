"""LearningLoop — trace-aware learning from user feedback.

Re-exported from the standalone agents.py to resolve package shadowing.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger()


class LearningLoop:
    """Trace-aware learning: converts user feedback into prioritized training data.

    Three tiers:
      auto_learned   (priority 1) — every successful query
      user_verified  (priority 2) — thumbs up
      user_corrected (priority 3) — thumbs down + correction; highest retrieval weight
    """

    def __init__(self, example_store: Any):
        self._store = example_store

    async def on_thumbs_up(self, nl_query: str, sql: str, source_id: str = "") -> str:
        if self._store:
            return await self._store.add(
                nl_query=nl_query, sql=sql, source_id=source_id,
                generator="user_verified", verified=True,
            )
        return ""

    async def on_correction(
        self,
        nl_query: str,
        corrected_sql: str,
        original_sql: str = "",
        trace_events: list = None,
        failure_type: str = "",
        failed_node: str = "",
        correction_note: str = "",
        source_id: str = "",
    ) -> dict:
        analysis = _analyze_trace_for_failure(
            trace_events or [], failure_type, failed_node, original_sql, corrected_sql
        )

        context_parts = []
        if correction_note:
            context_parts.append(correction_note)
        if analysis["failed_stage"]:
            stage_lessons = {
                "schema": "schema pruning selected wrong tables/columns",
                "retrieval": "wrong example retrieved from memory",
                "planning": "incorrect query strategy chosen",
                "generation": "SQL logic was incorrect",
                "filtering": "WHERE/HAVING conditions were wrong",
            }
            lesson = stage_lessons.get(analysis["failed_stage"], analysis["failed_stage"])
            if not correction_note or lesson not in correction_note:
                context_parts.append(f"fix: {lesson}")
        if analysis["schema_hint"]:
            context_parts.append(f"use: {analysis['schema_hint']}")

        nl_annotated = nl_query
        if context_parts:
            nl_annotated = f"{nl_query}\n[learn: {'; '.join(context_parts)}]"

        pair_id = ""
        if self._store:
            pair_id = await self._store.add(
                nl_query=nl_annotated, sql=corrected_sql,
                source_id=source_id, generator="user_corrected", verified=True,
            )

        return {
            "pair_id": pair_id,
            "failed_stage": analysis["failed_stage"],
            "schema_hint": analysis["schema_hint"],
            "message": _correction_message(analysis),
        }


def _analyze_trace_for_failure(
    trace_events: list, failure_type: str, failed_node: str,
    original_sql: str, corrected_sql: str,
) -> dict:
    FAILURE_TYPE_TO_STAGE = {
        "wrong_tables": "schema", "wrong_columns": "schema",
        "bad_example": "retrieval", "wrong_plan": "planning",
        "wrong_logic": "generation", "wrong_filter": "filtering",
        "wrong_aggregation": "generation",
    }
    NODE_TO_STAGE = {
        "prune": "schema", "retrieve": "retrieval", "plan": "planning",
        "generate": "generation", "execute": "execution", "correct": "correction",
    }

    failed_stage = ""
    if failed_node:
        failed_stage = NODE_TO_STAGE.get(failed_node, failed_node)
    elif failure_type:
        failed_stage = FAILURE_TYPE_TO_STAGE.get(failure_type, failure_type)
    elif trace_events:
        for evt in trace_events:
            if evt.get("status") == "failed":
                failed_stage = NODE_TO_STAGE.get(evt.get("node", ""), "")
                break
        if not failed_stage:
            failed_stage = "generation"

    schema_hint = _extract_schema_hint(original_sql, corrected_sql)
    return {"failed_stage": failed_stage, "schema_hint": schema_hint}


def _extract_schema_hint(original_sql: str, corrected_sql: str) -> str:
    if not corrected_sql or not original_sql:
        return ""
    try:
        pattern = re.compile(r'\b(?:FROM|JOIN)\s+(["`\w]+(?:\.["`\w]+)?)', re.IGNORECASE)
        orig_tables = {t.strip('`"').lower() for t in pattern.findall(original_sql)}
        corr_tables = {t.strip('`"').lower() for t in pattern.findall(corrected_sql)}
        new_tables = corr_tables - orig_tables
        if new_tables:
            return "table: " + ", ".join(sorted(new_tables))
    except Exception:
        pass
    return ""


def _correction_message(analysis: dict) -> str:
    MESSAGES = {
        "schema": "Schema context updated — agent will include correct tables next time",
        "retrieval": "Example store updated — correction surfaced first on similar queries",
        "planning": "Query strategy saved — agent will plan differently for this pattern",
        "generation": "SQL pattern registered — agent will generate correct SQL next time",
        "filtering": "Filter logic saved — agent will apply correct conditions",
        "": "Correction registered — agent will improve on similar queries",
    }
    return MESSAGES.get(analysis.get("failed_stage", ""), MESSAGES[""])
