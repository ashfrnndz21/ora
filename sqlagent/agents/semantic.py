"""Semantic Agent — iterative entity resolution with protocol interface.

Wraps the existing reason_about_query with the AgentRequest/AgentResponse protocol.
Receives decomposition from Ora, returns resolved entities with confidence.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from sqlagent.agents.protocol import AgentRequest, AgentResponse
from sqlagent.graph.ora_react import AgentTrace

logger = structlog.get_logger()


class SemanticAgent:
    """Resolves user terms to exact DB values via iterative reasoning."""

    def __init__(self, services: Any):
        self._services = services

    async def handle(self, request: AgentRequest, trace: AgentTrace) -> AgentResponse:
        """Process a resolution request from Ora."""
        start = time.monotonic()

        query = request.context.get("query", "")
        decomposition = request.context.get("decomposition", {})
        source_ids = request.context.get("source_ids", [])
        workspace_id = request.context.get("workspace_id", "")

        # Load semantic contexts
        from sqlagent.semantic_agent import load_context as load_sem_ctx
        sem_ctxs = {}
        effective_sources = source_ids or list(self._services.connectors.keys())
        for sid in effective_sources:
            ctx = load_sem_ctx(sid, workspace_id) if workspace_id else None
            if ctx:
                sem_ctxs[sid] = ctx

        # Call the iterative reasoning function
        from sqlagent.semantic_agent import reason_about_query
        reasoning = await reason_about_query(
            question=query,
            source_ids=effective_sources,
            workspace_id=workspace_id,
            connectors=self._services.connectors,
            llm=self._services.llm,
            semantic_contexts=sem_ctxs if sem_ctxs else None,
        )

        latency = int((time.monotonic() - start) * 1000)

        # Build AgentResponse
        unresolved = []
        if reasoning.confidence < 0.5:
            unresolved.append("low_confidence")

        # Record to trace
        trace.record(
            agent="semantic_agent",
            action="Resolved entities",
            status="completed" if reasoning.confidence >= 0.5 else "partial",
            input_context=f"Query: {query[:80]}, Decomposition: {len(decomposition.get('parts', []))} parts",
            output=(
                f"Filters: {len(reasoning.filters)}, "
                f"Tables: {reasoning.tables}, "
                f"Metrics: {reasoning.metrics}, "
                f"Aliases: {reasoning.new_aliases}"
            ),
            reasoning=reasoning.reasoning,
            latency_ms=latency,
            confidence=reasoning.confidence,
        )

        return AgentResponse(
            from_agent="semantic_agent",
            status="completed" if not unresolved else "partial",
            result={
                "resolved_query": reasoning.resolved_query,
                "filters": reasoning.filters,
                "tables": reasoning.tables,
                "metrics": reasoning.metrics,
                "calculations": [],  # from reasoning if available
                "new_aliases": reasoning.new_aliases,
                "reasoning_obj": reasoning,  # pass the full object for downstream use
            },
            unresolved=unresolved,
            confidence=reasoning.confidence,
            reasoning=reasoning.reasoning,
        )
