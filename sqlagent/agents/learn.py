"""Learn Agent — evolves the semantic layer after successful queries."""

from __future__ import annotations

import time

import structlog

from sqlagent.agents.protocol import AgentRequest, AgentResponse
from sqlagent.graph.ora_react import AgentTrace

logger = structlog.get_logger()


class LearnAgent:
    """Updates aliases, relationships, patterns, and column meanings."""

    async def handle(self, request: AgentRequest, trace: AgentTrace) -> AgentResponse:
        """Evolve semantic layer from a successful query."""
        start = time.monotonic()

        workspace_id = request.context.get("workspace_id", "")
        semantic_reasoning = request.context.get("semantic_reasoning")
        sources = request.context.get("sources", [])
        sql = request.context.get("sql", "")
        nl_query = request.context.get("nl_query", "")

        learned = {"aliases": 0, "relationships": 0, "patterns": 0}

        if workspace_id and semantic_reasoning:
            try:
                from sqlagent.semantic_agent import evolve_semantic_layer, strengthen_alias

                result = evolve_semantic_layer(
                    workspace_id=workspace_id,
                    query_result={
                        "semantic_reasoning": semantic_reasoning,
                        "target_sources": sources,
                        "sql": sql,
                        "nl_query": nl_query,
                        "succeeded": True,
                    },
                )
                learned.update(result)

                # Strengthen aliases
                new_aliases = semantic_reasoning.get("new_aliases", {})
                if new_aliases:
                    for alias, canonical in new_aliases.items():
                        for sid in sources:
                            strengthen_alias(workspace_id, sid, alias, canonical)

            except Exception as exc:
                logger.debug("learn.evolve_failed", error=str(exc))

        latency = int((time.monotonic() - start) * 1000)

        trace.record("learn", "Semantic layer updated",
                     output=f"Aliases: {learned.get('aliases',0)}, "
                            f"Relationships: {learned.get('relationships',0)}, "
                            f"Patterns: {learned.get('patterns',0)}",
                     latency_ms=latency)

        return AgentResponse(
            from_agent="learn", status="completed",
            result=learned, confidence=1.0,
            reasoning="Semantic layer evolved",
        )
