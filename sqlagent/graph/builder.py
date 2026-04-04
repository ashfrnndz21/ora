"""Compile the LangGraph StateGraph for query orchestration.

REACT PIPELINE (v2.0 final):

  ora_react → respond → learn

That's it. Ora does EVERYTHING:
  - Calls Semantic Agent (iterative reasoning + Schema Agent interaction)
  - Gets schema context
  - Builds query spec
  - Calls SQL Agent (generates SQL)
  - Validates SQL (fixes values inline)
  - Executes SQL
  - If error: diagnoses, fixes, retries at the right level
  - If empty: re-reasons, adjusts filters, retries
  - Evolves semantic layer on success
  → Only passes to Respond when it has a good answer or exhausted attempts

NO separate generate/execute/correct/validate nodes.
NO conditional routing.
ONE node does all the thinking. Then respond + learn.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from sqlagent.graph.state import QueryState
from sqlagent.graph.ora_react import ora_react
from sqlagent.graph.nodes import (
    make_respond_node,
    make_learn_node,
)


def compile_query_graph(services: Any) -> Any:
    """Compile the ReAct query orchestration graph.

    Three nodes:
      ora_react → respond → learn → END

    Args:
        services: PipelineServices with llm, connectors, ensemble, policy, etc.

    Returns:
        A compiled LangGraph.
    """
    graph = StateGraph(QueryState)

    # ── Ora ReAct — the single orchestrator ──────────────────────────────
    async def _ora_node(state: QueryState) -> dict:
        return await ora_react(state, services)

    graph.add_node("ora", _ora_node)
    graph.add_node("respond", make_respond_node(services))
    graph.add_node("learn", make_learn_node(services))

    # ── Linear flow ──────────────────────────────────────────────────────
    graph.set_entry_point("ora")
    graph.add_edge("ora", "respond")
    graph.add_edge("respond", "learn")
    graph.add_edge("learn", END)

    return graph.compile()
