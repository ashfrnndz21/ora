"""Compile the LangGraph StateGraph for query orchestration.

SIMPLIFIED PIPELINE (v2.0):

  ora → generate → execute → validate → respond → learn
                      ↓ error
                    correct → execute (retry)

ONE path. No routing decisions. No fan_out/synthesize/decompose.

Ora does ALL the thinking:
  - Calls Semantic Agent (iterative reasoning with Schema Agent interaction)
  - Calls Schema Agent (prune relevant tables/columns)
  - Retrieves similar examples from memory
  - Produces a COMPLETE query specification

SQL Agent does ALL the writing:
  - Receives Ora's structured query spec
  - Translates to syntactically correct SQL
  - Handles simple, compound, multi-table, JOIN queries — all the same way

After execution:
  - Validate checks if result matches the question
  - Correct retries on SQL errors or validation mismatches
  - Respond generates NL summary + confidence + chart
  - Learn saves everything to the semantic layer
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from sqlagent.graph.state import QueryState
from sqlagent.graph.nodes import (
    make_ora_node,
    make_generate_node,
    make_execute_node,
    make_correct_node,
    make_validate_node,
    make_respond_node,
    make_learn_node,
    # Legacy nodes still importable but not in the main graph:
    # make_prune_node, make_retrieve_node, make_plan_node,
    # make_fan_out_node, make_synthesize_node,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def route_after_execute(state: QueryState) -> str:
    """After execution: success → validate, error → correct (if budget allows)."""
    error = state.get("execution_error", "")
    if not error:
        return "validate"

    correction_round = state.get("correction_round", 0)
    max_corrections = state.get("max_corrections", 3)
    if correction_round < max_corrections:
        return "correct"

    return "validate"  # max corrections exceeded — validate anyway (will pass to respond)


def route_after_validate(state: QueryState) -> str:
    """After validation: passed → respond, mismatch → correct for retry."""
    error = state.get("execution_error", "")
    if error and "Validation:" in error:
        correction_round = state.get("correction_round", 0)
        max_corrections = state.get("max_corrections", 3)
        if correction_round < max_corrections:
            return "correct"
    return "respond"


def route_after_correct(state: QueryState) -> str:
    """After correction: always retry execution."""
    return "execute"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════


def compile_query_graph(services: Any) -> Any:
    """Compile the simplified query orchestration graph.

    ONE linear path:
      ora → generate → execute → validate → respond → learn

    With correction loop:
      execute → correct → execute (retry, up to max_corrections)

    Args:
        services: PipelineServices with llm, connectors, ensemble, policy, etc.

    Returns:
        A compiled LangGraph that can be invoked with:
            result = await graph.ainvoke(initial_state)
    """
    graph = StateGraph(QueryState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("ora", make_ora_node(services))
    graph.add_node("generate", make_generate_node(services))
    graph.add_node("execute", make_execute_node(services))
    graph.add_node("correct", make_correct_node(services))
    graph.add_node("validate", make_validate_node(services))
    graph.add_node("respond", make_respond_node(services))
    graph.add_node("learn", make_learn_node(services))

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("ora")

    # ── Linear flow ──────────────────────────────────────────────────────────
    graph.add_edge("ora", "generate")
    graph.add_edge("generate", "execute")

    # ── After execute: validate (success) or correct (error) ─────────────────
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "validate": "validate",
            "correct": "correct",
        },
    )

    # ── After validate: respond (passed) or correct (mismatch) ───────────────
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "respond": "respond",
            "correct": "correct",
        },
    )

    # ── After correct: retry execution ───────────────────────────────────────
    graph.add_conditional_edges(
        "correct",
        route_after_correct,
        {
            "execute": "execute",
        },
    )

    # ── Terminal ─────────────────────────────────────────────────────────────
    graph.add_edge("respond", "learn")
    graph.add_edge("learn", END)

    return graph.compile()
